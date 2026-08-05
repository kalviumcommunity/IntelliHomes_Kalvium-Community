import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.embed_corpus as embed_corpus  # noqa: E402
from ingestion.loader import Document  # noqa: E402
from scripts.embed_corpus import (  # noqa: E402
    _embed_batch_with_retry,
    _is_retryable,
    attach_vectors,
    build_run_summary,
    build_sample_output,
    chunk_corpus,
    embed_live,
    embed_offline,
    estimate_cost,
    estimate_tokens,
    existing_records,
    load_store,
    partition_cached,
    price_per_1m_tokens,
    save_store,
)

DOCS = [
    Document(
        source="a.txt",
        path="/x/a.txt",
        format="text",
        text="First paragraph of A.\n\nSecond paragraph of A.",
    ),
    Document(
        source="b.txt",
        path="/x/b.txt",
        format="text",
        text="Only one paragraph in B.",
    ),
]


def test_chunk_corpus_count_and_metadata():
    chunks = chunk_corpus(DOCS)
    assert len(chunks) == 3  # a.txt -> 2, b.txt -> 1
    assert chunks[0]["metadata"] == {
        "source": "a.txt",
        "section": "Section 1",
        "position": 0,
    }
    assert chunks[1]["metadata"]["section"] == "Section 2"
    assert chunks[1]["metadata"]["position"] == 1
    assert chunks[2]["metadata"]["source"] == "b.txt"
    assert chunks[2]["text"] == "Only one paragraph in B."


def test_chunk_ids_are_unique():
    ids = [c["id"] for c in chunk_corpus(DOCS)]
    assert len(ids) == len(set(ids))
    assert ids[0] == "a.txt#0"


def test_embed_offline_uniform_dimension():
    chunks = chunk_corpus(DOCS)
    vectors = embed_offline([c["text"] for c in chunks])
    dims = {len(v) for v in vectors}
    assert len(dims) == 1
    assert dims.pop() == 768


def test_embed_offline_is_deterministic():
    texts = [c["text"] for c in chunk_corpus(DOCS)]
    assert embed_offline(texts) == embed_offline(texts)


def test_attach_vectors_length_mismatch_raises():
    chunks = chunk_corpus(DOCS)
    try:
        attach_vectors(chunks, [[0.0] * 3])
    except ValueError:
        return
    raise AssertionError("expected ValueError on count mismatch")


def test_store_round_trip_preserves_records(tmp_path):
    chunks = chunk_corpus(DOCS)
    vectors = embed_offline([c["text"] for c in chunks])
    attach_vectors(chunks, vectors)
    store = {"corpus": "test", "dim": 768, "chunk_count": 3, "records": chunks}
    path = tmp_path / "store.json"
    save_store(store, path)
    reloaded = load_store(path)
    assert reloaded["chunk_count"] == 3
    assert reloaded["records"][0]["id"] == "a.txt#0"
    assert reloaded["records"][0]["vector"] == chunks[0]["vector"]


def test_sample_output_reports_count_dim_and_values():
    chunks = chunk_corpus(DOCS)
    vectors = embed_offline([c["text"] for c in chunks])
    attach_vectors(chunks, vectors)
    store = {
        "corpus": "test",
        "documents": 2,
        "chunk_count": 3,
        "dim": 768,
        "expected_dim": None,
        "model": "simulated",
        "endpoint": "n/a",
        "mode": "simulated",
        "output_file": "store.json",
        "records": chunks,
    }
    report = build_sample_output(store)
    assert "Chunks embedded      : 3" in report
    assert "Vector length        : 768" in report
    assert "All vectors same len : True" in report
    assert "TASK 2 — SAMPLE VECTOR VALUES" in report
    assert "first 8 values" in report
    assert "a.txt#0" in report
    assert "(len 768)" in report
    assert "'First paragraph of A.'" in report
    assert "TASK 4 — RUN SUMMARY (batching, retries, cost, skips)" in report
    assert "RUN SUMMARY" in report


# ── Task 2: retry with backoff ────────────────────────────────────────────


class FakeHTTPError(Exception):
    """Minimal stand-in for an OpenAI API error carrying an HTTP status."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, vectors):
        self.data = [_FakeItem(i, v) for i, v in enumerate(vectors)]


class FakeClient:
    """Records every batch it receives and serves canned vectors."""

    def __init__(self, vectors_factory):
        self.embeddings = _FakeEmbeddings(vectors_factory)


class _FakeEmbeddings:
    def __init__(self, vectors_factory):
        self.vectors_factory = vectors_factory  # callable(batch) -> vectors
        self.batches = []

    def create(self, model, input):
        self.batches.append(list(input))
        return _FakeResponse(self.vectors_factory(list(input)))


class FlakyClient:
    """Raises *error* until ``succeed_after`` calls, then serves vectors."""

    def __init__(self, error, succeed_after=None):
        self.embeddings = self
        self.error = error
        self.succeed_after = succeed_after
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        if self.succeed_after is None or self.calls <= self.succeed_after:
            raise self.error
        return _FakeResponse([[0.5] * 3 for _ in list(input)])


class _NoJitter:
    """Deterministic stand-in for ``random``: no jitter, no real sleeps."""

    @staticmethod
    def uniform(_a, _b):
        return 0.0


def _no_sleep(_seconds):
    return None


def test_is_retryable_status_codes():
    assert _is_retryable(FakeHTTPError(429))  # rate limit
    assert _is_retryable(FakeHTTPError(500))  # server error
    assert _is_retryable(FakeHTTPError(503))
    assert not _is_retryable(FakeHTTPError(400))  # permanent
    assert not _is_retryable(FakeHTTPError(401))
    assert not _is_retryable(FakeHTTPError(403))
    assert _is_retryable(ConnectionError("refused"))  # transport
    assert _is_retryable(TimeoutError("slow"))


def test_embed_batch_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(embed_corpus, "sleep", _no_sleep)
    monkeypatch.setattr(embed_corpus, "random", _NoJitter)
    client = FlakyClient(FakeHTTPError(429), succeed_after=2)
    vectors, retries = _embed_batch_with_retry(
        client, "m", ["a", "b"], max_retries=5, base_delay=0.01
    )
    assert retries == 2
    assert client.calls == 3  # initial attempt + 2 retries
    assert vectors == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]


def test_embed_batch_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(embed_corpus, "sleep", _no_sleep)
    monkeypatch.setattr(embed_corpus, "random", _NoJitter)
    client = FlakyClient(FakeHTTPError(429))
    with pytest.raises(FakeHTTPError):
        _embed_batch_with_retry(client, "m", ["a"], max_retries=2, base_delay=0.01)
    assert client.calls == 3  # initial attempt + 2 retries, then give up


def test_embed_batch_permanent_error_does_not_retry(monkeypatch):
    monkeypatch.setattr(embed_corpus, "sleep", _no_sleep)
    client = FlakyClient(FakeHTTPError(400))
    with pytest.raises(FakeHTTPError):
        _embed_batch_with_retry(client, "m", ["a"], max_retries=5, base_delay=0.01)
    assert client.calls == 1  # permanent errors never retried


# ── Task 1: batching ──────────────────────────────────────────────────────


def test_embed_live_batches_texts_and_orders_by_index(monkeypatch):
    def factory(batch):
        # Reversed values to prove vectors are re-sorted by index.
        return [[float(i)] * 2 for i in reversed(range(len(batch)))]

    client = FakeClient(factory)
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)

    texts = [f"t{i}" for i in range(10)]
    vectors, stats = embed_live(texts, batch_size=4, max_retries=0, base_delay=0)

    assert client.embeddings.batches == [
        ["t0", "t1", "t2", "t3"],
        ["t4", "t5", "t6", "t7"],
        ["t8", "t9"],
    ]
    assert len(vectors) == 10
    assert all(v is not None for v in vectors)
    assert stats["failed_batches"] == 0
    assert vectors[0] == [3.0, 3.0]  # index 0 -> value 3.0 after sorting


def test_embed_live_records_failed_batches(monkeypatch):
    client = FlakyClient(FakeHTTPError(429))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(embed_corpus, "sleep", _no_sleep)
    monkeypatch.setattr(embed_corpus, "random", _NoJitter)

    texts = [f"t{i}" for i in range(6)]
    vectors, stats = embed_live(texts, batch_size=4, max_retries=1, base_delay=0.01)

    assert vectors == [None] * 6  # failed positions marked, run continues
    assert stats["failed_batches"] == 2
    assert stats["failed_starts"] == [0, 4]
    assert stats["retries"] == 2  # one retry per batch


# ── Task 3: totals and approximate cost ───────────────────────────────────


def test_price_per_1m_tokens_known_and_local():
    assert price_per_1m_tokens("text-embedding-3-small") == 0.02
    assert price_per_1m_tokens("text-embedding-3-large") == 0.13
    assert price_per_1m_tokens("nomic-embed-text") == 0.0  # local = free


def test_price_override_wins(monkeypatch):
    monkeypatch.setattr(embed_corpus, "PRICE_OVERRIDE", "0.05")
    assert price_per_1m_tokens("anything") == 0.05


def test_estimate_cost_scales_with_tokens():
    price, cost = estimate_cost(1_000_000, "text-embedding-3-small")
    assert price == 0.02
    assert cost == pytest.approx(0.02)
    assert estimate_cost(2_000_000, "text-embedding-3-small")[1] == pytest.approx(0.04)


def test_estimate_tokens_counts_something():
    assert estimate_tokens(["hello world"]) >= 2
    assert estimate_tokens([]) == 0


def test_build_run_summary_reports_totals_cost_and_skips():
    store = {
        "total_chunks": 100,
        "chunk_count": 97,
        "embedded": 90,
        "skipped": 7,
        "failed": 3,
        "batches": 4,
        "batch_size": 64,
        "tokens": 50_000,
        "mode": "live",
        "model": "text-embedding-3-small",
        "stats": {"retries": 4, "failed_batches": 1, "failed_starts": [64]},
        "output_file": "store.json",
    }
    summary = build_run_summary(store)
    assert "Total chunks      : 100" in summary
    assert "Embedded          : 90" in summary
    assert "Skipped (cached)  : 7" in summary
    assert "Failed            : 3" in summary
    assert "Retries (live)    : 4" in summary
    assert "Approx. tokens    : 50,000" in summary
    assert "$0.0010" in summary  # 50k tokens * $0.02 / 1M
    assert "Failed batches at : [64]" in summary


# ── Task 4: skip already-embedded chunks on re-runs ───────────────────────


def test_partition_cached_skips_matching_and_reembeds_changed(tmp_path):
    records = [
        {"id": "a.txt#0", "text": "same", "metadata": {}},
        {"id": "a.txt#1", "text": "changed", "metadata": {}},
        {"id": "b.txt#0", "text": "new", "metadata": {}},
    ]
    cached = {
        "a.txt#0": {"id": "a.txt#0", "text": "same", "vector": [0.1]},
        "a.txt#1": {"id": "a.txt#1", "text": "old text", "vector": [0.2]},
    }
    to_embed, skipped = partition_cached(records, cached)
    assert [r["id"] for r in skipped] == ["a.txt#0"]
    assert [r["id"] for r in to_embed] == ["a.txt#1", "b.txt#0"]


def test_existing_records_reads_previous_store(tmp_path):
    path = tmp_path / "store.json"
    assert existing_records(path) == {}  # no store yet
    save_store({"records": [{"id": "a.txt#0", "text": "x", "vector": [1.0]}]}, path)
    records = existing_records(path)
    assert records["a.txt#0"]["text"] == "x"


def test_main_rerun_skips_cached_chunks(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text(
        "Alpha paragraph.\n\nBeta paragraph.", encoding="utf-8"
    )

    def fake_embedder(texts, **kwargs):
        vectors = [[float(i)] * 3 for i in range(len(texts))]
        return "live", vectors, {"retries": 0, "failed_batches": 0, "failed_starts": []}

    monkeypatch.setattr(embed_corpus, "CORPUS_DIR", str(corpus))
    monkeypatch.setattr(embed_corpus, "EMBEDDING_OUTPUT", str(tmp_path / "store.json"))
    monkeypatch.setattr(embed_corpus, "SAMPLE_OUTPUT", str(tmp_path / "sample.txt"))
    monkeypatch.setattr(embed_corpus, "embed_corpus_chunks", fake_embedder)

    # First run: everything is embedded.
    assert embed_corpus.main([]) == 0
    first = capsys.readouterr().out
    assert "2 to embed, 0 skipped" in first

    # Re-run: both chunks already cached, nothing sent to the API.
    assert embed_corpus.main([]) == 0
    second = capsys.readouterr().out
    assert "0 to embed, 2 skipped" in second

    store = load_store(tmp_path / "store.json")
    assert store["skipped"] == 2
    assert store["embedded"] == 0
    assert store["tokens"] == 0
    assert store["cost_usd"] == 0.0
    assert len(store["records"]) == 2
