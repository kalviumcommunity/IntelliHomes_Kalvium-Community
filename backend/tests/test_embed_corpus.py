import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.loader import Document
from scripts.embed_corpus import (
    attach_vectors,
    build_sample_output,
    chunk_corpus,
    embed_offline,
    load_store,
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
