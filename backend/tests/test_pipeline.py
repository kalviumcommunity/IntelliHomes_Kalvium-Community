"""End-to-end + per-stage tests for the query-to-answer RAG pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402

from pipeline import (  # noqa: E402
    assemble_stage,
    embed_stage,
    generate_stage,
    retrieve_stage,
    run_pipeline,
)
from retrieval.vector_search import open_collection, search  # noqa: E402
from scripts.embed_corpus import embed_offline  # noqa: E402
from scripts.history_manager import count_tokens  # noqa: E402

# A small property-domain corpus: each document is one chunk, mirroring the
# shape of the real cleaned_corpus store (3 chunks, 768-dim vectors).
DOCS = [
    ("a.txt", "How to transfer the title of a property to a new owner."),
    ("b.txt", "Property tax receipts confirm that all taxes have been paid."),
    ("c.txt", "The cafeteria menu has pasta for lunch today."),
]


@pytest.fixture
def collection(tmp_path):
    """A ChromaDB collection indexed with the simulated 768-dim embedder."""
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    _, col = open_collection(client=client)
    texts = [text for _, text in DOCS]
    col.add(
        ids=[f"{source}#0" for source, _ in DOCS],
        documents=texts,
        metadatas=[
            {"source": source, "section": "Section 1", "position": 0}
            for source, _ in DOCS
        ],
        embeddings=embed_offline(texts),
    )
    return client, col


@pytest.fixture
def store_path(tmp_path):
    """A minimal embeddings store JSON so the header can be read."""
    store = tmp_path / "embeddings.json"
    records = []
    for source, text in DOCS:
        records.append(
            {
                "id": f"{source}#0",
                "text": text,
                "metadata": {"source": source, "section": "Section 1", "position": 0},
                "vector": embed_offline([text])[0],
            }
        )
    store.write_text(
        __import__("json").dumps(
            {
                "corpus": "tmp",
                "chunk_count": len(records),
                "total_chunks": len(records),
                "dim": 768,
                "model": "nomic-embed-text",
                "endpoint": "http://localhost:11434/v1",
                "mode": "simulated",
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return store


@pytest.fixture
def client_collection(collection, store_path):
    """Return (client, collection, store_path) for the full pipeline tests."""
    client, col = collection
    return client, col, store_path


# ── Stage 1 · embed ──────────────────────────────────────────────────────


def test_embed_stage_returns_query_vector(store_path):
    result = embed_stage("how do I pay property taxes", store_path=store_path)
    assert result["query"] == "how do I pay property taxes"
    assert result["mode"] == "simulated"
    assert result["model"] == "nomic-embed-text"
    assert result["dim"] == 768
    assert len(result["vector"]) == 768
    assert all(isinstance(x, float) for x in result["vector"])


def test_embed_stage_forces_mode(store_path):
    result = embed_stage("title transfer", mode="simulated", store_path=store_path)
    assert result["mode"] == "simulated"


def test_embed_stage_rejects_empty_query():
    with pytest.raises(ValueError):
        embed_stage("   ")


# ── Stage 2 · retrieve ───────────────────────────────────────────────────


def test_retrieve_stage_consumes_pre_embedded_vector(collection, store_path):
    client, col = collection
    embedded = embed_stage("how do I pay property taxes", store_path=store_path)
    result = retrieve_stage(
        "how do I pay property taxes",
        embedded["vector"],
        k=2,
        mode=embedded["mode"],
        model=embedded["model"],
        client=client,
        name=col.name,
    )
    assert len(result["results"]) == 2
    scores = [hit["score"] for hit in result["results"]]
    assert scores == sorted(scores, reverse=True)
    # the tax query must rank the tax document first
    assert result["results"][0]["id"] == "b.txt#0"


def test_retrieve_stage_matches_search_without_vector(collection, store_path):
    client, col = collection
    embedded = embed_stage("how do I pay property taxes", store_path=store_path)
    via_vector = retrieve_stage(
        "how do I pay property taxes",
        embedded["vector"],
        k=3,
        mode=embedded["mode"],
        model=embedded["model"],
        client=client,
        name=col.name,
    )
    plain = search(
        "how do I pay property taxes",
        k=3,
        mode="simulated",
        client=client,
        name=col.name,
    )
    assert [h["id"] for h in via_vector["results"]] == [
        h["id"] for h in plain["results"]
    ]
    assert via_vector["results"][0]["score"] == pytest.approx(
        plain["results"][0]["score"]
    )


def test_retrieve_stage_rejects_empty_vector(collection, store_path):
    client, col = collection
    with pytest.raises(ValueError):
        retrieve_stage("query", [], client=client, name=col.name)


# ── Stage 3 · assemble ───────────────────────────────────────────────────


def test_assemble_stage_builds_context_and_sources(collection, store_path):
    client, col = collection
    embedded = embed_stage("how do I pay property taxes", store_path=store_path)
    retrieval = retrieve_stage(
        "how do I pay property taxes",
        embedded["vector"],
        k=2,
        mode=embedded["mode"],
        model=embedded["model"],
        client=client,
        name=col.name,
    )
    assembled = assemble_stage(retrieval)

    # context block is numbered and labelled with the source document
    assert "[1]" in assembled["context"] and "[2]" in assembled["context"]
    assert "b.txt" in assembled["context"]
    assert "score" in assembled["context"]
    # sources carry the structured records, best hit first
    assert len(assembled["sources"]) == 2
    assert assembled["sources"][0]["source"] == "b.txt"
    assert assembled["sources"][0]["id"] == "b.txt#0"
    assert assembled["sources"][0]["score"] == retrieval["results"][0]["score"]
    assert assembled["sources"][0]["text"] == retrieval["results"][0]["text"]
    assert assembled["hit_count"] == 2
    assert not assembled["truncated"]
    assert assembled["context_tokens"] > 0


def test_assemble_stage_empty_retrieval():
    assembled = assemble_stage({"results": []})
    assert assembled["context"] == ""
    assert assembled["sources"] == []
    assert assembled["context_tokens"] == 0
    assert assembled["hit_count"] == 0
    assert not assembled["truncated"]


def test_assemble_stage_token_cap_truncates_from_lowest_rank():
    retrieval = {
        "results": [
            {
                "id": "a#0",
                "text": "alpha alpha alpha",
                "metadata": {"source": "a.txt", "section": "Section 1", "position": 0},
                "score": 0.9,
            },
            {
                "id": "b#0",
                "text": "beta beta beta",
                "metadata": {"source": "b.txt", "section": "Section 1", "position": 0},
                "score": 0.8,
            },
        ]
    }
    full = assemble_stage(retrieval)
    assert full["hit_count"] == 2
    assert not full["truncated"]

    # A cap that fits the top-1 chunk but not the full block must drop the
    # lowest-ranked chunk (b.txt) and keep rank order.
    first_line = full["context"].split("\n\n")[0]
    cap = count_tokens(first_line)
    tiny = assemble_stage(retrieval, max_context_tokens=cap)
    assert tiny["truncated"] is True
    assert [s["id"] for s in tiny["sources"]] == ["a#0"]
    assert "b.txt" not in tiny["context"]
    assert full["context"].startswith(tiny["context"])
    assert tiny["context_tokens"] <= cap


# ── Stage 4 · generate ───────────────────────────────────────────────────


def _fake_llm(messages):
    """Deterministic test LLM that echoes the grounded context size."""
    system = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"), ""
    )
    return f"canned answer over {len(system)} context chars"


def test_generate_stage_with_pluggable_llm():
    context = "[1] b.txt — Section 1 (score 0.9)\nProperty tax receipts."
    result = generate_stage("how do I pay taxes", context, llm=_fake_llm)
    assert result["answer"].startswith("canned answer")
    assert result["live"] is False
    assert result["model"] == "simulated"
    assert (
        "Property tax receipts" in result["system_prompt"]
    )  # grounded context injected
    assert "how do I pay taxes" in result["user_prompt"]
    assert result["prompt_tokens"] > 0


def test_generate_stage_rejects_empty_query():
    with pytest.raises(ValueError):
        generate_stage("  ", "some context")


# ── Full pipeline · end to end ───────────────────────────────────────────


def test_run_pipeline_end_to_end(client_collection):
    client, col, store_path = client_collection
    result = run_pipeline(
        "how do I pay property taxes",
        k=3,
        store_path=store_path,
        client=client,
        name=col.name,
        llm=_fake_llm,
    )

    # every stage ran and produced its contract
    assert result["query"] == "how do I pay property taxes"
    assert result["embedding"]["mode"] == "simulated"
    assert result["embedding"]["dim"] == 768
    assert len(result["retrieval"]["results"]) == 3
    assert result["retrieval"]["results"][0]["id"] == "b.txt#0"
    assert result["context"]["hit_count"] == 3
    assert result["context"]["sources"][0]["id"] == "b.txt#0"
    assert result["answer"]["answer"].startswith("canned answer")
    assert result["answer"]["live"] is False

    # the answer's system prompt is grounded in the retrieved context
    assert "b.txt" in result["answer"]["system_prompt"]
    # returned sources match the retrieval ranking exactly
    assert [s["id"] for s in result["context"]["sources"]] == [
        h["id"] for h in result["retrieval"]["results"]
    ]
    # timing metadata is present for all four stages
    assert set(result["timings_ms"]) >= {
        "embed_ms",
        "retrieve_ms",
        "assemble_ms",
        "generate_ms",
        "total_ms",
    }


def test_run_pipeline_hybrid_and_filter(client_collection):
    client, col, store_path = client_collection
    result = run_pipeline(
        "property taxes",
        k=1,
        store_path=store_path,
        client=client,
        name=col.name,
        metadata_filter={"source": "b.txt"},
        hybrid=True,
        llm=_fake_llm,
    )
    assert len(result["retrieval"]["results"]) == 1
    assert result["retrieval"]["results"][0]["id"] == "b.txt#0"
    assert result["retrieval"]["total_matching"] == 1
    assert result["answer"]["answer"].startswith("canned answer")


def test_run_pipeline_no_results_still_generates(client_collection):
    client, col, store_path = client_collection
    result = run_pipeline(
        "how do I pay property taxes",
        k=3,
        store_path=store_path,
        client=client,
        name=col.name,
        metadata_filter={"source": "nonexistent.txt"},
        llm=_fake_llm,
    )
    assert result["retrieval"]["results"] == []
    assert result["context"]["context"] == ""
    assert result["context"]["sources"] == []
    assert result["answer"]["answer"].startswith("canned answer")
