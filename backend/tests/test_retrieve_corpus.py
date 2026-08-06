import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402

from retrieval import vector_search  # noqa: E402
from retrieval.vector_search import (  # noqa: E402
    embed_query,
    open_collection,
    search,
    store_header,
)
from scripts.embed_corpus import embed_offline  # noqa: E402

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


def test_search_returns_top_k_sorted_by_score(collection):
    _, col = collection
    result = search(
        "how do I pay property taxes", k=2, client=collection[0], name=col.name
    )
    assert len(result["results"]) == 2
    scores = [hit["score"] for hit in result["results"]]
    assert scores == sorted(scores, reverse=True)
    ids = [hit["id"] for hit in result["results"]]
    assert all(chunk_id in {"a.txt#0", "b.txt#0", "c.txt#0"} for chunk_id in ids)


def test_search_includes_scores_text_and_metadata(collection):
    _, col = collection
    result = search(
        "how do I pay property taxes", k=1, client=collection[0], name=col.name
    )
    hit = result["results"][0]
    assert isinstance(hit["id"], str) and hit["id"]
    assert isinstance(hit["text"], str) and hit["text"]
    assert hit["metadata"]["source"] in {"a.txt", "b.txt", "c.txt"}
    assert hit["metadata"]["position"] == 0
    assert hit["metadata"]["section"] == "Section 1"
    assert isinstance(hit["distance"], float)
    assert isinstance(hit["score"], float)
    # cosine similarity: score = 1 - distance
    assert hit["score"] == pytest.approx(1.0 - hit["distance"], abs=1e-6)
    assert hit["score"] <= 1.0 + 1e-9


def test_k_values_change_the_retrieved_results(collection):
    _, col = collection
    k1 = search("how do I pay property taxes", k=1, client=collection[0], name=col.name)
    k3 = search("how do I pay property taxes", k=3, client=collection[0], name=col.name)
    assert len(k1["results"]) == 1
    assert len(k3["results"]) == 3
    # the top-1 chunk is stable across k, and k=1 only returns the best one
    assert k1["results"][0]["id"] == k3["results"][0]["id"]
    assert k1["results"][0]["score"] >= k3["results"][1]["score"]


def test_k_larger_than_collection_is_clamped(collection):
    _, col = collection
    result = search("anything", k=100, client=collection[0], name=col.name)
    assert result["requested_k"] == 100
    assert result["k"] == 3  # clamped to the number of chunks
    assert len(result["results"]) == 3


def test_identical_query_scores_one(collection):
    _, col = collection
    query = "How to transfer the title of a property to a new owner."
    result = search(query, k=3, client=collection[0], name=col.name)
    assert result["results"][0]["id"] == "a.txt#0"
    assert result["results"][0]["score"] >= 0.9999  # identical text -> identical vector


def test_embed_query_simulated_matches_store_mode():
    mode, vectors, stats = embed_query(["hello world"], mode="simulated")
    assert mode == "simulated"
    assert len(vectors) == 1
    assert len(vectors[0]) == 768
    assert stats["retries"] == 0


def test_search_reads_mode_from_store_header(collection, store_path):
    _, col = collection
    # No explicit mode: it must come from the store header ("simulated").
    result = search(
        "how do I pay property taxes",
        k=1,
        store_path=store_path,
        client=collection[0],
        name=col.name,
    )
    assert result["mode"] == "simulated"
    assert result["model"] == "nomic-embed-text"
    assert result["dim"] == 768


def test_store_header_returns_empty_for_missing_store(tmp_path):
    header = store_header(tmp_path / "does-not-exist.json")
    assert header == {}


def test_relevant_query_ranks_the_right_document(collection):
    _, col = collection
    result = search(
        "how do I pay property taxes", k=1, client=collection[0], name=col.name
    )
    assert result["results"][0]["id"] == "b.txt#0"
