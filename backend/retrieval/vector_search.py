"""
vector_search.py — Top-k semantic retrieval over the ChromaDB vector store.

IntelliHomes RAG pipeline — Retrieval stage.

Given a user query, this module:

1. embeds the query with the SAME embedding backend that produced the
   indexed document chunks (the embedding mode/model are read from the
   embeddings store header, so the query and the chunks live in the same
   vector space);
2. runs a top-k similarity search against the ChromaDB collection;
3. returns each hit with its similarity score, source text and metadata
   (source document, section, chunk position) so a later stage can ground
   the model's answer in the retrieved context.

Scores are cosine similarities: ChromaDB is configured with the cosine
space, in which it returns a cosine *distance* (0 for identical vectors),
and we report ``score = 1 - distance`` so higher always means more similar.

Environment:
    CHROMA_PATH        ChromaDB persist directory (default: chroma_db).
    COLLECTION_NAME    Collection name (default: property_chunks).
    EMBEDDINGS_STORE   Path of the embeddings JSON store (default:
                       ../data/embeddings/cleaned_corpus-embeddings.json).
    OPENAI_API_KEY / EMBEDDING_MODEL / OPENAI_BASE_URL — passed through to
                       the embedding backend (see scripts/embed_corpus.py).

Usage
-----
    from retrieval.vector_search import search

    result = search("What document proves legal ownership?", k=3)
    for hit in result["results"]:
        print(hit["id"], hit["score"], hit["metadata"]["source"], hit["text"])
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the module is imported
# from, so the shared embedding helpers in scripts.embed_corpus resolve.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import chromadb  # noqa: E402

from scripts.embed_corpus import (  # noqa: E402
    embed_corpus_chunks,
    embed_live,
    embed_offline,
    load_store,
)

# ── Configuration (everything from the environment) ───────────────────────

CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "property_chunks")
EMBEDDINGS_STORE = os.environ.get("EMBEDDINGS_STORE", "")


# ── Embeddings store helpers ──────────────────────────────────────────────


def default_store_path() -> Path:
    """Default embeddings JSON store written by scripts/embed_corpus.py."""
    if EMBEDDINGS_STORE:
        p = Path(EMBEDDINGS_STORE).expanduser()
        return p if p.is_absolute() else BACKEND_ROOT / p
    return BACKEND_ROOT.parent / "data" / "embeddings" / "cleaned_corpus-embeddings.json"


def store_header(store_path: str | Path | None = None) -> dict:
    """Return the embeddings store document (header + records), or {}.

    The header records how the indexed chunks were embedded (mode, model,
    dimension). It is used to embed the query in the same vector space. An
    unreadable store degrades gracefully to {} so retrieval can still run.
    """
    path = Path(store_path) if store_path else default_store_path()
    try:
        return load_store(path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {}


# ── Task 1: embed the user query ──────────────────────────────────────────


def embed_query(
    texts: list[str], *, mode: str | None = None
) -> tuple[str, list[list[float]], dict]:
    """Embed *texts* with the same backend used for the document chunks.

    *mode* selects the backend:

    * ``None``       — try the live endpoint (Ollama/OpenAI) and fall back to
                       the simulated embedder, exactly like embed_corpus.py.
    * ``"simulated"``— offline deterministic fastText-style embedder (matches
                       a store generated without a reachable endpoint).
    * ``"live"``     — live endpoint only; raises if it fails.

    Returns ``(mode, vectors, stats)``, vectors aligned with *texts*.
    """
    if mode == "simulated":
        return (
            "simulated",
            embed_offline(texts),
            {"retries": 0, "failed_batches": 0, "failed_starts": []},
        )
    if mode == "live":
        vectors, stats = embed_live(texts)
        if any(v is None for v in vectors):
            raise RuntimeError("live embedding failed for the query")
        return "live", vectors, stats
    return embed_corpus_chunks(texts)


# ── Vector database helpers ───────────────────────────────────────────────


def open_collection(
    client: chromadb.ClientAPI | None = None,
    *,
    path: str | None = None,
    name: str | None = None,
) -> tuple[chromadb.ClientAPI, chromadb.Collection]:
    """Return ``(client, collection)`` for the persistent ChromaDB store.

    The collection is created with the cosine space and *no* default
    embedding function: this pipeline supplies its own vectors (produced by
    embed_corpus.py / embed_query), so ChromaDB must never embed text itself.
    """
    db_path = Path(path or CHROMA_PATH).expanduser()
    if not db_path.is_absolute():
        db_path = BACKEND_ROOT / db_path
    client = client or chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name or COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    return client, collection


# ── Task 2 + 3: top-k similarity search with scores and metadata ──────────


def search(
    query: str,
    k: int,
    *,
    mode: str | None = None,
    store_path: str | Path | None = None,
    client: chromadb.ClientAPI | None = None,
    path: str | None = None,
    name: str | None = None,
) -> dict:
    """Embed *query* and return the top-*k* most similar chunks in the store.

    Returns a dict::

        {"query": "…", "requested_k": 5, "k": 3, "mode": "simulated",
         "model": "nomic-embed-text", "dim": 768, "total_chunks": 3,
         "results": [
            {"id": "property_guide.txt#0", "text": "…",
             "metadata": {"source": "property_guide.txt",
                          "section": "Section 1", "position": 0},
             "distance": 0.123, "score": 0.877},
            …]}

    * *score*    — cosine similarity of the query and the chunk
                   (``1 - cosine distance``; higher is more similar).
    * *text*     — the chunk's source text (what later grounds the answer).
    * *metadata* — source document, section and chunk position.

    The query is embedded with the same mode/model used for the chunks
    (read from the store header when *mode* is None). *k* is clamped to the
    number of chunks in the collection.
    """
    store = store_header(store_path)
    if mode is None:
        # Match the embedding backend that produced the indexed chunks, so
        # the query vector and the chunk vectors live in the same space.
        mode = store.get("mode")
    model = store.get("model") or os.environ.get(
        "EMBEDDING_MODEL", "nomic-embed-text"
    )

    _, vectors, _stats = embed_query([query], mode=mode)
    vector = vectors[0]

    _, collection = open_collection(client=client, path=path, name=name)
    total = collection.count()
    effective_k = max(0, min(int(k), total))

    if effective_k == 0:
        return {
            "query": query,
            "requested_k": int(k),
            "k": 0,
            "mode": mode,
            "model": model,
            "dim": len(vector),
            "total_chunks": total,
            "results": [],
        }

    result = collection.query(
        query_embeddings=[vector],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    results = []
    for i, chunk_id in enumerate(ids):
        results.append(
            {
                "id": chunk_id,
                "text": documents[i],
                "metadata": metadatas[i] or {},
                "distance": round(float(distances[i]), 6),
                "score": round(1.0 - float(distances[i]), 6),
            }
        )

    return {
        "query": query,
        "requested_k": int(k),
        "k": effective_k,
        "mode": mode,
        "model": model,
        "dim": len(vector),
        "total_chunks": total,
        "results": results,
    }
