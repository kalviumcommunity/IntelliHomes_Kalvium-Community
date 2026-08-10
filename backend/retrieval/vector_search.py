"""
vector_search.py — Top-k semantic retrieval over the ChromaDB vector store.

IntelliHomes RAG pipeline — Retrieval stage.

Given a user query, this module:

1. embeds the query with the SAME embedding backend that produced the
   indexed document chunks (the embedding mode/model are read from the
   embeddings store header, so the query and the chunks live in the same
   vector space);
2. runs a top-k similarity search against the ChromaDB collection, scoped
   by an optional metadata filter (source, section, category, …) so
   retrieval can be restricted to the right document type before ranking;
3. returns each hit with its similarity score, source text and metadata
   (source document, section, chunk position, category) so a later stage
   can ground the model's answer in the retrieved context;
4. offers a hybrid search that re-ranks the vector candidates with
   keyword/lexical matching, so exact terms, names and IDs can be boosted
   even when the vector similarity is low.

Scores are cosine similarities: ChromaDB is configured with the cosine
space, in which it returns a cosine *distance* (0 for identical vectors),
and we report ``score = 1 - distance`` so higher always means more similar.
Hybrid scores combine that cosine score with a keyword overlap score (see
:func:`hybrid_search`).

Environment:
    CHROMA_PATH        ChromaDB persist directory (default: chroma_db).
    COLLECTION_NAME    Collection name (default: property_chunks).
    EMBEDDINGS_STORE   Path of the embeddings JSON store (default:
                       ../data/embeddings/cleaned_corpus-embeddings.json).
    OPENAI_API_KEY / EMBEDDING_MODEL / OPENAI_BASE_URL — passed through to
                       the embedding backend (see scripts/embed_corpus.py).

Usage
-----
    from retrieval.vector_search import search, hybrid_search

    # Unfiltered top-k over the whole corpus.
    result = search("What document proves legal ownership?", k=3)
    for hit in result["results"]:
        print(hit["id"], hit["score"], hit["metadata"]["source"], hit["text"])

    # Same query scoped to one source/category before ranking.
    taxes = search("What document proves legal ownership?", k=3,
                   metadata_filter={"category": "tax"})

    # Hybrid: vector + keyword matching, e.g. for exact IDs.
    hybrid = hybrid_search("inspection report INSP-2024-001", k=3)
"""

from __future__ import annotations

import json
import os
import re
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
    return (
        BACKEND_ROOT.parent / "data" / "embeddings" / "cleaned_corpus-embeddings.json"
    )


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


def _count_matching(
    collection: chromadb.Collection, metadata_filter: dict | None
) -> int:
    """Number of chunks in *collection* satisfying *metadata_filter*.

    ``Collection.count`` has no ``where`` argument in chromadb 1.5.9, so we
    count matches through ``get(where=…)`` (ids only, no documents or
    embeddings are loaded).
    """
    if not metadata_filter:
        return collection.count()
    return len(collection.get(where=metadata_filter, include=[])["ids"])


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
    metadata_filter: dict | None = None,
) -> dict:
    """Embed *query* and return the top-*k* most similar chunks in the store.

    *metadata_filter* (optional) restricts retrieval to chunks whose metadata
    matches a ChromaDB ``where`` clause — e.g. ``{"source": "taxes.txt"}`` or
    ``{"category": "legal"}`` — so searches can be scoped to a source,
    section, document type or date range before similarity ranking.

    Returns a dict::

        {"query": "…", "requested_k": 5, "k": 3, "mode": "simulated",
         "model": "nomic-embed-text", "dim": 768, "total_chunks": 3,
         "total_matching": 1, "metadata_filter": {"source": "taxes.txt"},
         "results": [
            {"id": "property_guide.txt#0", "text": "…",
             "metadata": {"source": "property_guide.txt",
                          "section": "Section 1", "position": 0},
             "distance": 0.123, "score": 0.877},
            …]}

    * *score*    — cosine similarity of the query and the chunk
                   (``1 - cosine distance``; higher is more similar).
    * *text*     — the chunk's source text (what later grounds the answer).
    * *metadata* — source document, section, chunk position and (when
                   indexed) category.
    * *total_matching* — chunks in the collection satisfying
                   *metadata_filter* (equals *total_chunks* when unfiltered).

    The query is embedded with the same mode/model used for the chunks
    (read from the store header when *mode* is None). *k* is clamped to the
    number of matching chunks in the collection.
    """
    store = store_header(store_path)
    if mode is None:
        # Match the embedding backend that produced the indexed chunks, so
        # the query vector and the chunk vectors live in the same space.
        mode = store.get("mode")
    model = store.get("model") or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

    _, vectors, _stats = embed_query([query], mode=mode)
    vector = vectors[0]

    _, collection = open_collection(client=client, path=path, name=name)
    total = collection.count()
    total_matching = _count_matching(collection, metadata_filter)
    effective_k = max(0, min(int(k), total_matching))

    base = {
        "query": query,
        "requested_k": int(k),
        "k": effective_k,
        "mode": mode,
        "model": model,
        "dim": len(vector),
        "total_chunks": total,
        "total_matching": total_matching,
        "metadata_filter": dict(metadata_filter) if metadata_filter else None,
    }

    if effective_k == 0:
        return {**base, "results": []}

    query_kwargs: dict = {
        "query_embeddings": [vector],
        "n_results": effective_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    result = collection.query(**query_kwargs)

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

    return {**base, "results": results}


# ── Task 4: metadata introspection ────────────────────────────────────────


def metadata_value_counts(
    client: chromadb.ClientAPI | None = None,
    *,
    path: str | None = None,
    name: str | None = None,
    key: str | None = None,
) -> dict:
    """Count distinct metadata values across the collection.

    With *key* (e.g. ``"source"``) returns ``{value: count}`` for that key;
    without, returns ``{key: {value: count}}`` for every metadata key. Useful
    for showing which filters are available before running a filtered search.
    """
    _, collection = open_collection(client=client, path=path, name=name)
    metadatas = collection.get(include=["metadatas"])["metadatas"]

    if key:
        counts: dict = {}
        for meta in metadatas:
            value = (meta or {}).get(key)
            counts[value] = counts.get(value, 0) + 1
        return counts

    nested: dict = {}
    for meta in metadatas:
        for k, value in sorted((meta or {}).items()):
            nested.setdefault(k, {})
            nested[k][value] = nested[k].get(value, 0) + 1
    return nested


# ── Task 3: keyword / hybrid matching ─────────────────────────────────────


def keyword_score(
    query: str, text: str, *, exact_phrase_bonus: float = 0.25
) -> float:
    """Lexical overlap of the *query* terms with *text*, in [0, 1].

    Tokenizes both sides into lowercase alphanumeric terms and returns the
    fraction of unique query terms that appear in *text* (so exact terms,
    names, IDs and code-like tokens match even when the vectors disagree).
    When the whole query appears verbatim in *text*, an *exact_phrase_bonus*
    is added (capped at 1.0) to reward exact-phrase hits.
    """
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    matched = sum(1 for term in query_terms if term in text_lower)
    score = matched / len(query_terms)
    if exact_phrase_bonus:
        phrase = re.sub(r"\s+", " ", query.strip().lower())
        if phrase and phrase in text_lower:
            score = min(1.0, score + exact_phrase_bonus)
    return round(score, 6)


def hybrid_search(
    query: str,
    k: int,
    *,
    mode: str | None = None,
    store_path: str | Path | None = None,
    client: chromadb.ClientAPI | None = None,
    path: str | None = None,
    name: str | None = None,
    metadata_filter: dict | None = None,
    vector_weight: float = 1.0,
    keyword_weight: float = 0.3,
    candidate_multiplier: int = 3,
) -> dict:
    """Vector search re-ranked with keyword matching (hybrid scoring).

    Retrieves a candidate pool of the top ``candidate_multiplier * k`` chunks
    from :func:`search` (or all matching chunks when the pool is smaller),
    then re-ranks them by a weighted combination of vector similarity and
    lexical overlap::

        hybrid_score = vector_weight * cosine_score
                     + keyword_weight * keyword_score(query, text)

    Exact terms, names and IDs get a lexical boost even when the vector
    similarity is low, so hybrid search can rescue chunks a pure top-k misses.
    Returns the same shape as :func:`search`, with each hit carrying both
    component scores: ``vector_score``, ``keyword_score`` and the combined
    ``score``.
    """
    store = store_header(store_path)
    if mode is None:
        mode = store.get("mode")
    model = store.get("model") or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

    _, vectors, _stats = embed_query([query], mode=mode)
    vector = vectors[0]

    _, collection = open_collection(client=client, path=path, name=name)
    total = collection.count()
    total_matching = _count_matching(collection, metadata_filter)
    if total_matching == 0:
        return {
            "query": query,
            "requested_k": int(k),
            "k": 0,
            "mode": mode,
            "model": model,
            "dim": len(vector),
            "total_chunks": total,
            "total_matching": total_matching,
            "metadata_filter": dict(metadata_filter) if metadata_filter else None,
            "hybrid": {"vector_weight": vector_weight,
                       "keyword_weight": keyword_weight},
            "results": [],
        }

    candidate_k = max(int(k), min(total_matching, int(k) * candidate_multiplier))
    query_kwargs: dict = {
        "query_embeddings": [vector],
        "n_results": candidate_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    got = collection.query(**query_kwargs)

    scored = []
    for i, chunk_id in enumerate(got["ids"][0]):
        text = got["documents"][0][i]
        vector_score = 1.0 - float(got["distances"][0][i])
        lexical = keyword_score(query, text)
        combined = vector_weight * vector_score + keyword_weight * lexical
        scored.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": got["metadatas"][0][i] or {},
                "distance": round(1.0 - vector_score, 6),
                "vector_score": round(vector_score, 6),
                "keyword_score": lexical,
                "score": round(combined, 6),
            }
        )

    scored.sort(key=lambda hit: hit["score"], reverse=True)
    results = scored[: int(k)]

    return {
        "query": query,
        "requested_k": int(k),
        "k": len(results),
        "mode": mode,
        "model": model,
        "dim": len(vector),
        "total_chunks": total,
        "total_matching": total_matching,
        "metadata_filter": dict(metadata_filter) if metadata_filter else None,
        "hybrid": {
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            "candidates_scored": len(scored),
        },
        "results": results,
    }
