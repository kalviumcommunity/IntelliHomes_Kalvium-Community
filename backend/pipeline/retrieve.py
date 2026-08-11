"""
retrieve.py — Pipeline stage 2 of 4: retrieve the most relevant chunks.

IntelliHomes RAG pipeline — Query-to-answer flow.

Consumes the vector produced by :func:`pipeline.embed.embed_stage` and runs a
top-k similarity search against the ChromaDB collection. Reuses the proven
ranking from :func:`retrieval.vector_search.search` (cosine similarity,
optional metadata filter, k clamped to the collection size) but WITHOUT
re-embedding the query: the stage-1 vector is passed straight to the vector
store, so a full pipeline run embeds the query exactly once.

Contract
--------
    retrieve_stage(query, vector) ->  # same shape as retrieval.vector_search.search
        {
          "query", "requested_k", "k", "mode", "model", "dim",
          "total_chunks", "total_matching", "metadata_filter",
          "results": [{"id", "text", "metadata", "distance", "score"}, ...],
        }

Environment (passed through to the vector store):
    CHROMA_PATH / COLLECTION_NAME / EMBEDDINGS_STORE
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the module is imported
# from, so the shared retrieval helpers resolve.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from retrieval.vector_search import hybrid_search, search  # noqa: E402


def retrieve_stage(
    query: str,
    vector: list[float],
    *,
    k: int = 3,
    mode: str | None = None,
    model: str | None = None,
    client=None,
    path: str | None = None,
    name: str | None = None,
    metadata_filter: dict | None = None,
    hybrid: bool = False,
    vector_weight: float = 1.0,
    keyword_weight: float = 0.3,
) -> dict:
    """Return the top-*k* chunks most similar to the pre-embedded *vector*.

    *vector* is the query embedding produced by ``embed_stage``; passing it
    in means the query is embedded exactly once per pipeline run.

    *hybrid* switches from pure vector search to the hybrid scorer (vector
    similarity + lexical overlap) — useful when the query contains exact
    terms, names or IDs that the vectors might under-weight.

    Returns the same shape as :func:`retrieval.vector_search.search`.
    """
    if not vector:
        raise ValueError("retrieve_stage requires a non-empty query vector")

    if hybrid:
        return hybrid_search(
            query,
            k,
            vector=vector,
            mode=mode,
            model=model,
            client=client,
            path=path,
            name=name,
            metadata_filter=metadata_filter,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )
    return search(
        query,
        k,
        vector=vector,
        mode=mode,
        model=model,
        client=client,
        path=path,
        name=name,
        metadata_filter=metadata_filter,
    )
