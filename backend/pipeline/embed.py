"""
embed.py — Pipeline stage 1 of 4: embed the user query.

IntelliHomes RAG pipeline — Query-to-answer flow.

Turns the raw user query into a vector in the SAME space as the indexed
document chunks. The embedding backend (mode, model, dimension) is read
from the embeddings store header — exactly like the indexing stage embedded
the chunks — so the query vector is always comparable with the chunk
vectors (live vectors are never mixed with simulated ones).

This stage produces the vector that stage 2 (retrieve) consumes, so the
query is embedded exactly once per pipeline run.

Contract
--------
    embed_stage(query) -> {
        "query":  str,             # the input query
        "vector": list[float],     # query embedding, ready for retrieval
        "mode":   "live" | "simulated",
        "model":  str,             # embedding model used
        "dim":    int,             # vector length
        "stats":  dict,            # embedding backend stats (retries, ...)
    }

Environment (passed through to the embedding backend):
    OPENAI_API_KEY / EMBEDDING_MODEL / OPENAI_BASE_URL / EMBEDDINGS_STORE
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the module is imported
# from, so the shared retrieval helpers resolve.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from retrieval.vector_search import embed_query, store_header  # noqa: E402


def embed_stage(
    query: str,
    *,
    mode: str | None = None,
    store_path: str | Path | None = None,
) -> dict:
    """Embed *query* into the chunk vector space (stage 1 of the pipeline).

    *mode* may force an embedding backend ("live" or "simulated"); when
    ``None`` it is read from the embeddings store header, so the query is
    embedded with the same backend that produced the indexed chunks.

    Raises ``ValueError`` for an empty query and ``RuntimeError`` when the
    backend produced no vector.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    store = store_header(store_path)
    if mode is None:
        # Match the embedding backend that produced the indexed chunks.
        mode = store.get("mode")
    model = store.get("model") or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

    resolved_mode, vectors, stats = embed_query([query], mode=mode)
    vector = vectors[0]
    if vector is None:
        raise RuntimeError(
            f"embedding failed for the query (mode={resolved_mode!r}); "
            "no vector was produced"
        )

    return {
        "query": query,
        "vector": vector,
        "mode": resolved_mode,
        "model": model,
        "dim": len(vector),
        "stats": stats,
    }
