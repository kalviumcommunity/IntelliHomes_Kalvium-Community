"""
runner.py — Compose the four pipeline stages into one query-to-answer flow.

IntelliHomes RAG pipeline — Query-to-answer orchestrator.

    user query
        │
        ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  1. embed   │───▶│  2. retrieve│───▶│  3. assemble│───▶│  4. generate│
    │  embed_stage│    │retrieve_... │    │ assemble_...│    │ generate_...│
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
        query vector     top-k chunks      context + sources     answer

Each stage is a small, independently testable function (pipeline.embed /
pipeline.retrieve / pipeline.assemble / pipeline.generate); this module only
wires them together and reports the full trace (stages + timings).

Contract
--------
    run_pipeline(query, k=3) -> {
        "query":      str,           # the user question
        "embedding":  {...},         # stage 1 output (mode/model/dim/stats)
        "retrieval":  {...},         # stage 2 output (top-k chunks + scores)
        "context":    {...},         # stage 3 output (context block + sources)
        "answer":     {...},         # stage 4 output (answer, model, prompts)
        "timings_ms": {...},         # per-stage wall-clock time
        "total_ms":   float,
    }
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the module is imported
# from, so the stage modules resolve.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pipeline.assemble import assemble_stage  # noqa: E402
from pipeline.embed import embed_stage  # noqa: E402
from pipeline.generate import generate_stage  # noqa: E402
from pipeline.retrieve import retrieve_stage  # noqa: E402


def run_pipeline(
    query: str,
    *,
    k: int = 3,
    mode: str | None = None,
    store_path: str | Path | None = None,
    client=None,
    path: str | None = None,
    name: str | None = None,
    metadata_filter: dict | None = None,
    hybrid: bool = False,
    vector_weight: float = 1.0,
    keyword_weight: float = 0.3,
    max_context_tokens: int | None = None,
    llm=None,
    live: bool | None = None,
    model: str | None = None,
) -> dict:
    """Run the full query-to-answer RAG flow for *query*.

    Keyword arguments mirror the individual stages (see each stage's
    docstring); the ones used most often are *k* (how many chunks to
    retrieve), *metadata_filter* (scope retrieval to a source/category),
    *hybrid* (vector+keyword re-ranking), *llm* (plug in a custom model
    callable, e.g. in tests) and *live* (force live/simulated generation).
    """
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    embedding = embed_stage(query, mode=mode, store_path=store_path)
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    retrieval = retrieve_stage(
        query,
        embedding["vector"],
        k=k,
        mode=embedding["mode"],
        model=embedding["model"],
        client=client,
        path=path,
        name=name,
        metadata_filter=metadata_filter,
        hybrid=hybrid,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    context = assemble_stage(retrieval, max_context_tokens=max_context_tokens)
    timings["assemble_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    answer = generate_stage(
        query,
        context["context"],
        llm=llm,
        live=live,
        model=model,
    )
    timings["generate_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    timings["total_ms"] = round(sum(timings.values()), 2)

    return {
        "query": query,
        "embedding": embedding,
        "retrieval": retrieval,
        "context": context,
        "answer": answer,
        "timings_ms": timings,
    }
