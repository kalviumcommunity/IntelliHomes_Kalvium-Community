"""
assemble.py — Pipeline stage 3 of 4: assemble grounded context.

IntelliHomes RAG pipeline — Query-to-answer flow.

Turns the ranked retrieval hits into:

* a plain-text, numbered **context block** injected into the generation
  prompt — every chunk labelled with its source document, section and
  similarity score so the model can cite where each fact comes from; and
* a structured **sources list** returned to the caller, so the UI can show
  WHERE the answer came from (source document, section, chunk position,
  category, score) with the chunk text.

An optional token cap keeps the context inside the model's window: whole
chunks are dropped from the lowest-ranked end until the block fits (the
model never sees a half-chunk).

Contract
--------
    assemble_stage(retrieval) -> {
        "context":        str,   # numbered grounded context block
        "sources":        list,  # [{id, source, section, position, category,
                                 #   score, text}, ...] in rank order
        "context_tokens": int,   # tiktoken estimate of the context block
        "truncated":      bool,  # True if max_context_tokens dropped chunks
        "hit_count":      int,   # number of retrieved chunks considered
    }
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the module is imported
# from, so the shared token counter resolves.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.history_manager import count_tokens  # noqa: E402


def _label(hit: dict) -> str:
    """Human-readable label for one retrieval hit, e.g. ``taxes.txt — Section 2 (score 0.61)``."""
    meta = hit.get("metadata") or {}
    parts = [meta.get("source") or hit.get("id") or "unknown"]
    if meta.get("section"):
        parts.append(f"{meta['section']}")
    if hit.get("score") is not None:
        parts.append(f"score {hit['score']}")
    return " — ".join(parts)


def _source_entry(hit: dict) -> dict:
    """Structured source record for one retrieval hit."""
    meta = hit.get("metadata") or {}
    return {
        "id": hit.get("id"),
        "source": meta.get("source"),
        "section": meta.get("section"),
        "position": meta.get("position"),
        "category": meta.get("category"),
        "score": hit.get("score"),
        "text": hit.get("text"),
    }


def assemble_stage(
    retrieval: dict,
    *,
    max_context_tokens: int | None = None,
) -> dict:
    """Build the grounded context block and sources list from *retrieval*.

    *retrieval* is the dict returned by ``retrieve_stage`` (or
    ``retrieval.vector_search.search``): its ``results`` are ranked best
    first. When *max_context_tokens* is given, whole chunks are dropped from
    the end (lowest-ranked) until the block fits.
    """
    hits = retrieval.get("results", [])
    if not hits:
        return {
            "context": "",
            "sources": [],
            "context_tokens": 0,
            "truncated": False,
            "hit_count": 0,
        }

    lines = []
    sources = []
    for i, hit in enumerate(hits, start=1):
        sources.append(_source_entry(hit))
        lines.append(f"[{i}] {_label(hit)}\n{hit.get('text', '')}")

    context = "\n\n".join(lines)
    truncated = False

    if max_context_tokens is not None:
        kept = []
        total = 0
        for line in lines:
            cost = count_tokens(line)
            if total + cost > max_context_tokens:
                truncated = True
                break
            kept.append(line)
            total += cost
        if truncated:
            context = "\n\n".join(kept)
            sources = sources[: len(kept)]

    return {
        "context": context,
        "sources": sources,
        "context_tokens": count_tokens(context),
        "truncated": truncated,
        "hit_count": len(sources),
    }
