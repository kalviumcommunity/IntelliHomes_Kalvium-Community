"""
run_pipeline.py — Run the full query-to-answer RAG pipeline end to end.

IntelliHomes RAG pipeline — Query-to-answer demo.

Demonstrates the complete flow for a sample user query:

    TASK 1 — the query is embedded with the SAME embedding backend that
             produced the indexed chunk vectors (mode/model are read from
             the embeddings store header);
    TASK 2 — a top-k similarity search retrieves the most relevant chunks
             (k=3 by default), using the stage-1 query vector directly;
    TASK 3 — the retrieved chunks are assembled into a numbered, grounded
             context block plus a structured sources list;
    TASK 4 — the grounded prompt is rendered and an answer is generated
             (live OpenAI-compatible endpoint when OPENAI_API_KEY is set,
             otherwise a deterministic simulated reply);
    TASK 5 — the answer AND the sources it was grounded on are printed and
             written to a sample output file.

Environment:
    SAMPLE_QUERY       Query to run (default: "What document proves legal
                       ownership of a property?").
    PIPELINE_K         Number of chunks to retrieve (default: 3).
    PIPELINE_OUTPUT    Sample output path (default: pipeline_sample_output.txt).
    OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL — generation backend
    EMBEDDING_MODEL / EMBEDDINGS_STORE — embedding backend
    CHROMA_PATH / COLLECTION_NAME — vector store

Usage
-----
    python scripts/run_pipeline.py                 # from backend/
    python scripts/run_pipeline.py "custom query"  # extra queries, same k
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the script is run from.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pipeline.runner import run_pipeline  # noqa: E402

# ── Configuration (everything from the environment) ───────────────────────

SAMPLE_QUERY = os.environ.get(
    "SAMPLE_QUERY",
    "What document proves legal ownership of a property?",
)
PIPELINE_K = int(os.environ.get("PIPELINE_K", "3"))
PIPELINE_OUTPUT = os.environ.get("PIPELINE_OUTPUT", "pipeline_sample_output.txt")


# ── Human-readable report ─────────────────────────────────────────────────


def _format_sources(sources: list[dict]) -> str:
    lines = []
    for i, source in enumerate(sources, start=1):
        location = source.get("source") or "?"
        if source.get("section"):
            location += f" — {source['section']}"
        lines.append(
            f"    [{i}] {location}  (score {source.get('score')})\n"
            f"        chunk: {source.get('id')}\n"
            f"        text:  {source.get('text')}"
        )
    return "\n".join(lines)


def build_sample_output(query: str, result: dict) -> str:
    """Render the pipeline run as a readable report (committed artifact)."""
    lines = [
        "=" * 72,
        "IntelliHomes RAG pipeline — query-to-answer run",
        "=" * 72,
        f"Query   : {query}",
        f"k       : {result['retrieval']['requested_k']}",
        "",
        "── STAGE 1 · EMBED ────────────────────────────────────────────────",
        f"  mode   : {result['embedding']['mode']}",
        f"  model  : {result['embedding']['model']}",
        f"  dim    : {result['embedding']['dim']}",
        f"  stats  : {result['embedding']['stats']}",
        "",
        "── STAGE 2 · RETRIEVE ─────────────────────────────────────────────",
        f"  top-{result['retrieval']['k']} of {result['retrieval']['total_chunks']} chunks "
        f"(mode={result['retrieval']['mode']}, model={result['retrieval']['model']})",
        "",
        "── STAGE 3 · ASSEMBLE ─────────────────────────────────────────────",
        f"  context tokens: {result['context']['context_tokens']}  "
        f"(truncated: {result['context']['truncated']})",
        "",
        "  Grounded context passed to the model:",
        (
            "  " + result["context"]["context"].replace("\n", "\n  ")
            if result["context"]["context"]
            else "  (no context retrieved)"
        ),
        "",
        "── STAGE 4 · GENERATE ─────────────────────────────────────────────",
        f"  model : {result['answer']['model']}  (live={result['answer']['live']})",
        f"  tokens: {result['answer']['prompt_tokens']}",
        "",
        "  Answer:",
        f"  {result['answer']['answer']}",
        "",
        "── RETURNED SOURCES ───────────────────────────────────────────────",
        (
            _format_sources(result["context"]["sources"])
            if result["context"]["sources"]
            else "  (no sources)"
        ),
        "",
        "── TIMINGS ────────────────────────────────────────────────────────",
        f"  {result['timings_ms']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    queries = list(argv or [])
    if not queries:
        queries = [SAMPLE_QUERY]

    output_path = Path(PIPELINE_OUTPUT).expanduser()
    if not output_path.is_absolute():
        output_path = BACKEND_ROOT / output_path

    sections = []
    for query in queries:
        print(f"Running pipeline for: {query!r}  (k={PIPELINE_K})")
        result = run_pipeline(query, k=PIPELINE_K)
        sections.append(build_sample_output(query, result))

    report = "\n".join(sections)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Sample output written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
