"""
retrieve_corpus.py — Top-k retrieval demo over the ChromaDB vector store.

IntelliHomes RAG pipeline — Retrieval stage (query -> top-k chunks).

Demonstrates the complete retrieval step for a sample user query:

    TASK 1 — the query is embedded with the SAME embedding backend that
             produced the indexed chunk vectors (mode/model are read from
             the embeddings store header);
    TASK 2 — a top-k similarity search runs against the ChromaDB collection
             and returns the most relevant chunks;
    TASK 3 — every hit is reported with its similarity score, source text
             and metadata (source document, section, chunk position);
    TASK 4 — the same query is run with several k values to show how the
             retrieved results change (and that k is clamped to the size
             of the collection).

Environment:
    CHROMA_PATH        ChromaDB persist directory (default: chroma_db).
    COLLECTION_NAME    Collection name (default: property_chunks).
    EMBEDDINGS_STORE   Path of the embeddings JSON store (default:
                       ../data/embeddings/cleaned_corpus-embeddings.json).
    SAMPLE_QUERY       Query to run (default: "What document proves legal
                       ownership of a property?").
    K_VALUES           Comma-separated k values to demonstrate
                       (default: "1,3,5").
    RETRIEVAL_OUTPUT   Sample output path (default: retrieval_sample_output.txt).

Usage
-----
    python scripts/retrieve_corpus.py          # from backend/
    python scripts/retrieve_corpus.py "query"  # custom query, default k values
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

from retrieval.vector_search import (  # noqa: E402
    default_store_path,
    embed_query,
    search,
    store_header,
)

# ── Configuration (everything from the environment) ───────────────────────

SAMPLE_QUERY = os.environ.get(
    "SAMPLE_QUERY",
    "What document proves legal ownership of a property?",
)
K_VALUES = [int(k) for k in os.environ.get("K_VALUES", "1,3,5").split(",") if k.strip()]
RETRIEVAL_OUTPUT = os.environ.get("RETRIEVAL_OUTPUT", "retrieval_sample_output.txt")


# ── Retrieval runs ────────────────────────────────────────────────────────


def run_retrieval(
    query: str,
    k_values: list[int],
    *,
    mode: str | None = None,
    store_path: str | Path | None = None,
    client=None,
    path: str | None = None,
    name: str | None = None,
) -> dict:
    """Run the full retrieval demo and return a structured result dict.

    *query* is embedded once (same backend as the chunks) and searched with
    every k in *k_values*. The returned dict feeds :func:`build_report`.
    """
    store = store_header(store_path)
    if mode is None:
        mode = store.get("mode")

    # Task 1 — embed the user query (same backend that produced the chunks).
    _, vectors, stats = embed_query([query], mode=mode)
    query_vector = vectors[0]

    # Tasks 2 + 3 + 4 — top-k searches for every requested k.
    searches = {}
    for k in sorted(set(k_values)):
        searches[k] = search(
            query,
            k,
            mode=mode,
            store_path=store_path,
            client=client,
            path=path,
            name=name,
        )

    return {
        "query": query,
        "mode": searches[k_values[0]]["mode"] if searches else mode,
        "model": searches[k_values[0]]["model"] if searches else store.get("model"),
        "dim": len(query_vector) if query_vector else 0,
        "query_vector": query_vector,
        "stats": stats,
        "store": {
            "corpus": store.get("corpus"),
            "chunk_count": store.get("chunk_count", store.get("total_chunks")),
            "endpoint": store.get("endpoint"),
        },
        "searches": searches,
    }


def _format_metadata(metadata: dict) -> str:
    """Render metadata as a compact `key=value, …` line."""
    return ", ".join(f"{key}={value}" for key, value in sorted(metadata.items()))


def build_report(data: dict) -> str:
    """Render the retrieval demo report (printed and saved to a file)."""
    lines: list[str] = []
    bar = "=" * 62

    lines.append(bar)
    lines.append("TOP-K RETRIEVAL — QUERY -> CHROMA VECTOR STORE")
    lines.append(bar)
    lines.append(f"Query       : {data['query']!r}")
    lines.append(f"Mode        : {data['mode']}")
    lines.append(f"Model       : {data['model']}")
    if data["store"].get("corpus"):
        lines.append(f"Corpus      : {data['store']['corpus']} "
                     f"({data['store'].get('chunk_count', '?')} chunks)")
    if data["store"].get("endpoint"):
        lines.append(f"Endpoint    : {data['store']['endpoint']}")
    lines.append(f"Vector dim  : {data['dim']}")

    lines.append("\n" + "-" * 62)
    lines.append("TASK 1 — EMBED THE USER QUERY (same model as the chunks)")
    lines.append("-" * 62)
    head = ", ".join(f"{v:+.4f}" for v in data["query_vector"][:8])
    lines.append(f"\nQuery vector first 8 values:")
    lines.append(f"  [{head}, …]  (len {data['dim']})")

    for k, result in sorted(data["searches"].items()):
        lines.append("\n" + "-" * 62)
        lines.append(
            f"TASKS 2 + 3 — TOP-{k} SEARCH "
            f"(k requested={result['requested_k']}, returned={result['k']}, "
            f"chunks in DB={result['total_chunks']})"
        )
        lines.append("-" * 62)
        if not result["results"]:
            lines.append("\n(no chunks in the collection to retrieve)")
            continue
        for i, hit in enumerate(result["results"], start=1):
            lines.append(f"\n[{i}] {hit['id']}")
            lines.append(f"    score    : {hit['score']:+.4f} "
                         f"(cosine similarity; distance {hit['distance']:.4f})")
            lines.append(f"    metadata : {_format_metadata(hit['metadata'])}")
            lines.append(f"    text     : {hit['text']!r}")

    lines.append("\n" + "-" * 62)
    lines.append("TASK 4 — HOW THE RESULTS CHANGE WITH k")
    lines.append("-" * 62)
    lines.append("\n  k  | returned | top-1 id                | top-1 score")
    lines.append("  ---+----------+-------------------------+------------")
    tops: dict[int, str] = {}
    for k, result in sorted(data["searches"].items()):
        if result["results"]:
            top = result["results"][0]
            tops[k] = top["id"]
            lines.append(
                f"  {k:>2} | {result['k']:>8} | {top['id']:<23} | {top['score']:+.4f}"
            )
        else:
            tops[k] = "—"
            lines.append(f"  {k:>2} | {result['k']:>8} | {'—':<23} | —")
    k_sorted = sorted(data["searches"])
    if len(k_sorted) >= 2:
        first, last = k_sorted[0], k_sorted[-1]
        same_top = tops[first] == tops[last]
        lines.append(f"\nTop-1 chunk is identical for k={first} and k={last}: "
                     f"{same_top}")
        lines.append(
            "Increasing k widens the retrieved context (more chunks returned); "
            "the ranking of the top results stays stable, and k is clamped to "
            "the number of chunks in the collection."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    query = args[0] if args else SAMPLE_QUERY

    data = run_retrieval(query, K_VALUES)
    report = build_report(data)

    out_path = Path(RETRIEVAL_OUTPUT)
    if not out_path.is_absolute():
        out_path = BACKEND_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print(report)
    print(f"\nSample output written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
