"""
embed_corpus.py — Embed a prepared corpus into vectors and store them.

IntelliHomes RAG pipeline — Embedding stage.

Takes a folder of plain-text documents (the prepared corpus), splits each
document into chunks, embeds every chunk through an OpenAI-compatible
embeddings API, and stores each vector together with the source text and the
metadata needed for retrieval (source document, section, chunk position).

Outputs
-------
* JSON store      — every chunk: id, text, metadata {source, section, position}
                    and its embedding vector (full precision, for retrieval).
* Sample output   — human-readable verification report: number of chunks
                    embedded, vector length, sample vector values, and every
                    stored record with its trimmed vector.

Configuration (environment variables; nothing secret/model-related is
hardcoded):
    OPENAI_API_KEY    API key for the embeddings endpoint (Ollama ignores it).
    EMBEDDING_MODEL   Embedding model name (default: nomic-embed-text).
    OPENAI_BASE_URL   Base URL of an OpenAI-compatible endpoint
                      (default: http://localhost:11434/v1 — local Ollama).
    EMBEDDING_DIM     Optional expected vector length; the run fails if the API
                      returns vectors of a different size.
    CORPUS_DIR        Folder with the prepared corpus (default: documents).
    EMBEDDING_OUTPUT  Path for the JSON vector store
                      (default: ../data/embeddings/<corpus>-embeddings.json).
    SAMPLE_OUTPUT     Path for the sample verification output
                      (default: embeddings_sample_output.txt).

Usage
-----
    python scripts/embed_corpus.py                    # from backend/
    python scripts/embed_corpus.py cleaned_corpus     # any prepared corpus

The script works against any OpenAI-compatible endpoint. If no endpoint is
reachable it falls back to a deterministic offline embedder (labelled
"simulated") so the pipeline still runs for demos and tests.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the script is run from,
# so the other pipeline stages (ingestion.loader, chunking.*) resolve.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from chunking.metadata import attach_metadata  # noqa: E402
from chunking.strategies import paragraph_chunk  # noqa: E402
from ingestion.loader import load_folder, print_intake_report  # noqa: E402


# ── Configuration (Task 3: everything from the environment) ───────────────

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")  # Ollama accepts any key
EXPECTED_DIM = os.environ.get("EMBEDDING_DIM")  # optional validation
CORPUS_DIR = os.environ.get("CORPUS_DIR", "documents")
EMBEDDING_OUTPUT = os.environ.get("EMBEDDING_OUTPUT", "")  # "" -> default
SAMPLE_OUTPUT = os.environ.get("SAMPLE_OUTPUT", "")


def _resolve(path: str, default: Path) -> Path:
    """Resolve *path* against BACKEND_ROOT; fall back to *default*."""
    if not path:
        return default
    p = Path(path).expanduser()
    return p if p.is_absolute() else BACKEND_ROOT / p


# ── Task 2: chunking with retrieval metadata ──────────────────────────────


def chunk_corpus(documents: list) -> list[dict]:
    """Split documents into chunks and attach source/section/position metadata.

    Returns one record per chunk::

        {"id": "ownership.txt#0",
         "text": "…",
         "metadata": {"source": "ownership.txt",
                      "section": "Section 1",
                      "position": 0}}
    """
    records: list[dict] = []
    for doc in documents:
        for item in attach_metadata(paragraph_chunk(doc.text), doc.source):
            records.append(
                {
                    "id": f"{doc.source}#{item['metadata']['position']}",
                    "text": item["text"],
                    "metadata": item["metadata"],
                }
            )
    return records


# ── Task 1: embeddings through an OpenAI-compatible API ───────────────────


def embed_live(
    texts: list[str],
    *,
    model: str = EMBEDDING_MODEL,
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed *texts* with a real model via an OpenAI-compatible endpoint."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors.extend(list(item.embedding) for item in ordered)
    return vectors


def embed_offline(texts: list[str]) -> list[list[float]]:
    """Deterministic offline fallback (fastText-style, 768-dim)."""
    from scripts.embeddings_demo import simulated_embed

    return simulated_embed(texts)


def embed_corpus_chunks(
    texts: list[str],
    *,
    use_live: bool = True,
) -> tuple[str, list[list[float]]]:
    """Embed chunk texts, preferring the live API and falling back offline.

    Returns ``(mode, vectors)`` where *mode* is "live" or "simulated".
    """
    if use_live:
        try:
            return "live", embed_live(texts)
        except Exception as exc:  # noqa: BLE001 - endpoint unreachable etc.
            print(f"WARNING: live embedding failed ({exc}); using offline fallback")
    return "simulated", embed_offline(texts)


def attach_vectors(records: list[dict], vectors: list[list[float]]) -> list[dict]:
    """Merge the embedding vectors into their corresponding chunk records."""
    if len(records) != len(vectors):
        raise ValueError(
            f"chunk/text count mismatch: {len(records)} records vs {len(vectors)} vectors"
        )
    for record, vector in zip(records, vectors):
        record["vector"] = vector
    return records


# ── Storage (Task 2: vectors + source text + metadata) ────────────────────


def save_store(store: dict, path: str | Path) -> None:
    """Write the vector store (records + header) as indented JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, ensure_ascii=False)


def load_store(path: str | Path) -> dict:
    """Read a store written by :func:`save_store`."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Verification output (Task 4) ──────────────────────────────────────────


def _trimmed(vector: list[float], head: int = 4, tail: int = 2) -> str:
    """Render a trimmed vector: ``[ v0, v1, …, vn-2, vn-1 ] (len N)``."""
    parts = [f"{v:+.4f}" for v in vector[:head]]
    parts.append("…")
    parts.extend(f"{v:+.4f}" for v in vector[-tail:])
    return f"[ {', '.join(parts)} ] (len {len(vector)})"


def build_sample_output(store: dict) -> str:
    """Render the verification report for the run (printed and saved)."""
    records = store["records"]
    dim = store["dim"]
    uniform = all(len(r["vector"]) == dim for r in records)
    lines: list[str] = []
    bar = "=" * 62

    lines.append(bar)
    lines.append("EMBEDDING PIPELINE — CORPUS -> VECTORS (SAMPLE OUTPUT)")
    lines.append(bar)
    lines.append(f"Corpus      : {store['corpus']}")
    lines.append(f"Documents   : {store['documents']}")
    lines.append(f"Endpoint    : {store['endpoint']}")
    lines.append(f"Model       : {store['model']}")
    lines.append(f"Mode        : {store['mode']}")

    lines.append("\n" + "-" * 62)
    lines.append("TASK 1 — VECTORS RETURNED WITH EXPECTED DIMENSION")
    lines.append("-" * 62)
    lines.append(f"\nChunks embedded      : {store['chunk_count']}")
    lines.append(f"Vector length        : {dim}")
    lines.append(f"All vectors same len : {uniform}")
    if store.get("expected_dim"):
        match = dim == int(store["expected_dim"])
        lines.append(
            f"Expected dimension   : {store['expected_dim']} -> "
            f"{'match' if match else 'MISMATCH !!'}"
        )

    lines.append("\n" + "-" * 62)
    lines.append("TASK 2 — SAMPLE VECTOR VALUES")
    lines.append("-" * 62)
    first = records[0]
    head = ", ".join(f"{v:+.4f}" for v in first["vector"][:8])
    lines.append(f"\nfirst chunk {first['id']!r} first 8 values:")
    lines.append(f"  [{head}, …]")

    lines.append("\n" + "-" * 62)
    lines.append("TASK 3 — STORED RECORDS (text + metadata + trimmed vector)")
    lines.append("-" * 62)
    for i, rec in enumerate(records):
        lines.append(f"\n[{i}] {rec['id']}")
        lines.append(f"    text     : {rec['text']!r}")
        lines.append(f"    metadata : {rec['metadata']}")
        lines.append(f"    vector   : {_trimmed(rec['vector'])}")

    lines.append("\n" + bar)
    lines.append("SUMMARY")
    lines.append(bar)
    lines.append(f"\nChunks embedded : {store['chunk_count']}")
    lines.append(f"Vector length   : {dim}")
    lines.append(f"All same length : {uniform}")
    lines.append(f"Store written to: {store['output_file']}")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    corpus_arg = args[0] if args else CORPUS_DIR
    corpus_path = _resolve(corpus_arg, BACKEND_ROOT / "documents")

    if not corpus_path.is_dir():
        print(f"ERROR: corpus directory not found: {corpus_path}")
        return 1

    print(f"Loading corpus from {corpus_path} …")
    result = load_folder(corpus_path)
    if not result.documents:
        print("ERROR: no documents loaded from the corpus directory")
        return 1
    print_intake_report(result)

    records = chunk_corpus(result.documents)
    texts = [r["text"] for r in records]
    if not texts:
        print("ERROR: corpus produced no chunks to embed")
        return 1

    mode, vectors = embed_corpus_chunks(texts)
    print(
        f"\nEmbedded {len(texts)} chunks "
        f"({'live: ' + EMBEDDING_MODEL if mode == 'live' else 'simulated'})"
    )

    # Task 1: confirm every vector has the expected dimension.
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        print(f"ERROR: mixed vector lengths returned (expected all {dim})")
        return 1
    if EXPECTED_DIM and dim != int(EXPECTED_DIM):
        print(
            f"ERROR: expected dimension {EXPECTED_DIM} "
            f"but the API returned {dim}"
        )
        return 1

    attach_vectors(records, vectors)

    corpus_name = corpus_path.name
    default_store = (
        BACKEND_ROOT.parent / "data" / "embeddings" / f"{corpus_name}-embeddings.json"
    )
    store_path = _resolve(EMBEDDING_OUTPUT, default_store)
    store = {
        "corpus": corpus_name,
        "documents": len(result.documents),
        "chunk_count": len(records),
        "dim": dim,
        "expected_dim": EXPECTED_DIM,
        "model": EMBEDDING_MODEL,
        "endpoint": BASE_URL,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_file": str(store_path),
        "records": records,
    }
    save_store(store, store_path)

    sample_path = _resolve(SAMPLE_OUTPUT, BACKEND_ROOT / "embeddings_sample_output.txt")
    sample = build_sample_output(store)
    sample_path.write_text(sample + "\n", encoding="utf-8")

    # Task 4: print the verification output.
    print(sample)
    print(f"\nSample output saved to {sample_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
