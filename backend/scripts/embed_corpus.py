"""
embed_corpus.py — Batch-embed a prepared corpus into vectors and store them.

IntelliHomes RAG pipeline — Embedding stage.

Takes a folder of plain-text documents (the prepared corpus), splits each
document into chunks, embeds every chunk through an OpenAI-compatible
embeddings API, and stores each vector together with the source text and the
metadata needed for retrieval (source document, section, chunk position).

Batch pipeline behaviour
------------------------
* Batching          — chunks are sent in batches of EMBEDDING_BATCH_SIZE
                      (default 64) instead of one API call per chunk.
* Retry + backoff   — rate-limit (429) and transient server/transport errors
                      are retried with exponential backoff; a run summary
                      reports how many retries happened.
* Skip on re-run    — chunks whose id and text already exist in the previous
                      store are skipped, so re-runs do not pay for duplicate
                      API calls.
* Cost estimate     — the run reports an approximate token count and dollar
                      cost (configurable per model; free for local models).
* Failures reported — batches that exhaust their retries are counted as
                      failures and listed in the run summary instead of
                      silently producing bad vectors.

Outputs
-------
* JSON store      — every chunk: id, text, metadata {source, section, position}
                    and its embedding vector (full precision, for retrieval).
* Sample output   — human-readable verification report: number of chunks
                    embedded, vector length, sample vector values, every
                    stored record with its trimmed vector, and a run summary.
* Run summary     — totals, retries, failures, skips and approximate cost,
                    printed to stdout and included in the sample output.

Configuration (environment variables; nothing secret/model-related is
hardcoded):
    OPENAI_API_KEY             API key for the embeddings endpoint
                               (Ollama ignores it).
    EMBEDDING_MODEL            Embedding model name
                               (default: nomic-embed-text).
    OPENAI_BASE_URL            Base URL of an OpenAI-compatible endpoint
                               (default: http://localhost:11434/v1 — Ollama).
    EMBEDDING_DIM              Optional expected vector length; the run fails
                               if the API returns vectors of a different size.
    CORPUS_DIR                 Folder with the prepared corpus
                               (default: cleaned_corpus).
    EMBEDDING_OUTPUT           Path for the JSON vector store
                               (default: ../data/embeddings/<corpus>-embeddings.json).
    SAMPLE_OUTPUT              Path for the sample verification output
                               (default: embeddings_sample_output.txt).
    EMBEDDING_BATCH_SIZE       Chunks per API request (default: 64).
    EMBEDDING_MAX_RETRIES      Retries per batch on transient errors
                               (default: 5).
    EMBEDDING_RETRY_BASE_DELAY Backoff base in seconds; delay doubles per
                               retry (default: 1.0).
    EMBEDDING_PRICE_PER_1M     USD per 1M tokens, overrides the model default
                               (default: 0.0 for local models; known OpenAI
                               embedding prices are built in).

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
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

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

# ── Configuration (everything from the environment) ───────────────────────

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")  # Ollama accepts any key
EXPECTED_DIM = os.environ.get("EMBEDDING_DIM")  # optional validation
CORPUS_DIR = os.environ.get("CORPUS_DIR", "cleaned_corpus")
EMBEDDING_OUTPUT = os.environ.get("EMBEDDING_OUTPUT", "")  # "" -> default
SAMPLE_OUTPUT = os.environ.get("SAMPLE_OUTPUT", "")

# Batch pipeline tuning.
BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
MAX_RETRIES = int(os.environ.get("EMBEDDING_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("EMBEDDING_RETRY_BASE_DELAY", "1.0"))
PRICE_OVERRIDE = os.environ.get("EMBEDDING_PRICE_PER_1M", "")  # "" -> by model

# Known OpenAI embedding pricing (USD per 1M tokens) used for the cost
# estimate. Local/self-hosted models default to free unless overridden.
MODEL_PRICE_PER_1M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


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


# ── Task 1 & 2: batched embeddings with retry + backoff ───────────────────


def _is_retryable(exc: Exception) -> bool:
    """Return True when *exc* is a rate limit, server error, or transport error.

    Rate limits (429) and server errors (5xx) carry an HTTP status code and
    are transient by definition. API errors like 400/401/403 are permanent —
    retrying would not help. Exceptions without a status code come from the
    transport layer (connection refused, timeouts, DNS), which are transient.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return True


def _embed_batch_with_retry(
    client,
    model: str,
    batch: list[str],
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> tuple[list[list[float]], int]:
    """Embed one batch, retrying transient failures with exponential backoff.

    The delay doubles on every retry (base_delay * 2 ** n) with a small random
    jitter so parallel workers do not retry in lockstep. Permanent errors raise
    immediately; transient errors raise once *max_retries* are exhausted.

    Returns ``(vectors, retries_used)`` — vectors ordered by the API's
    ``index`` field so they line up with *batch*.
    """
    retries = 0
    while True:
        try:
            response = client.embeddings.create(model=model, input=batch)
            ordered = sorted(response.data, key=lambda d: d.index)
            return [list(item.embedding) for item in ordered], retries
        except Exception as exc:  # noqa: BLE001 - inspected by _is_retryable
            if not _is_retryable(exc) or retries >= max_retries:
                exc.retries_used = retries  # surfaced in the run summary
                raise
            retries += 1
            delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, base_delay)
            print(
                f"    batch retry {retries}/{max_retries} after {delay:.1f}s — {exc!r}"
            )
            sleep(delay)


def embed_live(
    texts: list[str],
    *,
    model: str = EMBEDDING_MODEL,
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
    batch_size: int = BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> tuple[list[list[float] | None], dict]:
    """Embed *texts* with a real model via an OpenAI-compatible endpoint.

    Texts are processed in batches of *batch_size*; each batch is retried on
    transient errors with exponential backoff (see :func:`_embed_batch_with_retry`).
    A permanent error on one batch aborts the run. A batch that exhausts its
    retries contributes ``None`` at every position it covers instead of
    crashing the whole run — those positions are reported as failures.

    Returns ``(vectors, stats)`` where *vectors* is aligned with *texts*
    (``None`` marks failed positions) and *stats* holds:

        {"retries": total retries across all batches,
         "failed_batches": number of batches that exhausted their retries,
         "failed_starts": batch start indices that failed}
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    vectors: list[list[float] | None] = []
    stats = {"retries": 0, "failed_batches": 0, "failed_starts": []}

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_vectors, retries = _embed_batch_with_retry(
                client, model, batch, max_retries=max_retries, base_delay=base_delay
            )
            stats["retries"] += retries
            vectors.extend(batch_vectors)
        except Exception as exc:  # noqa: BLE001 - recorded as a failure
            print(
                f"    ERROR: batch {start // batch_size} failed after retries — {exc!r}"
            )
            stats["failed_batches"] += 1
            stats["failed_starts"].append(start)
            stats["retries"] += getattr(exc, "retries_used", 0)
            vectors.extend(None for _ in batch)

    return vectors, stats


def embed_offline(texts: list[str]) -> list[list[float]]:
    """Deterministic offline fallback (fastText-style, 768-dim)."""
    from scripts.embeddings_demo import simulated_embed

    return simulated_embed(texts)


def embed_corpus_chunks(
    texts: list[str],
    *,
    use_live: bool = True,
    batch_size: int = BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> tuple[str, list[list[float] | None], dict]:
    """Embed chunk texts, preferring the live API and falling back offline.

    Returns ``(mode, vectors, stats)``:

    * *mode*    — "live" or "simulated".
    * *vectors* — aligned with *texts*; ``None`` marks failed positions.
    * *stats*   — {"retries", "failed_batches", "failed_starts"}.

    The offline fallback is used only when the live endpoint is unreachable or
    rejects the request outright (every batch failed) — mixing live and
    simulated vectors in one store would corrupt retrieval.
    """
    if use_live:
        try:
            vectors, stats = embed_live(
                texts,
                batch_size=batch_size,
                max_retries=max_retries,
                base_delay=base_delay,
            )
            if any(v is not None for v in vectors):
                return "live", vectors, stats
            print(
                "WARNING: live embedding failed for every batch; using offline fallback"
            )
        except Exception as exc:  # noqa: BLE001 - endpoint unreachable etc.
            print(f"WARNING: live embedding failed ({exc}); using offline fallback")
    return (
        "simulated",
        embed_offline(texts),
        {"retries": 0, "failed_batches": 0, "failed_starts": []},
    )


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


def existing_records(path: str | Path) -> dict[str, dict]:
    """Return {chunk_id: record} for records already stored at *path*.

    Used to skip already-embedded chunks on re-runs. Returns an empty dict if
    there is no previous store or it cannot be read.
    """
    try:
        store = load_store(path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return {}
    return {r["id"]: r for r in store.get("records", [])}


def partition_cached(
    records: list[dict], cached: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Split *records* into ``(to_embed, skipped)`` using previously stored chunks.

    A chunk counts as cached only when the existing record has the same id AND
    the same text — if the source text changed, it is re-embedded so stale
    vectors are never reused.
    """
    to_embed: list[dict] = []
    skipped: list[dict] = []
    for rec in records:
        old = cached.get(rec["id"])
        if old is not None and old.get("text") == rec["text"]:
            skipped.append(old)
        else:
            to_embed.append(rec)
    return to_embed, skipped


# ── Task 3: token + cost estimation ───────────────────────────────────────


def estimate_tokens(texts: list[str]) -> int:
    """Approximate the number of tokens in *texts*.

    Uses the same BPE encoding family as OpenAI's embedding models
    (``cl100k_base``, already a project dependency via tiktoken). Falls back to
    a whitespace word count if tiktoken is unavailable.
    """
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoder.encode(text)) for text in texts)
    except Exception:  # noqa: BLE001 - never fail a run over token counting
        return sum(len(text.split()) for text in texts)


def price_per_1m_tokens(model: str) -> float:
    """USD per 1M tokens for *model*; 0.0 for local/free models.

    An explicit ``EMBEDDING_PRICE_PER_1M`` wins over the built-in table, which
    only knows OpenAI's embedding models. Anything else (Ollama models, custom
    endpoints) is assumed to be free unless the operator says otherwise.
    """
    if PRICE_OVERRIDE:
        return float(PRICE_OVERRIDE)
    return MODEL_PRICE_PER_1M.get(model, 0.0)


def estimate_cost(tokens: int, model: str) -> tuple[float, float]:
    """Return ``(price_per_1m, cost_usd)`` for *tokens* under *model*."""
    price = price_per_1m_tokens(model)
    return price, tokens / 1_000_000 * price


def format_cost(price: float, cost: float) -> str:
    """Render the cost line, noting when the model is assumed to be free."""
    if price == 0.0:
        return "$0.00 (assumed free — local model, or set EMBEDDING_PRICE_PER_1M)"
    return f"${cost:,.4f}"


# ── Task 3: run summary ───────────────────────────────────────────────────


def build_run_summary(store: dict) -> str:
    """Render the run summary block (totals, retries, failures, skips, cost)."""
    stats = store.get("stats", {})
    tokens = store.get("tokens", 0)
    price, cost = estimate_cost(tokens, store.get("model", ""))

    lines = [
        "RUN SUMMARY",
        "===========",
        f"Total chunks      : {store.get('total_chunks', store.get('chunk_count', 0))}",
        f"Embedded          : {store.get('embedded', 0)}",
        f"Skipped (cached)  : {store.get('skipped', 0)}",
        f"Failed            : {store.get('failed', 0)}",
        f"Retries (live)    : {stats.get('retries', 0)}",
        f"Batches           : {store.get('batches', 0)} (batch size {store.get('batch_size', BATCH_SIZE)})",
        f"Mode / model      : {store.get('mode', '?')} ({store.get('model', '?')})",
        f"Approx. tokens    : {tokens:,}",
        f"Price / 1M tokens : ${price:.4f}",
        f"Approx. cost      : {format_cost(price, cost)}",
        f"Store             : {store.get('output_file', '?')}",
    ]
    failed_starts = stats.get("failed_starts", [])
    if failed_starts:
        lines.append(f"Failed batches at : {failed_starts}")
    return "\n".join(lines)


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

    lines.append("\n" + "-" * 62)
    lines.append("TASK 4 — RUN SUMMARY (batching, retries, cost, skips)")
    lines.append("-" * 62)
    lines.append("\n" + build_run_summary(store))

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
    corpus_path = _resolve(corpus_arg, BACKEND_ROOT / "cleaned_corpus")

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
    if not records:
        print("ERROR: corpus produced no chunks to embed")
        return 1

    corpus_name = corpus_path.name
    default_store = (
        BACKEND_ROOT.parent / "data" / "embeddings" / f"{corpus_name}-embeddings.json"
    )
    store_path = _resolve(EMBEDDING_OUTPUT, default_store)

    # Task 4: skip chunks already embedded on a previous run. A chunk counts
    # as cached only when both its id and its text match — if the source text
    # changed, it is re-embedded so stale vectors are never reused.
    to_embed, skipped = partition_cached(records, existing_records(store_path))

    print(
        f"\nChunks: {len(records)} total, {len(to_embed)} to embed, "
        f"{len(skipped)} skipped (already cached)"
    )

    texts = [r["text"] for r in to_embed]
    mode, vectors, stats = (
        "cached",
        [],
        {"retries": 0, "failed_batches": 0, "failed_starts": []},
    )
    if texts:
        mode, vectors, stats = embed_corpus_chunks(
            texts,
            batch_size=BATCH_SIZE,
            max_retries=MAX_RETRIES,
            base_delay=RETRY_BASE_DELAY,
        )
        print(
            f"\nEmbedded {len(texts)} chunks "
            f"({'live: ' + EMBEDDING_MODEL if mode == 'live' else 'simulated'})"
        )

    # Merge: freshly embedded chunks (failed ones are dropped and counted),
    # then cached chunks carried over from the previous store.
    stored: list[dict] = []
    failures = 0
    for rec, vec in zip(to_embed, vectors):
        if vec is None:
            failures += 1
            print(
                f"  FAILED chunk {rec['id']} — retries exhausted, excluded from store"
            )
            continue
        rec["vector"] = vec
        stored.append(rec)
    stored.extend(skipped)

    if not stored:
        print("ERROR: no chunks embedded and no cached chunks to keep")
        return 1

    # Dimension checks: the store must be uniform (a model change since the
    # last run would otherwise corrupt retrieval), and the expected dimension
    # must hold when EMBEDDING_DIM is set.
    dims = {len(r["vector"]) for r in stored}
    if len(dims) > 1:
        print(
            f"ERROR: store would mix vector lengths {sorted(dims)} — "
            "model changed since the last run?"
        )
        return 1
    dim = dims.pop()
    if EXPECTED_DIM and dim != int(EXPECTED_DIM):
        print(
            f"ERROR: expected dimension {EXPECTED_DIM} " f"but the API returned {dim}"
        )
        return 1

    batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE if texts else 0
    tokens = estimate_tokens(texts)
    price, cost = estimate_cost(tokens, EMBEDDING_MODEL)

    store = {
        "corpus": corpus_name,
        "documents": len(result.documents),
        "chunk_count": len(stored),
        "total_chunks": len(records),
        "embedded": len(to_embed) - failures,
        "skipped": len(skipped),
        "failed": failures,
        "batches": batches,
        "batch_size": BATCH_SIZE,
        "dim": dim,
        "expected_dim": EXPECTED_DIM,
        "model": EMBEDDING_MODEL,
        "endpoint": BASE_URL,
        "mode": mode,
        "stats": stats,
        "tokens": tokens,
        "price_per_1m": price,
        "cost_usd": round(cost, 6),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_file": str(store_path),
        "records": stored,
    }
    save_store(store, store_path)

    sample_path = _resolve(SAMPLE_OUTPUT, BACKEND_ROOT / "embeddings_sample_output.txt")
    sample = build_sample_output(store)
    sample_path.write_text(sample + "\n", encoding="utf-8")

    # Task 3/4: print the verification output and the run summary.
    print(sample)
    print(f"\nSample output saved to {sample_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
