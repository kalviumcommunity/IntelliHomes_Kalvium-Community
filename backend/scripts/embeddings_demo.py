"""
embeddings_demo.py — Embeddings Fundamentals & Vector Representation.

Demonstrates the core idea behind semantic search in the IntelliHomes RAG
pipeline: text is turned into numeric vectors (embeddings) whose positions
in vector space encode *meaning*. Similar texts land near each other;
unrelated texts land far apart.

It generates embeddings for a few short sample texts, reports the vector
dimension, checks that every text produces a vector of the same length,
and compares similar vs. dissimilar texts with cosine similarity.

Usage
-----
    python scripts/embeddings_demo.py                 # live (Ollama/OpenAI), falls back to simulated
    OPENAI_BASE_URL=https://api.openai.com/v1 \
    OPENAI_API_KEY=sk-... \
    EMBEDDING_MODEL=text-embedding-3-small \
    python scripts/embeddings_demo.py                 # any OpenAI-compatible endpoint

The script works out of the box against a local Ollama server
(``http://localhost:11434/v1``, default model ``nomic-embed-text``).
If no embedding endpoint is reachable it falls back to a deterministic,
offline simulated embedder so the demo always runs.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import textwrap

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

# Any OpenAI-compatible endpoint works: Ollama (/v1) or OpenAI itself.
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
# Dimension used by the *simulated* embedder. Real models define their own:
#   nomic-embed-text -> 768, text-embedding-3-small -> 1536,
#   all-MiniLM-L6-v2 -> 384
SIMULATED_DIM = 768


# ── Task 1: Sample texts ──────────────────────────────────────────────────

# One pair with similar meaning, one clearly unrelated text, and a second
# similar pair drawn from the IntelliHomes property domain.
TEXTS = [
    "How do I reset my account password?",  # 0
    "Steps to recover access to my login",  # 1  (similar to 0)
    "The cafeteria menu has pasta today",  # 2  (dissimilar)
    "What documents do I need to transfer property ownership?",  # 3
    "Steps to transfer the title of a property to a new owner",  # 4  (similar to 3)
]

# ── Task 3: Cosine similarity ─────────────────────────────────────────────


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: dot product of unit vectors (direction only)."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ── Embedding backends ────────────────────────────────────────────────────


def embed_live(texts: list[str]) -> list[list[float]]:
    """Embed *texts* with a real model via an OpenAI-compatible endpoint."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"), base_url=BASE_URL, timeout=120
    )
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda d: d.index)
    return [list(item.embedding) for item in ordered]


def _feature_index(feature: str) -> int:
    """Deterministic 32-bit hash of a feature (stable across runs)."""
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def simulated_embed(texts: list[str], dim: int = SIMULATED_DIM) -> list[list[float]]:
    """Offline, deterministic fastText-style embedder (char n-grams + words).

    Each word contributes its own bucket plus character n-gram buckets
    (signed hashing, like fastText). Text sharing substrings/words gets
    overlapping vectors, so *surface* similarity shows up even without a
    trained model. It is a stand-in for demonstration only — production
    RAG uses a pretrained model (see ``embed_live``).
    """
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        words = re.findall(r"[a-z0-9]+", text.lower())
        for word in words:
            features = [f"w:{word}"]
            if len(word) >= 3:
                for n in (3, 4):
                    for i in range(len(word) - n + 1):
                        features.append(f"n:{word[i : i + n]}")
            for feature in features:
                idx = _feature_index(feature) % dim
                vec[idx] += 1.0 if _feature_index(feature) % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


# ── Report ────────────────────────────────────────────────────────────────


def build_report(embeddings: list[list[float]], live: bool) -> str:
    """Render the full demo report as text (also written to a file)."""
    dim = len(embeddings[0])
    uniform = all(len(e) == dim for e in embeddings)
    lines: list[str] = []

    lines.append("=" * 62)
    lines.append("EMBEDDINGS FUNDAMENTALS & VECTOR REPRESENTATION")
    lines.append("=" * 62)
    lines.append(
        f"\nMode     : {'live' if live else 'simulated'} "
        f"({'default: Ollama + %s' % EMBEDDING_MODEL if live else 'offline deterministic'})"
    )
    lines.append(f"Endpoint : {BASE_URL}")
    lines.append(
        f"Model    : {EMBEDDING_MODEL if live else 'simulated fastText-style'}"
    )

    # Task 1 + 2 — embeddings and dimension
    lines.append("\n" + "-" * 62)
    lines.append("TASK 1 — SAMPLE TEXTS -> EMBEDDINGS")
    lines.append("-" * 62)
    for text, vec in zip(TEXTS, embeddings):
        first8 = ", ".join(f"{v:+.4f}" for v in vec[:8])
        lines.append(f"\n{text!r}")
        lines.append(f"  first 8 values : [{first8}, …]")

    lines.append("\n" + "-" * 62)
    lines.append("TASK 2 — VECTOR DIMENSION")
    lines.append("-" * 62)
    lines.append(f"\nDimension of embeddings[0] : {dim}")
    lines.append(f"All texts same length       : {uniform}")
    if not uniform:
        lines.append("  !! MISMATCH — check the embedder !!")

    # Task 3 — similarity
    similar = cosine(embeddings[0], embeddings[1])
    similar_2 = cosine(embeddings[3], embeddings[4])
    dissimilar = cosine(embeddings[0], embeddings[2])
    dissimilar_2 = cosine(embeddings[1], embeddings[2])

    lines.append("\n" + "-" * 62)
    lines.append("TASK 3 — COSINE SIMILARITY (higher = closer meaning)")
    lines.append("-" * 62)
    lines.append(f"\n[0] {TEXTS[0]!r}")
    lines.append(f"[1] {TEXTS[1]!r}")
    lines.append(f"[2] {TEXTS[2]!r}")
    lines.append(f"[3] {TEXTS[3]!r}")
    lines.append(f"[4] {TEXTS[4]!r}")
    lines.append(f"\nsimilar    [0] vs [1]  : {similar:+.4f}")
    lines.append(f"similar    [3] vs [4]  : {similar_2:+.4f}")
    lines.append(f"dissimilar [0] vs [2]  : {dissimilar:+.4f}")
    lines.append(f"dissimilar [1] vs [2]  : {dissimilar_2:+.4f}")
    ok = similar > dissimilar and similar_2 > dissimilar_2
    lines.append(f"\nSimilar pair scores higher : {ok}")
    if not ok:
        lines.append("  !! Unexpected ranking — embeddings not capturing similarity !!")

    # Task 4 — what vectors represent
    lines.append("\n" + "=" * 62)
    lines.append("TASK 4 — WHAT DO THESE VECTORS REPRESENT?")
    lines.append("=" * 62)
    lines.append(textwrap.dedent("""\

            Each vector is a numeric representation of *meaning*. The model
            was trained so that texts about the same topic land close together
            in vector space and unrelated texts land far apart.

            The numbers are NOT:
            - random IDs — the same text always produces the same vector;
            - keyword counts — "login" and "password" share no words, yet
              their vectors point in nearly the same direction.

            They ARE a dense coordinate system for semantics: every dimension
            responds to a pattern the model learned across billions of texts.
            This is why RAG retrieval can match meaning, not just keywords.

            In RAG each chunk is embedded and stored in a vector database.
            A user question is embedded too, and retrieval becomes a
            nearest-neighbor search over chunk vectors.
            """))
    lines.append("\n" + "=" * 62)
    return "\n".join(lines)


def main() -> None:
    live = False
    embeddings: list[list[float]]
    try:
        embeddings = embed_live(TEXTS)
        live = True
        print("✓ Connected to embedding endpoint, generating real embeddings …\n")
    except Exception as exc:
        print(f"⚠  Live embedding failed ({exc.__class__.__name__}: {exc})")
        print("   Falling back to the offline simulated embedder.\n")
        embeddings = simulated_embed(TEXTS)

    report = build_report(embeddings, live=live)
    print(report)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "embeddings_report.md",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Embeddings Report\n\n```text\n{report}\n```\n")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
