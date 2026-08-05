import math
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")


@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int]
    embedding: list[float]


def text_to_embedding(text: str) -> list[float]:
    """Convert text to a simple deterministic embedding vector for sample ranking.

    This is a lightweight stand-in for real model embeddings. The values are
    counts of common keywords so we can demonstrate cosine similarity ranking
    without requiring API calls.
    """
    keywords = [
        "ownership",
        "title",
        "deed",
        "survey",
        "boundaries",
        "tax",
        "payments",
        "building",
        "permits",
        "approval",
        "documents",
        "property",
    ]
    normalized = text.lower().replace(".", "").replace("?", "")
    tokens = normalized.split()
    return [float(tokens.count(term)) for term in keywords]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between vectors, which is useful for
    comparing embedding direction while reducing the impact of vector length.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def build_sample_chunks() -> list[Chunk]:
    texts = [
        "Buying property requires verifying ownership documents.",
        "A Title Deed confirms legal ownership.",
        "A Survey Plan defines property boundaries.",
        "Property Tax Receipts confirm tax payments.",
        "Building permits verify construction approval.",
    ]
    source = "sample_document.txt"
    chunks: list[Chunk] = []
    for position, text in enumerate(texts):
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    "source": source,
                    "section": f"Section {position + 1}",
                    "position": position,
                },
                embedding=text_to_embedding(text),
            )
        )
    return chunks


def rank_chunks(query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
    query_embedding = text_to_embedding(query)
    scored = [
        (chunk, cosine_similarity(query_embedding, chunk.embedding))
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def format_ranked_results(ranked: list[tuple[Chunk, float]]) -> str:
    lines: list[str] = []
    lines.append("Query similarity ranking")
    lines.append("=" * 24)
    for rank, (chunk, score) in enumerate(ranked, start=1):
        lines.append(f"Rank {rank}: score={score:.4f}")
        lines.append(f"Text: {chunk.text}")
        lines.append(f"Metadata: {chunk.metadata}")
        lines.append("")
    return "\n".join(lines).strip()


def main() -> None:
    query = "Which documents confirm ownership and property boundaries?"
    chunks = build_sample_chunks()
    ranked_results = rank_chunks(query, chunks)
    print(format_ranked_results(ranked_results))
    print(
        "\nJustification: cosine similarity is used because it compares the direction of "
        "embedding vectors and matches semantic similarity even when vector magnitudes differ."
    )


if __name__ == "__main__":
    main()
