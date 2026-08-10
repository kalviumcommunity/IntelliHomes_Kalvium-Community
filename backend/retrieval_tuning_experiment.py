from __future__ import annotations

from pathlib import Path

from chunking.metadata import attach_metadata
from chunking.strategies import paragraph_chunk
from retrieval.similarity import format_retrieval_tuning_report


DOCUMENTS = {
    "ownership.txt": """
Property ownership should always be verified before purchase.

The seller should provide valid ownership records.
""",
    "property_guide.txt": """
Buying property requires verifying ownership documents.

A Title Deed confirms legal ownership.

A Survey Plan defines property boundaries.
""",
    "taxes.txt": """
Property Tax Receipts confirm that all taxes have been paid.

Outstanding taxes should be cleared before purchase.
""",
}

TEST_CASES = [
    {
        "query": "What proves legal ownership?",
        "expected_sources": ["property_guide.txt"],
    },
    {
        "query": "How do I verify property taxes?",
        "expected_sources": ["taxes.txt"],
    },
    {
        "query": "Who should provide ownership records?",
        "expected_sources": ["ownership.txt"],
    },
]

SETTINGS = [
    {"name": "k1_min0", "top_k": 1, "min_score": 0.0},
    {"name": "k3_min0", "top_k": 3, "min_score": 0.0},
    {"name": "k3_min0_2", "top_k": 3, "min_score": 0.2},
]


def build_chunks() -> list[dict]:
    all_chunks = []
    for filename, text in DOCUMENTS.items():
        chunks = paragraph_chunk(text.strip())
        all_chunks.extend(attach_metadata(chunks, filename))
    return all_chunks


def main() -> None:
    chunks = build_chunks()
    report = format_retrieval_tuning_report(TEST_CASES, chunks, SETTINGS)
    output_path = Path(__file__).with_name("retrieval_tuning_report.md")
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
