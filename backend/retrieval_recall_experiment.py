from __future__ import annotations

from pathlib import Path

from chunking.metadata import attach_metadata
from chunking.strategies import paragraph_chunk
from retrieval.similarity import format_recall_precision_report


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

LABELLED_CASES = [
    {
        "query": "What proves legal ownership?",
        "relevant_chunk_ids": ["property_guide-2"],
        "expected_sources": ["property_guide.txt"],
    },
    {
        "query": "How do I verify property taxes?",
        "relevant_chunk_ids": ["taxes-1"],
        "expected_sources": ["taxes.txt"],
    },
    {
        "query": "Who should provide ownership records?",
        "relevant_chunk_ids": ["ownership-1"],
        "expected_sources": ["ownership.txt"],
    },
]

SETTINGS = [
    {"name": "k1", "top_k": 1},
    {"name": "k3", "top_k": 3},
]


def build_chunks() -> list[dict]:
    all_chunks = []
    for filename, text in DOCUMENTS.items():
        chunks = paragraph_chunk(text.strip())
        annotated = attach_metadata(chunks, filename)
        for index, chunk in enumerate(annotated):
            chunk_id = f"{Path(filename).stem}-{index + 1}"
            chunk["metadata"]["chunk_id"] = chunk_id
        all_chunks.extend(annotated)
    return all_chunks


def main() -> None:
    chunks = build_chunks()
    report = format_recall_precision_report(LABELLED_CASES, chunks, SETTINGS)
    output_path = Path(__file__).with_name("retrieval_recall_report.md")
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
