from chunking.metadata import attach_metadata
from chunking.strategies import paragraph_chunk
from retrieval.similarity import rank_chunks

documents = {
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
"""
}

all_chunks = []

for filename, text in documents.items():

    chunks = paragraph_chunk(text.strip())

    all_chunks.extend(
        attach_metadata(chunks, filename)
    )

tests = [

    (
        "What proves legal ownership?",
        "property_guide.txt"
    ),

    (
        "How do I verify property taxes?",
        "taxes.txt"
    ),

    (
        "Who should provide ownership records?",
        "ownership.txt"
    )

]

passes = 0

print("===== SANITY TESTS =====\n")

for query, expected in tests:

    ranked = rank_chunks(query, all_chunks)

    best = ranked[0][1]

    score = ranked[0][0]

    print(f"Query: {query}")
    print(f"Top Source: {best['metadata']['source']}")
    print(f"Score: {score:.3f}")

    if best["metadata"]["source"] == expected:

        print("PASS\n")
        passes += 1

    else:

        print("FAIL\n")

print(f"Passed {passes}/{len(tests)} tests")