from retrieval.similarity import evaluate_recall_precision


def test_evaluate_recall_precision_reports_recall_and_precision() -> None:
    chunks = [
        {
            "text": "A Title Deed confirms legal ownership.",
            "metadata": {"source": "property_guide.txt", "chunk_id": "property_guide-2"},
        },
        {
            "text": "Property Tax Receipts confirm taxes are paid.",
            "metadata": {"source": "taxes.txt", "chunk_id": "taxes-1"},
        },
    ]

    cases = [
        {
            "query": "What proves legal ownership?",
            "relevant_chunk_ids": ["property_guide-2"],
            "expected_sources": ["property_guide.txt"],
        }
    ]

    settings = [
        {"name": "k1", "top_k": 1},
        {"name": "k2", "top_k": 2},
    ]

    results = evaluate_recall_precision(cases, chunks, settings)

    assert results[0]["name"] == "k1"
    assert results[0]["recall_at_1"] == 1.0
    assert results[0]["precision_at_1"] == 1.0
    assert results[1]["recall_at_2"] == 1.0
    assert results[1]["precision_at_2"] == 0.5
