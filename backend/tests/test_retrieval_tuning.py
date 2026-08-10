from retrieval.similarity import evaluate_retrieval_settings


def test_evaluate_retrieval_settings_reports_hits() -> None:
    chunks = [
        {
            "text": "A Title Deed confirms legal ownership.",
            "metadata": {"source": "property_guide.txt"},
        },
        {
            "text": "Property Tax Receipts confirm taxes are paid.",
            "metadata": {"source": "taxes.txt"},
        },
    ]

    cases = [
        {
            "query": "What proves legal ownership?",
            "expected_sources": ["property_guide.txt"],
        }
    ]

    settings = [
        {"name": "top1", "top_k": 1, "min_score": 0.0},
        {"name": "top2", "top_k": 2, "min_score": 0.0},
    ]

    results = evaluate_retrieval_settings(cases, chunks, settings)

    assert results[0]["name"] == "top1"
    assert results[0]["top1_hit"] is True
    assert results[0]["topk_hit"] is True
    assert results[1]["name"] == "top2"
    assert results[1]["topk_hit"] is True
