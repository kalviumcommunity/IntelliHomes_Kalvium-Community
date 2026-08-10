from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def similarity(query, text):
    return SequenceMatcher(None, query.lower(), text.lower()).ratio()


def rank_chunks(query, chunks):

    ranked = []

    for chunk in chunks:

        score = similarity(query, chunk["text"])

        ranked.append((score, chunk))

    ranked.sort(reverse=True, key=lambda x: x[0])

    return ranked


def _get_chunk_source(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    return metadata.get("source", "unknown")


def evaluate_retrieval_settings(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    settings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate retrieval settings for a small set of test queries.

    Each case contains a query and expected_sources. Each setting defines a
    retrieval configuration (for example top_k and min_score). The returned
    list contains per-setting metrics such as top-1 hit and top-k hit rate.
    """

    results: list[dict[str, Any]] = []

    for setting in settings:
        hits = 0
        top1_hits = 0
        topk_hits = 0
        detailed: list[dict[str, Any]] = []

        for case in cases:
            ranked = rank_chunks(case["query"], chunks)
            filtered = [item for item in ranked if item[0] >= setting.get("min_score", 0.0)]
            top_candidates = filtered[: setting.get("top_k", 3)]
            retrieved_sources = [_get_chunk_source(item[1]) for item in top_candidates]
            expected_sources = case.get("expected_sources", [])

            top1_hit = bool(retrieved_sources and retrieved_sources[0] in expected_sources)
            topk_hit = any(source in expected_sources for source in retrieved_sources)

            if top1_hit:
                top1_hits += 1
            if topk_hit:
                topk_hits += 1
            if topk_hit:
                hits += 1

            detailed.append(
                {
                    "query": case["query"],
                    "expected_sources": expected_sources,
                    "retrieved_sources": retrieved_sources,
                    "top1_hit": top1_hit,
                    "topk_hit": topk_hit,
                }
            )

        results.append(
            {
                "name": setting.get("name", "setting"),
                "top_k": setting.get("top_k", 3),
                "min_score": setting.get("min_score", 0.0),
                "top1_hit": bool(top1_hits > 0),
                "topk_hit": bool(topk_hits > 0),
                "top1_hit_rate": round(top1_hits / len(cases), 2) if cases else 0.0,
                "topk_hit_rate": round(topk_hits / len(cases), 2) if cases else 0.0,
                "hit_rate": round(hits / len(cases), 2) if cases else 0.0,
                "details": detailed,
            }
        )

    return results


def format_retrieval_tuning_report(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    settings: list[dict[str, Any]],
) -> str:
    results = evaluate_retrieval_settings(cases, chunks, settings)
    best = max(results, key=lambda item: (item["hit_rate"], item["topk_hit_rate"], item["top1_hit_rate"]))

    lines = [
        "# Retrieval relevance tuning",
        "",
        "## Test queries",
        "",
    ]
    for case in cases:
        lines.append(f"- Query: {case['query']}")
        lines.append(f"  Expected sources: {', '.join(case['expected_sources'])}")

    lines.extend([
        "",
        "## Compared settings",
        "",
    ])

    for result in results:
        lines.append(
            f"- {result['name']}: top_k={result['top_k']}, min_score={result['min_score']}, "
            f"top1 hit rate={result['top1_hit_rate']:.2f}, top-k hit rate={result['topk_hit_rate']:.2f}"
        )

    lines.extend([
        "",
        "## Best settings",
        "",
        f"- {best['name']}: top_k={best['top_k']}, min_score={best['min_score']}",
        f"- Reason: it achieved the highest hit rate ({best['hit_rate']:.2f}) while keeping top-k coverage strong.",
    ])

    return "\n".join(lines)