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


def evaluate_recall_precision(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    settings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure recall@k and precision@k for labelled retrieval cases."""

    results: list[dict[str, Any]] = []

    for setting in settings:
        top_k = setting.get("top_k", 3)
        recall_hits = 0
        precision_hits = 0
        detailed: list[dict[str, Any]] = []

        for case in cases:
            ranked = rank_chunks(case["query"], chunks)
            top_results = ranked[:top_k]
            retrieved_ids = []
            for _, chunk in top_results:
                metadata = chunk.get("metadata", {})
                chunk_id = metadata.get("chunk_id")
                if chunk_id:
                    retrieved_ids.append(chunk_id)
                else:
                    retrieved_ids.append(metadata.get("source", "unknown"))

            relevant_ids = case.get("relevant_chunk_ids", [])
            retrieved_relevant = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_ids]

            recall_at_k = 1.0 if any(chunk_id in relevant_ids for chunk_id in retrieved_ids) else 0.0
            precision_at_k = len(retrieved_relevant) / max(top_k, 1)

            if recall_at_k:
                recall_hits += 1
            precision_hits += precision_at_k

            detailed.append(
                {
                    "query": case["query"],
                    "relevant_chunk_ids": relevant_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "recall_at_k": recall_at_k,
                    "precision_at_k": round(precision_at_k, 2),
                }
            )

        recall_value = round(recall_hits / len(cases), 2) if cases else 0.0
        precision_value = round(precision_hits / len(cases), 2) if cases else 0.0

        result = {
            "name": setting.get("name", "setting"),
            "top_k": top_k,
            "recall_at_k": recall_value,
            "precision_at_k": precision_value,
            "details": detailed,
        }
        result[f"recall_at_{top_k}"] = recall_value
        result[f"precision_at_{top_k}"] = precision_value
        results.append(result)

    return results


def format_recall_precision_report(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    settings: list[dict[str, Any]],
) -> str:
    results = evaluate_recall_precision(cases, chunks, settings)
    best = max(results, key=lambda item: (item["recall_at_k"], item["precision_at_k"]))

    lines = [
        "# Retrieval recall and precision evaluation",
        "",
        "## Labelled query set",
        "",
    ]

    for case in cases:
        lines.append(f"- Query: {case['query']}")
        lines.append(f"  Relevant chunk IDs: {', '.join(case['relevant_chunk_ids'])}")

    lines.extend([
        "",
        "## Recall/precision results",
        "",
    ])

    for result in results:
        lines.append(
            f"- {result['name']}: recall@{result['top_k']}={result['recall_at_k']:.2f}, precision@{result['top_k']}={result['precision_at_k']:.2f}"
        )

    lines.extend([
        "",
        "## Failure analysis",
        "",
        f"- Best setting: {best['name']} (recall@{best['top_k']}={best['recall_at_k']:.2f}, precision@{best['top_k']}={best['precision_at_k']:.2f})",
        "- Likely causes for lower-scoring cases: chunking may be too coarse, the query wording may be too broad, or the retrieval signal may need a stronger keyword overlap or metadata filter.",
    ])

    return "\n".join(lines)