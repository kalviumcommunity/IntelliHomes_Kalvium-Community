import re

def rerank(query, candidates):
    query_words = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
    reranked = []

    for candidate in candidates:
        text_words = set(
            re.findall(r"\b[a-zA-Z]+\b", candidate["document"].lower())
        )

        overlap = query_words.intersection(text_words)

        candidate_copy = candidate.copy()
        candidate_copy["rerank_score"] = len(overlap)
        candidate_copy["matched_words"] = sorted(overlap)

        reranked.append(candidate_copy)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked
