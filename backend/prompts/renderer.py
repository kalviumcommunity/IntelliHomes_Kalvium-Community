from __future__ import annotations

from typing import Any

from prompts.templates import SYSTEM_TEMPLATE, USER_TEMPLATE


def render_prompt(context: str, question: str):
    return (
        SYSTEM_TEMPLATE.format(context=context),
        USER_TEMPLATE.format(question=question),
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _chunk_metadata(chunk: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata", {}))
    source = metadata.get("source") or f"chunk-{index}"
    chunk_id = metadata.get("chunk_id") or metadata.get("id") or f"{source}#{metadata.get('position', index - 1)}"
    section = metadata.get("section") or metadata.get("heading") or "N/A"
    page = metadata.get("page")
    position = metadata.get("position")
    return {
        "source": source,
        "chunk_id": chunk_id,
        "section": section,
        "page": page,
        "position": position,
        "chunk_index": index,
    }


def build_citation_map(chunks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return the citation metadata for each chunk so answers can be traced to a real source."""

    citation_map: dict[int, dict[str, Any]] = {}
    for index, chunk in enumerate(chunks, start=1):
        citation_map[index] = _chunk_metadata(chunk, index)
    return citation_map


def build_cited_answer(answer: str, chunks: list[dict[str, Any]], cited_indices: list[int] | None = None) -> str:
    """Attach only verifiable citations for the chunks used by the answer."""

    if not chunks or not cited_indices:
        return "I don't have enough information to answer confidently."

    citation_map = build_citation_map(chunks)
    valid_indices = [idx for idx in cited_indices if idx in citation_map and citation_map[idx].get("source")]
    if not valid_indices:
        return "I don't have enough information to answer confidently."

    citation_parts: list[str] = []
    for idx in valid_indices:
        metadata = citation_map[idx]
        source = metadata["source"]
        chunk_id = metadata.get("chunk_id")
        section = metadata.get("section")
        page = metadata.get("page")
        snippet = chunks[idx - 1].get("text", "").strip()
        details: list[str] = [source]
        if chunk_id:
            details.append(f"chunk_id={chunk_id}")
        if section and section != "N/A":
            details.append(f"section={section}")
        if page is not None:
            details.append(f"page={page}")
        if snippet:
            details.append(f"text={snippet!r}")
        citation_parts.append(f"[{idx}] {'; '.join(details)}")

    return f"{answer.strip()} {' '.join(citation_parts)}".strip()


def assemble_retrieved_context(
    chunks: list[dict[str, Any]],
    token_budget: int = 400,
    reserved_tokens: int = 80,
) -> str:
    """Assemble retrieved chunks into prompt context while respecting a token budget."""

    available = max(0, token_budget - reserved_tokens)
    parts: list[str] = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        metadata = _chunk_metadata(chunk, index)
        source = metadata["source"]
        text = chunk.get("text", "")
        citation_details = [f"chunk_id={metadata['chunk_id']}"]
        if metadata.get("section") and metadata["section"] != "N/A":
            citation_details.append(f"section={metadata['section']}")
        if metadata.get("page") is not None:
            citation_details.append(f"page={metadata['page']}")
        marker = f"[{index}] {source} ({'; '.join(citation_details)})"
        entry = f"{marker}\n{text}"
        entry_tokens = _estimate_tokens(entry)

        if used + entry_tokens > available:
            break

        parts.append(entry)
        used += entry_tokens

    return "\n\n".join(parts)


def build_augmented_prompt(
    chunks: list[dict[str, Any]],
    question: str,
    token_budget: int = 400,
    reserved_tokens: int = 80,
) -> tuple[str, str]:
    """Build a system/user prompt pair with injected context and grounding guidance."""

    context = assemble_retrieved_context(chunks, token_budget=token_budget, reserved_tokens=reserved_tokens)
    system_prompt = SYSTEM_TEMPLATE.format(context=context)
    user_prompt = USER_TEMPLATE.format(question=question)
    return system_prompt, user_prompt