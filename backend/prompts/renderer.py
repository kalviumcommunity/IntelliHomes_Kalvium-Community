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
        source = chunk.get("metadata", {}).get("source", f"chunk-{index}")
        text = chunk.get("text", "")
        marker = f"[{index}] {source}"
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