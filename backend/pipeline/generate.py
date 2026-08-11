"""
generate.py — Pipeline stage 4 of 4: generate the grounded answer.

IntelliHomes RAG pipeline — Query-to-answer flow.

Renders the grounded context block from stage 3 (assemble) into the
IntelliHomes prompt templates and calls an LLM to produce the final answer.

Two backends are supported:

* **live** — any OpenAI-compatible chat-completions endpoint (Ollama by
  default, or OpenAI); used when ``OPENAI_API_KEY`` is set and *live* is
  not explicitly disabled;
* **simulated** — a deterministic offline responder that summarises the
  grounded context, so the pipeline still runs end to end for demos and
  tests without any network or model.

The LLM itself is pluggable: pass any callable ``llm(messages) -> str`` to
:func:`generate_stage` (e.g. for tests), or use :func:`make_llm`.

Contract
--------
    generate_stage(query, context, llm=None, live=None) -> {
        "answer":        str,    # the generated answer
        "model":         str,    # model name used (or "simulated")
        "live":          bool,   # whether a real endpoint was used
        "system_prompt": str,    # rendered system prompt (for debugging)
        "user_prompt":   str,    # rendered user prompt (for debugging)
        "prompt_tokens": int,    # tiktoken estimate of both prompts
    }

Environment:
    OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make `backend/` importable no matter which directory the module is imported
# from, so the prompt templates and token counter resolve.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from prompts.renderer import render_prompt  # noqa: E402
from scripts.history_manager import count_tokens  # noqa: E402

# ── Configuration (everything from the environment) ───────────────────────

LIVE_MODEL = os.environ.get("OPENAI_MODEL", "llama3.1:8b")


def make_llm(live: bool | None = None):
    """Return a callable ``llm(messages) -> assistant_text``.

    *live* is ``True`` (real endpoint) when ``OPENAI_API_KEY`` is set and
    *live* is not explicitly ``False``; otherwise a deterministic simulated
    responder is returned so the pipeline always runs.
    """
    if live is None:
        live = bool(os.environ.get("OPENAI_API_KEY"))

    if live:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=300,
        )

        def _live_llm(messages: list[dict]) -> str:
            resp = client.chat.completions.create(model=LIVE_MODEL, messages=messages)
            return resp.choices[0].message.content

        return _live_llm

    def _simulated_llm(messages: list[dict]) -> str:
        # Deterministic offline responder: ground the reply in the retrieved
        # context (the system prompt), so demos/tests are reproducible.
        system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"),
            "",
        )
        return (
            "Based on the retrieved documents, the relevant context is "
            "provided above. Verify the cited sources (title deed, "
            "encumbrance certificate) with a legal expert before proceeding. "
            f"(simulated reply; context characters: {len(system)})"
        )

    return _simulated_llm


def generate_stage(
    query: str,
    context: str,
    *,
    llm=None,
    live: bool | None = None,
    model: str | None = None,
) -> dict:
    """Generate a grounded answer for *query* given the *context* block.

    *llm* is an optional callable ``llm(messages) -> str``; when omitted,
    :func:`make_llm` builds one (*live* unless ``OPENAI_API_KEY`` is unset
    or *live* is ``False``). *model* only labels the reply (for reporting);
    the live model itself is configured through ``OPENAI_MODEL``.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    system_prompt, user_prompt = render_prompt(context=context, question=query)

    if llm is None:
        llm = make_llm(live=live)
        live = live if live is not None else bool(os.environ.get("OPENAI_API_KEY"))
    else:
        live = bool(live) if live is not None else False

    answer = llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {
        "answer": answer,
        "model": model or (LIVE_MODEL if live else "simulated"),
        "live": live,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_tokens": count_tokens(system_prompt) + count_tokens(user_prompt),
    }
