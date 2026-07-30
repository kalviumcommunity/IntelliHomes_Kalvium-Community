"""
structured_output.py — Structured Output & JSON Response Handling.

Teaches the app to ask the model for structured JSON, parse it, validate it,
and recover gracefully when things go wrong.

Usage
-----
    python scripts/structured_output.py          # simulated mode (demo)
    OPENAI_API_KEY=... python scripts/structured_output.py   # live mode
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

MODEL = os.environ.get("OPENAI_MODEL", "llama3.1:8b")

# Every response MUST contain these keys
REQUIRED_FIELDS = ("answer", "source")

# ── Task 1: Prompt for a defined JSON structure ───────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are IntelliHomes AI, a helpful assistant for real-estate staff.

    Reply with ONLY a JSON object in exactly this format — no extra text,
    no markdown, no explanation:

    {
        "answer": "your concise answer to the user's question",
        "source": "the document or section name where this info comes from"
    }

    Valid JSON only.  Every response MUST be parseable by json.loads().""")


# ── Tasks 2, 3 & 4: Parse, handle malformed JSON, validate fields ────────


def parse_json_response(
    raw: str, required: tuple[str, ...] = REQUIRED_FIELDS
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse *raw* model output as JSON and validate required fields.

    Returns ``(data_dict, None)`` on success or ``(None, error_message)``
    on failure.
    """
    # ── Task 3: Detect malformed JSON ──
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON — {exc}"

    # ── Task 4: Validate required fields ──
    missing = [k for k in required if k not in data]
    if missing:
        return None, f"missing required fields: {missing}"

    return data, None


def retry_once(
    llm, messages: list[dict], raw_failure: str, err: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Retry once with a stricter instruction after a parse failure."""
    print(f"    ⚠  Parse failed: {err}")
    print(f"    🔁  Retrying with stricter instruction …")

    messages.append({"role": "assistant", "content": raw_failure})
    messages.append(
        {
            "role": "user",
            "content": (
                "Your previous response was not valid JSON. "
                "Respond with ONLY a valid JSON object containing "
                f'"{REQUIRED_FIELDS[0]}" and "{REQUIRED_FIELDS[1]}" fields. '
                "No other text."
            ),
        }
    )

    raw2 = llm(messages)
    return parse_json_response(raw2)


# ── LLM factory (live / simulated) ────────────────────────────────────────

# Simulated responses — deliberately problematic so we can demonstrate recovery.
# The simulated LLM cycles through these in order.
_SIMULATED_RESPONSES = [
    # 0 — Perfect JSON (success path)
    json.dumps(
        {
            "answer": "Title deed, sale deed, and encumbrance certificate.",
            "source": "Property Documentation Guide",
        }
    ),
    # 1 — Malformed: trailing comma in object  (recovery path)
    (
        '{"answer": "Check the EC for outstanding loan entries.",'
        ' "source": "Encumbrance Certificate Guide",}'
    ),
    # 2 — Malformed: prose wrapping JSON (recovery path)
    (
        "Here is the information you requested: "
        '{"answer": "Stamp duty is 5-7%% of the agreement value.",'
        ' "source": "Stamp Duty Act"}'
    ),
    # 3 — Valid JSON but missing "source" field (validation failure)
    json.dumps({"answer": "You need NOC from the builder and society."}),
    # 4 — Perfect JSON again
    json.dumps(
        {
            "answer": "An NOC confirms no legal disputes on the property.",
            "source": "No-Objection Certificate guidelines",
        }
    ),
]


def _make_llm(live: bool = False):
    """Return a callable that takes messages and returns assistant content."""
    if live:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=300,
        )

        def _llm(messages):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},  # Task 1 — enforce JSON
                temperature=0,
            )
            return resp.choices[0].message.content

    else:

        def _llm(_messages):
            # Cycle through the list so each call gets the next response
            idx = _llm.call_count % len(_SIMULATED_RESPONSES)
            _llm.call_count += 1
            return _SIMULATED_RESPONSES[idx]

        _llm.call_count = 0

    return _llm


# ── High-level helper ─────────────────────────────────────────────────────


def ask_structured(llm, question: str) -> dict[str, Any]:
    """Ask *question* and return a validated JSON dict.

    If parsing or validation fails, the method retries **once** with a
    stricter instruction before giving up and returning a fallback dict.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    raw = llm(messages)

    # ── Try 1 (success or first failure) ──
    data, err = parse_json_response(raw)
    if data is not None:
        return data

    # ── Recovery: retry once ──
    data, err2 = retry_once(llm, messages, raw, err)
    if data is not None:
        return data

    # ── Give up — return a safe fallback ──
    print(f"    ❌  Retry also failed: {err2}")
    return {
        "answer": f"[Could not parse model response — {err2}]",
        "source": "unknown",
    }


# ── Demo runner ────────────────────────────────────────────────────────────


def _print_box(text: str) -> None:
    try:
        width = min(70, os.get_terminal_size().columns - 2)
    except OSError:
        width = 68  # fallback when stdout is piped
    print(f"\n  ╔{'═' * width}╗")
    for line in text.splitlines():
        print(f"  ║ {line:<{width}} ║")
    print(f"  ╚{'═' * width}╝")


QUESTIONS = [
    "What documents verify property ownership?",
    "How do I check for outstanding loans on a property?",
    "Explain stamp duty calculation.",
    "What is an NOC and why is it needed?",
    "How to verify clear title of a property?",
]


def main() -> None:
    live = bool(os.environ.get("OPENAI_API_KEY"))
    llm = _make_llm(live)

    print(f"  Structured Output Demo  |  Model: {MODEL}")
    print(f"  Mode: {'🔴 Live' if live else '🟡 Simulated'}")
    print(f"  Required fields: {REQUIRED_FIELDS}")
    print(f"  Questions: {len(QUESTIONS)}")
    print(
        f"  Simulated responses: {len(_SIMULATED_RESPONSES)} items"
        f" (some deliberately malformed for demo)"
    )
    print()

    for i, question in enumerate(QUESTIONS, 1):
        _print_box(f"[{i}] Q: {question}")
        print()

        result = ask_structured(llm, question)

        print(
            f"\n    ✅  Parsed & validated successfully"
            if "Could not parse" not in result.get("answer", "")
            else f"\n    ⚠️  Used fallback"
        )

        print(f"    {'─' * 50}")
        for key, value in result.items():
            print(f"    {key:12s} : {value}")

        print(f"    {'─' * 50}")
        print()

    # ── Summary ────────────────────────────────────────────────────────
    # Count results from the current run (tracked during the loop)
    _print_box("DEMONSTRATED CAPABILITIES")
    print(f"""
    ✓ Task 1 : Prompt for a defined JSON structure
              SYSTEM_PROMPT asks for {{"answer", "source"}}
              Live mode uses response_format={{"type": "json_object"}}

    ✓ Task 2 : Parse JSON into a usable object
              json.loads() converts the raw string to a dict

    ✓ Task 3 : Handle malformed JSON gracefully
              json.JSONDecodeError caught — no unhandled crashes
              Retry mechanism re-asks with stricter instruction

    ✓ Task 4 : Validate required fields before use
              Checks every key in {REQUIRED_FIELDS} is present
              Missing fields are reported explicitly

    ✓ Task 5 : Commit with sample parsed results
              Run with:  python scripts/structured_output.py
              Live mode: OPENAI_API_KEY=... python scripts/structured_output.py
""")


if __name__ == "__main__":
    main()
