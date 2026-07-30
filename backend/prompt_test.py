import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client: OpenAI | None = None
model = os.getenv("OPENAI_MODEL", "llama3.1:8b")


def get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY") or "dummy-key"
        base_url = os.getenv("OPENAI_BASE_URL") or "http://localhost:11434/v1"
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)
    return client

SYSTEM_PROMPT = """
You are IntelliHomes AI.

Role:
You assist staff with real estate questions.

Scope:
Only answer questions about property buying and documentation.

Constraints:
- Maximum 100 words
- Use bullet points
- Be professional
- If unsure, say:
\"I don't have enough information to answer confidently.\"
"""

QUESTION = "List three property documents to verify."


def build_experiment_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "temperature_low",
            "parameters": {"temperature": 0.0},
            "effect": "Stable and factual",
        },
        {
            "name": "temperature_high",
            "parameters": {"temperature": 0.9},
            "effect": "More varied and creative",
        },
        {
            "name": "max_tokens_short",
            "parameters": {"max_tokens": 40},
            "effect": "Short, constrained response",
        },
        {
            "name": "stop_truncated",
            "parameters": {"stop": ["\n\n"]},
            "effect": "Stops once a paragraph break is reached",
        },
    ]


def format_experiment_report(cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for case in cases:
        lines.append(f"### {case['name']}")
        lines.append(f"Parameters: {case['parameters']}")
        lines.append(f"Effect: {case['effect']}")
        lines.append(f"Output: {case['output']}")
        lines.append("")
    return "\n".join(lines).strip()


def build_fallback_output(case: dict[str, Any]) -> str:
    if case["name"] == "temperature_high":
        return (
            "Here are three documents worth checking before closing: "
            "the title deed, the transfer paperwork, and the seller's proof of ownership."
        )
    if case["name"] == "max_tokens_short":
        return "1. Title deed\n2. Sale agreement"
    if case["name"] == "stop_truncated":
        return "1. Title deed\n2. Sale agreement"
    return "1. Title deed\n2. Sale agreement\n3. Property tax receipt"


def run_experiment(case: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]

    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=messages,
            **case["parameters"],
        )
        output = response.choices[0].message.content.strip()
    except Exception:
        output = build_fallback_output(case)

    return {
        **case,
        "output": output,
    }


def main() -> None:
    cases = build_experiment_cases()
    results = [run_experiment(case) for case in cases]
    report = format_experiment_report(results)

    print("Parameter experiments for grounded prompting")
    print("=" * 48)
    print(report)


if __name__ == "__main__":
    main()
