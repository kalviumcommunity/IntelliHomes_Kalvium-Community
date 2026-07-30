"""
main.py — IntelliHomes backend entry point.

Demonstrates the context-window-aware history manager with a short
live (or simulated) conversation.
"""

import os
import textwrap

from dotenv import load_dotenv

from scripts.history_manager import ChatHistory, count_tokens

load_dotenv()

SYSTEM_PROMPT = textwrap.dedent("""\
    You are IntelliHomes AI, a helpful assistant for real-estate staff.
    Answer concisely. Use retrieved context when provided.
""")


def _make_llm(live: bool = False):
    """Return a callable that takes messages and returns an assistant reply."""
    if live:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=300,
        )
        model = os.environ.get("OPENAI_MODEL", "llama3.1:8b")

        def _llm(messages):
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content

    else:

        def _llm(_messages):
            return (
                "Here are the property documents you need. "
                "Always verify the title deed and encumbrance certificate "
                "before proceeding with any property transaction. "
                "Consult a legal expert for thorough due diligence."
            )

    return _llm


def main():
    live = bool(os.environ.get("OPENAI_API_KEY"))
    budget = 2000

    chat = ChatHistory(
        system_prompt=SYSTEM_PROMPT,
        token_budget=budget,
        strategy="trim",
    )
    llm = _make_llm(live)

    print(f"IntelliHomes RAG Assistant (budget={budget}, strategy={chat.strategy})")
    print(
        f"System tokens: {count_tokens(SYSTEM_PROMPT)}  |  {'Live' if live else 'Simulated'} mode"
    )
    print("-" * 55)

    questions = [
        "What documents verify property ownership?",
        "How do I check for outstanding loans?",
        "Explain stamp duty calculation.",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}] User: {q}")
        reply = chat.ask(q, llm_completion=llm)
        print(
            f"    Tokens: {chat.total_tokens():>4}  |  Messages: {len(chat.messages)}"
        )
        print(f"    Assistant: {reply[:100]}…")

    print(f"\n{'=' * 55}")
    print(f"Final history: {len(chat.messages)} messages, {chat.total_tokens()} tokens")
    print(
        f"Strategy '{chat.strategy}' kept it under {budget}: {chat.total_tokens() <= budget}"
    )


if __name__ == "__main__":
    main()
