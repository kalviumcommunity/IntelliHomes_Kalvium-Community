import os
from pathlib import Path
import tiktoken

# tokenizer used by many OpenAI models
enc = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(enc.encode(text))


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
docs_dir = PROJECT_ROOT / "docs"

# ---------- Sample 1 ----------
question = "How do I reset my smart home's WiFi?"

# ---------- Sample 2 ----------
with open(docs_dir / "PRD.md", encoding="utf-8") as f:
    document = f.read()


samples = {"Question": question, "Document": document}


INPUT_PRICE = 0.0005  # per 1K tokens
OUTPUT_PRICE = 0.0015

EXPECTED_OUTPUT = 300  # assume assistant returns ~300 tokens


print("=" * 60)

for name, text in samples.items():

    chars = len(text)
    tokens = token_count(text)

    input_cost = (tokens / 1000) * INPUT_PRICE
    output_cost = (EXPECTED_OUTPUT / 1000) * OUTPUT_PRICE
    total = input_cost + output_cost

    print(f"\n{name}")
    print("-" * 40)
    print(f"Characters : {chars}")
    print(f"Tokens      : {tokens}")
    print(f"Estimated Cost : ${total:.6f}")

print("\nRelationship Demo")
print("-" * 40)

examples = [
    "refund",
    "supercalifragilisticexpialidocious",
    "print('Hello World')",
    "こんにちは",
]

for text in examples:
    print(f"{text:40} chars={len(text):2} tokens={token_count(text)}")
