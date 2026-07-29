import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=180,
)

model = os.getenv("OPENAI_MODEL")


def test_prompt(system_prompt, user_prompt):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    print("\n===================================")
    print("SYSTEM PROMPT")
    print("===================================")
    print(system_prompt)

    print("\nUSER QUESTION")
    print(user_prompt)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    print("\nMODEL RESPONSE")
    print(response.choices[0].message.content)


# -----------------------------
# Prompt A (Vague)
# -----------------------------

prompt_a = """
You are a helpful assistant.
"""

# -----------------------------
# Prompt B (Clear)
# -----------------------------

prompt_b = """
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
"I don't have enough information to answer confidently."
"""

question = "List three property documents to verify."

print("\n\n******** PROMPT A ********")
test_prompt(prompt_a, question)

print("\n\n******** PROMPT B ********")
test_prompt(prompt_b, question)