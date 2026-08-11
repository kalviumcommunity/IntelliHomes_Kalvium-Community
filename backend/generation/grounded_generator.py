import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)

MODEL = os.getenv("OPENAI_MODEL", "llama3.1:8b")


def generate_answer(question, context):
    """
    Generate an answer using only the supplied retrieved context.
    Falls back when no supporting context is available.
    """

    if not context or not context.strip():
        return (
            "I don't have enough information in the provided "
            "context to answer confidently."
        )

    system_prompt = """You are IntelliHomes AI, a helpful real estate assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not add information that is not supported by the context.
- If the context does not contain enough information, say:
  "I don't have enough information in the provided context to answer confidently."
- Keep the answer concise and clear.
"""

    user_prompt = f"""Retrieved context:

{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        timeout=60,
    )

    return response.choices[0].message.content
