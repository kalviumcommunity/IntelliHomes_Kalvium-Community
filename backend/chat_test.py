import os

from dotenv import load_dotenv
from openai import OpenAI
from prompts.renderer import render_prompt

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=180,
)

model = os.getenv("OPENAI_MODEL")


def main():
    system_prompt, user_prompt = render_prompt(
        context="Buying residential property",
        question="List three property documents to verify."
    )

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

    print("===== SYSTEM PROMPT =====")
    print(system_prompt)

    print("\n===== USER PROMPT =====")
    print(user_prompt)

    print("\nSending request...\n")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    print("===== RESPONSE =====")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()