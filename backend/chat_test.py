import json
import os

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=300,
)

model = os.environ["OPENAI_MODEL"]

messages = [
    {
        "role": "system",
        "content": "You are IntelliHomes AI."
    },
    {
        "role": "user",
        "content": "List three property documents to verify."
    }
]

print("===== REQUEST =====")
print(json.dumps(messages, indent=2))

print("\nSending request to Ollama...\n")

try:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    print("Response received!\n")

    print("===== RESPONSE =====")
    print(response.choices[0].message.content)

    if response.usage:
        print("\n===== TOKEN USAGE =====")
        print(response.usage)

except AuthenticationError:
    print("❌ Authentication failed.")

except RateLimitError:
    print("❌ Rate limit exceeded.")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
