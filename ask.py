"""Ask a question and print the DeepSeek-V4-Flash answer.

The API key is read from DEEPSEEK_API_KEY in .env.
The endpoint is reachable on the PolyU campus network only.
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

ENDPOINT = "https://genai.comp.polyu.edu.hk/api/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash"


def main() -> int:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("DEEPSEEK_API_KEY is missing. Check your .env file.", file=sys.stderr)
        return 1

    question = input("Question: ").strip()
    if not question:
        print("Please enter a question.", file=sys.stderr)
        return 1

    try:
        response = requests.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 2000,
                "stream": False,
            },
            timeout=120,
        )
    except requests.RequestException as exc:
        print(f"Could not reach the API: {exc}", file=sys.stderr)
        print("The endpoint is available on the PolyU campus network only.", file=sys.stderr)
        return 1

    try:
        data = response.json()
    except ValueError:
        print(f"API error ({response.status_code}): {response.text}", file=sys.stderr)
        return 1

    if response.status_code != 200 or "choices" not in data:
        print(f"API error ({response.status_code}): {json.dumps(data, indent=2)}", file=sys.stderr)
        return 1

    message = data["choices"][0]["message"]
    answer = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not answer:
        print(json.dumps(data, indent=2))
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
