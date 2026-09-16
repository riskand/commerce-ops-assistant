# ask_once.py — the smallest possible call to the Messages API
import asyncio, json, os

import httpx

from app.tools import GET_ORDER_SCHEMA

MODEL = "claude-sonnet-5"       # check docs.claude.com for the current list
QUESTION = "What is the status of order SO-1042?"

async def main() -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                     "anthropic-version": "2023-06-01"},
            json={"model": MODEL, "max_tokens": 1024,
                  "tools": [GET_ORDER_SCHEMA],
                  "messages": [{"role": "user", "content": QUESTION}]},
        )
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))

asyncio.run(main())
