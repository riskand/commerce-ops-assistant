# run.py  (scratch, repo root)
import asyncio
from app.loop import run_loop, TOOL_SCHEMAS

QUESTION = "What is the status of order SO-1042 and did it sync to the channel?"

async def main() -> None:
    text, history = await run_loop(
        [{"role": "user", "content": QUESTION}], TOOL_SCHEMAS)
    print(text)
    print(f"--- messages: {len(history)}")

asyncio.run(main())
