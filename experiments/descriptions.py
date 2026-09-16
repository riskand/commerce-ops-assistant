# experiments/descriptions.py
"""The description experiment. Records the mistakes the model makes
calling this server's tools -- wrong tool, wrong argument shape,
over-calling -- each of which is fixed ONLY by editing tool docstrings in
mcp_server/server.py, never by touching this harness. Run once, edit
descriptions, run again, and a before/after pair of
results/descriptions-<timestamp>.json files is a diff instead of a
memory.

Needs a real ANTHROPIC_API_KEY -- app/config.py's placeholder authenticates
with nothing but a 401 -- and the docker-compose database seed_domain
writes into. Spends real API calls every time it runs; see
experiments/README.md before running it.
"""
import argparse
import asyncio
import json
import pathlib
import sys
import time
import uuid

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.loop import call_llm, final_text, tool_result
from app.mcp_client import ToolRouter
from seed_domain import seed_domain

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO_TENANT_ID = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MAX_ITERATIONS = 5

# "sync-plus-docs" is an ops question that forces list_sync_errors then
# search_docs. The rest are picked to surface each of the three named
# mistakes:
# wrong tool and wrong argument shape show up as an unexpected name or a
# malformed argument in a question's recorded tool_calls; over-calling
# shows up only on a question that needs no tool at all ("no-tool-
# arithmetic") -- a model that calls one anyway is the mistake itself,
# not a missing feature to add.
QUESTIONS = [
    {"id": "sync-plus-docs",
     "question": ("Which orders failed to sync to channel B today, and "
                  "what do our docs say about that error code?"),
     "note": "two tools composed: list_sync_errors then search_docs"},
    {"id": "order-lookup",
     "question": "What is the status of order SO-1042?",
     "note": "single tool: get_order"},
    {"id": "policy-lookup",
     "question": "What is our return window for a buyer?",
     "note": "single tool: search_docs"},
    {"id": "channel-a-errors",
     "question": "List the sync failures on channel A.",
     "note": "single tool: list_sync_errors -- argument shape check"},
    {"id": "no-tool-arithmetic",
     "question": "What is 17 plus 25?",
     "note": "needs no tool -- calling one anyway is over-calling"},
]


async def run_question(router: ToolRouter, question: str,
                       max_iterations: int = MAX_ITERATIONS) -> dict:
    """Runs one question through the loop with every tool the router
    knows about offered, and records what the experiment needs: the
    tools chosen, in order, with their arguments; how many model calls
    it took; and the final answer."""
    messages = [{"role": "user", "content": question}]
    tools = router.schemas()
    tool_calls = []
    model_calls = 0
    for _ in range(max_iterations):
        resp = await call_llm(messages, tools)
        model_calls += 1
        messages.append({"role": "assistant", "content": resp["content"]})
        if resp["stop_reason"] != "tool_use":
            return {"tool_calls": tool_calls, "model_calls": model_calls,
                    "final_answer": final_text(resp)}
        results = []
        for block in resp["content"]:
            if block["type"] == "tool_use":
                tool_calls.append({"name": block["name"],
                                   "arguments": block["input"]})
                out = await router.call(block["name"], block["input"])
                results.append(tool_result(block["id"], out))
        messages.append({"role": "user", "content": results})
    return {"tool_calls": tool_calls, "model_calls": model_calls,
            "final_answer": None, "error": "step budget exceeded"}


async def main(tenant_id: uuid.UUID, out_dir: pathlib.Path) -> pathlib.Path:
    await seed_domain(tenant_id)

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"],
        cwd=str(ROOT), env={"MCP_TENANT_ID": str(tenant_id)})

    rows = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            router = ToolRouter()
            router.register("saas-tools", session, tools)

            for q in QUESTIONS:
                result = await run_question(router, q["question"])
                rows.append({"id": q["id"], "question": q["question"],
                            "note": q["note"], **result})

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"descriptions-{stamp}.json"
    out_path.write_text(json.dumps(rows, indent=2))

    print(f"{'id':<20} {'calls':>5}  {'tools called':<40} answer")
    for row in rows:
        tool_names = ",".join(t["name"] for t in row["tool_calls"]) or "-"
        answer = (row["final_answer"] or "")[:40]
        print(f"{row['id']:<20} {row['model_calls']:>5}  "
              f"{tool_names:<40} {answer}")
    print(f"\nwrote {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tenant-id", default=str(DEMO_TENANT_ID))
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    asyncio.run(main(uuid.UUID(args.tenant_id), pathlib.Path(args.out_dir)))
