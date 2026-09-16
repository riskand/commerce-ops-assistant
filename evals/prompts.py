# evals/prompts.py
"""Score every version of one prompt against the same fixed question set,
and print the board.

This is the loop the rest of this package exists to enable. A technique
that does not move this number is a technique you do not ship, and the
only way to know which is which is to run both against the same set."""
import argparse, asyncio, json, pathlib

import httpx
from sqlalchemy import select

from app.config import settings
from app.db import Session
from app.models import PromptVersion
from evals.run import judge


def leaderboard(rows: list[dict]) -> list[dict]:
    """Sorted best-first, with exactly one winner. Ties break on the
    lower version number: an older prompt that matches a newer one is
    not an improvement, and marking both 'best' hides that."""
    board = sorted(rows, key=lambda r: (-r["answer_accuracy"], r["version"]))
    return [{**row, "best": i == 0} for i, row in enumerate(board)]


async def versions_of(name: str) -> list[int]:
    async with Session() as session:
        rows = (await session.execute(
            select(PromptVersion.version)
            .where(PromptVersion.name == name)
            .order_by(PromptVersion.version))).scalars().all()
    return list(rows)


async def score(version: int, qa: list[dict],
                client: httpx.AsyncClient) -> dict:
    # An empty eval set is a broken run, not a score of zero -- reporting
    # 0.0 accuracy here would read as a real measurement.
    if not qa:
        raise ValueError("empty question set: nothing to score")
    correct = 0
    for row in qa:
        r = await client.post("/query", json={
            "tenant_id": row["tenant_id"], "question": row["question"],
            "prompt_version": version})
        body = r.json()
        correct += await judge(row["question"], row["expected_answer"],
                               body["answer"])
    return {"version": version, "n": len(qa),
            "answer_accuracy": round(correct / len(qa), 2)}


async def main(name: str, qa_path: str) -> None:
    qa = [json.loads(l)
          for l in pathlib.Path(qa_path).read_text().splitlines() if l]
    rows = []
    async with httpx.AsyncClient(base_url=settings.app_url,
                                 timeout=120) as client:
        for version in await versions_of(name):
            rows.append(await score(version, qa, client))
    print(f"{'ver':>4}  {'accuracy':>8}  best")
    for row in leaderboard(rows):
        mark = "<-" if row["best"] else ""
        print(f"{row['version']:>4}  {row['answer_accuracy']:>8}  {mark}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="rag_answer")
    p.add_argument("--qa", default="evals/qa.jsonl")
    a = p.parse_args()
    asyncio.run(main(a.name, a.qa))
