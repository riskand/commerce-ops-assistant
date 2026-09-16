# evals/run.py
import argparse, asyncio, json, pathlib

import httpx

from app.config import settings
from app.loop import call_llm
from app.prompts import render


async def judge(question: str, expected: str, actual: str) -> bool:
    prompt = await render("eval_judge", {"question": question,
                                         "expected": expected,
                                         "actual": actual})
    resp = await call_llm([{"role": "user", "content": prompt}], tools=[])
    text = "".join(b["text"] for b in resp["content"] if b["type"] == "text")
    return text.strip().upper().startswith("YES")


async def main(qa: str, out: str, mode: str) -> None:
    rows = [json.loads(l) for l in pathlib.Path(qa).read_text().splitlines() if l]
    answerable = [r for r in rows if r["must_cite_doc"]]
    unanswerable = [r for r in rows if not r["must_cite_doc"]]
    retrieved = correct = refused = 0

    async with httpx.AsyncClient(base_url=settings.app_url,
                                 timeout=120) as client:
        for row in rows:
            r = await client.post("/query", json={
                "tenant_id": row["tenant_id"], "question": row["question"],
                "mode": mode})
            body = r.json()
            verdict = await judge(row["question"], row["expected_answer"],
                                  body["answer"])
            if row["must_cite_doc"]:
                cited = {c["document_id"] for c in body["chunks"]}
                retrieved += row["must_cite_doc"] in cited
                correct += verdict
            else:
                refused += verdict

    n = len(answerable)
    result = {"n": n,
              "recall_at_5": round(retrieved / n, 2),
              "answer_accuracy": round(correct / n, 2)}
    if unanswerable:
        result["refusal_accuracy"] = round(refused / len(unanswerable), 2)
    print(json.dumps(result))
    pathlib.Path(out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default="evals/qa.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["naive", "fusion", "rerank"],
                   default="rerank")
    a = p.parse_args()
    asyncio.run(main(a.qa, a.out, a.mode))
