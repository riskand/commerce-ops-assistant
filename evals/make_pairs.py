# evals/make_pairs.py
"""Build a training set for the embedder out of the corpus itself.

    python -m evals.make_pairs --out evals/pairs.jsonl

A retrieval model learns from (question, passage) pairs, and you have
passages but no questions. So a model writes them: for each chunk, a few
questions that chunk answers. This is standard practice for domain
adaptation, and it is the only reason a fine-tune is possible at all on a
corpus nobody has ever queried.

**It defaults to the local model, and that is the point.** A corpus of
three thousand chunks is three thousand generations, and this is the
shape of work where a frontier model buys you almost nothing and charges
for every one: the questions are short, the passage is right there, and a
small model is entirely capable of asking "how long is the returns
window?". The card you bought for the fallback pays for itself here.
Pass --generator api when you want the expensive opinion.

**evals/qa.jsonl is not touched, and must never be.** Those questions are
human-written and are the held-out set; training on them and then
reporting the recall they produce measures memorisation, not retrieval.
The corpus is necessarily shared -- adapting to this corpus is the whole
point -- but the questions are not.
"""
import argparse
import asyncio
import json
import pathlib

from sqlalchemy import select

from app.db import Session
from app.loop import call_llm, final_text, local_generate
from app.models import Chunk

PROMPT = """Below is one passage from a commerce-operations knowledge base.

Write {n} short questions that this passage answers, as a seller or an
operator would actually ask them. Use the passage's own vocabulary for
identifiers (SKUs, error codes, policy names) and ordinary words for
everything else. Do not number them. One question per line, nothing else.

Passage:
{content}"""


async def questions_for(content: str, n: int, generator: str) -> list[str]:
    messages = [{"role": "user",
                 "content": PROMPT.format(n=n, content=content)}]
    resp = (await local_generate(messages) if generator == "local"
            else await call_llm(messages, tools=[]))
    # Small models pad. Strip list markers, keep only what is actually a
    # question, and drop the preamble -- cheap generation is worth a
    # strict parser, not a trusting one.
    lines = [l.strip(" -*\t0123456789.")
             for l in final_text(resp).splitlines()]
    return [l for l in lines if l.endswith("?") and len(l) > 12][:n]


async def main(out: str, n: int, limit: int | None,
               generator: str) -> None:
    async with Session() as session:
        stmt = select(Chunk.id, Chunk.content).order_by(Chunk.id)
        rows = list(await session.execute(stmt.limit(limit) if limit else stmt))

    path = pathlib.Path(out)
    written = 0
    with path.open("w") as fh:
        for chunk_id, content in rows:
            for question in await questions_for(content, n,
                                                generator):
                fh.write(json.dumps({"question": question,
                                     "chunk_id": chunk_id,
                                     "positive": content}) + "\n")
                written += 1
            print(f"{written} pairs from {chunk_id}", flush=True)
    print(f"wrote {written} pairs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="evals/pairs.jsonl")
    p.add_argument("--n", type=int, default=3, help="questions per chunk")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--generator", choices=["local", "api"], default="local")
    a = p.parse_args()
    asyncio.run(main(a.out, a.n, a.limit, a.generator))
