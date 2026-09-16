# reindex.py
"""Re-embed the chunks an embedding-model change left behind.

Run after the active embedding model changes -- MODEL in
app/embedder.py, or EMBEDDING_MODEL in the environment:

    python reindex.py                 # report how many chunks are stale
    python reindex.py --yes           # re-embed them, a page at a time

Pages are committed as they finish, so an interrupted run resumes where
it stopped rather than starting over -- the query selects by staleness,
not by offset, so rows already done simply stop matching.
"""
import asyncio
import sys

from sqlalchemy import func, select

from app.db import Session
from app.embedder import active_model, get_embedder
from app.ingest import embed_all
from app.models import Chunk

PAGE = 256          # chunks re-embedded per transaction


async def stale_count() -> int:
    async with Session() as session:
        return await session.scalar(
            select(func.count()).select_from(Chunk)
            .where(Chunk.embedding_model != active_model()))


async def reindex(page: int = PAGE) -> int:
    embedder = await get_embedder()
    done = 0
    while True:
        async with Session() as session, session.begin():
            rows = list(await session.scalars(
                select(Chunk).where(
                    Chunk.embedding_model != active_model())
                .order_by(Chunk.id).limit(page)))
            if not rows:
                return done
            vectors = await embed_all(embedder, [r.content for r in rows])
            for row, vector in zip(rows, vectors):
                row.embedding, row.embedding_model = vector, embedder.name
            done += len(rows)
        print(f"re-embedded {done}", flush=True)


async def main() -> None:
    pending = await stale_count()
    print(f"{pending} chunks are not on {active_model()}")
    if "--yes" not in sys.argv:
        print("re-run with --yes to re-embed them")
        return
    print(f"done: {await reindex()} chunks")


if __name__ == "__main__":
    asyncio.run(main())
