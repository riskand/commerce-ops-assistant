# seed_domain.py
"""Seeds a demo tenant's orders and sync_errors so the MCP tools in mcp_server/
have something real to query, in the Inspector and in a real host. Idempotent
the way /ingest is: delete this tenant's rows, then insert, in one
transaction, so re-running never duplicates rows.

Structured as a callable rather than a bare script: the experiment harnesses'
integration tests call seed_domain(tenant_id) directly from a pytest
fixture, rather than depending on someone having run this by hand.
"""
import asyncio
import uuid

from sqlalchemy import delete

from app.db import Session
from app.models import Chunk, Order, SyncError
from evals.corpus import DOCUMENTS

DEMO_TENANT_ID = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


async def seed_domain(tenant_id: uuid.UUID) -> dict:
    async with Session() as session, session.begin():
        await session.execute(delete(Order).where(Order.tenant_id == tenant_id))
        await session.execute(
            delete(SyncError).where(SyncError.tenant_id == tenant_id))

        orders = [
            Order(id=uuid.uuid4(), tenant_id=tenant_id, order_ref="SO-1042",
                  status="sync_failed", channel="channel_b",
                  line_items=[{"sku": "WIDGET-1", "qty": 2}]),
            Order(id=uuid.uuid4(), tenant_id=tenant_id, order_ref="SO-1043",
                  status="sync_failed", channel="channel_a",
                  line_items=[{"sku": "WIDGET-2", "qty": 1}]),
            Order(id=uuid.uuid4(), tenant_id=tenant_id, order_ref="SO-1044",
                  status="pending", channel="channel_b",
                  line_items=[{"sku": "WIDGET-1", "qty": 5}]),
        ]
        session.add_all(orders)

        session.add_all([
            SyncError(tenant_id=tenant_id, order_ref="SO-1042",
                      channel="channel_b", code="8541",
                      detail="Address validation failed: missing postal code"),
            SyncError(tenant_id=tenant_id, order_ref="SO-1044",
                      channel="channel_b", code="4102",
                      detail="Inventory hold expired before sync retried"),
            SyncError(tenant_id=tenant_id, order_ref="SO-1043",
                      channel="channel_a", code="8541",
                      detail="Address validation failed: missing postal code"),
        ])

    return {"orders": len(orders), "sync_errors": 3}


# The four policy documents search_docs is meant to find. Kept here rather
# than in a fixture because the whole demo — "which orders failed to
# sync, and what do our docs say about that error code?" — needs both halves
# of the corpus: the failures, and the document explaining the code. Seeding
# only the orders leaves search_docs with nothing to return, and a model
# facing an empty tool calls it again and again rather than giving up.
# They live in evals/corpus.py, beside the eval questions asked of them.


async def seed_documents(tenant_id: uuid.UUID) -> dict:
    """Ingest the policy corpus so search_docs has something to find.

    Separate from seed_domain because this one needs the embedding model,
    and therefore torch. Integration tests call seed_domain alone and stay
    free of that dependency; the experiments and a live demo call both.
    """
    from app.chunker import chunk
    from app.embedder import get_embedder

    embedder = await get_embedder()
    async with Session() as session, session.begin():
        await session.execute(delete(Chunk).where(Chunk.tenant_id == tenant_id))

    total = 0
    for document_id, text in DOCUMENTS.items():
        parts = chunk(text)
        vectors = await embedder.embed(parts)
        async with Session() as session, session.begin():
            session.add_all([
                Chunk(tenant_id=tenant_id, document_id=uuid.UUID(document_id),
                      ordinal=i, content=part, embedding=vector)
                for i, (part, vector) in enumerate(zip(parts, vectors))])
        total += len(parts)
    return {"documents": len(DOCUMENTS), "chunks": total}


async def _seed_all(tenant_id: uuid.UUID) -> None:
    print(await seed_domain(tenant_id))
    print(await seed_documents(tenant_id))


if __name__ == "__main__":
    # One asyncio.run for both halves. app/db.py's engine is a module-level
    # singleton whose pooled asyncpg connections belong to the loop that
    # opened them, so a second asyncio.run finds them attached to a loop
    # that no longer exists and fails with "attached to a different loop".
    asyncio.run(_seed_all(DEMO_TENANT_ID))
