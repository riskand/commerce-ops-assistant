# mcp_server/server.py
"""Three tools: a model-facing surface over the same orders and
sync_errors a support engineer would query directly, minus any argument
that could name a tenant. Tenant scope comes from settings.mcp_tenant_id
(app/config.py), fixed once when this process starts, never from a
parameter a caller supplies."""
import uuid

from mcp.server import MCPServer
from sqlalchemy import select

from app.config import settings
from app.db import Session
from app.embedder import get_embedder
from app.models import Order, SyncError
from app.reranker import get_reranker
from app.retrieval import hybrid_search

mcp = MCPServer("saas-tools")


def _tenant() -> uuid.UUID:
    """The tenant this server speaks for, fixed at startup."""
    return uuid.UUID(settings.mcp_tenant_id)


@mcp.tool()
async def search_docs(query: str) -> list[dict]:
    """Search this tenant's indexed documents (policies, help articles,
    prior support notes) for passages relevant to a question, using the
    same hybrid vector-and-keyword search the /docsearch route uses.

    Use this for questions about a written policy or procedure, such as
    "what is the return window" or "how do we handle a chargeback". It
    does not know the live status of a specific order; for that, use
    get_order or list_sync_errors instead.

    Args:
        query: The question or phrase to search for, in plain language.
            It is sent to both a semantic and a keyword search, so
            ordinary wording works better than a list of keywords.

    Returns:
        Up to 5 hits, ranked best first, each a dict with:
            document_id: str, the source document's identifier.
            content: str, the matching passage's text.
            score: float, this hit's rank score. Higher means more
                relevant; scores are only comparable within one call.
        An empty list means nothing matched well enough to return.
    """
    embedder = await get_embedder()
    reranker = await get_reranker()
    qvec = (await embedder.embed([query]))[0]
    async with Session() as session:
        hits = await hybrid_search(session, _tenant(), query, qvec,
                                   reranker, k=5)
    return [{"document_id": str(doc_id), "content": content, "score": score}
            for doc_id, content, score in hits]


@mcp.tool()
async def get_order(order_ref: str) -> dict | None:
    """Look up one order by its reference number, for this tenant only.

    Use this when a question names a specific order, for example "what
    is the status of SO-1042", to see its current status, sales channel
    and line items. It does not explain why an order failed to sync to
    a channel; for that, call list_sync_errors with the order's channel.

    Args:
        order_ref: The order's reference, e.g. "SO-1042". Matched
            exactly, case-sensitive.

    Returns:
        A dict with order_ref (str), status (str), channel (str) and
        line_items (a list of dicts, each with the line's sku and qty),
        or None if this tenant has no order with that reference.
    """
    async with Session() as session:
        result = await session.execute(
            select(Order).where(Order.tenant_id == _tenant(),
                                Order.order_ref == order_ref))
        order = result.scalar_one_or_none()
    if order is None:
        return None
    return {"order_ref": order.order_ref, "status": order.status,
            "channel": order.channel, "line_items": order.line_items}


@mcp.tool()
async def list_sync_errors(channel: str) -> list[dict]:
    """List this tenant's recent sync failures for one sales channel.

    Use this to answer "why do orders on channel_b keep failing to
    sync", or to explain a specific order's sync_failed status once
    get_order has named its channel: pass that channel here to see the
    underlying error.

    Args:
        channel: The sales channel to check, e.g. "channel_b". Matched
            exactly, case-sensitive.

    Returns:
        A list of dicts, newest first, each with order_ref (str),
        channel (str), code (str, the upstream error code), detail
        (str, a human-readable explanation) and occurred_at (str, an
        ISO 8601 timestamp). An empty list means this tenant has no
        recorded failures on that channel.
    """
    async with Session() as session:
        result = await session.execute(
            select(SyncError)
            .where(SyncError.tenant_id == _tenant(),
                   SyncError.channel == channel)
            .order_by(SyncError.occurred_at.desc()))
        errors = result.scalars().all()
    return [{"order_ref": e.order_ref, "channel": e.channel, "code": e.code,
             "detail": e.detail, "occurred_at": e.occurred_at.isoformat()}
            for e in errors]


async def _serve() -> None:
    """Load the models before serving, not on the first tool call.

    search_docs needs an embedder and a reranker. Left lazy, the first call
    pays for loading both — on a cold cache that is minutes, and any client
    with a timeout on tool calls gives up long before it finishes, reporting
    a timeout rather than a download. A server that is not ready should not
    be accepting calls, so the loading happens here, before stdio opens.
    """
    from app.embedder import get_embedder
    from app.reranker import get_reranker

    await get_embedder()
    await get_reranker()
    await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_serve())
