# app/retrieval.py
import uuid

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk
from app.reranker import Reranker

Hit = tuple[uuid.UUID, str, float]        # document_id, content, score


async def search(session: AsyncSession, tenant_id: uuid.UUID,
                 qvec: list[float], k: int = 5) -> list[Hit]:
    stmt = (
        select(Chunk.document_id, Chunk.content,
               (1 - Chunk.embedding.cosine_distance(qvec)).label("score"))
        .where(Chunk.tenant_id == tenant_id)     # tenant isolation IS this line
        .order_by(Chunk.embedding.cosine_distance(qvec))
        .limit(k)
    )
    rows = (await session.execute(stmt)).all()
    return [(r.document_id, r.content, r.score) for r in rows]


async def fts_search(session: AsyncSession, tenant_id: uuid.UUID,
                     q: str, k: int = 20) -> list[Hit]:
    tsq = func.websearch_to_tsquery("english", q)
    stmt = (
        select(Chunk.document_id, Chunk.content,
               func.ts_rank(Chunk.fts, tsq).label("score"))
        .where(Chunk.tenant_id == tenant_id, Chunk.fts.op("@@")(tsq))
        .order_by(desc("score"))
        .limit(k)
    )
    rows = (await session.execute(stmt)).all()
    return [(r.document_id, r.content, r.score) for r in rows]


def rrf_scores(*ranked: list[uuid.UUID], k: int = 60) -> dict[uuid.UUID, float]:
    """Each id's summed 1/(k + rank) across every list it appears in."""
    scores: dict[uuid.UUID, float] = {}
    for lst in ranked:
        for rank, doc_id in enumerate(lst, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def rrf(*ranked: list[uuid.UUID], k: int = 60) -> list[uuid.UUID]:
    """Reciprocal Rank Fusion. Pure, so it is trivially testable."""
    scores = rrf_scores(*ranked, k=k)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


async def fused_search(session: AsyncSession, tenant_id: uuid.UUID,
                       q: str, qvec: list[float], k: int = 20) -> list[Hit]:
    vec = await search(session, tenant_id, qvec, k=20)
    fts = await fts_search(session, tenant_id, q, k=20)

    content = {doc_id: text for doc_id, text, _ in vec + fts}
    lanes = [d for d, _, _ in vec], [d for d, _, _ in fts]
    scores = rrf_scores(*lanes)
    return [(d, content[d], scores[d]) for d in rrf(*lanes)[:k]]


async def hybrid_search(session: AsyncSession, tenant_id: uuid.UUID,
                        q: str, qvec: list[float], reranker: Reranker,
                        k: int = 5, min_score: float = float("-inf")
                        ) -> list[Hit]:
    fused = await fused_search(session, tenant_id, q, qvec, k=20)
    docs = [text for _, text, _ in fused]

    ranked = await reranker.rerank(q, docs)
    return [(fused[i][0], docs[i], score)
            for i, score in ranked[:k] if score >= min_score]
