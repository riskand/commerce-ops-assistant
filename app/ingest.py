# app/ingest.py
"""The half of ingestion that must not run inside a request."""
import uuid

from sqlalchemy import delete

from app.chunker import chunk
from app.config import settings
from app.context import situate
from app.db import Session
from app.embedder import get_embedder
from app.models import Chunk, Document

BATCH = 32          # chunks handed to the model at once


async def embed_all(embedder, texts: list[str]) -> list[list[float]]:
    """One encode() per BATCH, not one for the whole document."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        vectors.extend(await embedder.embed(texts[i:i + BATCH]))
    return vectors


async def ingest_document(document_id: uuid.UUID) -> None:
    """Runs outside the request. Every exit path writes a terminal status."""
    async with Session() as session, session.begin():
        doc = await session.get(Document, document_id)
        doc.status = "embedding"
        body, tenant_id = doc.body, doc.tenant_id
    try:
        texts = chunk(body)
        if settings.contextual_retrieval:          # one model call per chunk
            texts = [f"{await situate(body, c)}\n\n{c}" for c in texts]
        embedder = await get_embedder()
        vectors = await embed_all(embedder, texts)
    except Exception as e:
        async with Session() as session, session.begin():
            doc = await session.get(Document, document_id)
            doc.status, doc.error = "failed", str(e)
        return
    async with Session() as session, session.begin():
        await session.execute(                        # delete then insert,
            delete(Chunk).where(                      # one transaction
                Chunk.tenant_id == tenant_id,
                Chunk.document_id == document_id))
        session.add_all([
            Chunk(tenant_id=tenant_id, document_id=document_id,
                  ordinal=i, content=t, embedding=v,
                  embedding_model=embedder.name)
            for i, (t, v) in enumerate(zip(texts, vectors))
        ])
        doc = await session.get(Document, document_id)
        doc.status, doc.chunks, doc.error = "ready", len(texts), None
