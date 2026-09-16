# app/routes/docsearch.py
import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.db import Session
from app.embedder import get_embedder
from app.ingest import ingest_document
from app.loop import call_llm
from app.models import Document
from app.prompts import render
from app.reranker import get_reranker
from app.retrieval import fused_search, hybrid_search, search

router = APIRouter(tags=["docsearch"])

NOT_FOUND = "Not found in your documents."


class IngestRequest(BaseModel):
    tenant_id: uuid.UUID
    document_id: uuid.UUID
    text: str


class QueryRequest(BaseModel):
    tenant_id: uuid.UUID
    question: str
    mode: Literal["naive", "fusion", "rerank"] = "rerank"   # where to stop
    # Pinning a version is for evaluation: scoring v1 against v2 by
    # flipping the active row would mutate global state and race every
    # other caller. Production sends nothing and gets the active one.
    prompt_version: int | None = None


@router.post("/ingest", status_code=202)
async def ingest(req: IngestRequest, background: BackgroundTasks) -> dict:
    async with Session() as session, session.begin():
        await session.merge(Document(
            id=req.document_id, tenant_id=req.tenant_id, body=req.text,
            status="queued", chunks=0, error=None))
    background.add_task(ingest_document, req.document_id)
    return {"document_id": str(req.document_id), "status": "queued"}


@router.get("/documents/{document_id}")
async def document_status(document_id: uuid.UUID) -> dict:
    async with Session() as session:
        doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "no such document")
    return {"document_id": str(doc.id), "status": doc.status,
            "chunks": doc.chunks, "error": doc.error}


@router.post("/query")
async def query(req: QueryRequest) -> dict:
    embedder = await get_embedder()
    qvec = (await embedder.embed([req.question]))[0]
    async with Session() as session:
        if req.mode == "naive":
            hits = await search(session, req.tenant_id, qvec, k=5)
        elif req.mode == "fusion":
            hits = await fused_search(session, req.tenant_id, req.question,
                                      qvec, k=5)
        else:
            reranker = await get_reranker()
            hits = await hybrid_search(session, req.tenant_id, req.question,
                                       qvec, reranker, k=5,
                                       min_score=settings.min_rerank_score)
    if not hits:                       # nothing relevant enough to answer from
        return {"answer": NOT_FOUND, "chunks": []}

    context = "\n\n".join(f"[{n}] {text}"
                          for n, (_, text, _s) in enumerate(hits, 1))
    prompt = await render("rag_answer",
                          {"context": context, "question": req.question},
                          version=req.prompt_version)
    resp = await call_llm([{"role": "user", "content": prompt}], tools=[])

    return {
        "answer": "".join(b["text"] for b in resp["content"]
                          if b["type"] == "text"),
        "chunks": [{"n": n, "document_id": str(doc_id), "text": text,
                    "score": round(sc, 3)}
                   for n, (doc_id, text, sc) in enumerate(hits, 1)],
    }
