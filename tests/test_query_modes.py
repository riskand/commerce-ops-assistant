# tests/test_query_modes.py
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.config import settings
from app.db import Session
from app.main import app
from app.models import Chunk, Document
from tests.fakes import FakeEmbedder, ReversingReranker

DOCS = [
    "Refunds are issued within 30 days of an approved return request.",
    "Shipping to remote postcodes takes an extra 5 business days.",
    "Sellers must respond to buyer messages within 24 hours.",
]
QUESTION = "When are refunds issued?"     # the keyword lane matches DOCS[0]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    async def fake_call_llm(messages, tools):
        return {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Within 30 days. [1]"}]}

    monkeypatch.setattr("app.embedder._embedder", FakeEmbedder())
    monkeypatch.setattr("app.reranker._reranker", ReversingReranker())
    monkeypatch.setattr("app.routes.docsearch.call_llm", fake_call_llm)
    monkeypatch.setattr(settings, "contextual_retrieval", False)  # no model at ingest


@pytest.fixture
async def tenant():
    tenant_id = uuid.uuid4()          # its rows are deleted by id, never truncated
    yield tenant_id
    async with Session() as session, session.begin():
        await session.execute(delete(Chunk).where(Chunk.tenant_id == tenant_id))
        await session.execute(
            delete(Document).where(Document.tenant_id == tenant_id))


def ask(client: TestClient, tenant: uuid.UUID, **mode: str) -> list[dict]:
    r = client.post("/query", json={"tenant_id": str(tenant),
                                    "question": QUESTION, **mode})
    assert r.status_code == 200
    return r.json()["chunks"]


def texts(hits: list[dict]) -> list[str]:
    return [h["text"] for h in hits]


@pytest.mark.integration
def test_each_mode_stops_the_pipeline_somewhere_else(tenant):
    with TestClient(app) as client:
        for text in DOCS:
            client.post("/ingest", json={"tenant_id": str(tenant),
                                         "document_id": str(uuid.uuid4()),
                                         "text": text})
        naive = ask(client, tenant, mode="naive")
        fusion = ask(client, tenant, mode="fusion")
        rerank = ask(client, tenant, mode="rerank")
        default = ask(client, tenant)

    assert sorted(texts(naive)) == sorted(DOCS)             # every document
    assert all(h["score"] <= 1.0 for h in naive)            # cosines
    assert all(h["score"] <= round(2 / 61, 3) for h in fusion)   # RRF sums
    assert texts(rerank) == texts(fusion)[::-1]             # the reranker ran
    assert default == rerank                                # rerank by default


def test_an_unknown_mode_is_a_422():
    with TestClient(app) as client:
        r = client.post("/query", json={"tenant_id": str(uuid.uuid4()),
                                        "question": QUESTION, "mode": "bm25"})
    assert r.status_code == 422
