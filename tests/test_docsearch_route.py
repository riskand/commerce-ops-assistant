# tests/test_docsearch_route.py
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.embedder import DIMS
from app.main import app
from app.routes.docsearch import NOT_FOUND
from tests.fakes import FakeEmbedder, FakeReranker


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr("app.embedder._embedder", FakeEmbedder())
    monkeypatch.setattr("app.reranker._reranker", FakeReranker())


async def test_the_fake_embedder_matches_the_declared_dimensions():
    vectors = await FakeEmbedder().embed(["one", "two"])
    assert len(vectors) == 2
    assert all(len(v) == DIMS for v in vectors)
    assert abs(sum(v * v for v in vectors[0]) - 1.0) < 1e-6


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_ingest_then_query_answers_from_the_ingested_text(monkeypatch):
    tenant = uuid.uuid4()

    async def fake_call_llm(messages, tools):
        return {"stop_reason": "end_turn",
                "content": [{"type": "text",
                             "text": "Thirty days. [1]"}]}

    monkeypatch.setattr("app.routes.docsearch.call_llm", fake_call_llm)
    with TestClient(app) as client:
        document = uuid.uuid4()
        ingested = client.post("/ingest", json={
            "tenant_id": str(tenant),
            "document_id": str(document),
            "text": "Returns are accepted within 30 days of delivery.",
        })
        assert ingested.status_code == 202
        status = client.get(f"/documents/{document}").json()
        assert status["status"] == "ready"
        assert status["chunks"] == 1

        queried = client.post("/query", json={
            "tenant_id": str(tenant),
            "question": "How long do buyers have to return an item?",
        })

    assert queried.status_code == 200
    body = queried.json()
    assert body["answer"] == "Thirty days. [1]"
    assert len(body["chunks"]) == 1


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_a_question_below_the_floor_is_refused_without_calling_the_model(
        monkeypatch):
    """The refusal has to happen before generation, not inside it: a model
    handed the nearest chunks answers from them, relevant or not."""
    tenant = uuid.uuid4()

    async def must_not_run(messages, tools):
        raise AssertionError("call_llm was reached for a refused question")

    monkeypatch.setattr("app.routes.docsearch.call_llm", must_not_run)
    monkeypatch.setattr(settings, "min_rerank_score", 1e9)
    with TestClient(app) as client:
        document = uuid.uuid4()
        client.post("/ingest", json={
            "tenant_id": str(tenant),
            "document_id": str(document),
            "text": "Returns are accepted within 30 days of delivery.",
        })
        assert client.get(f"/documents/{document}").json()["status"] == "ready"
        queried = client.post("/query", json={
            "tenant_id": str(tenant),
            "question": "How long do buyers have to return an item?",
        })

    assert queried.status_code == 200
    assert queried.json() == {"answer": NOT_FOUND, "chunks": []}
