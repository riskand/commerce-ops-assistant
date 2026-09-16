# tests/test_ingest.py
"""Ingestion off the request path."""
import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import ingest
from app.config import settings
from app.db import Session
from app.embedder import DIMS
from app.main import app
from app.models import Chunk
from tests.fakes import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    name = "counting-fake"

    def __init__(self) -> None:
        self.batches: list[int] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return await super().embed(texts)


async def test_embedding_is_batched_and_keeps_its_order():
    embedder = CountingEmbedder()
    texts = [f"chunk {i}" for i in range(ingest.BATCH * 2 + 6)]
    vectors = await ingest.embed_all(embedder, texts)
    assert embedder.batches == [ingest.BATCH, ingest.BATCH, 6]
    assert len(vectors) == len(texts)
    assert all(len(v) == DIMS for v in vectors)
    # order preserved across batch boundaries: the vector for the first
    # chunk of the third batch is the one that embedder produced for it
    solo = await CountingEmbedder().embed([texts[ingest.BATCH * 2]])
    assert vectors[ingest.BATCH * 2] == solo[0]


@pytest.fixture
def fake_embedder(monkeypatch):
    embedder = CountingEmbedder()
    monkeypatch.setattr("app.embedder._embedder", embedder)
    return embedder


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_ingest_returns_immediately_and_the_status_goes_ready(fake_embedder):
    tenant, document = uuid.uuid4(), uuid.uuid4()
    with TestClient(app) as client:
        accepted = client.post("/ingest", json={
            "tenant_id": str(tenant), "document_id": str(document),
            "text": "Returns are accepted within 30 days of delivery.",
        })
        assert accepted.status_code == 202
        assert accepted.json() == {"document_id": str(document),
                                   "status": "queued"}
        # TestClient runs background tasks before returning, so by here
        # the work a real deployment would still be doing is done.
        status = client.get(f"/documents/{document}").json()
    assert status["status"] == "ready"
    assert status["chunks"] == 1
    assert status["error"] is None


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_every_chunk_records_the_model_that_embedded_it(fake_embedder):
    tenant, document = uuid.uuid4(), uuid.uuid4()
    with TestClient(app) as client:
        client.post("/ingest", json={
            "tenant_id": str(tenant), "document_id": str(document),
            "text": "Payouts settle on the fifteenth of each month.",
        })

    async def models() -> list[str]:
        async with Session() as session:
            rows = await session.execute(
                select(Chunk.embedding_model).where(
                    Chunk.document_id == document))
            return list(rows.scalars())

    recorded = asyncio.run(models())
    assert recorded and set(recorded) == {"counting-fake"}


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_a_failing_embedder_leaves_a_terminal_failed_status(monkeypatch):
    class Broken(FakeEmbedder):
        name = "broken"

        async def embed(self, texts):
            raise RuntimeError("model is not loaded")

    monkeypatch.setattr("app.embedder._embedder", Broken())
    tenant, document = uuid.uuid4(), uuid.uuid4()
    with TestClient(app) as client:
        assert client.post("/ingest", json={
            "tenant_id": str(tenant), "document_id": str(document),
            "text": "anything at all",
        }).status_code == 202
        status = client.get(f"/documents/{document}").json()
    assert status["status"] == "failed"
    assert "model is not loaded" in status["error"]


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_re_ingesting_replaces_the_chunks_rather_than_doubling_them(
        fake_embedder):
    tenant, document = uuid.uuid4(), uuid.uuid4()
    body = {"tenant_id": str(tenant), "document_id": str(document),
            "text": "Returns are accepted within 30 days of delivery."}
    with TestClient(app) as client:
        client.post("/ingest", json=body)
        client.post("/ingest", json=body)
        status = client.get(f"/documents/{document}").json()
    assert status["chunks"] == 1


@pytest.mark.integration
def test_an_unknown_document_is_a_404():
    with TestClient(app) as client:
        assert client.get(f"/documents/{uuid.uuid4()}").status_code == 404


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_contextual_retrieval_stores_the_note_with_the_chunk(fake_embedder,
                                                            monkeypatch):
    """The note has to reach `content`, not only the vector: the fts column
    is computed from content, which is how the keyword lane sees it too."""
    async def fake_situate(document, chunk_text):
        return "From the returns policy."

    monkeypatch.setattr("app.ingest.situate", fake_situate)
    monkeypatch.setattr(settings, "contextual_retrieval", True)
    tenant, document = uuid.uuid4(), uuid.uuid4()
    with TestClient(app) as client:
        client.post("/ingest", json={
            "tenant_id": str(tenant), "document_id": str(document),
            "text": "Returns are accepted within 30 days of delivery.",
        })

    async def contents() -> list[str]:
        async with Session() as session:
            rows = await session.execute(
                select(Chunk.content).where(Chunk.document_id == document))
            return list(rows.scalars())

    assert asyncio.run(contents()) == [
        "From the returns policy.\n\n"
        "Returns are accepted within 30 days of delivery."]


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
def test_with_contextual_retrieval_off_no_model_is_called(fake_embedder,
                                                         monkeypatch):
    async def must_not_run(document, chunk_text):
        raise AssertionError("situate ran with contextual_retrieval off")

    monkeypatch.setattr("app.ingest.situate", must_not_run)
    monkeypatch.setattr(settings, "contextual_retrieval", False)
    tenant, document = uuid.uuid4(), uuid.uuid4()
    with TestClient(app) as client:
        client.post("/ingest", json={
            "tenant_id": str(tenant), "document_id": str(document),
            "text": "Returns are accepted within 30 days of delivery.",
        })
        status = client.get(f"/documents/{document}").json()
    assert status["status"] == "ready"
