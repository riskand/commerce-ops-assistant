# tests/test_reindex.py
"""Changing the embedding model without a full rebuild."""
import uuid

import pytest
from sqlalchemy import select

import reindex
from app.db import Session
from app.models import Chunk
from tests.fakes import FakeEmbedder


class NewModel(FakeEmbedder):
    """A different model: same shape, different vectors."""
    name = "new-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await super().embed([f"v2::{t}" for t in texts])


@pytest.fixture
def new_model(monkeypatch):
    embedder = NewModel()
    monkeypatch.setattr("app.config.settings.embedding_model",
                        "new-model")
    monkeypatch.setattr("app.embedder._embedder", embedder)
    return embedder


async def _insert(tenant, document, texts, model):
    old = await FakeEmbedder().embed(texts)
    async with Session() as session, session.begin():
        session.add_all([
            Chunk(tenant_id=tenant, document_id=document, ordinal=i,
                  content=t, embedding=v, embedding_model=model)
            for i, (t, v) in enumerate(zip(texts, old))])


async def _models(document) -> list[str]:
    async with Session() as session:
        return list(await session.scalars(
            select(Chunk.embedding_model)
            .where(Chunk.document_id == document)))


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
async def test_only_the_stale_chunks_are_counted_and_re_embedded(new_model):
    tenant, stale_doc, fresh_doc = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _insert(tenant, stale_doc, ["one", "two", "three"], "old-model")
    await _insert(tenant, fresh_doc, ["four"], "new-model")

    assert await reindex.stale_count() == 3
    assert await reindex.reindex(page=2) == 3       # two pages, then empty
    assert await reindex.stale_count() == 0
    assert await _models(stale_doc) == ["new-model"] * 3


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
async def test_the_vectors_actually_change(new_model):
    tenant, document = uuid.uuid4(), uuid.uuid4()
    await _insert(tenant, document, ["returns policy"], "old-model")
    async with Session() as session:
        before = list(await session.scalars(
            select(Chunk.embedding).where(Chunk.document_id == document)))

    await reindex.reindex()

    async with Session() as session:
        after = list(await session.scalars(
            select(Chunk.embedding).where(Chunk.document_id == document)))
    assert list(after[0]) != list(before[0])
    expected = (await new_model.embed(["returns policy"]))[0]
    assert [round(v, 6) for v in after[0]] == [round(v, 6) for v in expected]


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
async def test_a_second_run_is_a_no_op(new_model):
    tenant, document = uuid.uuid4(), uuid.uuid4()
    await _insert(tenant, document, ["one", "two"], "old-model")
    assert await reindex.reindex() == 2
    assert await reindex.reindex() == 0
