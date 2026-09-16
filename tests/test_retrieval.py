# tests/test_retrieval.py
import uuid

import pytest
from sqlalchemy import select

from app.db import Session
from app.models import Chunk
from app.retrieval import fts_search, search
from tests.fakes import FakeEmbedder

# Both tests here insert rows into `chunks`; opt in to the truncate.
pytestmark = pytest.mark.usefixtures("clean_chunks_table")


@pytest.mark.integration
async def test_search_orders_by_distance_and_isolates_the_tenant():
    """Guards app/retrieval.py's `.where(Chunk.tenant_id == tenant_id)` —
    the comment there calls it "tenant isolation IS this line". Isolation
    is checked by identity (document_id), not by count or score: theirs's
    row carries the *exact* embedding of the query (a tie for best
    possible distance) and its own distinct document_id, so if the tenant
    filter were ever dropped it would either appear in mine's result set
    outright, or silently displace one of mine's own three documents from
    the top-k — either way `mine_ids` stops matching `set(mine_doc_ids)`
    exactly. Verified red-then-green: removing the `.where(...)` clause
    from app/retrieval.py and rerunning this test fails on
    `theirs_doc_id not in mine_ids`; restoring it passes again (see the
    report's Fix round 2 section for the transcript)."""
    mine, theirs = uuid.uuid4(), uuid.uuid4()
    embedder = FakeEmbedder()
    texts = ["refunds within 30 days", "shipping takes 5 days",
             "a completely unrelated clause"]
    vectors = await embedder.embed(texts)
    mine_doc_ids = [uuid.uuid4() for _ in texts]
    theirs_doc_id = uuid.uuid4()

    async with Session() as session, session.begin():
        session.add_all([
            Chunk(tenant_id=mine, document_id=mine_doc_ids[i], ordinal=i,
                  content=t, embedding=v)
            for i, (t, v) in enumerate(zip(texts, vectors))])
        session.add(Chunk(tenant_id=theirs, document_id=theirs_doc_id,
                          ordinal=0, content=texts[0],
                          embedding=vectors[0]))

    qvec = (await embedder.embed([texts[0]]))[0]
    async with Session() as session:
        mine_hits = await search(session, mine, qvec, k=3)
        theirs_hits = await search(session, theirs, qvec, k=3)

    mine_ids = {doc_id for doc_id, _text, _score in mine_hits}
    theirs_ids = {doc_id for doc_id, _text, _score in theirs_hits}

    assert theirs_doc_id not in mine_ids
    assert mine_ids == set(mine_doc_ids)
    assert theirs_ids == {theirs_doc_id}

    assert mine_hits[0][1] == texts[0]
    assert mine_hits[0][2] > mine_hits[-1][2]


@pytest.mark.integration
async def test_the_generated_fts_column_populates_and_is_searchable():
    """Guards against a plain, non-generated fts column ever coming back:
    the migration must compute fts from content on write, not leave it
    NULL. Proven two ways — the column itself is non-NULL after a plain
    ORM insert (no trigger, no application code touched it), and
    fts_search's websearch_to_tsquery/@@ actually matches a word that only
    appears in content."""
    tenant = uuid.uuid4()
    embedder = FakeEmbedder()
    content = "Returns are accepted within 30 days of delivery."
    vector = (await embedder.embed([content]))[0]

    async with Session() as session, session.begin():
        session.add(Chunk(tenant_id=tenant, document_id=uuid.uuid4(),
                          ordinal=0, content=content, embedding=vector))

    async with Session() as session:
        fts_value = (await session.execute(
            select(Chunk.fts).where(Chunk.tenant_id == tenant)
        )).scalar_one()
        assert fts_value is not None

        hits = await fts_search(session, tenant, "returns", k=5)

    assert len(hits) == 1
    assert hits[0][1] == content
