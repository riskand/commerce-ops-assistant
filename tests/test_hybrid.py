# tests/test_hybrid.py
import uuid

import pytest

from app.db import Session
from app.models import Chunk
from app.retrieval import fts_search, hybrid_search, rrf, search
from tests.fakes import FakeEmbedder, FakeReranker, ReversingReranker

# Every test here seeds `chunks` via _seed(); opt in to the truncate.
pytestmark = pytest.mark.usefixtures("clean_chunks_table")

DOCS = [
    "Refunds are issued within 30 days of an approved return request.",
    "Shipping to remote postcodes takes an extra 5 business days.",
    "Sellers must respond to buyer messages within 24 hours.",
]


async def _seed(tenant: uuid.UUID) -> list[uuid.UUID]:
    vectors = await FakeEmbedder().embed(DOCS)
    doc_ids = [uuid.uuid4() for _ in DOCS]
    async with Session() as session, session.begin():
        session.add_all([
            Chunk(tenant_id=tenant, document_id=doc_id, ordinal=i,
                  content=text, embedding=vector)
            for i, (text, vector, doc_id) in
            enumerate(zip(DOCS, vectors, doc_ids))])
    return doc_ids


@pytest.mark.integration
async def test_fts_matches_a_word_the_vector_fake_cannot_know():
    tenant = uuid.uuid4()
    await _seed(tenant)
    async with Session() as session:
        hits = await fts_search(session, tenant, "refunds", k=10)
    assert [h[1] for h in hits] == [DOCS[0]]


@pytest.mark.integration
async def test_fts_respects_the_tenant_boundary():
    """Asserted by identity, not count: both tenants seed the same three
    texts, so a tenant filter that silently returned theirs's row instead
    of mine's would still leave len(hits) == 1. mine_doc_ids[0] is the
    document_id behind DOCS[0], the only one that matches "refunds" — a
    broken filter surfaces as hits[0][0] belonging to theirs instead."""
    mine, theirs = uuid.uuid4(), uuid.uuid4()
    mine_doc_ids = await _seed(mine)
    theirs_doc_ids = await _seed(theirs)
    async with Session() as session:
        hits = await fts_search(session, mine, "refunds", k=10)
    assert len(hits) == 1
    assert hits[0][0] == mine_doc_ids[0]
    assert hits[0][0] not in theirs_doc_ids


@pytest.mark.integration
async def test_hybrid_search_returns_exactly_k_hits_when_enough_exist():
    tenant = uuid.uuid4()
    await _seed(tenant)
    qvec = (await FakeEmbedder().embed(["refunds"]))[0]
    async with Session() as session:
        hits = await hybrid_search(session, tenant, "refunds", qvec,
                                   FakeReranker(), k=2)
    # Exact, not <=: _seed wrote 3 rows, so anything short of 2 means
    # rows were lost (wrong tenant, a failed insert) rather than trimmed.
    assert len(hits) == 2
    assert all(isinstance(h[0], uuid.UUID) for h in hits)


@pytest.mark.integration
async def test_hybrid_search_actually_applies_the_reranker():
    """The original version of this test used FakeReranker and asserted
    its output was sorted by descending length. For this exact corpus and
    query, the pre-rerank fused order already happens to be sorted by
    descending length, so that assertion passed identically whether or not
    hybrid_search ever called reranker.rerank() — it gave false confidence
    about the one thing it was named for.

    This test computes the fused order the same way hybrid_search does,
    independently of any reranker, then checks that ReversingReranker's
    effect is visible in hybrid_search's actual output. Reversing 3
    distinct documents can never equal the identity permutation, so this
    fails loudly if the rerank call is removed or its result is ignored.

    Verified red-then-green by commenting out the
    `ranked = await reranker.rerank(q, docs)` line in app/retrieval.py
    (falling back to `ranked = list(enumerate(float(len(d)) for d in
    docs))`, i.e. the fused/identity order) and rerunning: this test goes
    red while test_hybrid_search_returns_exactly_k_hits_when_enough_exist
    stays green. Restoring the real call turns it green again."""
    tenant = uuid.uuid4()
    await _seed(tenant)
    qvec = (await FakeEmbedder().embed(["refunds"]))[0]

    async with Session() as session:
        vec = await search(session, tenant, qvec, k=20)
        fts = await fts_search(session, tenant, "refunds", k=20)
    content = {doc_id: text for doc_id, text, _ in vec + fts}
    fused = rrf([d for d, _, _ in vec], [d for d, _, _ in fts])[:20]
    natural_order = [content[d] for d in fused]

    async with Session() as session:
        hits = await hybrid_search(session, tenant, "refunds", qvec,
                                   ReversingReranker(), k=len(fused))

    assert [h[1] for h in hits] == list(reversed(natural_order))


@pytest.mark.integration
async def test_min_score_drops_everything_below_the_floor():
    """hits == [] is also what a broken _seed (wrong tenant, a failed
    migration, truncate ordering) would produce with any min_score, so
    prove a hit exists at the default floor first — only then does an
    empty result at a deliberately impossible floor mean the floor did
    the filtering."""
    tenant = uuid.uuid4()
    await _seed(tenant)
    qvec = (await FakeEmbedder().embed(["refunds"]))[0]

    async with Session() as session:
        hits = await hybrid_search(session, tenant, "refunds", qvec,
                                   FakeReranker(), k=5)
    assert hits != []

    async with Session() as session:
        hits = await hybrid_search(session, tenant, "refunds", qvec,
                                   FakeReranker(), k=5, min_score=1e9)
    assert hits == []
