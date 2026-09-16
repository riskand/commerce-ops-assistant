# tests/test_mcp_tools.py
"""Note on channel naming: seed_domain.py (Task 1) seeds "channel_b" and
"channel_a" with an underscore. list_sync_errors matches its argument
exactly, so the integration tests below query "channel_b" to match the
real seeded rows."""
import inspect
import uuid

import pytest

from app.config import settings
from app.db import Session
from app.models import Chunk
from mcp_server import server
from tests.fakes import FakeEmbedder, FakeReranker


def test_no_tool_accepts_a_tenant_argument():
    """Tenant scope is configuration, not an argument. A tenant_id parameter
    is a value the model can be talked into supplying, which is exactly the
    hole the tenant note exists to close."""
    for fn in (server.search_docs, server.get_order, server.list_sync_errors):
        params = set(inspect.signature(fn).parameters)
        assert "tenant_id" not in params, f"{fn.__name__} takes a tenant_id"


def test_every_tool_is_annotated_and_documented():
    """The SDK derives each tool's schema from annotations and the docstring,
    so an unannotated argument ships a schema the model cannot use."""
    for fn in (server.search_docs, server.get_order, server.list_sync_errors):
        sig = inspect.signature(fn)
        assert fn.__doc__ and len(fn.__doc__.strip()) > 40, fn.__name__
        for name, p in sig.parameters.items():
            assert p.annotation is not inspect.Parameter.empty, \
                f"{fn.__name__}({name}) is unannotated"
        assert sig.return_annotation is not inspect.Parameter.empty


@pytest.fixture
async def seeded_demo_tenant():
    """Seeds the demo tenant's orders and sync_errors before a test reads
    them, rather than assuming someone ran seed_domain.py by hand."""
    from seed_domain import seed_domain
    await seed_domain(uuid.UUID(settings.mcp_tenant_id))


@pytest.mark.integration
@pytest.mark.usefixtures("seeded_demo_tenant")
async def test_get_order_finds_a_seeded_order():
    result = await server.get_order("SO-1042")
    assert result["order_ref"] == "SO-1042"
    assert "status" in result and "channel" in result


@pytest.mark.integration
@pytest.mark.usefixtures("seeded_demo_tenant")
async def test_list_sync_errors_filters_by_channel():
    errors = await server.list_sync_errors("channel_b")
    assert errors, "seed_domain.py should have created a channel_b failure"
    assert all(e["channel"] == "channel_b" for e in errors)


@pytest.mark.integration
async def test_tools_cannot_reach_another_tenants_rows(monkeypatch):
    """The whole point of the tenant note, as a test."""
    monkeypatch.setattr("app.config.settings.mcp_tenant_id", str(uuid.uuid4()))
    assert await server.get_order("SO-1042") is None
    assert await server.list_sync_errors("channel_b") == []


@pytest.mark.integration
@pytest.mark.usefixtures("clean_chunks_table")
async def test_search_docs_cannot_reach_another_tenants_chunks(monkeypatch):
    """get_order and list_sync_errors are proven above; search_docs is the
    one tool whose corpus is shared across every tenant, so a wrong
    variable in the line that passes _tenant() into hybrid_search would
    leak another tenant's documents into the model's context. app/
    retrieval.py's own search() is already tenant-tested (see
    test_retrieval.py), so this is not re-testing the query — it is
    testing that search_docs actually forwards the scoped tenant rather
    than, say, a stray argument or a hardcoded id.

    Both tenants get a chunk with the *same* content and a distinct
    document_id, mirroring test_retrieval.py's approach: identical text
    means a dropped tenant filter surfaces the other tenant's row in the
    same top-k rather than just changing a score, which a count-only
    assertion could miss."""
    monkeypatch.setattr("app.embedder._embedder", FakeEmbedder())
    monkeypatch.setattr("app.reranker._reranker", FakeReranker())

    mine, theirs = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr("app.config.settings.mcp_tenant_id", str(mine))

    content = "Refunds are issued within 30 days of an approved return."
    vector = (await FakeEmbedder().embed([content]))[0]
    mine_doc_id, theirs_doc_id = uuid.uuid4(), uuid.uuid4()

    async with Session() as session, session.begin():
        session.add_all([
            Chunk(tenant_id=mine, document_id=mine_doc_id, ordinal=0,
                  content=content, embedding=vector),
            Chunk(tenant_id=theirs, document_id=theirs_doc_id, ordinal=0,
                  content=content, embedding=vector),
        ])

    hits = await server.search_docs("refund window")

    doc_ids = {h["document_id"] for h in hits}
    assert doc_ids == {str(mine_doc_id)}
    assert str(theirs_doc_id) not in doc_ids
