# tests/test_mcp_protocol.py
import pathlib
import sys
import uuid

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.config import settings
from seed_domain import seed_domain

ROOT = pathlib.Path(__file__).resolve().parent.parent

PARAMS = StdioServerParameters(
    command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT))


@pytest.mark.integration
async def test_the_server_advertises_three_tools_with_derived_schemas():
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools

    names = {t.name for t in tools}
    assert names == {"search_docs", "get_order", "list_sync_errors"}

    for t in tools:
        assert t.description and len(t.description) > 40, t.name
        props = t.input_schema["properties"]
        assert props, f"{t.name} has no arguments in its schema"
        assert "tenant_id" not in props, f"{t.name} exposes tenant_id"


@pytest.mark.integration
async def test_calling_a_tool_over_the_protocol_returns_data():
    """Before this fix, this asserted only `result.content` was truthy --
    but get_order("SO-1042") on an unseeded database returns None, and
    None still serialises to a non-empty "null" text content block, so
    a test named "returns data" passed on a call that found nothing.
    Seeding the demo tenant first and reading the order back out of
    structured_content proves the protocol actually carried SO-1042's
    row, not just a truthy wrapper around an absent one."""
    await seed_domain(uuid.UUID(settings.mcp_tenant_id))

    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_order",
                                             {"order_ref": "SO-1042"})

    assert result.structured_content is not None
    order = result.structured_content["result"]
    assert order["order_ref"] == "SO-1042"
    assert order["status"] == "sync_failed"
    assert order["channel"] == "channel_b"
