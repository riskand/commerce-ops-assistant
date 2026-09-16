# tests/test_chat_integration.py
"""The two-server path, exercised against the real protocol rather than
fakes: two live sessions, one merged tool list. The public filesystem and
fetch reference servers need `npx` and a network, so they stay in the
documented manual run; this spawns two subprocesses of our own server
instead, which needs
neither, and collides on purpose by registering both under names that
already overlap -- proving the same code path ToolRouter.register uses
for two genuinely different servers."""
import pathlib
import sys
import uuid

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from sqlalchemy import delete

from app.db import Session
from app.mcp_client import ToolRouter
from app.models import Order

ROOT = pathlib.Path(__file__).resolve().parent.parent

TOOL_NAMES = {"search_docs", "get_order", "list_sync_errors"}


def _params(tenant_id: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"],
        cwd=str(ROOT), env={"MCP_TENANT_ID": tenant_id})


@pytest.mark.integration
async def test_two_live_servers_merge_into_one_prefixed_tool_list():
    """Before this fix, both calls below answered None -- which is also
    exactly what the earlier _unwrap bug returned for every call,
    regardless of routing. A test that only checks "both are None" cannot
    tell correct unwrapping-and-routing apart from that bug. Seeding each
    tenant with its own distinct, real order and asserting each prefixed
    name comes back with the right tenant's row proves both unwrapping
    (a real dict came back, not a bug's None) and routing (the right
    session answered) in one assertion."""
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    async with Session() as session, session.begin():
        session.add_all([
            Order(id=uuid.uuid4(), tenant_id=tenant_a, order_ref="SO-9001",
                  status="pending", channel="channel_a", line_items=[]),
            Order(id=uuid.uuid4(), tenant_id=tenant_b, order_ref="SO-9002",
                  status="fulfilled", channel="channel_b", line_items=[]),
        ])

    router = ToolRouter()
    try:
        async with stdio_client(_params(str(tenant_a))) as (read_a, write_a):
            async with ClientSession(read_a, write_a) as session_a:
                await session_a.initialize()
                tools_a = (await session_a.list_tools()).tools

                async with stdio_client(_params(str(tenant_b))) as (
                        read_b, write_b):
                    async with ClientSession(read_b, write_b) as session_b:
                        await session_b.initialize()
                        tools_b = (await session_b.list_tools()).tools

                        router.register("alpha", session_a, tools_a)
                        router.register("beta", session_b, tools_b)

                        names = {s["name"] for s in router.schemas()}
                        assert names == TOOL_NAMES | {
                            f"beta__{n}" for n in TOOL_NAMES}

                        # each prefixed name routes to the server that
                        # owns it, and each server answers only for its
                        # own tenant: router.call already unwrapped the
                        # CallToolResult by the time it gets here (see
                        # app/mcp_client.py:_unwrap).
                        result_a = await router.call(
                            "get_order", {"order_ref": "SO-9001"})
                        result_b = await router.call(
                            "beta__get_order", {"order_ref": "SO-9002"})
                        assert result_a["order_ref"] == "SO-9001"
                        assert result_a["status"] == "pending"
                        assert result_b["order_ref"] == "SO-9002"
                        assert result_b["status"] == "fulfilled"

                        # cross-tenant misses stay None -- alpha's session
                        # cannot see tenant_b's order under either name.
                        assert await router.call(
                            "get_order", {"order_ref": "SO-9002"}) is None
    finally:
        async with Session() as session, session.begin():
            await session.execute(
                delete(Order).where(
                    Order.tenant_id.in_([tenant_a, tenant_b])))


@pytest.mark.integration
def test_real_lifespan_registers_real_tools(monkeypatch):
    """The gap a manual smoke script cannot close: the lifespan this repo
    actually ships (app/config.py's default mcp_servers, not a hand-built
    one) run through the app the way uvicorn would run it. The autouse
    fixture in conftest.py blanks settings.mcp_servers for every test;
    this test's own monkeypatch runs after that fixture's setup, so it
    can simply put the default back."""
    from fastapi.testclient import TestClient

    from app.config import Settings, settings
    from app.main import app
    from app.routes import chat

    monkeypatch.setattr(settings, "mcp_servers", Settings().mcp_servers)

    with TestClient(app):
        names = {s["name"] for s in chat.router_for_request().schemas()}

    assert TOOL_NAMES <= names


@pytest.mark.integration
def test_chat_completes_a_real_tool_call_through_the_real_lifespan(
        monkeypatch):
    """The extension of the test above that would have caught the
    CallToolResult-is-not-JSON-serialisable bug: connect for real, then
    actually finish a /chat request that calls a real tool, rather than
    only checking what got registered."""
    from fastapi.testclient import TestClient

    from app.config import Settings, settings
    from app.main import app
    from app.routes import chat

    monkeypatch.setattr(settings, "mcp_servers", Settings().mcp_servers)

    responses = [
        {"stop_reason": "tool_use",
         "content": [{"type": "tool_use", "id": "toolu_1",
                      "name": "get_order", "input": {"order_ref": "SO-1042"}}]},
        {"stop_reason": "end_turn",
         "content": [{"type": "text", "text": "done"}]},
    ]

    async def fake(messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(chat, "_call_llm", fake)
    with TestClient(app) as client:
        resp = client.post("/chat", json={"messages": [
            {"role": "user", "content": "look up SO-1042"}]})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "done"
