# tests/test_live.py
"""The tests that touch the real embedding model and the real API. All
three are integration-marked: the first downloads ~1.3GB on a cold
cache, the other two spend money. Each skips cleanly (rather than
failing) when what it needs isn't there, so a red result in this file
always means a real problem, not a missing local dependency."""
import contextlib
import pathlib
import sys
import uuid

import httpx
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.config import settings
from app.embedder import DIMS, LocalEmbedder
from app.loop import TOOL_SCHEMAS, call_llm, final_text, tool_result
from app.mcp_client import ToolRouter

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.integration
async def test_the_real_embedder_matches_what_the_schema_declares():
    pytest.importorskip(
        "sentence_transformers",
        reason="requires the sentence-transformers package, which is not "
               "installed (pip install -r requirements-local.txt)")

    vectors = await LocalEmbedder().embed(
        ["Returns are accepted within 30 days.",
         "Refunds take up to a month after an approved return.",
         "Shipping to remote postcodes takes five extra days."])

    assert all(len(v) == DIMS for v in vectors)
    for v in vectors:
        assert abs(sum(x * x for x in v) - 1.0) < 1e-4   # normalize_embeddings

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    # unit length makes cosine similarity a dot product, which is the whole
    # reason normalize_embeddings=True is there
    assert dot(vectors[0], vectors[1]) > dot(vectors[0], vectors[2])


@pytest.mark.integration
async def test_one_live_round_trip_asks_for_the_tool():
    try:
        resp = await call_llm(
            [{"role": "user",
              "content": "Look up order SO-1042 and tell me its sync state."}],
            TOOL_SCHEMAS)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            pytest.skip("requires a real ANTHROPIC_API_KEY; got 401 "
                        "Unauthorized from the placeholder/missing key")
        raise

    assert resp["stop_reason"] == "tool_use"
    used = [b for b in resp["content"] if b["type"] == "tool_use"]
    assert [b["name"] for b in used] == ["get_order"]
    assert used[0]["input"]["order_id"] == "SO-1042"
    assert used[0]["id"].startswith("toolu_")


@pytest.mark.integration
async def test_mcp_tools_driven_by_a_real_model_end_to_end():
    """The gap the test above leaves: that one calls the fixed
    TOOL_SCHEMAS directly, never touching MCP. This spawns the real
    server over stdio, discovers its tools the way app/routes/chat.py's
    lifespan does, and drives a real model through a full tool call and
    back to a final answer, exercised for
    real rather than through fakes (see tests/test_chat_integration.py
    for the fake-model version of this same path).

    Needs the same docker-compose database most of this file's
    integration siblings need (get_order queries it), plus a real
    ANTHROPIC_API_KEY -- skips cleanly, matching the round-trip test
    above, when the key is a placeholder.

    The skip has to happen after the stdio session below has closed,
    not while pytest.skip()'s exception is still unwinding through it:
    stdio_client and ClientSession both run their own anyio task groups,
    and letting the skip propagate straight through those re-wraps it as
    an opaque "unhandled errors in a TaskGroup", which still skips but
    hides the actual reason. contextlib.AsyncExitStack lets the whole
    session close cleanly first; the skip (if any) fires outside it."""
    from seed_domain import seed_domain

    await seed_domain(uuid.UUID(settings.mcp_tenant_id))

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"],
        cwd=str(ROOT))

    skip_reason = None
    async with contextlib.AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = (await session.list_tools()).tools
        router = ToolRouter()
        router.register("saas-tools", session, tools)

        messages = [{"role": "user",
                    "content": "Look up order SO-1042 and tell me its "
                               "sync state."}]
        try:
            resp = await call_llm(messages, router.schemas())
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 401:
                raise
            skip_reason = ("requires a real ANTHROPIC_API_KEY; got 401 "
                          "Unauthorized from the placeholder/missing key")
        else:
            assert resp["stop_reason"] == "tool_use"
            used = [b for b in resp["content"] if b["type"] == "tool_use"]
            assert used and used[0]["name"] == "get_order"

            messages.append({"role": "assistant", "content": resp["content"]})
            out = await router.call(used[0]["name"], used[0]["input"])
            assert out["order_ref"] == "SO-1042"
            messages.append({"role": "user", "content": [
                tool_result(used[0]["id"], out)]})

            resp = await call_llm(messages, router.schemas())
            assert resp["stop_reason"] != "tool_use"
            assert final_text(resp), "expected a final answer, not silence"

    if skip_reason:
        pytest.skip(skip_reason)
