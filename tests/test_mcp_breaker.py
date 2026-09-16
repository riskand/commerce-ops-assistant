# tests/test_mcp_breaker.py
"""One flaky server must not cost every chat turn."""
import asyncio

import pytest

from app.mcp_client import ToolCallFailed, ToolRouter


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = "a tool that does a thing, described at length"
        self.input_schema = {"type": "object"}


class Flaky:
    """Fails until `heal()` is called, then succeeds."""

    def __init__(self):
        self.calls, self.healthy = 0, False

    async def call_tool(self, name, args):
        self.calls += 1
        if not self.healthy:
            raise ConnectionError("pipe closed")
        return _Result("fine now")

    def heal(self):
        self.healthy = True


class _Result:
    def __init__(self, value):
        self.structured_content = {"result": value}
        self.content, self.is_error = [], False


def _router(session, **kw):
    r = ToolRouter(allowlist=[], **kw)
    r.register("alpha", session, [FakeTool("search_docs")])
    return r


async def _fail(router, times):
    for _ in range(times):
        with pytest.raises(ToolCallFailed):
            await router.call("search_docs", {})


async def test_the_server_is_still_called_below_the_threshold():
    session = Flaky()
    router = _router(session, threshold=3, cooldown=60)
    await _fail(router, 2)
    assert session.calls == 2


async def test_the_breaker_opens_and_stops_touching_the_server():
    session = Flaky()
    router = _router(session, threshold=3, cooldown=60)
    await _fail(router, 3)
    assert session.calls == 3
    with pytest.raises(ToolCallFailed, match="is not being called"):
        await router.call("search_docs", {})
    assert session.calls == 3          # the fourth call never left the process


async def test_an_open_breaker_fails_fast_rather_than_waiting():
    """The point of the breaker: a call that would have burned the whole
    timeout returns immediately instead."""
    class Silent:
        async def call_tool(self, name, args):
            await asyncio.Event().wait()

    router = _router(Silent(), threshold=1, cooldown=60, call_timeout=0.2)
    with pytest.raises(ToolCallFailed, match="timed out"):
        await router.call("search_docs", {})
    started = asyncio.get_running_loop().time()
    with pytest.raises(ToolCallFailed, match="is not being called"):
        await router.call("search_docs", {})
    assert asyncio.get_running_loop().time() - started < 0.05


async def test_after_the_cooldown_one_call_is_let_through_and_can_close_it():
    session = Flaky()
    router = _router(session, threshold=2, cooldown=0.05)
    await _fail(router, 2)
    with pytest.raises(ToolCallFailed, match="is not being called"):
        await router.call("search_docs", {})
    await asyncio.sleep(0.06)
    session.heal()
    assert await router.call("search_docs", {}) == "fine now"
    assert await router.call("search_docs", {}) == "fine now"   # closed


async def test_a_failed_probe_re_opens_the_breaker():
    session = Flaky()
    router = _router(session, threshold=2, cooldown=0.05)
    await _fail(router, 2)
    await asyncio.sleep(0.06)
    await _fail(router, 1)                     # the probe fails
    assert session.calls == 3
    with pytest.raises(ToolCallFailed, match="is not being called"):
        await router.call("search_docs", {})
    assert session.calls == 3                  # shut again, not probing


async def test_a_success_resets_the_failure_count():
    session = Flaky()
    router = _router(session, threshold=3, cooldown=60)
    await _fail(router, 2)
    session.heal()
    await router.call("search_docs", {})
    session.healthy = False
    await _fail(router, 2)                     # count restarted, not at 4
    assert session.calls == 5


async def test_the_allowlist_decides_what_is_exposed():
    router = ToolRouter(allowlist=["search_docs"])
    router.register("alpha", Flaky(),
                    [FakeTool("search_docs"), FakeTool("delete_everything")])
    assert [t["name"] for t in router.schemas()] == ["search_docs"]
    with pytest.raises(ToolCallFailed, match="unknown tool"):
        await router.call("delete_everything", {})


async def test_an_empty_allowlist_exposes_whatever_was_discovered():
    router = ToolRouter(allowlist=[])
    router.register("alpha", Flaky(),
                    [FakeTool("search_docs"), FakeTool("read_file")])
    assert {t["name"] for t in router.schemas()} == {"search_docs",
                                                     "read_file"}
