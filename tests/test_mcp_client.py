# tests/test_mcp_client.py
import copy

import pytest

from app.mcp_client import ToolCallFailed, ToolRouter, translate


class FakeTool:
    def __init__(self, name, description="a tool that does a thing, at length",
                 schema=None):
        self.name = name
        self.description = description
        self.input_schema = schema or {"type": "object",
                                        "properties": {"q": {"type": "string"}},
                                        "required": ["q"]}


class FakeCallToolResult:
    """Shaped like the real mcp.types.CallToolResult -- structured_content
    wraps the tool's return value under "result", content carries text
    blocks -- not like a bare string. A FakeSession that returned a plain
    string here is exactly what let a prior version of app/mcp_client.py's
    ToolRouter._unwrap keep a dead "no structured_content attribute"
    branch alive: the branch existed solely because this fake disagreed
    with the real type it stood in for. See tests/test_chat_route.py's
    FakeCallToolResult, which this mirrors, for the same shape and the
    same reasoning."""

    def __init__(self, value):
        self.structured_content = {"result": value}
        self.content = []
        self.is_error = False


class FakeSession:
    def __init__(self, result="ok", raises=None):
        self.result = FakeCallToolResult(result)
        self.raises, self.calls = raises, []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if self.raises:
            raise self.raises
        return self.result


def test_translate_maps_the_three_fields_the_api_wants():
    out = translate(FakeTool("search_docs"))
    assert out["name"] == "search_docs"
    assert out["description"].startswith("a tool")
    assert out["input_schema"]["properties"]["q"]["type"] == "string"
    assert set(out) == {"name", "description", "input_schema"}


def test_translate_is_pure():
    tool = FakeTool("get_order")
    before = dict(tool.input_schema)
    translate(tool)
    assert tool.input_schema == before


def test_translate_returns_an_independent_copy_of_the_schema():
    """translate hands out a schema the model and downstream code may
    inspect and even annotate; if it were the tool's own dict, a mutation
    anywhere downstream — including a nested field — would silently
    corrupt the schema the server advertised for every future call."""
    tool = FakeTool("get_order")
    before = copy.deepcopy(tool.input_schema)

    out = translate(tool)
    out["input_schema"]["properties"]["q"]["type"] = "integer"
    out["input_schema"]["properties"]["new_field"] = {"type": "boolean"}

    assert tool.input_schema == before


async def test_a_tool_routes_to_the_session_that_owns_it():
    a, b = FakeSession("from-a"), FakeSession("from-b")
    r = ToolRouter()
    r.register("alpha", a, [FakeTool("one")])
    r.register("beta", b, [FakeTool("two")])
    assert await r.call("two", {"q": "x"}) == "from-b"
    assert a.calls == [] and b.calls == [("two", {"q": "x"})]


async def test_a_colliding_name_is_prefixed_with_its_server():
    a, b = FakeSession("from-a"), FakeSession("from-b")
    r = ToolRouter()
    r.register("alpha", a, [FakeTool("search")])
    r.register("beta", b, [FakeTool("search")])

    names = {s["name"] for s in r.schemas()}
    assert names == {"search", "beta__search"}

    assert await r.call("search", {"q": "x"}) == "from-a"
    assert await r.call("beta__search", {"q": "x"}) == "from-b"
    assert b.calls == [("search", {"q": "x"})], "prefix must be stripped"


async def test_a_dead_server_raises_rather_than_hanging():
    """Killing one MCP server produces a clean error,
    not a hung request."""
    r = ToolRouter()
    r.register("alpha", FakeSession(raises=ConnectionError("pipe closed")),
               [FakeTool("one")])
    with pytest.raises(ToolCallFailed) as excinfo:
        await r.call("one", {"q": "x"})
    assert "alpha" in str(excinfo.value)


async def test_an_unknown_tool_raises_rather_than_returning_none():
    r = ToolRouter()
    with pytest.raises(ToolCallFailed):
        await r.call("nope", {})


class _Text:
    def __init__(self, text):
        self.text = text


class FakeUnannotatedResult:
    """Shaped like a real mcp.types.CallToolResult from a tool with no
    return annotation: structured_content is None (there is no return
    value to wrap), content carries only text blocks. This is exactly
    what the public filesystem and fetch reference servers return, and
    what none of this repo's own three tools
    ever do -- see tests/test_chat_route.py's FakeCallToolResult for the
    shape a *successful, annotated* tool result takes, and its comment
    for why a fake must mirror the real type instead of handing back a
    plain value."""

    def __init__(self, text):
        self.structured_content = None
        self.content = [_Text(text)]
        self.is_error = False


class FakeSessionReturningUnannotatedResult:
    def __init__(self, text):
        self.result = FakeUnannotatedResult(text)

    async def call_tool(self, name, args):
        return self.result


async def test_a_result_with_no_structured_content_falls_back_to_its_text():
    """Before this fix, _unwrap returned None here every time, silently --
    not a bug for any tool this repo ships (all three carry concrete
    return annotations, so MCPServer always populates structured_content)
    but very much one for a public reference server, which returns plain
    text content blocks and nothing else. Without this fallback, every
    call to such a server would return None, with no error telling the
    caller why."""
    r = ToolRouter()
    r.register("fs", FakeSessionReturningUnannotatedResult(
        "hello.txt\nworld.txt"), [FakeTool("list_dir")])
    assert await r.call("list_dir", {}) == "hello.txt\nworld.txt"
