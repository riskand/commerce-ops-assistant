# app/mcp_client.py
"""Translation and dispatch: the bridge between MCP tool definitions and the
Anthropic tools array, and the router that sends a call back to the session
that owns it. This module only discovers and routes — the agent loop that
decides which tools to call lives in app/loop.py."""
import asyncio
import copy
import time
from typing import Any

from app.config import settings

DEFAULT_CALL_TIMEOUT = 30  # seconds; a silent server should not hang a request
BREAKER_THRESHOLD = 5      # consecutive failures before a server is cut off
BREAKER_COOLDOWN = 30.0    # seconds before one probe call is let through


class ToolCallFailed(Exception):
    """Raised when a registered session's call_tool raises or times out, so
    a dead OR merely silent MCP server surfaces as a clean error instead of
    a hang, and when a caller asks for a tool name that was never
    registered."""


def translate(tool) -> dict:
    """One MCP tool definition to one Anthropic tools-array entry. Reads
    tool.name, tool.description and tool.input_schema; returns a new dict
    with exactly those three keys. Never mutates the tool it reads, and
    never hands out a reference into it either: input_schema is a JSON
    schema (nested dicts and lists), and a shallow copy would still share
    every nested value with the tool's own schema, so a deep copy is the
    only version of "doesn't mutate its input" that also holds up once
    something downstream mutates the schema it was handed."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": copy.deepcopy(tool.input_schema),
    }


def _unwrap(result: Any) -> Any:
    """MCP's call_tool returns a CallToolResult, not the tool's own return
    value -- content is a list of TextContent/ImageContent blocks, and
    structured_content wraps the actual value one level down, under
    "result". Unwrapping here, at the MCP boundary, means everything past
    this point (json.dumps in app/loop.py's tool_result, most of all) sees
    the same plain dict/list/str/None a local Python function would have
    returned, exactly like translate() unwraps a schema at the same
    boundary.

    A real CallToolResult can still have structured_content=None: every
    tool this repo ships has a concrete return annotation, so MCPServer
    always populates it, but a public reference server -- the
    filesystem or fetch examples published by the MCP org, say -- returns
    plain text content blocks and nothing else. Discarding that (returning
    None) would make every call to such a server look like a tool that ran
    and found nothing, with no error to explain why; falling back to its
    text keeps the result a caller can actually use."""
    if getattr(result, "is_error", False):
        text = "".join(getattr(b, "text", "") for b in result.content)
        return {"error": text or "tool call reported an error"}
    if result.structured_content is not None:
        return result.structured_content.get("result")
    text = "".join(getattr(b, "text", "") for b in result.content)
    return text or None


class _Breaker:
    """One per server. Closed while calls succeed; opens after `threshold`
    consecutive failures and refuses calls for `cooldown` seconds without
    touching the server; then lets exactly one call through and decides
    again on its result.

    The timeout in ToolRouter.call bounds one call. This bounds the
    server: without it, a flaky dependency costs every chat turn the full
    timeout, so one dead MCP server degrades the whole copilot to the
    speed of its slowest failure. Failing in microseconds is the feature."""

    def __init__(self, threshold: int, cooldown: float) -> None:
        self.threshold, self.cooldown = threshold, cooldown
        self.failures, self.opened_at = 0, 0.0

    def blocked(self, now: float) -> bool:
        return (self.failures >= self.threshold
                and now - self.opened_at < self.cooldown)

    def succeeded(self) -> None:
        self.failures, self.opened_at = 0, 0.0

    def failed(self, now: float) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = now       # re-arms on a failed probe, too


class ToolRouter:
    """The tool_name -> session map. Each server is registered with the
    tools its session exposes; a name already claimed by an earlier
    registration is disambiguated by prefixing the new one with its
    server name."""

    def __init__(self, call_timeout: float = DEFAULT_CALL_TIMEOUT,
                 allowlist: list[str] | None = None,
                 threshold: int = BREAKER_THRESHOLD,
                 cooldown: float = BREAKER_COOLDOWN):
        self._sessions = {}
        self._tools = {}
        self._call_timeout = call_timeout
        # Discovery is dynamic; exposure should not be. An empty list
        # means "whatever the servers offer", which is right for a laptop
        # and wrong for a deployment, where a server adding a tool should
        # not silently hand the model a new capability.
        self._allowlist = (settings.mcp_tool_allowlist if allowlist is None
                           else allowlist)
        self._breakers: dict[str, _Breaker] = {}
        self._threshold, self._cooldown = threshold, cooldown

    def register(self, server_name: str, session, tools: list) -> None:
        for tool in tools:
            if self._allowlist and tool.name not in self._allowlist:
                continue
            name = tool.name
            if name in self._sessions:
                name = f"{server_name}__{tool.name}"
            self._sessions[name] = (server_name, session, tool.name)
            schema = translate(tool)
            schema["name"] = name
            self._tools[name] = schema

    def schemas(self) -> list[dict]:
        return list(self._tools.values())

    async def call(self, name: str, args: dict) -> Any:
        if name not in self._sessions:
            raise ToolCallFailed(f"unknown tool: {name}")
        server_name, session, real_name = self._sessions[name]
        breaker = self._breakers.setdefault(
            server_name, _Breaker(self._threshold, self._cooldown))
        now = time.monotonic()
        if breaker.blocked(now):
            raise ToolCallFailed(
                f"server {server_name!r} is not being called: "
                f"{breaker.failures} consecutive failures, next attempt in "
                f"{breaker.cooldown - (now - breaker.opened_at):.0f}s")
        try:
            result = await asyncio.wait_for(
                session.call_tool(real_name, args), timeout=self._call_timeout)
        except asyncio.TimeoutError as exc:
            # str(asyncio.TimeoutError()) is always "" -- naming the
            # timeout explicitly is the only way this detail says
            # anything at all, rather than trailing off after "failed: ".
            breaker.failed(time.monotonic())
            raise ToolCallFailed(
                f"call to {name!r} on server {server_name!r} failed: "
                f"timed out after {self._call_timeout}s"
            ) from exc
        except Exception as exc:
            breaker.failed(time.monotonic())
            raise ToolCallFailed(
                f"call to {name!r} on server {server_name!r} failed: {exc}"
            ) from exc
        breaker.succeeded()
        return _unwrap(result)
