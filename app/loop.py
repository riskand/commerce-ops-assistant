# app/loop.py
import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from app.config import settings
from app.tools import get_order, GET_ORDER_SCHEMA

MAX_ATTEMPTS = 4
TIMEOUT = httpx.Timeout(60.0, connect=5.0)
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
CALLS_LOG = Path(__file__).resolve().parents[1] / "calls.jsonl"
LOCAL_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
TOOLS: dict[str, Callable] = {"get_order": get_order}
TOOL_SCHEMAS = [GET_ORDER_SCHEMA]

async def post(client: httpx.AsyncClient, model: str, messages: list[dict],
               tools: list[dict], system: str | None = None) -> httpx.Response:
    body = {"model": model, "max_tokens": 1024,
            "tools": tools, "messages": messages}
    if system is not None:                    # omitted, not sent as null
        body["system"] = system
    return await client.post(
        settings.llm_api_url,
        headers={"x-api-key": settings.anthropic_api_key,
                 "anthropic-version": "2023-06-01"},
        json=body,
    )

def backoff(attempt: int, resp: httpx.Response | None) -> float:
    if resp is not None and resp.headers.get("retry-after", "").isdigit():
        return float(resp.headers["retry-after"])
    return min(8.0, 0.5 * 2 ** attempt) * (0.5 + random.random())

def log_call(model: str, attempts: int, started: float,
             resp: httpx.Response | None = None,
             exc: Exception | None = None) -> dict:
    """One JSON line per call, answered or not, returning the parsed body
    so that a call site reads `return log_call(...)`."""
    body = resp.json() if resp is not None else {}
    usage = body.get("usage", {})
    status = getattr(getattr(exc, "response", None), "status_code", "")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "model": body.get("model", model),   # who answered, not who was asked
        "attempts": attempts,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "error": f"{type(exc).__name__} {status}".strip() if exc else None,
    }
    with CALLS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return body


def can_run_locally(tools: list[dict]) -> bool:
    """Whether the local model is a substitute for this particular call.

    A fallback that cannot do the job is not a fallback. The local model
    is small -- small enough to fit beside the embedder on one card --
    and small models do not do reliable tool use, so a call that needs
    tools is one the local rung must decline rather than botch. Text-only
    calls are the opposite case: a RAG answer written a little worse
    beats an outage, which is the same trade this module makes when it chooses
    a cheaper model for the second rung."""
    return bool(settings.serving_url) and not tools


async def local_generate(messages: list[dict], system: str | None = None,
                         attempts: int = 1) -> dict:
    """The GPU tier's /generate, which answers in the Anthropic shape the
    rest of this application already speaks. `attempts` is passed in
    rather than counted here, because by the time this rung runs the two
    above it have already spent theirs."""
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=LOCAL_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.serving_url.rstrip('/')}/generate",
            json={"messages": messages, "system": system})
    resp.raise_for_status()
    return log_call("local", attempts, started, resp)


async def call_llm(messages: list[dict], tools: list[dict],
                   system: str | None = None) -> dict:
    attempts, model, started = 0, settings.llm_model, time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for attempt in range(MAX_ATTEMPTS):
                resp, attempts = None, attempts + 1
                started = time.perf_counter()   # the request, not the wait
                try:
                    resp = await post(client, model, messages, tools, system)
                except httpx.TransportError:
                    pass                      # DNS, connect, read timeout
                else:
                    if resp.status_code not in RETRY_STATUS:
                        resp.raise_for_status()   # 400, 401, 404: your bug
                        return log_call(model, attempts, started, resp)
                await asyncio.sleep(backoff(attempt, resp))
            try:
                attempts, model = attempts + 1, settings.llm_fallback_model
                started = time.perf_counter()
                resp = await post(client, model, messages, tools, system)
                resp.raise_for_status()
                return log_call(model, attempts, started, resp)
            except httpx.HTTPError:       # TransportError is one of these
                if not can_run_locally(tools):
                    raise
        # Outside the `async with`: the commercial client is released
        # before the local one is opened, because the whole reason we are
        # here is that the first client could not reach anything.
        attempts, model = attempts + 1, "local"
        return await local_generate(messages, system, attempts)
    except httpx.HTTPError as exc:
        log_call(model, attempts, started, exc=exc)
        raise


def final_text(resp: dict) -> str:
    return "".join(b["text"] for b in resp["content"] if b["type"] == "text")

def tool_result(tool_use_id: str, out: Any) -> dict:
    return {"type": "tool_result",
            "tool_use_id": tool_use_id,          # must match the id you got
            "content": json.dumps(out)}


async def run_loop(messages, tools, max_iterations=5):
    for _ in range(max_iterations):
        resp = await call_llm(messages, tools)
        messages.append({"role": "assistant", "content": resp["content"]})
        if resp["stop_reason"] != "tool_use":
            return final_text(resp), messages
        results = []
        for block in resp["content"]:
            if block["type"] == "tool_use":
                try:
                    out = TOOLS[block["name"]](**block["input"])
                except Exception as e:        # let the model recover once
                    out = {"error": str(e)}
                results.append(tool_result(block["id"], out))
        messages.append({"role": "user", "content": results})
    raise RuntimeError("step budget exceeded")
