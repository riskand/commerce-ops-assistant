# tests/test_calls_log.py
"""The one JSON line per call that call_llm appends.

The same substitutions as test_gateway.py -- app.loop.post replaced, so
no socket is opened, and asyncio.sleep replaced, so no wait is really
waited. tests/conftest.py points CALLS_LOG at a tmp_path for every test
in the suite, so nothing here writes into the repo.
"""
import json
import time

import httpx
import pytest

from app import loop
from app.config import settings

REQ = httpx.Request("POST", settings.llm_api_url)


def _resp(status: int, **kw) -> httpx.Response:
    body = kw.pop("json", {"model": "claude-sonnet-5-20260114",
                           "content": [{"type": "text", "text": "ok"}],
                           "stop_reason": "end_turn",
                           "usage": {"input_tokens": 412,
                                     "output_tokens": 96}})
    return httpx.Response(status, json=body, request=REQ, **kw)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(loop.asyncio, "sleep", fake_sleep)


def _sequence(monkeypatch, *responses):
    served = list(responses)

    async def fake_post(client, model, messages, tools, system=None):
        out = served.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(loop, "post", fake_post)


def _lines() -> list[dict]:
    if not loop.CALLS_LOG.exists():
        return []
    return [json.loads(line) for line in
            loop.CALLS_LOG.read_text().splitlines()]


async def test_an_answered_call_writes_exactly_one_line(monkeypatch):
    _sequence(monkeypatch, _resp(200))
    await loop.call_llm([], [])
    line, = _lines()
    assert line["attempts"] == 1
    assert line["input_tokens"] == 412 and line["output_tokens"] == 96
    assert line["error"] is None
    assert isinstance(line["latency_ms"], int)


async def test_the_line_names_the_model_that_answered(monkeypatch):
    """The setting says claude-sonnet-5; the response says which build of
    it replied. A cost table built from the setting cannot tell a
    fallback from a primary, which is the one thing it exists to show."""
    _sequence(monkeypatch, _resp(200))
    await loop.call_llm([], [])
    assert _lines()[0]["model"] == "claude-sonnet-5-20260114"
    assert settings.llm_model == "claude-sonnet-5"


async def test_a_body_without_a_model_falls_back_to_the_one_asked_for(
        monkeypatch):
    _sequence(monkeypatch, _resp(200, json={"content": [], "usage": {}}))
    await loop.call_llm([], [])
    assert _lines()[0]["model"] == settings.llm_model


async def test_a_retried_call_is_one_line_counting_every_attempt(monkeypatch):
    _sequence(monkeypatch, _resp(429), httpx.ConnectError("no route"),
              _resp(200))
    await loop.call_llm([], [])
    line, = _lines()
    assert line["attempts"] == 3


async def test_a_fallback_records_the_fallback_model(monkeypatch):
    _sequence(monkeypatch, *([_resp(529)] * loop.MAX_ATTEMPTS),
              _resp(200, json={"content": [], "usage": {}}))
    await loop.call_llm([], [])
    line, = _lines()
    assert line["model"] == settings.llm_fallback_model
    assert line["attempts"] == loop.MAX_ATTEMPTS + 1


async def test_a_dead_ladder_still_writes_a_line_and_still_raises(monkeypatch):
    _sequence(monkeypatch, *([_resp(529)] * (loop.MAX_ATTEMPTS + 1)))
    with pytest.raises(httpx.HTTPStatusError):
        await loop.call_llm([], [])
    line, = _lines()
    assert line["error"] == "HTTPStatusError 529"
    assert line["model"] == settings.llm_fallback_model
    assert line["input_tokens"] is None and line["output_tokens"] is None


async def test_a_wrong_key_is_one_line_after_one_attempt(monkeypatch):
    _sequence(monkeypatch, _resp(401))
    with pytest.raises(httpx.HTTPStatusError):
        await loop.call_llm([], [])
    line, = _lines()
    assert line["error"] == "HTTPStatusError 401"
    assert line["attempts"] == 1


async def test_a_transport_error_has_no_status_to_report(monkeypatch):
    _sequence(monkeypatch, *([httpx.ConnectError("refused")]
                             * (loop.MAX_ATTEMPTS + 1)))
    with pytest.raises(httpx.ConnectError):
        await loop.call_llm([], [])
    assert _lines()[0]["error"] == "ConnectError"


async def test_every_line_carries_the_same_keys(monkeypatch):
    """One shape, so `jq` and any reporting on top read the file without a
    branch per record."""
    _sequence(monkeypatch, _resp(200), _resp(401))
    await loop.call_llm([], [])
    with pytest.raises(httpx.HTTPStatusError):
        await loop.call_llm([], [])
    answered, failed = _lines()
    assert answered.keys() == failed.keys()


async def test_latency_measures_the_request_not_the_backoff(monkeypatch):
    """The number that makes a p95 investigation take a week is one that
    silently includes the waits. Here a retry sleeps 200ms for real while
    the request itself is instant, so a latency built from the whole call
    would be over 200 and the one we write is nowhere near it."""
    async def slow_sleep(seconds):
        time.sleep(0.2)

    monkeypatch.setattr(loop.asyncio, "sleep", slow_sleep)
    _sequence(monkeypatch, _resp(429), _resp(200))
    await loop.call_llm([], [])
    line, = _lines()
    assert line["attempts"] == 2
    assert line["latency_ms"] < 100


async def test_the_local_rung_writes_a_line_naming_the_local_model(
        monkeypatch):
    """The third rung logs from inside local_generate, so the count it
    reports has to include the two rungs that already failed above it."""
    monkeypatch.setattr(loop.settings, "serving_url", "http://serving:8001")

    async def fake_local_post(url, json=None):
        return httpx.Response(
            200, request=httpx.Request("POST", url),
            json={"model": "Qwen/Qwen2.5-0.5B-Instruct",
                  "stop_reason": "end_turn",
                  "content": [{"type": "text", "text": "answered locally"}],
                  "usage": {"input_tokens": 5, "output_tokens": 3}})

    class Client:
        post = staticmethod(fake_local_post)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    _sequence(monkeypatch, *([_resp(529)] * (loop.MAX_ATTEMPTS + 1)))
    monkeypatch.setattr(loop.httpx, "AsyncClient", lambda **kw: Client())
    await loop.call_llm([{"role": "user", "content": "hi"}], [])
    line, = _lines()
    assert line["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert line["attempts"] == loop.MAX_ATTEMPTS + 2
