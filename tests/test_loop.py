# tests/test_loop.py
import pytest

from app import loop
from app.loop import final_text, run_loop, tool_result


def _text(s: str) -> dict:
    return {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": s}]}


def _use(name: str, args: dict, id: str = "toolu_1") -> dict:
    return {"stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": id,
                         "name": name, "input": args}]}


def _results(history: list[dict]) -> list[dict]:
    return [b for m in history if isinstance(m["content"], list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"]


def test_final_text_joins_only_the_text_blocks():
    resp = {"content": [{"type": "text", "text": "a"},
                        {"type": "tool_use", "id": "t", "name": "n",
                         "input": {}},
                        {"type": "text", "text": "b"}]}
    assert final_text(resp) == "ab"


def test_tool_result_echoes_the_id_it_was_given():
    block = tool_result("toolu_abc", {"ok": True})
    assert block["tool_use_id"] == "toolu_abc"
    assert block["content"] == '{"ok": true}'


async def test_a_plain_answer_ends_the_loop(monkeypatch):
    calls = []

    async def fake(messages, tools):
        calls.append(messages)
        return _text("done")

    monkeypatch.setattr(loop, "call_llm", fake)
    text, history = await run_loop([{"role": "user", "content": "hi"}], [])
    assert text == "done"
    assert len(calls) == 1


async def test_a_tool_use_dispatches_through_the_registry(monkeypatch):
    responses = [_use("get_order", {"order_id": "SO-1042"}),
                 _text("It failed on channel-b.")]

    async def fake(messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(loop, "call_llm", fake)
    text, history = await run_loop([{"role": "user", "content": "why?"}], [])
    assert text == "It failed on channel-b."
    assert len(_results(history)) == 1
    assert "partially_synced" in _results(history)[0]["content"]


async def test_a_raising_tool_comes_back_as_information(monkeypatch):
    responses = [_use("get_order", {"not_a_parameter": 1}), _text("ok")]

    async def fake(messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(loop, "call_llm", fake)
    _answer, history = await run_loop([{"role": "user", "content": "x"}], [])
    assert "error" in _results(history)[0]["content"]


async def test_the_step_budget_is_enforced(monkeypatch):
    async def fake(messages, tools):
        return _use("get_order", {"order_id": "SO-1"})

    monkeypatch.setattr(loop, "call_llm", fake)
    with pytest.raises(RuntimeError, match="step budget exceeded"):
        await run_loop([{"role": "user", "content": "x"}], [],
                       max_iterations=3)
