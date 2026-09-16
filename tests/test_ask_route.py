# tests/test_ask_route.py
from fastapi.testclient import TestClient

from app import loop
from app.main import app


def test_ask_returns_the_answer_and_the_tools_it_used(monkeypatch):
    responses = [
        {"stop_reason": "tool_use",
         "content": [{"type": "tool_use", "id": "toolu_1",
                      "name": "get_order",
                      "input": {"order_id": "SO-1042"}}]},
        {"stop_reason": "end_turn",
         "content": [{"type": "text", "text": "It failed on channel-b."}]},
    ]

    async def fake(messages, tools):
        return responses.pop(0)

    monkeypatch.setattr(loop, "call_llm", fake)
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "why is SO-1042 stuck?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "It failed on channel-b."
    assert [c["name"] for c in body["tool_calls"]] == ["get_order"]
    assert body["turns"] == 4
