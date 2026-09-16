# tests/test_prompts_retrofit.py
"""Moving a prompt into the registry must change where the text lives and
nothing else. These capture the prompt actually sent, rather than trusting
that it was assembled correctly."""
import uuid

import pytest
from fastapi.testclient import TestClient

import app.routes.docsearch as docsearch
from app.main import app

pytestmark = pytest.mark.integration


def test_query_builds_its_prompt_from_the_registry(monkeypatch):
    seen = {}

    async def fake_call_llm(messages, tools):
        seen["prompt"] = messages[0]["content"]
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(docsearch, "call_llm", fake_call_llm)
    with TestClient(app) as client:
        r = client.post("/query", json={"tenant_id": str(uuid.uuid4()),
                                        "question": "what is the policy?"})
    assert r.status_code == 200
    assert seen["prompt"].startswith("Answer the question using only")
    assert "what is the policy?" in seen["prompt"]


def test_query_can_pin_a_prompt_version(monkeypatch):
    seen = {}

    async def fake_call_llm(messages, tools):
        seen["prompt"] = messages[0]["content"]
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(docsearch, "call_llm", fake_call_llm)
    with TestClient(app) as client:
        r = client.post("/query", json={"tenant_id": str(uuid.uuid4()),
                                        "question": "q", "prompt_version": 1})
    assert r.status_code == 200
    assert seen["prompt"].startswith("Answer the question using only")


def test_no_prompt_literal_survives_in_the_retrofitted_modules():
    """These strings are rows now, not literals. A
    reintroduced literal is a regression this catches cheaply."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for path in ("app/routes/docsearch.py", "evals/run.py"):
        src = (root / path).read_text()
        assert "{context}" not in src, path
        assert "Reply exactly YES or NO" not in src, path
