# tests/test_prompts_route.py
"""The endpoints exist so a prompt change is reviewable and reversible
the way a code change is: see the versions, read the diff, switch, switch
back."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration


def test_listing_names_the_active_version_of_each_prompt():
    with TestClient(app) as client:
        body = client.get("/prompts").json()
    by_name = {p["name"]: p for p in body}
    assert by_name["rag_answer"]["active_version"] == 1


def test_showing_one_prompt_returns_its_versions_newest_first():
    with TestClient(app) as client:
        body = client.get("/prompts/rag_answer").json()
    assert [v["version"] for v in body] == sorted(
        [v["version"] for v in body], reverse=True)
    assert "{context}" in body[-1]["template"]


def test_an_unknown_prompt_is_a_404():
    with TestClient(app) as client:
        r = client.get("/prompts/no_such_prompt")
    assert r.status_code == 404
    assert "no_such_prompt" in r.json()["detail"]


def test_diffing_a_version_against_itself_is_empty():
    with TestClient(app) as client:
        body = client.get("/prompts/rag_answer/diff",
                          params={"a": 1, "b": 1}).json()
    assert body["diff"] == ""


def test_activating_an_unknown_version_is_a_404_and_changes_nothing():
    with TestClient(app) as client:
        before = client.get("/prompts").json()
        assert any(p["name"] == "rag_answer" for p in before)
        r = client.post("/prompts/rag_answer/activate", json={"version": 999})
        assert r.status_code == 404
        assert "999" in r.json()["detail"]
        assert client.get("/prompts").json() == before
