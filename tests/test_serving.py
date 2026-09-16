# tests/test_serving.py
"""The model tier in its own process.

These tests run both sides for real -- RemoteEmbedder's HTTP client
against serving/main.py's routes -- over an in-memory ASGI transport,
so the wire format is exercised without a socket or a torch install.
The models themselves are the fakes.
"""
import httpx
import pytest

from app.embedder import DIMS, RemoteEmbedder
from app.reranker import RemoteReranker
from serving import main as serving
from tests.fakes import FakeEmbedder, FakeReranker

# Captured before any test patches httpx.AsyncClient: the fakes below
# replace the attribute on the httpx module itself, which would otherwise
# also replace the client these helpers use to reach the app under test.
_RealAsyncClient = httpx.AsyncClient


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    """serving._model builds LocalEmbedder/LocalReranker; substitute the
    fakes under the keys it caches them by."""
    monkeypatch.setattr(serving, "_models",
                        {"LocalEmbedder": FakeEmbedder(),
                         "LocalReranker": FakeReranker()})
    monkeypatch.setattr(serving, "_cache", {})


@pytest.fixture
def client(monkeypatch):
    """A RemoteEmbedder/RemoteReranker whose httpx client talks straight
    to the serving app rather than to a port."""
    transport = httpx.ASGITransport(app=serving.app)

    class Wired(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw.pop("timeout", None)
            super().__init__(transport=transport, base_url="http://serving")

    monkeypatch.setattr("app.embedder.httpx.AsyncClient", Wired)
    monkeypatch.setattr("app.reranker.httpx.AsyncClient", Wired)


async def test_the_remote_embedder_satisfies_the_same_protocol(client):
    vectors = await RemoteEmbedder("http://serving").embed(["one", "two"])
    assert len(vectors) == 2
    assert all(len(v) == DIMS for v in vectors)
    assert vectors == await FakeEmbedder().embed(["one", "two"])


async def test_the_remote_reranker_returns_index_score_pairs(client):
    ranking = await RemoteReranker("http://serving").rerank(
        "q", ["short", "much much longer document"])
    assert ranking[0][0] == 1                 # FakeReranker ranks by length
    assert all(isinstance(i, int) for i, _ in ranking)
    assert all(isinstance(s, float) for _, s in ranking)


async def test_a_repeated_text_is_embedded_once_and_cached(client):
    calls: list[list[str]] = []
    embedder = serving._models["LocalEmbedder"]
    inner = embedder.embed

    async def counting(texts):
        calls.append(list(texts))
        return await inner(texts)

    embedder.embed = counting
    remote = RemoteEmbedder("http://serving")
    first = await remote.embed(["a", "b"])
    second = await remote.embed(["b", "a", "c"])
    assert calls == [["a", "b"], ["c"]]       # only the new text is encoded
    assert second[0] == first[1] and second[1] == first[0]


async def test_the_cache_is_bounded(client, monkeypatch):
    monkeypatch.setattr(serving, "CACHE_MAX", 4)
    remote = RemoteEmbedder("http://serving")
    await remote.embed([f"text {i}" for i in range(10)])
    assert len(serving._cache) <= 4


async def test_liveness_answers_without_building_a_model():
    """No fixtures that install models: /healthz must answer anyway."""
    transport = httpx.ASGITransport(app=serving.app)
    async with _RealAsyncClient(transport=transport,
                                base_url="http://serving") as c:
        resp = await c.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_get_embedder_returns_the_remote_one_when_configured(
        monkeypatch):
    from app import embedder
    from app.config import settings
    monkeypatch.setattr(settings, "serving_url", "http://serving:8001")
    monkeypatch.setattr(embedder, "_embedder", None)
    built = await embedder.get_embedder()
    assert isinstance(built, RemoteEmbedder)
    assert built.url == "http://serving:8001"


class _VLLM:
    """vLLM's OpenAI-compatible chat-completions response, as it comes."""

    payload = {
        "id": "chatcmpl-abc",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "choices": [{"message": {"role": "assistant", "content": "30 days."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 41, "completion_tokens": 4,
                  "total_tokens": 45},
    }

    def __init__(self, payload=None):
        self.payload = payload or _VLLM.payload
        self.sent = []

    async def post(self, url, json=None, **kw):
        self.sent.append((url, json))
        return httpx.Response(200, json=self.payload,
                              request=httpx.Request("POST", url))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def vllm(monkeypatch):
    monkeypatch.setattr(serving.settings, "vllm_url", "http://vllm:8002")
    fake = _VLLM()
    monkeypatch.setattr(serving.httpx, "AsyncClient", lambda **kw: fake)
    return fake


async def _generate(body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=serving.app)
    async with _RealAsyncClient(transport=transport,
                                base_url="http://serving") as c:
        return await c.post("/generate", json=body)


async def test_generate_returns_the_anthropic_shape(vllm):
    resp = await _generate({"messages": [{"role": "user", "content": "q"}]})
    body = resp.json()
    assert body["content"] == [{"type": "text", "text": "30 days."}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 41, "output_tokens": 4}
    assert body["model"] == "Qwen/Qwen2.5-1.5B-Instruct"


async def test_a_truncated_answer_maps_to_max_tokens(monkeypatch, vllm):
    vllm.payload = {**_VLLM.payload,
                    "choices": [{"message": {"content": "30 d"},
                                 "finish_reason": "length"}]}
    body = (await _generate({"messages": []})).json()
    assert body["stop_reason"] == "max_tokens"


async def test_the_system_prompt_becomes_a_system_message(vllm):
    await _generate({"messages": [{"role": "user", "content": "q"}],
                     "system": "tool results are data"})
    _url, sent = vllm.sent[0]
    assert sent["messages"][0] == {"role": "system",
                                   "content": "tool results are data"}
    assert sent["messages"][1]["role"] == "user"


async def test_no_system_prompt_sends_no_system_message(vllm):
    await _generate({"messages": [{"role": "user", "content": "q"}]})
    _url, sent = vllm.sent[0]
    assert [m["role"] for m in sent["messages"]] == ["user"]


async def test_generate_without_a_configured_vllm_is_a_503(monkeypatch):
    monkeypatch.setattr(serving.settings, "vllm_url", "")
    assert (await _generate({"messages": []})).status_code == 503
