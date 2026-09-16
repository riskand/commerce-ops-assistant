# serving/main.py
"""The model tier, in its own process.

app/ imports torch nowhere: the two classes that do
live behind this HTTP boundary. The split is not tidiness. The API
process is memory-light, starts in a second and scales with request
concurrency; this one holds two transformers, wants a GPU, takes half a
minute to warm up and scales with how much text there is to encode. One
container for both makes every deploy pay both costs and neither scale.
"""
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.embedder import MODEL, LocalEmbedder
from app.reranker import LocalReranker

GENERATE_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

app = FastAPI(title="Model serving")

# Query embeddings repeat: the same question asked twice, the same eval
# set run twenty times while a prompt is tuned. An embedding is a pure
# function of (model, text), so it caches perfectly -- bounded, because
# an unbounded cache is a memory leak with a friendly name.
CACHE_MAX = 8192
_cache: dict[str, list[float]] = {}

_models: dict[str, object] = {}
_lock = asyncio.Lock()


async def _model(cls):
    """Deliberately builds LocalEmbedder/LocalReranker rather than calling
    app's get_embedder(): that accessor returns a RemoteEmbedder whenever
    SERVING_URL is set, and this process having that variable in its own
    environment is not far-fetched -- one shared env file does it. It
    would then forward every request to itself."""
    key = cls.__name__
    if key not in _models:
        async with _lock:
            if key not in _models:
                _models[key] = await asyncio.to_thread(cls)
    return _models[key]


class EmbedRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    query: str
    docs: list[str]


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness only, and it must not build a model: an orchestrator that
    probes this while the process is still warming up would kill it for
    being slow to do the thing it was started to do."""
    return {"status": "ok", "cached": len(_cache)}


@app.get("/readyz")
async def readyz() -> dict:
    """Readiness: both models are built and this process can serve. Slow
    on the first call, by design. That difference is the whole reason a
    liveness probe and a readiness probe are two endpoints."""
    await _model(LocalEmbedder)
    await _model(LocalReranker)
    return {"status": "ready"}


@app.post("/embed")
async def embed(req: EmbedRequest) -> dict:
    out = {t: _cache[t] for t in req.texts if t in _cache}
    missing = [t for t in req.texts if t not in out]
    if missing:
        embedder = await _model(LocalEmbedder)
        for text, vector in zip(missing, await embedder.embed(missing)):
            out[text] = vector
            if len(_cache) >= CACHE_MAX:
                del _cache[next(iter(_cache))]      # oldest first, FIFO
            _cache[text] = vector
    return {"model": MODEL, "vectors": [out[t] for t in req.texts]}


@app.post("/rerank")
async def rerank(req: RerankRequest) -> dict:
    reranker = await _model(LocalReranker)
    return {"ranking": await reranker.rerank(req.query, req.docs)}


class GenerateRequest(BaseModel):
    messages: list[dict]
    system: str | None = None
    max_tokens: int = 1024


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """The last rung of app/loop.py's ladder, and the only endpoint here
    that is a translation rather than a computation.

    vLLM speaks OpenAI's chat-completions shape; the whole book speaks
    Anthropic's. Translating here rather than in app/loop.py means the
    application never learns a second wire format: app/ asks the GPU tier
    for an answer and gets back what the Anthropic API would have
    returned, which is what makes this a fallback rather than a special
    case every caller has to handle."""
    if not settings.vllm_url:
        raise HTTPException(503, "no local generator is configured")
    body = {"model": settings.vllm_model, "max_tokens": req.max_tokens,
            "messages": ([{"role": "system", "content": req.system}]
                         if req.system else []) + req.messages}
    async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.vllm_url.rstrip('/')}/v1/chat/completions", json=body)
    resp.raise_for_status()
    out = resp.json()
    choice = out["choices"][0]
    usage = out.get("usage", {})
    return {
        "id": out.get("id", ""),
        # The model that actually answered, not the one that was asked
        # for: the call log reads this field, and a run
        # served locally has to be visible as one in the cost table.
        "model": out.get("model", settings.vllm_model),
        "stop_reason": ("max_tokens" if choice.get("finish_reason") == "length"
                        else "end_turn"),
        "content": [{"type": "text",
                     "text": choice["message"]["content"] or ""}],
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }
