# app/embedder.py
import asyncio
from typing import Protocol

import httpx

from app.config import settings

TIMEOUT = httpx.Timeout(60.0, connect=5.0)

DIMS = 1024        # must match Vector(1024) in models.py
MODEL = "BAAI/bge-large-en-v1.5"       # the default, not the last word


def active_model() -> str:
    """The embedder this deployment is configured to use -- a Hub name or a
    local directory. Read without building anything, so reindex.py can ask
    what the current model is without loading half a gigabyte of weights
    just to run a COUNT."""
    return settings.embedding_model or MODEL


class Embedder(Protocol):
    name: str
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """In-process sentence-transformers. Free, and fast enough to learn on."""

    def __init__(self, model: str | None = None) -> None:
        model = model or active_model()
        from sentence_transformers import SentenceTransformer
        self.name = model
        self._model = SentenceTransformer(model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class RemoteEmbedder:
    """The same Protocol, over HTTP. This is what the Protocol was for:
    every caller of embed() is unchanged, and this process stops needing
    torch installed at all."""

    def __init__(self, url: str, name: str | None = None) -> None:
        self.url, self.name = url.rstrip("/"), name or active_model()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{self.url}/embed",
                                     json={"texts": texts})
        resp.raise_for_status()
        return resp.json()["vectors"]


_embedder: Embedder | None = None
_embedder_lock = asyncio.Lock()


async def _build() -> Embedder:
    if settings.serving_url:
        return RemoteEmbedder(settings.serving_url)
    return await asyncio.to_thread(LocalEmbedder)


async def get_embedder() -> Embedder:
    """Built on first use, in a thread, behind a lock."""
    global _embedder
    if _embedder is None:
        async with _embedder_lock:
            if _embedder is None:
                _embedder = await _build()
    return _embedder
