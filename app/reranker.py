# app/reranker.py
import asyncio
from typing import Protocol

import httpx

from app.config import settings

TIMEOUT = httpx.Timeout(60.0, connect=5.0)


class Reranker(Protocol):
    async def rerank(self, query: str,
                     docs: list[str]) -> list[tuple[int, float]]: ...


class LocalReranker:
    """Cross-encoder, in-process. Same interface trick as the embedder."""

    def __init__(self,
                 model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model)

    async def rerank(self, query: str,
                     docs: list[str]) -> list[tuple[int, float]]:
        pairs = [(query, d) for d in docs]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order]


class RemoteReranker:
    """The Reranker Protocol, over HTTP, for the same reason."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    async def rerank(self, query: str,
                     docs: list[str]) -> list[tuple[int, float]]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{self.url}/rerank",
                                     json={"query": query, "docs": docs})
        resp.raise_for_status()
        return [(int(i), float(s)) for i, s in resp.json()["ranking"]]


_reranker: Reranker | None = None
_reranker_lock = asyncio.Lock()


async def _build() -> Reranker:
    if settings.serving_url:
        return RemoteReranker(settings.serving_url)
    return await asyncio.to_thread(LocalReranker)


async def get_reranker() -> Reranker:
    """Built on first use, off the event loop, behind a lock — same reason
    as get_embedder."""
    global _reranker
    if _reranker is None:
        async with _reranker_lock:
            if _reranker is None:
                _reranker = await _build()
    return _reranker
