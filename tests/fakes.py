# tests/fakes.py
"""Dependency-free stand-ins. Both satisfy the Protocols in app/, which is
what the Protocols are for."""
import hashlib

from app.embedder import DIMS


class FakeEmbedder:
    """Deterministic and unit length, so cosine distance still behaves."""

    name = "fake-embedder"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [(digest[i % len(digest)] - 128) / 128.0
                   for i in range(DIMS)]
            norm = sum(v * v for v in raw) ** 0.5 or 1.0
            out.append([v / norm for v in raw])
        return out


class FakeReranker:
    """Ranks by descending length: deterministic, and obviously not
    semantic, so a test that depends on meaning fails loudly."""

    async def rerank(self, query: str,
                     docs: list[str]) -> list[tuple[int, float]]:
        order = sorted(range(len(docs)), key=lambda i: -len(docs[i]))
        return [(i, float(len(docs[i]))) for i in order]


class ReversingReranker:
    """A second fake exists because FakeReranker's descending-length order
    can coincide with the pre-rerank fused order for a given corpus and
    query, which lets a hybrid_search that never calls (or ignores)
    reranker.rerank() pass by accident (see tests/test_hybrid.py, fix
    round 1). Reversing 3+ distinct documents can never equal the identity
    permutation, so this one proves the rerank step actually ran no matter
    what the input order was."""

    async def rerank(self, query: str,
                     docs: list[str]) -> list[tuple[int, float]]:
        order = list(reversed(range(len(docs))))
        return [(i, float(len(order) - rank))
                for rank, i in enumerate(order)]
