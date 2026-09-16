# app/chunker.py
import re

CHARS_PER_TOKEN = 4          # approximation; the eval harness measures properly


def _tokens(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


def _split_paragraph(para: str, target: int) -> list[str]:
    """An oversized paragraph falls back to sentence boundaries."""
    out, buf = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", para):
        if buf and _tokens(buf) + _tokens(sentence) > target:
            out.append(buf.strip())
            buf = ""
        buf = f"{buf} {sentence}"
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk(text: str, target_tokens: int = 500,
          overlap_tokens: int = 50) -> list[str]:
    units: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if _tokens(para) <= target_tokens:
            units.append(para)
        else:
            units.extend(_split_paragraph(para, target_tokens))

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if buf and _tokens(buf) + _tokens(unit) > target_tokens:
            chunks.append(buf.strip())
            buf = buf[-overlap_tokens * CHARS_PER_TOKEN:]    # carry context
        buf = f"{buf}\n\n{unit}"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks
