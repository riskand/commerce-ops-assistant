# tests/test_chunker.py
from app.chunker import CHARS_PER_TOKEN, chunk


def test_empty_document():
    assert chunk("") == []


def test_one_liner_survives_intact():
    assert chunk("Returns are accepted within 30 days.") == \
        ["Returns are accepted within 30 days."]


def test_huge_paragraph_is_split():
    text = "The seller must respond. " * 2000        # one giant paragraph
    assert len(chunk(text)) > 1


def test_chunks_respect_target():
    text = "\n\n".join(("Policy clause %d. " % i) * 40 for i in range(50))
    assert all(len(c) < 500 * CHARS_PER_TOKEN * 1.5 for c in chunk(text))
