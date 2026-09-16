# tests/test_prompts.py
"""render_template's rules, which are the reason prompts stop being
f-strings. Everything here is pure -- no database -- so it runs in the
fast tier."""
import pytest

from app.prompts import PromptError, declared, render_template


def test_a_value_containing_a_placeholder_is_not_expanded():
    """Injection defence at the prompt layer. A user question that
    contains {context} must arrive in the prompt as those literal
    characters, not as a second substitution pass. str.format does not
    re-scan the values it substitutes, and this test is what stops
    someone 'simplifying' it into something that does."""
    out = render_template("Question: {question}",
                          {"question": "what does {context} mean?"})
    assert out == "Question: what does {context} mean?"


def test_the_prefix_before_the_first_placeholder_is_byte_stable():
    """Prompt caching keys on an exact prefix, so the cacheable head of a
    template must not move when the values change. A registry makes it
    easy to lose this by interpolating near the top; this test notices."""
    template = "Answer from the context.\n\n{context}\n\nQ: {question}"
    head = template.split("{", 1)[0]
    a = render_template(template, {"context": "x", "question": "y"})
    b = render_template(template, {"context": "zzz", "question": "w"})
    assert a.startswith(head) and b.startswith(head)


def test_an_undeclared_value_is_refused():
    with pytest.raises(PromptError) as e:
        render_template("Q: {question}", {"question": "a", "colour": "b"})
    assert "colour" in str(e.value)


def test_a_missing_value_is_refused():
    with pytest.raises(PromptError) as e:
        render_template("{context} / {question}", {"question": "a"})
    assert "context" in str(e.value)


def test_declared_finds_every_placeholder_once():
    assert declared("{a} {b} {a}") == {"a", "b"}
    assert declared("no placeholders") == set()
