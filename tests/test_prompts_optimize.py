# tests/test_prompts_optimize.py
"""The gate between a model's suggestion and your prompt registry. A
proposal is untrusted text: it may be fenced, it may be chatty, and it
may have dropped the one placeholder the caller depends on."""
import pytest

from app.prompts import PromptError
from evals.optimize import accept_proposal

CURRENT = "Use the context.\n\n{context}\n\nQuestion: {question}"


def test_a_proposal_that_drops_a_placeholder_is_rejected():
    with pytest.raises(PromptError) as e:
        accept_proposal(CURRENT, "Answer well.\n\nQuestion: {question}")
    assert "context" in str(e.value)


def test_a_proposal_that_invents_a_placeholder_is_rejected():
    with pytest.raises(PromptError) as e:
        accept_proposal(CURRENT, CURRENT + "\nTone: {tone}")
    assert "tone" in str(e.value)


def test_a_fenced_proposal_is_unwrapped():
    fenced = "```\n" + CURRENT + "\n```"
    assert accept_proposal(CURRENT, fenced) == CURRENT


def test_a_valid_proposal_is_returned_unchanged():
    better = CURRENT + "\n\nAnswer in at most three sentences."
    assert accept_proposal(CURRENT, better) == better


def test_a_chatty_reply_with_a_fenced_template_is_unwrapped():
    """The most common shape a model actually replies in. Before this
    was handled, the whole sentence plus its backticks was stored as
    the prompt, because a preamble does not change the placeholder set
    and so passed the contract check untouched."""
    reply = "Sure, here is an improved version:\n\n```\n" + CURRENT + "\n```"
    assert accept_proposal(CURRENT, reply) == CURRENT


def test_a_language_tagged_fence_is_unwrapped():
    assert accept_proposal(CURRENT, "```text\n" + CURRENT + "\n```") == CURRENT


def test_a_reply_still_carrying_a_fence_marker_is_refused():
    """Unrecognised fencing must fail loudly rather than be stored."""
    with pytest.raises(PromptError) as e:
        accept_proposal(CURRENT, "```" + CURRENT + "\n```")
    assert "fence" in str(e.value)


def test_a_proposal_with_an_unbalanced_brace_is_refused():
    with pytest.raises(PromptError) as e:
        accept_proposal(CURRENT, "Use {context and {question}")
    assert "not a valid template" in str(e.value)


def test_a_proposal_with_an_empty_placeholder_is_refused():
    """{} slips past the contract check because declared() drops empty
    field names, then breaks every render that uses the prompt."""
    with pytest.raises(PromptError) as e:
        accept_proposal(CURRENT, CURRENT + "\n{}")
    assert "does not render" in str(e.value)
