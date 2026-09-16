# tests/test_rrf.py
from app.retrieval import rrf


def test_agreement_beats_a_single_first_place():
    """Ranked second by both retrievers beats ranked first by only one."""
    assert rrf(["x", "b"], ["y", "b"])[0] == "b"


def test_single_list_preserves_order():
    assert rrf(["a", "b", "c"]) == ["a", "b", "c"]


def test_appearing_in_more_lists_wins():
    assert rrf(["x"], ["y"], ["y"])[0] == "y"


def test_equal_rank_sums_tie():
    """Mirror-image rankings genuinely tie — RRF sees positions, not scores."""
    fused = rrf(["a", "b"], ["b", "a"])
    assert set(fused) == {"a", "b"}
