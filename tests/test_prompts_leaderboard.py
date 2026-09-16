# tests/test_prompts_leaderboard.py
"""The leaderboard's arithmetic, without a database or a model. A
scoreboard that cannot be tested is a scoreboard nobody should trust."""
from evals.prompts import leaderboard


def test_rows_sort_by_accuracy_descending():
    rows = [{"version": 1, "answer_accuracy": 0.5},
            {"version": 2, "answer_accuracy": 0.9},
            {"version": 3, "answer_accuracy": 0.7}]
    assert [r["version"] for r in leaderboard(rows)] == [2, 3, 1]


def test_the_winner_is_marked_and_ties_do_not_both_win():
    """Listed newest-first on purpose: a stable sort with no version
    tiebreak would leave v2 first, so this ordering is what proves the
    tiebreak runs."""
    rows = [{"version": 2, "answer_accuracy": 0.8},
            {"version": 1, "answer_accuracy": 0.8}]
    out = leaderboard(rows)
    assert [r["version"] for r in out] == [1, 2]
    assert [r["best"] for r in out] == [True, False]


def test_an_empty_run_produces_an_empty_board():
    assert leaderboard([]) == []
