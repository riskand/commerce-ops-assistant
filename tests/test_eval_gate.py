# tests/test_eval_gate.py
"""The eval suite as a build gate."""
import json

import pytest

from evals.gate import TOLERANCE, compare, main

BASE = {"n": 30, "recall_at_5": 0.80, "answer_accuracy": 0.75}


def test_meeting_the_baseline_passes():
    assert compare(BASE, dict(BASE)) == []


def test_an_improvement_passes():
    assert compare(BASE, {**BASE, "recall_at_5": 0.91}) == []


def test_a_drop_inside_the_tolerance_is_noise_not_a_regression():
    nudged = {**BASE, "answer_accuracy": BASE["answer_accuracy"] - TOLERANCE / 2}
    assert compare(BASE, nudged) == []


def test_a_drop_past_the_tolerance_fails():
    dropped = {**BASE, "recall_at_5": BASE["recall_at_5"] - TOLERANCE - 0.01}
    failures = compare(BASE, dropped)
    assert len(failures) == 1
    assert "recall_at_5" in failures[0]


def test_a_metric_that_vanished_from_the_run_fails():
    failures = compare(BASE, {"n": 30, "recall_at_5": 0.80})
    assert any("answer_accuracy" in f and "missing" in f for f in failures)


def test_a_shrunken_question_set_fails_even_if_the_scores_rose():
    """The cheapest way to make an eval pass is to delete the questions
    it fails, so the size of the set is itself a gated metric."""
    failures = compare(BASE, {"n": 5, "recall_at_5": 1.0,
                              "answer_accuracy": 1.0})
    assert len(failures) == 1
    assert failures[0].startswith("n:")


def test_the_first_run_records_a_baseline_and_passes(tmp_path, monkeypatch):
    from evals import gate
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(gate, "BASELINE", baseline)
    results = tmp_path / "results.json"
    results.write_text(json.dumps(BASE))
    assert main(str(results)) == 0
    assert json.loads(baseline.read_text()) == BASE


def test_a_regression_exits_non_zero(tmp_path, monkeypatch):
    from evals import gate
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    monkeypatch.setattr(gate, "BASELINE", baseline)
    results = tmp_path / "results.json"
    results.write_text(json.dumps({**BASE, "answer_accuracy": 0.40}))
    assert main(str(results)) == 1
