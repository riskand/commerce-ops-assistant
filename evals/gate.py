# evals/gate.py
"""Turn an eval run into a build result.

    python -m evals.gate results.json

Exits 0 when every metric is at or above its recorded baseline (minus a
tolerance for the noise an LLM judge introduces), non-zero otherwise --
which is what makes it usable as a required check on a pull request.
Prompts are code; this is their test suite, and a test suite nothing
fails a build over is a report nobody reads.
"""
import json
import pathlib
import sys

BASELINE = pathlib.Path("evals/baseline.json")
# The judge is itself a model, so two runs of an unchanged system differ
# slightly. Below this, a drop is noise; at or above it, something moved.
TOLERANCE = 0.03


def compare(baseline: dict, results: dict) -> list[str]:
    """Every metric in the baseline must be met. A metric the baseline
    knows about and the results do not is a failure, not a skip: that is
    what an eval silently dropped from the suite looks like."""
    failures = []
    for metric, floor in baseline.items():
        if metric == "n" or not isinstance(floor, (int, float)):
            continue
        got = results.get(metric)
        if got is None:
            failures.append(f"{metric}: missing from this run")
        elif got < floor - TOLERANCE:
            failures.append(f"{metric}: {got} < {floor} (baseline)")
    if results.get("n", 0) < baseline.get("n", 0):
        failures.append(
            f"n: {results.get('n')} questions ran, baseline had {baseline['n']}")
    return failures


def main(path: str) -> int:
    results = json.loads(pathlib.Path(path).read_text())
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(results, indent=2) + "\n")
        print(f"no baseline yet; recorded this run as one: {results}")
        return 0
    baseline = json.loads(BASELINE.read_text())
    failures = compare(baseline, results)
    for line in failures:
        print(f"REGRESSION  {line}")
    if failures:
        print("\nIf the change is an improvement you meant to make, update "
              "evals/baseline.json in the same commit and say why.")
        return 1
    print(f"ok: {results} meets {baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
