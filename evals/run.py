"""Runs every case in cases.json against the real model and prints a score.

    .venv/bin/python -m evals.run

Unlike the tests in tests/, which replace the model with a fake, an eval asks
the real model, so each run costs API credits and the answers can differ from
run to run. Run it before and after a change (a new prompt, agentic RAG,
LangChain) and compare the scores.

Each case is asked twice, once with the whole business document ("full") and
once with RAG, so the two modes can be compared: the model and the prompt are
the same, so a case that passes in one mode and fails in the other shows what
retrieval changed.

With LANGSMITH_TRACING=true, every question is also traced to LangSmith, in
its own project (see observe.py).
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from spadesk.app import app

from .observe import ask_case
from .summary import Result, print_summary
from .validate import stale_cases

CASES = Path(__file__).parent / "cases.json"


def main() -> None:
    """Asks each case in both modes, grades the replies and prints the result."""
    cases = json.loads(CASES.read_text())
    # Stop before spending anything if a case no longer matches the document or data.
    if stale := stale_cases(cases):
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(stale))
    # TestClient calls the app in this process, without starting a server.
    client = TestClient(app)
    results: list[Result] = []

    for case in cases:
        for mode in ("full", "rag"):
            outcome = ask_case(client, case, mode)
            result = Result(case["id"], mode, outcome["problems"], outcome["seconds"])
            results.append(result)
            print(f"{'PASS' if result.passed else 'FAIL'}  {mode:4}  {result.seconds:5.1f}s  {case['id']}")
            for problem in result.problems:
                print(f"                    {problem}")

    print_summary(results)


if __name__ == "__main__":
    main()
