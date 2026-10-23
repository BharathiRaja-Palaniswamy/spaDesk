"""Runs every case in cases.json against the real model and prints a score.

    .venv/bin/python -m evals.run --label rerank

The label names the run in LangSmith (the `run:<label>` tag), so runs can be
compared there; it defaults to "unlabelled".

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

The answers always come from the open-source models (SPADESK_MODELS is set
to "opensource" here, whatever `.env` says), with no fallback, so a score
always measures them. The reasoning: if the open models answer well, Claude
can be expected to do as well or better. The judge stays on Claude Opus
(spadesk/models/profiles.py). The LangChain pipeline is used, since the
hand-built one calls Claude directly.
"""

import argparse
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from spadesk.app import app
from spadesk.models.profiles import get_profile

from .observe import ask_case
from .summary import Result, print_summary
from .validate import stale_cases

CASES = Path(__file__).parent / "cases.json"


def main() -> None:
    """Asks each case in both modes, grades the replies and prints the result."""
    parser = argparse.ArgumentParser(description="Run the evals against the real model.")
    parser.add_argument("--label", default="unlabelled", help='names this run in LangSmith, e.g. "rerank"')
    label = parser.parse_args().label
    # Settings are read on every call, so setting these here applies to the whole run.
    os.environ |= {"SPADESK_MODELS": "opensource", "SPADESK_MODEL_FALLBACK": "", "SPADESK_RAG_ENGINE": "langchain"}
    print(f"Answers: {get_profile('chat-default').primary}    Judge: {get_profile('judge').primary}")
    cases = json.loads(CASES.read_text())
    # Stop before spending anything if a case no longer matches the document or data.
    if stale := stale_cases(cases):
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(stale))
    # TestClient calls the app in this process, without starting a server.
    client = TestClient(app)
    results: list[Result] = []

    for case in cases:
        for mode in ("full", "rag"):
            outcome = ask_case(client, case, mode, label)
            result = Result(case["id"], mode, outcome["problems"], outcome["seconds"])
            results.append(result)
            print(f"{'PASS' if result.passed else 'FAIL'}  {mode:4}  {result.seconds:5.1f}s  {case['id']}")
            for problem in result.problems:
                print(f"                    {problem}")

    print_summary(results)


if __name__ == "__main__":
    main()
