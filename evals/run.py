"""Runs the golden cases against the real models and prints the scores.

    .venv/bin/python -m evals.run --label baseline                  # every case
    .venv/bin/python -m evals.run --target pebby --label baseline   # one target
    .venv/bin/python -m evals.run --only H05 E03                    # a few cases, by id

Unlike the tests in tests/, which replace the model with a stand-in, an eval
asks the real models, so each run costs API credits and the answers can
differ from run to run. Run it before and after a change and compare.

Every case in golden.json is played on its target (golden.py), what happened
is recorded (`observe`), and then graded layer by layer against its label
(`grade`): Pebby's conversations in pebby.py, the two chats in chats.py, the
answer checks shared in checks.py. The same two steps make a LangSmith
experiment (experiment.py). The cases are checked against the guide first
(validate.py), so nothing is spent on a case that is out of date.

The answers always come from the open-source models (SPADESK_MODELS is set
to "opensource" here, whatever `.env` says), with no fallback, so a score
always measures them. The reasoning: if the open models answer well, Claude
can be expected to do as well or better. The judge stays on Claude Opus
(spadesk/models/profiles.py). RAG is on, as customers get it, through the
LangChain pipeline.

With LANGSMITH_TRACING=true, each case is also traced to its own project
(observe.py); --label names the run there (the `run:<label>` tag).
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

from spadesk.models import gateway
from spadesk.models.profiles import get_profile

from . import chats, golden, pebby
from .observe import traced
from .summary import Result, summary
from .validate import stale_cases


def prepare(cases: list[golden.Case]) -> tuple[Path, object, list]:
    """Sets up a run of `cases`: the open-source models with no fallback, RAG through
    LangChain, handoffs and feedback cards in a temporary folder (a run never touches
    the real ones), and every model call counted. Stops before anything is spent if a
    case is out of date. Returns the folder, a client of the app, and the usage list."""
    folder = Path(tempfile.mkdtemp(prefix="spadesk-eval-"))
    os.environ |= {"SPADESK_MODELS": "opensource", "SPADESK_MODEL_FALLBACK": "", "SPADESK_RAG_ENGINE": "langchain",
                   "SPADESK_HANDOFFS": str(folder / "handoffs.jsonl"), "SPADESK_FEEDBACK": str(folder / "feedback.jsonl")}
    print(f"Answers: {get_profile('booking-agent').primary}    Judge: {get_profile('judge').primary}")
    if stale := stale_cases(cases):
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(stale))
    spent: list = []
    _counting(spent)
    return folder, chats.client(), spent


def play(case: golden.Case, label: str, folder: Path, chat) -> dict:
    """Plays the case on its target and records what happened (`observe`)."""
    if case["target"] == "pebby":
        return pebby.observe(case, f"eval-{label}-{case['id']}-{int(time.time())}", folder / "handoffs.jsonl")
    return chats.observe(case, chat)


def grade(case: golden.Case, seen: dict) -> dict[str, list[str]]:
    """Each layer's problems, from what `play` recorded, against the case's label."""
    return (pebby if case["target"] == "pebby" else chats).grade(case, seen)


def _counting(spent: list) -> None:
    """Have every model call also add its usage to `spent` (the gateway logs it as before)."""
    logged = gateway.record

    def record(usage):
        spent.append(usage)
        logged(usage)
    gateway.record = record


def _played(case: golden.Case, label: str, folder: Path, chat) -> dict:
    """Plays and grades the case. A case that cannot be played (a route's error, say)
    fails with the reason, and the run goes on."""
    try:
        seen = play(case, label, folder, chat)
    except Exception as error:
        return {"layers": {"errors": [f"{type(error).__name__}: {error}"[:300]]}, "conversation": []}
    return {"layers": grade(case, seen), "conversation": seen["conversation"]}


def _print(result: Result) -> None:
    """One line per case, then each problem under it."""
    layers = "  ".join(f"{layer[:5]} {'ok' if not problems else 'FAIL'}" for layer, problems in result.layers.items())
    print(f"{'PASS' if result.passed else 'FAIL'}  {result.seconds:5.1f}s {result.calls:2} calls  "
          f"{result.case:40} {layers}")
    for layer, problems in result.layers.items():
        for problem in problems:
            print(f"        {layer}: {problem}")


def main() -> None:
    """Plays each case, grades it and prints the scores."""
    parser = argparse.ArgumentParser(description="Run the golden evals against the real models.")
    parser.add_argument("--target", default="all", choices=["all", *golden.TARGETS])
    parser.add_argument("--only", nargs="*", help="only the cases whose id starts with one of these, e.g. H05 E03")
    parser.add_argument("--label", default="unlabelled", help='names the run in LangSmith, e.g. "baseline"')
    args = parser.parse_args()
    cases = golden.cases(args.target, args.only)
    folder, chat, spent = prepare(cases)
    results = []
    for case in cases:
        spent.clear()
        started = time.perf_counter()
        played = traced(case, args.label, lambda: _played(case, args.label, folder, chat))
        answering = [usage for usage in spent if usage.profile != "judge"]  # what the case cost, not its grading
        results.append(Result(case["id"], case["target"], case["split"], played["layers"],
                              time.perf_counter() - started, len(answering),
                              sum(u.input_tokens + u.output_tokens for u in answering), played["conversation"]))
        _print(results[-1])
    print("\n".join(summary(results)))


if __name__ == "__main__":
    main()
