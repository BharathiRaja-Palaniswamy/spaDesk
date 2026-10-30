"""Makes each eval case easy to find in LangSmith.

The app is already traced (spadesk/tracing.py), so an eval run sends traces
whenever LANGSMITH_TRACING=true. Without this file they would land in the
app's own project, mixed with real chats and with nothing to say which case
they were. Here each case is played inside a parent run:

    eval: H05-body-wrap-price (pebby)      <- this file: case, split, pass/fail
      ├─ pebby turn                         <- the app's own traces, one per turn
      │    ├─ booking-agent                    prompt, reply, tokens
      │    └─ search_guide                     the guide's excerpts
      └─ ...

and every run goes to its own project, LANGSMITH_EVAL_PROJECT (default
"spadesk-evals"), tagged "eval", the target, the split and "run:<label>". In
LangSmith a failed case can then be filtered by tag or metadata ("passed")
and opened to see why it failed.
"""

import os
from collections.abc import Callable
from typing import Any

from langsmith import get_current_run_tree, traceable, tracing_context

from .golden import Case

EVAL_PROJECT = os.getenv("LANGSMITH_EVAL_PROJECT", "spadesk-evals")


def _shown(inputs: dict[str, Any]) -> dict[str, Any]:
    """What the parent run shows as its input: the case as played, not the function that plays it."""
    case = inputs["case"]
    return {key: case[key] for key in ("id", "target", "split", "channel", "caller", "seed", "turns") if key in case}


@traceable(process_inputs=_shown)
def _graded(play: Callable[[], dict[str, Any]], case: Case) -> dict[str, Any]:
    """Plays the case. Its inputs and result (each layer's problems) are what
    LangSmith shows for the case's parent run."""
    result = play()
    if run := get_current_run_tree():  # filter failed cases in LangSmith by this
        run.metadata.update({"passed": not any(result["layers"].values())})
    return result


def traced(case: Case, label: str, play: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """`play()`, recorded in the eval project as one parent run named after the case."""
    with tracing_context(project_name=EVAL_PROJECT,
                         tags=["eval", case["target"], case["split"], f"run:{label}"],
                         metadata={"case": case["id"], "target": case["target"], "split": case["split"],
                                   "run": label}):
        return _graded(play, case, langsmith_extra={"name": f"eval: {case['id']} ({case['target']})"})
