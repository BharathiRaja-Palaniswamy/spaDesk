"""Runs the golden cases as a LangSmith experiment, to compare runs there.

    .venv/bin/python -m evals.experiment --name baseline-v1                  # every target
    .venv/bin/python -m evals.experiment --name fix-1 --target pebby --split edge

It plays the same cases as run.py, the same way (the open-source models, RAG
on, the judge on Claude), but from the LangSmith datasets (datasets.py: send
them first) and through LangSmith's `evaluate`, which records the experiment:

  target      plays one example on its target and returns what happened
              (run.play): the run's outputs, kept with the experiment
  evaluator   grades those outputs against the example's reference outputs,
              its label (run.grade), and gives one score per layer: routing,
              extraction, outcome, answer (1 passed, 0 failed, the problems
              as the comment), and `passed` for the whole case

Each target's dataset is its own experiment, named "<name>-<target>". In
LangSmith, two experiments on one dataset can be opened side by side (the
comparison view): case by case, which scores went up and which went down.
An experiment's metadata records what it ran with: the models, the golden
version, the retrieval settings and the code's commit, so a change in a score
can be traced to a change in one of them.

Cases run one at a time: Pebby's calendar and the usage count are shared by
the whole process. Each run costs API credits.
"""

import argparse
import subprocess
from typing import Any

from langsmith import Client, evaluate
from langsmith.run_trees import get_cached_client

from spadesk.config import get_settings
from spadesk.models.profiles import get_profile

from . import golden
from .datasets import dataset_name
from .run import grade, play, prepare
from .summary import LAYERS


def _commit() -> str:
    """The code's git commit, with "+changes" when the working tree is not clean."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return sha + ("+changes" if dirty else "")


def scores(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """The evaluator: one score per layer the label checks, and `passed`."""
    layers = grade({**inputs, "expected": reference_outputs["expected"]}, outputs)
    results = [{"key": layer, "score": int(not problems), "comment": "; ".join(problems) or None}
               for layer, problems in layers.items()]
    return {"results": results + [{"key": "passed", "score": int(not any(layers.values()))}]}


def experiment(client: Client, target: str, name: str, split: str | None, folder, chat) -> None:
    """Runs one target's dataset (at golden.json's version tag) as the experiment "<name>-<target>"."""
    version = golden.load()["version"]
    examples = client.list_examples(dataset_name=dataset_name(target), as_of=f"v{version}",
                                    splits=[split] if split else None)
    settings = get_settings()
    evaluate(lambda inputs: play(inputs, name, folder, chat), data=examples, evaluators=[scores],
             experiment_prefix=f"{name}-{target}", max_concurrency=0, client=client,
             description=f"golden v{version}, {target}" + (f", {split} only" if split else ""),
             metadata={"golden_version": version, "target": target, "split": split or "all", "commit": _commit(),
                       "models": str(get_profile("booking-agent").primary), "judge": str(get_profile("judge").primary),
                       "rag_engine": settings.rag_engine, "reranker": settings.reranker,
                       "top_k": settings.retrieval_top_k, "layers": [layer for layer in LAYERS]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden cases as LangSmith experiments.")
    parser.add_argument("--name", required=True, help='names the experiments, e.g. "baseline-v1"')
    parser.add_argument("--target", default="all", choices=["all", *golden.TARGETS])
    parser.add_argument("--split", choices=golden.SPLITS, help="only the cases of one split")
    args = parser.parse_args()
    folder, chat, _ = prepare(golden.cases(args.target))
    client = get_cached_client()  # the app's client, set up by prepare: personal details masked (tracing.py)
    for target in golden.TARGETS if args.target == "all" else [args.target]:
        experiment(client, target, args.name, args.split, folder, chat)


if __name__ == "__main__":
    main()
