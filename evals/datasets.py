"""Copies golden.json into LangSmith as datasets, one per target.

    .venv/bin/python -m evals.datasets          # what would change, sent nowhere
    .venv/bin/python -m evals.datasets --send   # make LangSmith match golden.json

A LangSmith dataset is a list of examples, each with inputs (what the target
is given) and reference outputs (what it should do). Here every golden case
becomes one example in "spadesk-golden-<target>":

  inputs     the case as it is played: id, turns, channel, caller, seed
  outputs    its label, `expected` (the evaluators compare against it)
  metadata   category, source, why: to filter and read a case in LangSmith
  splits     its split (happy, edge, known_failure, adversarial), so an
             experiment can be run on, or read for, one kind of case

golden.json stays the source of truth: the copy goes one way only, and any
change in LangSmith is overwritten at the next --send. Each example's id is
worked out from the case's id, so sending twice changes nothing, an edited
case is updated in place, and a case taken out of golden.json is deleted.
After a send, the dataset is tagged with golden.json's version ("v1"), so an
experiment can name exactly the cases it ran on.

Nothing here calls a model; it only needs LANGSMITH_API_KEY.
"""

import argparse
import uuid
from datetime import datetime, timezone
from typing import Any

from langsmith import Client

import spadesk.config  # noqa: F401  loads .env, where LANGSMITH_API_KEY is

from . import golden

NAMESPACE = uuid.UUID("5a6d3c1e-9f43-4c2e-8f3a-2b1d7e6c9a10")  # fixed: the same case always gets the same id
PLAYED = ("id", "target", "turns", "channel", "caller", "seed")  # what the target is given
NOTES = ("category", "source", "why")


def dataset_name(target: str) -> str:
    return f"spadesk-golden-{target}"


def example(case: golden.Case, version: int) -> dict[str, Any]:
    """One golden case as a LangSmith example."""
    return {"id": str(uuid.uuid5(NAMESPACE, case["id"])),
            "inputs": {key: case[key] for key in PLAYED if key in case},
            "outputs": {"expected": case["expected"]},
            "metadata": {**{key: case[key] for key in NOTES if key in case}, "split": case["split"],
                         "golden_version": version},
            "split": case["split"]}


def _ours(metadata: dict | None) -> dict:
    """An example's metadata without what LangSmith adds itself (its split, as "dataset_split")."""
    return {key: value for key, value in (metadata or {}).items() if key != "dataset_split"}


def plan(client: Client, target: str, version: int) -> dict[str, list]:
    """What a send would do to the target's dataset: the examples to add, to
    update (changed in golden.json), and the ids to delete (gone from it)."""
    wanted = {e["id"]: e for e in (example(c, version) for c in golden.cases(target))}
    name = dataset_name(target)
    present = {str(e.id): e for e in client.list_examples(dataset_name=name)} if client.has_dataset(dataset_name=name) else {}
    changed = [e for i, e in wanted.items() if i in present and (present[i].inputs, present[i].outputs,
                                                                  _ours(present[i].metadata)) != (e["inputs"], e["outputs"], e["metadata"])]
    return {"add": [e for i, e in wanted.items() if i not in present], "update": changed,
            "delete": [i for i in present if i not in wanted]}


def send(client: Client, target: str, changes: dict[str, list]) -> None:
    """Applies `changes` to the target's dataset, creating it first if needed."""
    name = dataset_name(target)
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description=f"spaDesk golden cases for {target} (copied from evals/golden.json).")
    if changes["add"]:
        client.create_examples(dataset_name=name, examples=changes["add"])
    for e in changes["update"]:
        client.update_example(e["id"], inputs=e["inputs"], outputs=e["outputs"], metadata=e["metadata"],
                              split=e["split"])
    for example_id in changes["delete"]:
        client.delete_example(example_id)


def tag(client: Client, target: str, version: int) -> None:
    """Tags the dataset as it is now with golden.json's version ("v1"): an experiment run
    on that tag gets exactly these examples, whatever is changed later."""
    client.update_dataset_tag(dataset_name=dataset_name(target), as_of=datetime.now(timezone.utc), tag=f"v{version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy golden.json into LangSmith datasets.")
    parser.add_argument("--send", action="store_true", help="make the changes; without it, only list them")
    args = parser.parse_args()
    client, version = Client(), golden.load()["version"]
    for target in golden.TARGETS:
        changes = plan(client, target, version)
        print(f"{dataset_name(target):28} add {len(changes['add']):3}  update {len(changes['update']):3}  "
              f"delete {len(changes['delete']):3}")
        if args.send:
            if any(changes.values()):
                send(client, target, changes)
            tag(client, target, version)
    print(f"Tagged v{version}." if args.send else "Nothing sent: run with --send to apply.")


if __name__ == "__main__":
    main()
