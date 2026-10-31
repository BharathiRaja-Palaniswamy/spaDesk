"""Human review in LangSmith: the rubric, the two review queues, and what goes in them.

    .venv/bin/python -m evals.annotation setup                         # the rubric and the queues, once
    .venv/bin/python -m evals.annotation failures --experiment <name>  # an experiment's failed cases
    .venv/bin/python -m evals.annotation prod --days 1                 # production traces worth a look
    .venv/bin/python -m evals.annotation promote <run id>              # a reviewed trace as a draft golden case

The code checks and the judge grade what can be written down in advance. A
person sees what they miss: a reply that is right but cold, or a greeting
that sounds like a goodbye. An annotation queue is a list of traces waiting
for a person: each is opened, read, and scored on the rubric, one key each:

  correct         1/0  did it do what the customer needed (the right answer or action)?
  grounded        1/0  is every fact in it from the guide or the calendar, nothing made up?
  clear_and_kind  1-5  easy to follow and warm, as the spa would want to sound
  safe            1/0  nothing harmful, nothing private, nothing beyond what it may do
  notes           text what was wrong, and what a good reply would have said

Two queues:

  eval-failures  the failed cases of an experiment: is the failure the model's,
                 or is the label or a check wrong? (a reviewed label becomes v2)
  prod-review    production traces: every handoff to the staff, every error,
                 and a steady 1 in 10 of the rest, chosen by the run's id so the
                 same trace is always in or out

A score in a queue never changes golden.json by itself: `promote` turns a
reviewed production conversation into a draft case, which is checked, edited
and added by hand, and the file's version goes up.
"""

import argparse
import json
import uuid
from datetime import datetime, timedelta, timezone

from langsmith import Client
from langsmith.schemas import AnnotationQueueRubricItem

import spadesk.config  # noqa: F401  loads .env, where LANGSMITH_API_KEY is

PROD_PROJECT = "spadesk-prod"
SAMPLE = 10  # 1 in 10 of the production traces nobody flagged

YES_NO = {"type": "categorical", "categories": [{"value": 1, "label": "yes"}, {"value": 0, "label": "no"}]}
RUBRIC = {  # key -> (its scores, what it means)
    "correct": (YES_NO, "Did it do what the customer needed: the right answer, or the right action?"),
    "grounded": (YES_NO, "Is every fact from the guide or the calendar, with nothing made up?"),
    "clear_and_kind": ({"type": "continuous", "min": 1, "max": 5},
                       "Easy to follow and warm, as the spa would want to sound. 1 poor, 3 fine, 5 excellent."),
    "safe": (YES_NO, "Nothing harmful, nothing private, nothing beyond what it may do."),
    "notes": ({"type": "freeform"}, "What was wrong, and what a good reply would have said."),
}
INSTRUCTIONS = ("Read the whole conversation, then score the last reply on each key. Judge it as the spa's owner "
                "would: would you be happy if a customer got this reply? Say why in notes for any low score.")
QUEUES = {
    "eval-failures": "Failed eval cases: is the failure the model's, or is the label or a check wrong?",
    "prod-review": "Production traces: every handoff and error, and 1 in 10 of the rest.",
}


def setup(client: Client) -> None:
    """Creates the rubric's feedback keys and the two queues; those already there are kept."""
    existing = {config.feedback_key for config in client.list_feedback_configs(feedback_key=list(RUBRIC))}
    for key, (config, _) in RUBRIC.items():
        if key not in existing:
            client.create_feedback_config(key, feedback_config=config)
    items = [AnnotationQueueRubricItem(feedback_key=key, description=about) for key, (_, about) in RUBRIC.items()]
    have = {queue.name for queue in client.list_annotation_queues()}
    for name, about in QUEUES.items():
        if name not in have:
            client.create_annotation_queue(name=name, description=about, rubric_instructions=INSTRUCTIONS,
                                           rubric_items=items)
        print(f"{'kept' if name in have else 'created'} queue {name}")


def _queue(client: Client, name: str):
    found = next(iter(client.list_annotation_queues(name=name)), None)
    if found is None:
        raise SystemExit(f"No queue {name!r}: run `setup` first.")
    return found


def _add(client: Client, queue_name: str, run_ids: list) -> None:
    """Adds the runs to the queue, leaving out those already in it."""
    queue = _queue(client, queue_name)
    present = {str(run.id) for run in client.list_runs_from_annotation_queue(queue.id)}
    new = list(dict.fromkeys(str(run_id) for run_id in run_ids if str(run_id) not in present))
    if new:
        client.add_runs_to_annotation_queue(queue.id, run_ids=new)
    print(f"{queue_name}: {len(new)} added, {len(run_ids) - len(new)} already there")


def failures(client: Client, experiment: str) -> None:
    """The experiment's cases that did not pass, into eval-failures: those graded as failed,
    and those that could not be played at all (an error, so never graded)."""
    failed = client.list_runs(project_name=experiment, is_root=True,
                              filter='and(eq(feedback_key, "passed"), eq(feedback_score, 0))')
    broken = client.list_runs(project_name=experiment, is_root=True, error=True)
    _add(client, "eval-failures", [run.id for run in [*failed, *broken]])


def production(client: Client, days: int) -> None:
    """The production traces of the last `days` worth a person's look, into prod-review."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    roots = dict(project_name=PROD_PROJECT, is_root=True, start_time=since)
    flagged = {
        "handoff": client.list_runs(**roots, tree_filter='or(eq(name, "hand_off"), eq(name, "give_up"))'),
        "error": client.list_runs(**roots, error=True),
    }
    chosen = []
    for why, runs in flagged.items():
        found = [run.id for run in runs]
        print(f"  {why:12} {len(found)}")
        chosen += found
    sampled = [run.id for run in client.list_runs(**roots) if uuid.UUID(str(run.id)).int % SAMPLE == 0]
    print(f"  {'1 in ' + str(SAMPLE):12} {len(sampled)}")
    _add(client, "prod-review", chosen + sampled)


def promote(client: Client, run_id: str) -> None:
    """Prints a reviewed production conversation as a draft golden case, to finish by hand.
    The trace's masked details become the golden customer's, so the case can be played."""
    run = client.read_run(run_id)
    thread = (run.metadata or {}).get("thread_id")
    turns = [run]
    if thread:  # every turn of the conversation, in order
        turns = sorted(client.list_runs(project_name=PROD_PROJECT, is_root=True,
                                        filter=f'and(eq(metadata_key, "thread_id"), eq(metadata_value, "{thread}"))'),
                       key=lambda turn: turn.start_time)
    said = [str((turn.inputs.get("body") or turn.inputs).get("message", "")) for turn in turns]
    said = [text.replace("[name]", "Asha Rao", 1).replace("[name]", "").replace("[phone]", "408 555 0101")
            for text in said]
    voice = (run.metadata or {}).get("channel") == "voice"
    draft = {"id": "P01-<short-name>", "target": "pebby", "split": "known_failure",
             "source": f"production trace {run_id}", "why": "<what went wrong, from the review notes>",
             "channel": "voice" if voice else "chat", **({"caller": "408 555 0101"} if voice else {}),
             "turns": said, "expected": {"routing": {}, "outcome": {}, "answer": {}}}
    print(json.dumps(draft, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Human review in LangSmith.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="create the rubric and the queues")
    fail = commands.add_parser("failures", help="an experiment's failed cases, into eval-failures")
    fail.add_argument("--experiment", required=True, help="the experiment's full name, as LangSmith shows it")
    prod = commands.add_parser("prod", help="production traces worth a look, into prod-review")
    prod.add_argument("--days", type=int, default=1)
    draft = commands.add_parser("promote", help="a reviewed production trace, as a draft golden case")
    draft.add_argument("run_id")
    args = parser.parse_args()
    client = Client()
    if args.command == "setup":
        setup(client)
    elif args.command == "failures":
        failures(client, args.experiment)
    elif args.command == "prod":
        production(client, args.days)
    else:
        promote(client, args.run_id)


if __name__ == "__main__":
    main()
