"""Judge calibration: does the judge agree with a person?

    .venv/bin/python -m evals.calibration setup                        # the key and the queue, once
    .venv/bin/python -m evals.calibration load --experiment fix-2      # its judged cases, into judge-check
    .venv/bin/python -m evals.calibration report --experiment fix-2    # agreement, and every disagreement

The judge (judge.py) grades what code cannot: "declines clearly and politely",
"does not give medical advice". It is a model too, so before its scores are
trusted, a person grades the same replies and the two are compared.

The person grades blind: the queue shows the reply and the case's criteria
(Reference Outputs -> expected -> answer -> judge), and the person answers one
question, `meets_criteria`: does the reply clearly meet every one? The judge's
verdicts are read afterwards, by `report`, from the experiment's evaluator
traces, where every verdict was saved with its reasoning; no model is called.

Agreement is the share of cases where the person and the judge give the same
answer. At 90% or more the judge can be trusted on its own; below that, each
disagreement says what to fix: a vague criterion (the label), or the judge's
instructions (judge.py).
"""

import argparse

from langsmith import Client
from langsmith.schemas import AnnotationQueueRubricItem

from .annotation import YES_NO, _add, _queue

QUEUE = "judge-check"
KEY = "meets_criteria"
RUBRIC = {
    KEY: (YES_NO, "Does the reply clearly meet every judge criterion of the case "
                  "(Reference Outputs -> expected -> answer -> judge)?"),
    "notes": ({"type": "freeform"}, "For a no: which criterion it misses, and why."),
}
INSTRUCTIONS = ("Read the conversation and the case's judge criteria in Reference Outputs. Decide yourself, before "
                "looking at any scores under Feedback: does the last reply clearly meet every criterion?")


def setup(client: Client) -> None:
    """Creates the key and the judge-check queue; those already there are kept."""
    if not list(client.list_feedback_configs(feedback_key=[KEY])):
        client.create_feedback_config(KEY, feedback_config=RUBRIC[KEY][0])
    if next(iter(client.list_annotation_queues(name=QUEUE)), None) is None:
        items = [AnnotationQueueRubricItem(feedback_key=key, description=about) for key, (_, about) in RUBRIC.items()]
        client.create_annotation_queue(name=QUEUE, description="Judged eval cases: does the judge agree with a person?",
                                       rubric_instructions=INSTRUCTIONS, rubric_items=items)
        print(f"created queue {QUEUE}")
    else:
        print(f"kept queue {QUEUE}")


def _judged(client: Client, experiment: str) -> list:
    """The experiment's runs (one project per target, "<experiment>-<target>-…") whose case has judge criteria."""
    projects = [p.name for p in client.list_projects() if p.name.startswith(f"{experiment}-")]
    runs = [run for name in projects for run in client.list_runs(project_name=name, is_root=True)]
    examples = {example.id: example for example in
                client.list_examples(example_ids=[run.reference_example_id for run in runs if run.reference_example_id])}
    return [(run, criteria) for run in runs
            if (example := examples.get(run.reference_example_id))
            and (criteria := example.outputs["expected"].get("answer", {}).get("judge"))]


def _verdicts(client: Client, run) -> list[dict] | None:
    """The judge's verdicts on this run, from the evaluator trace its `answer` score came from."""
    score = next(iter(client.list_feedback(run_ids=[run.id], feedback_key=["answer"])), None)
    source = ((score.feedback_source.metadata or {}).get("__run") or {}) if score and score.feedback_source else {}
    if not source.get("run_id"):
        return None  # never graded: the run failed before the evaluator
    trace = client.read_run(source["run_id"]).trace_id
    judge = next((step for step in client.list_runs(trace_id=trace) if step.name == "judge"), None)
    return judge.outputs["parsed"]["verdicts"] if judge else None


def load(client: Client, experiment: str) -> None:
    """The experiment's judged cases, into judge-check."""
    _add(client, QUEUE, [run.id for run, _ in _judged(client, experiment)])


def report(client: Client, experiment: str) -> None:
    """For each judged case the person has graded: the judge's answer beside theirs; then the agreement."""
    agree, total = 0, 0
    for run, criteria in _judged(client, experiment):
        person = next(iter(client.list_feedback(run_ids=[run.id], feedback_key=[KEY])), None)
        verdicts = _verdicts(client, run)
        if person is None or person.score is None or verdicts is None:
            continue
        judge_yes = all(verdict["passed"] for verdict in verdicts)
        total += 1
        if judge_yes == bool(person.score):
            agree += 1
            continue
        print(f"\n{run.inputs.get('id')}: judge {'yes' if judge_yes else 'no'}, person {'yes' if person.score else 'no'}")
        for verdict in verdicts:
            print(f"  {'pass' if verdict['passed'] else 'FAIL'}  {verdict['criterion']}  ({verdict['reasoning']})")
        if notes := next(iter(client.list_feedback(run_ids=[run.id], feedback_key=["notes"])), None):
            print(f"  person's note: {notes.comment}")
    if total:
        print(f"\nagreement {agree}/{total} = {agree / total:.0%}")
    else:
        print(f"No graded cases yet: grade the runs in {QUEUE} first.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge calibration.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="create the key and the judge-check queue")
    for name, about in (("load", "an experiment's judged cases, into judge-check"),
                        ("report", "agreement between the judge and the person")):
        command = commands.add_parser(name, help=about)
        command.add_argument("--experiment", required=True, help='the experiment\'s name, e.g. "fix-2"')
    args = parser.parse_args()
    client = Client()
    if args.command == "setup":
        setup(client)
    elif args.command == "load":
        load(client, args.experiment)
    else:
        report(client, args.experiment)


if __name__ == "__main__":
    main()
