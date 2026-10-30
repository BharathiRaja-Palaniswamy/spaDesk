"""Runs the booking conversations in booking_cases.json against the real models
and scores the agents in three layers.

    .venv/bin/python -m evals.booking --label booking-agent

Each case is a conversation, played turn by turn into the real agent, and is
graded in up to three layers, each in code (no judge model):

  1. routing     did Pebby take the right path for the message: the right tool
                 (booking, search_guide, record_feedback, hand_off) and, for a
                 booking, the right intent? `not_tools` must not have been used.
                 A routing failure is a wrong path: the wrong-path rate is the
                 share of cases that went down one.
  2. extraction  are the details Pebby sent the customer's own: the services
                 (exactly these; or only ones the customer named), a package,
                 their words for the day and time, a therapist, the bookings. And
                 missing info handled: a detail the customer did not give must
                 not be sent (it is asked for instead).
  3. downstream  did the conversation end as it should: asking for the
                 customer's yes or not, a reply that says what it must (and
                 nothing it must not), exactly as many bookings as approved
                 (a yes books once), no one's booking removed unasked, the
                 staff handed what is beyond one booking, and one feedback
                 card per piece of feedback.

The cases are balanced: about half are requests that should go through (a
question answered, a booking made, feedback filed), and half are requests
that must not (a detail missing, a service the spa does not offer, a "no",
another customer's bookings, a bulk, group or recurring request, and harmful
ones: instructions to ignore the rules, asking for Pebby's instructions, an
unsafe treatment, something the spa does not offer or must not do): there the
right outcome is a question, a polite refusal or the staff, never a booking or
a cancellation.

And what it cost: the model calls each conversation made, by job (Pebby's
`booking-agent`, and the small `cheap-router` that reads names and
feedback), with their tokens. A model that repeats a tool call shows up
here even when the answer was right.

Each layer is scored over the cases that check it, so a failure shows where
it comes from: a wrong path, a detail made up, or a wrong outcome. Reading
words into dates and times, and booking itself, are code; tests/ covers them
for free, so this measures only what a model can get wrong.

The calendar is an in-memory one (the tests' stand-in), empty except for the
case's `seed` bookings, with the clock at the real time now, so "Friday"
means the coming Friday. Nothing reaches Google Calendar; handoffs and
feedback are written to a temporary folder, and counted per case.

Traces go to the eval project (LANGSMITH_EVAL_PROJECT, "spadesk-evals"), as
for evals/run.py, not the app's: each case is one parent run, "eval: <case>
(booking)", tagged "eval", "booking", the model set and "run:<label>", with
"passed" in its metadata, and the conversation's turns inside it.

Like evals/run.py, the answers come from the open-source models unless
--models says otherwise, with no fallback. Each run costs API credits.
"""

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from langsmith import get_current_run_tree, traceable, tracing_context

from spadesk.agent import booking, executor, pebby, respond, yes
from spadesk.agent.details import named_services
from spadesk.agent.feedback import cards as feedback_cards
from spadesk.agent.flow import get_graph
from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge
from spadesk.config import get_settings
from spadesk.models import gateway
from spadesk.models.profiles import get_profile
from tests.standin_calendar import standin_desk

from .observe import EVAL_PROJECT

CASES = Path(__file__).parent / "booking_cases.json"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LAYERS = ("routing", "extraction", "downstream")
PHONE = "408 555 0101"  # the customer every case plays (a seed booking may name another)


def _desk(seed: list[dict]):
    """An in-memory calendar at the real time now, with the case's bookings (the customer's own,
    unless a seed names another `phone`), used by every part of the agent."""
    zone = ZoneInfo(get_settings().timezone)
    desk = standin_desk(now=lambda: datetime.now(zone))
    for seeded in seed:
        day = _next(seeded["weekday"], desk.now())
        hour, minute = map(int, seeded["time"].split(":"))
        desk.book(seeded["service"], seeded["therapist"], datetime(day.year, day.month, day.day, hour, minute),
                  seeded.get("customer", "Asha Rao"), seeded.get("phone", PHONE))
    for module in (pebby, yes, executor, booking):
        module.get_appointments = lambda: desk
    return desk


def _next(weekday: str, now: datetime):
    """The coming `weekday` (tomorrow at the earliest)."""
    ahead = (DAYS.index(weekday) - now.weekday() - 1) % 7 + 1
    return (now + timedelta(days=ahead)).date()


def _everyone(desk) -> list:
    """Every customer's upcoming bookings on the calendar, not only the case's customer."""
    now = desk.now()
    return desk.calendar.bookings(now - timedelta(days=1), now + timedelta(days=60))


def _lines(path: Path) -> int:
    """How many records a JSON-lines file holds (0 before the first is written)."""
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def _counting(spent: list) -> None:
    """Have every model call also add its usage to `spent` (the gateway logs it as before)."""
    logged = gateway.record

    def record(usage):
        spent.append(usage)
        logged(usage)
    gateway.record = record


def _calls(thread: str) -> list[dict]:
    """Every tool request Pebby made in the conversation, in order."""
    messages = get_graph().get_state({"configurable": {"thread_id": thread}}).values["messages"]
    return [call for m in messages for call in (getattr(m, "tool_calls", None) or [])]


def _routing(want: dict, calls: list[dict]) -> list[str]:
    """Layer 1: the right tool and intent; none of the wrong ones."""
    used, problems = [call["name"] for call in calls], []
    if (tool := want.get("tool")) and tool not in used:
        problems.append(f"did not use {tool} (used {used or 'no tools'})")
    problems += [f"used {name}" for name in want.get("not_tools", []) if name in used]
    intents = [call["args"].get("intent") for call in calls if call["name"] == "booking"]
    if "intent" in want and (not intents or intents[-1] not in want["intent"]):
        problems.append(f"intent {intents[-1] if intents else None!r}, expected one of {want['intent']}")
    return problems


def _extraction(want: dict, calls: list[dict], words: str) -> list[str]:
    """Layer 2: the details sent are the customer's own; nothing they did not give."""
    requests = [call["args"] for call in calls if call["name"] == "booking"]
    problems = []
    if want.get("only_named_services"):  # every booking request, not just the last
        problems += [f"chose {unnamed} for them" for got in requests
                     if (unnamed := named_services(got.get("services", []), got.get("package", ""), words)[1])]
    for field in ("day_said", "time_said"):  # missing info is asked for, never filled in
        if want.get(f"no_{field}"):
            problems += [f"{field} {got[field]!r}, but the customer said none" for got in requests if got.get(field)]
    asks = [key for key in ("services", "package", "day_said_has", "time_said_has", "therapist", "references")
            if key in want]
    if not asks:
        return problems
    if not requests:
        return problems + ["never called booking"]
    got = requests[-1]
    if "services" in want and sorted(got.get("services", [])) != sorted(want["services"]):
        problems.append(f"services {got.get('services')}, expected {want['services']}")
    if want.get("package") and not got.get("package"):
        problems.append("no package name")
    for field in ("day_said", "time_said"):
        if (part := want.get(f"{field}_has")) and part not in got.get(field, "").lower():
            problems.append(f"{field} {got.get(field)!r}, expected it to mention {part!r}")
    if "therapist" in want and got.get("therapist") != want["therapist"]:
        problems.append(f"therapist {got.get('therapist')!r}, expected {want['therapist']!r}")
    if want.get("references") and not got.get("references"):
        problems.append("no booking references")
    return problems


def _downstream(want: dict, turn, booked: int, handed: int, cards: int, removed: int) -> list[str]:
    """Layer 3: the conversation ended as it should."""
    problems = []
    if "confirming" in want and turn is not None and turn.confirming != want["confirming"]:
        problems.append(f"confirming is {turn.confirming}, expected {want['confirming']}")
    answer = turn.answer.lower() if turn else ""
    problems += [f"the reply does not mention {word!r}" for word in want.get("answer_has", [])
                 if word.lower() not in answer]
    problems += [f"the reply mentions {word!r}" for word in want.get("answer_lacks", []) if word.lower() in answer]
    if "booked" in want and booked != want["booked"]:
        problems.append(f"{booked} bookings made, expected {want['booked']}")
    if "removed" in want and removed != want["removed"]:
        problems.append(f"{removed} bookings removed from the calendar, expected {want['removed']}")
    if "handed_off" in want and bool(handed) != want["handed_off"]:
        problems.append(f"{'handed' if handed else 'not handed'} to the staff, expected the opposite")
    if "cards" in want and cards != want["cards"]:
        problems.append(f"{cards} feedback cards filed, expected {want['cards']}")
    return problems


def grade(case: dict, calls: list[dict], turn, booked: int = 0, handed: int = 0,
          cards: int = 0, removed: int = 0) -> dict[str, list[str]]:
    """The problems found in each layer the case checks; a layer passed when its list is empty.
    `booked`, `handed`, `cards`: the bookings, staff handoffs and feedback cards the conversation made;
    `removed`: the bookings (anyone's) it took off the calendar."""
    words = " ".join(case["turns"]).lower()
    checks = {"routing": lambda want: _routing(want, calls),
              "extraction": lambda want: _extraction(want, calls, words),
              "downstream": lambda want: _downstream(want, turn, booked, handed, cards, removed)}
    return {layer: checks[layer](case[layer]) for layer in LAYERS if layer in case}


def summary(graded: list[dict[str, list[str]]]) -> list[str]:
    """Each layer's pass rate, the wrong-path rate, and how many cases passed in every layer."""
    lines = []
    for layer in LAYERS:
        scored = [g[layer] for g in graded if layer in g]
        passed = sum(not problems for problems in scored)
        lines.append(f"{layer:12} {passed}/{len(scored)}" + (f"  ({passed / len(scored):.0%})" if scored else ""))
    routed = [g["routing"] for g in graded if "routing" in g]
    wrong = sum(bool(problems) for problems in routed)
    lines.append(f"{'wrong path':12} {wrong}/{len(routed)}" + (f"  ({wrong / len(routed):.0%})" if routed else ""))
    lines.append(f"{'all layers':12} {sum(not any(g.values()) for g in graded)}/{len(graded)}")
    return lines


def stale(cases: list[dict]) -> list[str]:
    """Cases naming a service or therapist the document no longer has."""
    schedule = read_schedule(get_knowledge())
    found = []
    for case in cases:
        want = case.get("extraction", {})
        named = want.get("services", []) + [b["service"] for b in case.get("seed", [])]
        found += [f"{case['id']}: no service {name!r}" for name in named if name not in schedule.services]
        people = [want["therapist"]] if "therapist" in want else []
        people += [b["therapist"] for b in case.get("seed", [])]
        found += [f"{case['id']}: no therapist {name!r}" for name in people if name not in schedule.therapists]
    return found


@traceable
def _play(case: dict, thread: str, desk, handoffs: Path, seeded: set[str],
          before: tuple[int, int, int]) -> dict[str, list[str]]:
    """Plays one case's turns into Pebby and grades the conversation. Its inputs and
    result (each layer's problems) are what LangSmith shows for the case's parent run."""
    turn = None
    for said in case["turns"]:
        turn = respond(thread, said)
    after = len(desk.upcoming(PHONE)), _lines(handoffs), len(feedback_cards())
    removed = len(seeded - {b.id for b in _everyone(desk)})
    graded = grade(case, _calls(thread), turn, *(a - b for a, b in zip(after, before)), removed)
    if run := get_current_run_tree():  # filter failed cases in LangSmith by this
        run.metadata.update({"passed": not any(graded.values())})
    return graded


def main() -> None:
    """Plays each case, grades it layer by layer and prints the scores."""
    parser = argparse.ArgumentParser(description="Run the booking evals against the real models.")
    parser.add_argument("--label", default="unlabelled", help="names the run in LangSmith (the run:<label> tag) and goes into each thread id")
    parser.add_argument("--models", default="opensource", choices=["opensource", "claude"])
    args = parser.parse_args()
    folder = tempfile.mkdtemp(prefix="spadesk-booking-eval-")
    handoffs = Path(folder) / "handoffs.jsonl"
    os.environ |= {"SPADESK_MODELS": args.models, "SPADESK_MODEL_FALLBACK": "",
                   "SPADESK_HANDOFFS": str(handoffs), "SPADESK_FEEDBACK": f"{folder}/feedback.jsonl"}
    print(f"Pebby: {get_profile('booking-agent').primary}")
    cases = json.loads(CASES.read_text())
    if problems := stale(cases):  # stop before spending anything
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(problems))
    graded, seconds, spent = [], [], []
    _counting(spent)
    calls = Counter()
    tokens = 0
    for case in cases:
        spent.clear()
        desk = _desk(case.get("seed", []))
        seeded = {b.id for b in _everyone(desk)}
        before = len(desk.upcoming(PHONE)), _lines(handoffs), len(feedback_cards())
        thread = f"eval-{args.label}-{case['id']}-{int(time.time())}"
        started = time.perf_counter()
        # In the eval project, not the app's: one parent run per case, its turns inside.
        with tracing_context(project_name=EVAL_PROJECT, tags=["eval", "booking", args.models, f"run:{args.label}"],
                             metadata={"case": case["id"], "run": args.label, "models": args.models,
                                       "thread_id": thread}):
            graded.append(_play(case, thread, desk, handoffs, seeded, before,
                                langsmith_extra={"name": f"eval: {case['id']} (booking)"}))
        seconds.append(time.perf_counter() - started)
        calls.update(usage.profile for usage in spent)
        tokens += sum(usage.input_tokens + usage.output_tokens for usage in spent)
        layers = "  ".join(f"{layer[:5]} {'ok' if not problems else 'FAIL'}" for layer, problems in graded[-1].items())
        print(f"{seconds[-1]:5.1f}s  {len(spent):2} calls  {case['id']:36} {layers}")
        for layer, problems in graded[-1].items():
            for problem in problems:
                print(f"         {layer}: {problem}")
    print("\n" + "\n".join(summary(graded)))
    print(f"{'avg time':12} {sum(seconds) / len(seconds):.1f}s a conversation")
    per = ", ".join(f"{profile} {count / len(cases):.1f}" for profile, count in calls.most_common())
    print(f"{'avg calls':12} {sum(calls.values()) / len(cases):.1f} a conversation ({per}), "
          f"{tokens / len(cases):,.0f} tokens")

if __name__ == "__main__":
    main()
