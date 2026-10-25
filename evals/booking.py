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
                 customer's yes or not, a reply that says what it must, and
                 exactly as many bookings as approved (a yes books once).

And what it cost: the model calls each conversation made, by job (Pebby's
`booking-agent`, the small `cheap-router` that reads names, the Booking
agent's `booking-team`), with their tokens. A model that repeats a tool call
shows up here even when the answer was right.

Each layer is scored over the cases that check it, so a failure shows where
it comes from: a wrong path, a detail made up, or a wrong outcome. Reading
words into dates and times, and booking itself, are code; tests/ covers them
for free, so this measures only what a model can get wrong.

The calendar is an in-memory one (the tests' stand-in), empty except for the
case's `seed` bookings, with the clock at the real time now, so "Friday"
means the coming Friday. Nothing reaches Google Calendar; handoffs and
feedback are written to a temporary folder.

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

from spadesk.agent import respond
from spadesk.agent.booking import agent as booking_agent
from spadesk.agent.booking import executor as approved
from spadesk.agent.booking import yes as confirm
from spadesk.agent.booking.collect import named_services
from spadesk.agent.pebby import identify, tools
from spadesk.agent.pebby.graph import get_graph
from spadesk.booking import actions
from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge
from spadesk.config import get_settings
from spadesk.models import gateway
from spadesk.models.profiles import get_profile
from tests.standin_calendar import standin_desk

CASES = Path(__file__).parent / "booking_cases.json"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LAYERS = ("routing", "extraction", "downstream")


def _desk(seed: list[dict], phone: str):
    """An in-memory calendar at the real time now, with the case's bookings, used by every part of the agent."""
    zone = ZoneInfo(get_settings().timezone)
    desk = standin_desk(now=lambda: datetime.now(zone))
    for booking in seed:
        day = _next(booking["weekday"], desk.now())
        hour, minute = map(int, booking["time"].split(":"))
        desk.book(booking["service"], booking["therapist"], datetime(day.year, day.month, day.day, hour, minute),
                  "Asha Rao", phone)
    for module in (tools, confirm, identify, actions, approved, booking_agent):
        module.get_appointments = lambda: desk
    return desk


def _next(weekday: str, now: datetime):
    """The coming `weekday` (tomorrow at the earliest)."""
    ahead = (DAYS.index(weekday) - now.weekday() - 1) % 7 + 1
    return (now + timedelta(days=ahead)).date()


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


def _downstream(want: dict, turn, booked: int) -> list[str]:
    """Layer 3: the conversation ended as it should."""
    problems = []
    if "confirming" in want and turn is not None and turn.confirming != want["confirming"]:
        problems.append(f"confirming is {turn.confirming}, expected {want['confirming']}")
    answer = turn.answer.lower() if turn else ""
    problems += [f"the reply does not mention {word!r}" for word in want.get("answer_has", [])
                 if word.lower() not in answer]
    if "booked" in want and booked != want["booked"]:
        problems.append(f"{booked} bookings made, expected {want['booked']}")
    return problems


def grade(case: dict, calls: list[dict], turn, booked: int = 0) -> dict[str, list[str]]:
    """The problems found in each layer the case checks; a layer passed when its list is empty.
    `booked`: the bookings the conversation made."""
    words = " ".join(case["turns"]).lower()
    checks = {"routing": lambda want: _routing(want, calls),
              "extraction": lambda want: _extraction(want, calls, words),
              "downstream": lambda want: _downstream(want, turn, booked)}
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


def main() -> None:
    """Plays each case, grades it layer by layer and prints the scores."""
    parser = argparse.ArgumentParser(description="Run the booking evals against the real models.")
    parser.add_argument("--label", default="unlabelled", help="goes into each conversation's thread id, to find the run in LangSmith")
    parser.add_argument("--models", default="opensource", choices=["opensource", "claude"])
    args = parser.parse_args()
    folder = tempfile.mkdtemp(prefix="spadesk-booking-eval-")
    os.environ |= {"SPADESK_MODELS": args.models, "SPADESK_MODEL_FALLBACK": "",
                   "SPADESK_HANDOFFS": f"{folder}/handoffs.jsonl", "SPADESK_FEEDBACK": f"{folder}/feedback.jsonl"}
    print(f"Pebby: {get_profile('booking-agent').primary}    Booking agent: {get_profile('booking-team').primary}")
    cases = json.loads(CASES.read_text())
    if problems := stale(cases):  # stop before spending anything
        raise SystemExit("Update these cases first:\n  " + "\n  ".join(problems))
    graded, seconds, spent = [], [], []
    _counting(spent)
    calls = Counter()
    tokens = 0
    for case in cases:
        spent.clear()
        desk = _desk(case.get("seed", []), "408 555 0101")
        before = len(desk.upcoming("408 555 0101"))
        thread = f"eval-{args.label}-{case['id']}-{int(time.time())}"
        started, turn = time.perf_counter(), None
        for said in case["turns"]:
            turn = respond(thread, said)
        seconds.append(time.perf_counter() - started)
        graded.append(grade(case, _calls(thread), turn, len(desk.upcoming("408 555 0101")) - before))
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
