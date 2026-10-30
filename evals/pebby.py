"""Plays one Pebby case, a conversation, into the real agent (`observe`), and
grades what happened against its label (`grade`).

Each customer message in `turns` is sent to Pebby in order, on the case's
`channel` ("chat", or "voice" with the `caller`'s number), and the
conversation is graded in four layers, each against its part of the label:

  routing     did Pebby take the right path: the right tool (booking,
              search_guide, record_feedback, hand_off) and, for a booking, the
              right intent? `not_tools` must not have been used. A routing
              failure is a wrong path.
  extraction  are the details Pebby sent the customer's own: the services
              (exactly these, or only ones the customer named), a package,
              their words for the day and time, a therapist? A detail the
              customer did not give must not be sent (it is asked for instead).
  outcome     did the conversation end as it should: asking for the customer's
              yes or not, as many bookings made and removed (anyone's) as
              meant, the staff handed what is beyond one booking, one feedback
              card per piece of feedback.
  answer      what the last reply says (checks.py), and the sections the
              guide search found.

A turn that raises an error is handled as the app handles it, with a handover
to the staff, and the case fails with the error under "errors".

Reading words into dates and times, and booking itself, are code that tests/
covers for free; this measures what a model can get wrong.

The calendar is an in-memory one (the tests' stand-in), empty except for the
case's `seed` bookings, with the clock at the real time now, so "Friday"
means the coming Friday. Nothing reaches Google Calendar; handoffs and
feedback are written to a temporary folder (run.py), and counted per case.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import ToolMessage

from spadesk.agent import booking, executor, hand_over, pebby, respond, yes
from spadesk.agent.details import named_services
from spadesk.agent.feedback import cards as feedback_cards
from spadesk.agent.flow import get_graph
from spadesk.config import get_settings
from tests.standin_calendar import standin_desk

from .checks import grade_answer
from .golden import Case

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CUSTOMER = ("Asha Rao", "408 555 0101")  # who a seed booking belongs to, unless it names another


def _desk(seed: list[dict]):
    """An in-memory calendar at the real time now, with the case's bookings, used by
    every part of the agent."""
    zone = ZoneInfo(get_settings().timezone)
    desk = standin_desk(now=lambda: datetime.now(zone))
    for seeded in seed:
        day = _next(seeded["weekday"], desk.now())
        hour, minute = map(int, seeded["time"].split(":"))
        desk.book(seeded["service"], seeded["therapist"], datetime(day.year, day.month, day.day, hour, minute),
                  seeded.get("customer", CUSTOMER[0]), seeded.get("phone", CUSTOMER[1]))
    for module in (pebby, yes, executor, booking):
        module.get_appointments = lambda: desk
    return desk


def _next(weekday: str, now: datetime):
    """The coming `weekday` (tomorrow at the earliest)."""
    ahead = (DAYS.index(weekday) - now.weekday() - 1) % 7 + 1
    return (now + timedelta(days=ahead)).date()


def _ids(desk) -> set[str]:
    """Every customer's upcoming bookings on the calendar, by id."""
    now = desk.now()
    return {b.id for b in desk.calendar.bookings(now - timedelta(days=1), now + timedelta(days=60))}


def _lines(path: Path) -> int:
    """How many records a JSON-lines file holds (0 before the first is written)."""
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def _messages(thread: str) -> list:
    return get_graph().get_state({"configurable": {"thread_id": thread}}).values["messages"]


def routing(want: dict, calls: list[dict]) -> list[str]:
    """The right tool and intent; none of the wrong ones."""
    used, problems = [call["name"] for call in calls], []
    if (tool := want.get("tool")) and tool not in used:
        problems.append(f"did not use {tool} (used {used or 'no tools'})")
    problems += [f"used {name}" for name in want.get("not_tools", []) if name in used]
    intents = [call["args"].get("intent") for call in calls if call["name"] == "booking"]
    if "intent" in want and (not intents or intents[-1] not in want["intent"]):
        problems.append(f"intent {intents[-1] if intents else None!r}, expected one of {want['intent']}")
    return problems


def extraction(want: dict, calls: list[dict], words: str) -> list[str]:
    """The details sent are the customer's own; nothing they did not give."""
    requests = [call["args"] for call in calls if call["name"] == "booking"]
    problems = []
    if want.get("only_named_services"):  # every booking request, not just the last
        problems += [f"chose {unnamed} for them" for got in requests
                     if (unnamed := named_services(got.get("services", []), got.get("package", ""), words)[1])]
    for field in ("day_said", "time_said"):  # missing info is asked for, never filled in
        if want.get(f"no_{field}"):
            problems += [f"{field} {got[field]!r}, but the customer said none" for got in requests if got.get(field)]
    asks = [key for key in ("services", "package", "day_said_has", "time_said_has", "therapist") if key in want]
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
        if (part := want.get(f"{field}_has")) and part not in (got.get(field) or "").lower():
            problems.append(f"{field} {got.get(field)!r}, expected it to mention {part!r}")
    if "therapist" in want and got.get("therapist") != want["therapist"]:
        problems.append(f"therapist {got.get('therapist')!r}, expected {want['therapist']!r}")
    return problems


def outcome(want: dict, confirming: bool, counts: dict[str, int]) -> list[str]:
    """The conversation ended as it should. `counts`: the bookings made and removed
    (anyone's), the staff handoffs and the feedback cards the conversation made."""
    problems = []
    if "confirming" in want and confirming != want["confirming"]:
        problems.append(f"confirming is {confirming}, expected {want['confirming']}")
    for key, word in (("made", "bookings made"), ("removed", "bookings removed"), ("cards", "feedback cards")):
        if key in want and counts[key] != want[key]:
            problems.append(f"{counts[key]} {word}, expected {want[key]}")
    if "handed_off" in want and bool(counts["handed_off"]) != want["handed_off"]:
        problems.append(f"{'handed' if counts['handed_off'] else 'not handed'} to the staff, expected the opposite")
    return problems


def _found_sections(messages: list) -> list[str]:
    """The guide sections the search handed Pebby in this conversation."""
    found = []
    for message in messages:
        if isinstance(message, ToolMessage):
            result = json.loads(message.content) if message.content.startswith("{") else {}
            for excerpt in result.get("excerpts", []):
                sections = excerpt["sections"]  # a list, or one string with a heading per line
                found += sections if isinstance(sections, list) else sections.split("\n")
    return list(dict.fromkeys(found))


def observe(case: Case, thread: str, handoffs: Path) -> dict[str, Any]:
    """Plays `case` into Pebby and records what happened, as plain data (the
    run's outputs in LangSmith): the conversation, the tool requests, what the
    calendar, the staff and the feedback cards got, and the last reply."""
    desk = _desk(case.get("seed", []))
    before_ids, before_handoffs, before_cards = _ids(desk), _lines(handoffs), len(feedback_cards())
    channel, caller = case.get("channel", "chat"), case.get("caller", "")
    said, turn, errors = [], None, []
    for message in case["turns"]:
        try:
            turn = respond(thread, message, channel, caller)
        except Exception as error:  # as the app does (agent/routes.py): any failure ends in a handover
            errors.append(f"{type(error).__name__}: {error}"[:300])
            turn = hand_over(thread, "The assistant hit an error.", message, caller)
        said += [f"Customer: {message}", f"Pebby: {turn.answer if turn else '(no reply)'}"]
    messages = _messages(thread)
    after_ids = _ids(desk)
    items = [f"{item.name} {item.detail}" for item in (turn.items if turn else [])]
    return {"conversation": said, "errors": errors,
            "calls": [{"name": c["name"], "args": c["args"]} for m in messages for c in (getattr(m, "tool_calls", None) or [])],
            "confirming": bool(turn and turn.confirming),
            "counts": {"made": len(after_ids - before_ids), "removed": len(before_ids - after_ids),
                       "handed_off": _lines(handoffs) - before_handoffs, "cards": len(feedback_cards()) - before_cards},
            "reply": {"question": "\n".join(said[:-1]), "text": " ".join([turn.answer if turn else "", *items]),
                      "sections": _found_sections(messages)}}


def grade(case: Case, seen: dict[str, Any]) -> dict[str, list[str]]:
    """Each layer's problems, from what `observe` recorded, against the case's label;
    a layer passed when its list is empty. Only the layers the label has."""
    want, words = case["expected"], " ".join(case["turns"]).lower()
    layers = {"routing": lambda w: routing(w, seen["calls"]),
              "extraction": lambda w: extraction(w, seen["calls"], words),
              "outcome": lambda w: outcome(w, seen["confirming"], seen["counts"]),
              "answer": lambda w: grade_answer(seen["reply"], w)}
    graded = {layer: check(want[layer]) for layer, check in layers.items() if layer in want}
    # A turn that failed is a failure whatever the layers say: the customer was handed to the staff.
    return graded | ({"errors": seen["errors"]} if seen.get("errors") else {})
