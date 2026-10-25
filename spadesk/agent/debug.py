"""Debug mode: what happened between the agents in one turn, step by step.

Each part of the team notes what it did as it does it (`note`): who acted,
whom it passed to, one line saying what, and the data behind it. For
example:

    Pebby    -> booking    intent new · Lip Wax · day "tomorrow" · time "afternoon"
    Collect  -> The yes    Lip Wax · Monday 26 October · after 12:00 · nothing missing
    The yes  -> Customer   "Just to check before I look for times: ..."

`steps()` collects one turn's notes, in order, for the chat to show under
Pebby's reply, as a flow (Customer -> Pebby -> booking -> ...) and a list.
Each actor has a kind, so the chat can tell an AI from code at a glance.

Who sees them: everyone using the chat, when SPADESK_DEBUG is on, and no
one when it is off (agent/routes.py). It is for testing: keep it off on a
site customers use. Notes are kept for the turn being run, and nowhere
else; the full record of every call is LangSmith's (spadesk/tracing.py).
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# What each actor is: "ai" (Pebby's model), "small-ai" (a small, fast model),
# "tool" (a tool Pebby calls), "person" (the customer); anything else is "code".
KINDS = {"Customer": "person", "Pebby": "ai", "Identify": "small-ai", "Feedback agent": "small-ai",
         "search_guide": "tool", "booking": "tool", "record_feedback": "tool", "hand_off": "tool"}

_steps: ContextVar[list[dict] | None] = ContextVar("debug_steps", default=None)


def kind(actor: str) -> str:
    """'ai', 'small-ai', 'tool', 'person' or 'code'."""
    return KINDS.get(actor, "code")


def note(who: str, to: str, what: str, data: Any = None) -> None:
    """Note one step of the turn being run: `who` passed to `to`, `what` in one
    line, and the data behind it (nothing happens outside `steps`)."""
    if (steps := _steps.get()) is not None:
        # Through JSON and back, so later changes to `data` do not change the note.
        steps.append({"who": who, "to": to, "kinds": [kind(who), kind(to)], "what": what,
                      "data": json.loads(json.dumps(data, default=str))})


def brief(fields: dict[str, Any]) -> str:
    """A request's fields in one line: 'intent new · Lip Wax · day "tomorrow"'."""
    parts = []
    for key, value in fields.items():
        if value in ("", None, [], {}):
            continue
        if isinstance(value, list):
            parts.append(", ".join(map(str, value)))
        elif key.endswith("_said"):
            parts.append(f'{key.removesuffix("_said")} "{value}"')
        else:
            parts.append(f"{key} {value}")
    return " · ".join(parts)


def outcome(result: Any) -> str:
    """A tool's or an agent's result in one line: 'plan_ready: Lip Wax 13:00 with Hannah Lee'."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if "error" in result:
        return f"error: {result['error']}"
    if "plan" in result and isinstance(result["plan"], dict):
        items = [f"{i['service']} {i['start'][11:16]} with {i['therapist']}" for i in result["plan"]["items"]]
        return f"{result.get('status', 'plan')}: " + ", then ".join(items)
    if booked := result.get("booked"):
        return "booked " + ", ".join(b["reference"] for b in booked)
    if cancelled := result.get("cancelled"):
        done = cancelled if isinstance(cancelled, list) else [cancelled]
        return "cancelled " + ", ".join(c["reference"] if isinstance(c, dict) else c for c in done)
    if "card" in result:
        return f"card {result['card']} filed" + (" · asked for more" if result.get("asked_for_more") else "")
    if "not_booked" in result:
        return "not booked: " + " ".join(result["not_booked"])
    parts = [result.get("status", "")]
    parts += [f"missing {', '.join(result['missing'])}"] if result.get("missing") else []
    parts += [f"{len(result['options'])} options"] if result.get("options") else []
    parts += [f"{len(result['bookings'])} bookings"] if "bookings" in result else []
    parts += [result["reason"]] if result.get("reason") else []
    parts += [f"{len(result['excerpts'])} excerpts from the guide"] if "excerpts" in result else []
    parts += ["recorded"] if result.get("recorded") else []
    parts += ["handed to the staff"] if result.get("handed_off") else []
    return " · ".join(p for p in parts if p) or "done"


@contextmanager
def steps() -> Iterator[list[dict]]:
    """Collect the notes of the code run inside the block, in order."""
    collected: list[dict] = []
    token = _steps.set(collected)
    try:
        yield collected
    finally:
        _steps.reset(token)
