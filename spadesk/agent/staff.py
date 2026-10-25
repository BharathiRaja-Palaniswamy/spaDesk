"""What is written down for the staff: handoffs, and the files they go in.

A handoff is a request for a person to call a customer back. The agents
hand off when they should not or cannot go on alone (the rules are in
rules.py): the customer asks for a person, has a complaint or a health
question the guide does not settle, wants something the tools cannot do,
cancels part of a package, or booking keeps failing. The conversation never
just stops: whenever something goes wrong (a model or the search cannot be
reached, a turn takes too long, an unexpected error), the customer hears
HANDOVER and the request is written down here.

Handoffs (SPADESK_HANDOFFS) and feedback (feedback.py, SPADESK_FEEDBACK) are
kept the same way: each is one JSON object per line, appended to its file
and stamped with the spa's local time. Appending never rewrites what is
there, and a person (or a dashboard, later) can read the file line by line.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import get_settings

log = logging.getLogger("spadesk.agent")

# What the customer is told whenever their request goes to the staff.
HANDOVER = "I need to hand this request over to our staff, and they will reach out to you shortly."


def stamp() -> str:
    """The spa's local time now, e.g. '2026-10-25T14:10:00-07:00'."""
    return datetime.now(ZoneInfo(get_settings().timezone)).isoformat(timespec="seconds")


def append(path: Path, entry: dict) -> None:
    """Add `entry` as one line at the end of `path` (created, with its folder, if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def record_handoff(name: str, phone: str, reason: str) -> dict:
    """Write down a request for the staff to call this customer back, and return it."""
    entry = {"at": stamp(), "name": name, "phone": phone, "reason": reason.strip()}
    append(get_settings().handoffs_path, entry)
    log.info("handed off to staff: %s", reason)
    return entry
