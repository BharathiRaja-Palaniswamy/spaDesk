"""Handing a conversation to a person: the request is written down for the staff.

The agent hands off when it should not or cannot go on alone: the customer
asks for a person, has a complaint or a health question the guide does not
settle, wants something the tools cannot do (a class, a package, a group),
or the calendar keeps failing. Each handoff is one line appended to a JSON
Lines file (SPADESK_HANDOFFS), with who to call back and why, for the staff
to read and act on.

The conversation never just stops: whenever something goes wrong (a model or
the search cannot be reached, a turn takes too long, an unexpected error),
the customer hears HANDOVER and the request is written down here.
"""

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from ..config import get_settings

log = logging.getLogger("spadesk.agent")

# What the customer is told whenever their request goes to the staff.
HANDOVER = "I need to hand this request over to our staff, and they will reach out to you shortly."


def record_handoff(name: str, phone: str, reason: str) -> dict:
    """Write down a request for the staff to call this customer back, and return it."""
    settings = get_settings()
    entry = {"at": datetime.now(ZoneInfo(settings.timezone)).isoformat(timespec="seconds"),
             "name": name, "phone": phone, "reason": reason.strip()}
    settings.handoffs_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.handoffs_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")
    log.info("handed off to staff: %s", reason)
    return entry
