"""Pebby's feedback tool: a small model reads what a customer says about their visit.

RESPONSIBLE FOR: turning a customer's feedback, in their own words, into a
record the spa can count and act on: its sentiment, its topic, a one-line
summary, and whether it is a complaint someone should follow up.

It is an LLM because feedback is language: "Priya was lovely but the room
was freezing" is mixed, about the facilities, and worth a call. It uses a
small, fast model (the low tier): reading one message into a fixed shape
is a simple job. It never talks to the customer.

    1. read    the model reads the words into a Reading
    2. record  one line in SPADESK_FEEDBACK (staff.py), with who said it
    3. follow  a complaint also goes to the staff (staff.py), so someone calls

The Master (Pebby) sends feedback here with the record_feedback tool, and
tells the customer what the answer says. If the model cannot be reached,
the words are still recorded, unread, and handed to the staff: feedback is
never lost.
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ...config import get_settings
from ...models import ModelError, structured
from ..rules import READER_MODEL
from ..staff import append, record_handoff, stamp

log = logging.getLogger("spadesk.agent")


READ = """You read feedback a customer gave a spa, in their own words, and record it in the Reading shape. \
Do not add anything they did not say."""


class Reading(BaseModel):
    """One customer's feedback, as the small model reads it."""

    sentiment: Literal["positive", "neutral", "negative", "mixed"] = Field(
        description="How the customer feels overall.")
    topic: Literal["treatment", "staff", "booking", "facilities", "price", "other"] = Field(
        description="What the feedback is mostly about.")
    summary: str = Field(description="One sentence for the staff, e.g. 'Loved the massage; the room was cold.'")
    complaint: bool = Field(description="True when something went wrong that a member of staff should follow "
                                        "up with the customer.")


def record_feedback(name: str, phone: str, words: str) -> dict:
    """Read, record and (for a complaint) hand on this customer's feedback.
    Returns what the Master is told."""
    try:
        reading = structured(READER_MODEL, [SystemMessage(READ), HumanMessage(words)], Reading).reply
    except ModelError as error:
        log.warning("feedback could not be read: %s", error)
        reading = None
    entry = {"at": stamp(), "name": name, "phone": phone, "words": words.strip(),
             **(reading.model_dump() if reading else {"unread": True})}
    append(get_settings().feedback_path, entry)
    if reading and not reading.complaint:
        return {"recorded": True, "next": "Thank the customer for their feedback."}
    record_handoff(name, phone, f"Feedback to follow up: {reading.summary if reading else words.strip()}")
    return {"recorded": True, "handed_off": True,
            "next": "Thank the customer, say you are sorry if something went wrong, and tell them our staff "
                    "will call them about it."}
