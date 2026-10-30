"""The Feedback agent: what a customer says about their visit.

RESPONSIBLE FOR: hearing a customer's feedback, in their own words, and
  1. reading it: how they feel, what it is about, a one-line summary, and
     whether something went wrong that the spa should follow up (a complaint)
  2. answering them: a short, caring reply, shown to the customer as Pebby's
     (one voice in the chat), with no second call to Pebby. When the
     feedback is too brief to act on (which service? which visit? what
     happened?), the reply asks for what is missing, once (FEEDBACK_QUESTIONS):
     the card is filed straight away, and their answer, passed back with the
     card's reference, is read with it and updates the same card
  3. filing it as a card for the owner (on /manage, the Feedback tab), open
     until the owner closes it, with a note if they like

It is an agent because feedback is language: "Priya was lovely but the room
was freezing" is mixed, about the facilities, and worth a call; and the
reply has to fit what was said. Pebby only recognises feedback and passes
it on (its record_feedback tool); everything after is done here.

The reply never promises what only the spa can decide (a refund, a free
visit) and never states a fact it was not given: for a complaint it says
sorry and that the team will look into it. If the model cannot be reached,
the card is still filed (marked unread) and the customer still thanked:
feedback is never lost.

Cards are kept in SPADESK_FEEDBACK as events, one JSON object per line
(staff.py appends them): a "card" when feedback arrives, a "closed" when the
owner closes it. Nothing is ever rewritten; `cards` reads the events back.
"""

import json
import logging
import secrets
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import get_settings
from ..models import ModelError, structured
from .debug import note
from .rules import FEEDBACK_MODEL, FEEDBACK_QUESTIONS
from .staff import append, stamp

log = logging.getLogger("spadesk.agent")

READ = """You are the feedback agent of a spa. A customer shared feedback about their visit, in their own \
words. Read it into the Reading shape, and write the reply they will see.

The reply: one to three warm, plain sentences, as the spa ("we"). Thank them. If something went wrong, say \
sorry for that part and that our team will look into it and may contact them. Never promise anything only \
the spa can decide (a refund, a discount, a free visit), and never state a fact they did not give you.

If the feedback is too brief for the spa to act on (it does not say which service, which visit, or what \
happened), set needs_more, and end the reply with one polite question asking for what is missing."""
# Said to the model instead when it may not ask again (FEEDBACK_QUESTIONS reached).
NO_MORE = "\n\nThey have already answered your question: do not ask anything; thank them."
# What the customer hears when the model cannot be reached: the card is still filed.
THANKS = "Thank you for telling us. I've passed this on to our team."
# The question added when the agent means to ask for more but its reply asks nothing:
# the customer must be able to see what to answer.
MORE = "Could you tell us which service and day it was, and what happened?"


class Reading(BaseModel):
    """One customer's feedback, as the Feedback agent reads it, and its reply."""

    sentiment: Literal["positive", "neutral", "negative", "mixed"] = Field(description="How they feel overall.")
    topic: Literal["treatment", "staff", "booking", "facilities", "price", "other"] = Field(
        description="What the feedback is mostly about.")
    summary: str = Field(description="One sentence for the owner, e.g. 'Loved the massage; the room was cold.'")
    complaint: bool = Field(description="True when something went wrong that the spa should follow up with "
                                        "the customer.")
    reply: str = Field(description="The reply the customer sees: one to three warm sentences.")
    needs_more: bool = Field(default=False, description="True when the feedback is too brief to act on, and "
                                                        "the reply asks for what is missing.")


def give_feedback(name: str, phone: str, words: str, card_id: str = "") -> dict[str, Any]:
    """Read the customer's feedback, file it as a card (or, with `card_id`, add their
    answer to the card it asked about), and return the reply for them."""
    earlier = next((c for c in cards() if c["id"] == card_id and c["phone"] == phone and c["status"] == "open"),
                   None) if card_id else None
    all_words = f"{earlier['words']}\n{words.strip()}" if earlier else words.strip()
    may_ask = (earlier or {}).get("questions", 0) < FEEDBACK_QUESTIONS
    try:
        reading = structured(FEEDBACK_MODEL, [SystemMessage(READ + ("" if may_ask else NO_MORE)),
                                              HumanMessage(all_words)], Reading).reply
    except ModelError as error:
        log.warning("feedback could not be read: %s", error)
        reading = None
    asks = bool(reading and reading.needs_more and may_ask)
    read = reading.model_dump(exclude={"reply", "needs_more"}) if reading else {"unread": True}
    questions = (earlier or {}).get("questions", 0) + asks
    card = {"id": earlier["id"] if earlier else "FB-" + secrets.token_hex(3).upper(), "words": all_words,
            "questions": questions, **read}
    event = {"event": "update", **card} if earlier else {"event": "card", "at": stamp(), "name": name,
                                                       "phone": phone, **card}
    append(get_settings().feedback_path, event)  # filed straight away, even while a question is open
    reply = reading.reply.strip() if reading and reading.reply.strip() else THANKS
    if asks and "?" not in reply:
        reply = f"{reply} {MORE}"
    note("Feedback agent", "Customer", reply, {k: v for k, v in event.items() if k != "event"})
    # `reply_to_customer` ends Pebby's turn with these words (flow.py tools).
    return {"card": card["id"], "complaint": bool(reading and reading.complaint), "asked_for_more": asks,
            "reply_to_customer": reply}


def _events() -> list[dict[str, Any]]:
    """Every event in the feedback file, oldest first."""
    path = get_settings().feedback_path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cards() -> list[dict[str, Any]]:
    """Every card, open ones first, newest first within each: what the owner sees."""
    found: dict[str, dict[str, Any]] = {}
    for n, event in enumerate(_events()):
        if event.get("event", "card") == "card":
            card_id = event.get("id") or f"FB-OLD{n}"  # a line written before cards had ids
            found[card_id] = {**{k: v for k, v in event.items() if k != "event"}, "id": card_id, "status": "open"}
        elif event["event"] == "update" and event["id"] in found:  # their answer to the agent's question
            found[event["id"]] |= {k: v for k, v in event.items() if k not in ("event", "id")}
            if "unread" not in event:
                found[event["id"]].pop("unread", None)
        elif event["event"] == "closed" and event["id"] in found:
            found[event["id"]] |= {"status": "closed", "closed_at": event["at"], "owner_note": event.get("note", "")}
    newest_first = sorted(found.values(), key=lambda card: card["at"], reverse=True)
    return [c for c in newest_first if c["status"] == "open"] + [c for c in newest_first if c["status"] != "open"]


def close(card_id: str, owner_note: str = "") -> dict[str, Any] | None:
    """Close an open card, with the owner's note; None when there is no such open card."""
    card = next((c for c in cards() if c["id"] == card_id and c["status"] == "open"), None)
    if card is None:
        return None
    append(get_settings().feedback_path, {"event": "closed", "id": card_id, "at": stamp(), "note": owner_note.strip()})
    return next(c for c in cards() if c["id"] == card_id)
