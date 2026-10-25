"""Pebby's tools, made for one customer. Pebby is the front agent: the only
one who talks to the customer. It decides what each message is about and
uses the tool for it:

    search_guide     a question about the spa: services, prices, policies
    booking          anything about bookings. The details are checked against what
                     the customer said (collect.py); what is missing comes back to
                     ask for; complete details are read back to the customer first;
                     then the Booking agent (booking/agent.py) answers, and a plan or
                     a cancellation waits for the customer's yes (booking/yes.py)
    record_feedback  a small model reads and records it; a complaint also goes to the staff (feedback.py)
    hand_off         asks the staff to call the customer back (staff.py)

The customer's name and phone number come from the conversation (identify.py),
never from the model: the tools are made for this customer, so the model
cannot book, list or cancel under anyone else's number.

Each tool needs a permission (NEEDS), and the model is only offered the
tools its role allows (spadesk/access/permissions.py). A customer's
conversation never holds the owner's tools (the reports), so no wording of
a question can make the model use them.
"""

import logging
from typing import Any

from ...access.permissions import Role, allowed
from ...booking.actions import guarded
from ...booking.appointments import get_appointments
from ...booking.schedule import read_schedule
from ...business.store import get_knowledge
from ...lc.retriever import get_retriever
from ...models import Tool
from ...rag.errors import ModelUnavailable
from ..booking.agent import describe, handle, missing_result
from ..booking.collect import collect
from ..debug import brief, note
from ..staff import record_handoff
from .feedback import record_feedback

log = logging.getLogger("spadesk.agent")

# Requests the graph runs itself (nodes._hold): they may end in a question for the customer's yes.
HELD = {"booking"}
# Tool -> the permission it needs. Every tool must be listed: one that is not is never offered.
NEEDS = {"search_guide": "search_guide", "booking": "book", "record_feedback": "contact_staff",
         "hand_off": "contact_staff"}


def search_guide(args: dict[str, Any]) -> dict[str, Any]:
    """The parts of the business document that best answer the question, or
    an error the model can read when the search cannot be reached."""
    try:
        documents = get_retriever().invoke(args["question"])
    except ModelUnavailable as error:
        log.warning("guide search unavailable: %s", error)
        return {"error": "The guide can't be searched right now."}
    return {"excerpts": [{"sections": doc.metadata["sections"], "text": doc.page_content} for doc in documents]}


def booking_parameters() -> dict[str, Any]:
    """The booking tool's inputs, with the document's services and team. The day and
    the time are the customer's own words; code works out the date and times."""
    schedule = read_schedule(get_knowledge())
    return {"type": "object", "required": ["intent"], "properties": {
        "intent": {"type": "string", "enum": ["new", "change", "cancel", "my_bookings", "availability"]},
        "services": {"type": "array", "items": {"type": "string", "enum": list(schedule.services)},
                     "description": "Every service, by its exact name; for a package, every service it includes."},
        "package": {"type": "string", "description": "The package's name, if they asked for one."},
        "add_ons": {"type": "array", "items": {"type": "string", "enum": list(schedule.add_ons)},
                    "description": "Extras they asked for with it (e.g. a scalp massage); to add one to an "
                                   "existing booking, a change of that booking."},
        "day_said": {"type": "string", "description": "The customer's own words for the day, as they wrote them "
                                                      "(e.g. 'Thursday', 'tomorrow', '30 October'); empty if "
                                                      "they have not said one."},
        "time_said": {"type": "string", "description": "The customer's own words for the time, as they wrote "
                                                       "them (e.g. 'after 3pm', 'morning', 'any time'); empty if "
                                                       "they have not said one."},
        "therapist": {"type": "string", "enum": list(schedule.therapists)},
        "references": {"type": "array", "items": {"type": "string"},
                       "description": "The bookings to change (one) or cancel (one or more), e.g. ['BK-63715A']."},
        "notes": {"type": "string", "description": "A wish no other field holds; usually empty."}}}


def booking(args: dict[str, Any], phone: str, words: str) -> dict[str, Any]:
    """The booking tool: what is missing, the details to read back, or the Booking agent's answer."""
    desk = get_appointments()
    upcoming = desk.upcoming(phone) if args.get("references") else []  # theirs only: no one else's can be named
    request, missing, unnamed = collect(args, words, desk.now().date(), upcoming)
    to = "Pebby" if missing else "The yes" if request.intent in ("new", "change") else "Booking agent"
    worked_out = brief(request.model_dump(exclude_defaults=True, exclude={"intent"}))
    note("Collect", to, f"{worked_out or 'nothing yet'} · " + (f"missing {', '.join(missing)}" if missing
                                                                else "nothing missing"),
         {"said": {key: args[key] for key in ("day_said", "time_said") if args.get(key)},
          "worked out": request.model_dump(exclude_defaults=True), "missing": missing,
          "not named by the customer": unnamed + ([args["therapist"]] if args.get("therapist") and
                                                   not request.therapist else [])
          + [r for r in args.get("references", []) if r not in request.references]})
    if missing:
        result = describe(missing_result(request, missing, phone))
        # No service they named: the ones chosen for them are offered as choices, never used.
        return result | ({"options": unnamed} if "services" in missing and unnamed else {})
    if request.intent in ("new", "change"):  # complete: the customer confirms the details first
        # The customer's own words for the day and time go with it, for the read-back.
        said = {"day": args.get("day_said", ""), "time": args.get("time_said", "")}
        return {"status": "confirm_details", "request": request.model_dump(exclude_defaults=True), "said": said}
    return describe(handle(request, phone))


def agent_tools(name: str, phone: str, role: Role = Role.CUSTOMER, words: str = "") -> list[Tool]:
    """The tools `role` may use, made for the customer with this name and number.
    `words`: what the customer has written so far (collect.customer_words)."""
    tools = [
        Tool("search_guide",
             "Searches the spa's guide (services, prices, team, hours, policies, suitability, offers, packages). "
             "Use it for any fact about the spa before you state it.",
             {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
             search_guide),
        Tool("booking",
             "Anything about bookings: a new booking, changing or cancelling one, their bookings, when someone "
             "is free. Call it once you have every detail from the customer (for a booking: the service they "
             "named, the day and the time), not for each one. The system then reads the details back to them "
             "and shows the times found for their yes; anything else it answers, tell them in your words.",
             booking_parameters(),
             guarded(lambda args: booking(args, phone, words))),
        Tool("record_feedback",
             "Records feedback about a visit: praise, a complaint, a suggestion. Not for questions or requests. "
             "Then tell the customer what it says to.",
             {"type": "object", "properties": {"words": {
                 "type": "string", "description": "The customer's feedback in their own words, all of it."}},
              "required": ["words"]},
             lambda args: record_feedback(name, phone, args["words"])),
        Tool("hand_off",
             "Asks the spa's staff to call this customer back. Use it when they ask for a person, have a "
             "health question the guide does not settle, want something you cannot do (a class, a group), "
             "or booking keeps failing (a complaint goes to record_feedback). Say why, in one sentence.",
             {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
             lambda args: {"handed_off": True, **record_handoff(name, phone, args["reason"])}),
    ]
    return [tool for tool in tools if allowed(role, NEEDS.get(tool.name, ""))]
