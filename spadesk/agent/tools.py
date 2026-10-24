"""The booking agent's tools, made for one customer.

    search_guide      the business document: what the spa offers, prices, policies
    find_free_times   free times for a service on a day       (booking/actions.py)
    book_appointment  books one: waits for the customer's yes (confirm.py)
    cancel_booking    cancels one: waits for the customer's yes
    my_bookings       this customer's upcoming bookings
    hand_off          asks the staff to call the customer back (handoffs.py)

The customer's name and phone number come from the conversation (identify.py),
never from the model: the tools are made for this customer, so the model
cannot book, list or cancel under anyone else's number.

Each tool needs a permission (NEEDS), and the model is only offered the
tools its role allows (spadesk/access/permissions.py). A customer's
conversation never holds the owner's tools (the reports), so no wording of
a question can make the model use them.

The lists of services and team members in the tools' descriptions are read
from the document each time the tools are made, so they never go stale.
"""

import logging
from typing import Any

from ..access.permissions import Role, allowed
from ..booking.actions import (
    book_appointment,
    cancel_booking,
    confirmation,
    find_free_times,
    guarded,
)
from ..booking.appointments import get_appointments
from ..booking.schedule import read_schedule
from ..business.store import get_knowledge
from ..lc.retriever import get_retriever
from ..models import Tool
from ..rag.errors import ModelUnavailable
from .handoffs import record_handoff

log = logging.getLogger("spadesk.agent")

# The tools that change a booking: the graph asks the customer before running them.
NEEDS_YES = {"book_appointment", "cancel_booking"}
# Tool -> the permission it needs. Every tool must be listed: one that is not is never offered.
NEEDS = {"search_guide": "search_guide", "find_free_times": "book", "book_appointment": "book",
         "cancel_booking": "book", "my_bookings": "book", "hand_off": "contact_staff"}


def search_guide(args: dict[str, Any]) -> dict[str, Any]:
    """The parts of the business document that best answer the question, or
    an error the model can read when the search cannot be reached."""
    try:
        documents = get_retriever().invoke(args["question"])
    except ModelUnavailable as error:
        log.warning("guide search unavailable: %s", error)
        return {"error": "The guide can't be searched right now."}
    return {"excerpts": [{"sections": doc.metadata["sections"], "text": doc.page_content} for doc in documents]}


def agent_tools(name: str, phone: str, role: Role = Role.CUSTOMER) -> list[Tool]:
    """The tools `role` may use, made for the customer with this name and number."""
    schedule = read_schedule(get_knowledge())
    services = {"type": "string", "enum": list(schedule.services)}
    therapists = {"type": "string", "enum": list(schedule.therapists)}
    minutes = {"type": "integer", "description": "The length, for a service that comes in more than one "
                                                 "(e.g. a 90-minute massage). Leave out for the standard length."}
    tools = [
        Tool("search_guide",
             "Searches the spa's guide (services, prices, team, hours, policies, suitability, offers). Use it "
             "for any fact about the spa before you state it.",
             {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
             search_guide),
        Tool("find_free_times",
             "Free start times for a service on one day, by therapist, in the spa's time zone. Give a "
             "therapist only if the customer asked for one. Use this before offering any time.",
             {"type": "object", "properties": {"service": services, "therapist": therapists, "minutes": minutes,
                                               "date": {"type": "string", "description": "The date, YYYY-MM-DD."}},
              "required": ["service", "date"]},
             guarded(find_free_times)),
        Tool("book_appointment",
             "Books a free time found with find_free_times, for this customer. Call it as soon as the customer "
             "has chosen the service, therapist, day and time: the system then asks them to confirm, so do "
             "not ask yourself.",
             {"type": "object", "properties": {
                 "service": services, "therapist": therapists, "minutes": minutes,
                 "start": {"type": "string", "description": "The start, YYYY-MM-DDTHH:MM, as find_free_times gave it."}},
              "required": ["service", "therapist", "start"]},
             guarded(lambda args: book_appointment({**args, "name": name, "phone": phone}))),
        Tool("cancel_booking",
             "Cancels one of this customer's bookings, by its reference (from my_bookings). Call it as soon as "
             "the customer says which booking: the system then asks them to confirm.",
             {"type": "object", "properties": {"reference": {"type": "string"}}, "required": ["reference"]},
             guarded(lambda args: cancel_booking({**args, "phone": phone}))),
        Tool("my_bookings", "This customer's upcoming bookings, with their references.",
             {"type": "object", "properties": {}},
             guarded(lambda args: {"bookings": [confirmation(b) for b in get_appointments().upcoming(phone)]})),
        Tool("hand_off",
             "Asks the spa's staff to call this customer back. Use it when they ask for a person, have a "
             "complaint, have a health question the guide does not settle, want something these tools cannot "
             "do (a class, a package, a group), or booking keeps failing. Say why, in one sentence.",
             {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
             lambda args: {"handed_off": True, **record_handoff(name, phone, args["reason"])}),
    ]
    return [tool for tool in tools if allowed(role, NEEDS.get(tool.name, ""))]
