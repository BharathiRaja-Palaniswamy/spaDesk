"""The booking agent's tools, made for one customer.

    search_guide      the business document: what the spa offers, prices, policies
    find_free_times   free times for a service on a day       (booking/tools.py)
    book_appointment  books one: waits for the customer's yes (confirm.py)
    cancel_booking    cancels one: waits for the customer's yes
    my_bookings       this customer's upcoming bookings
    hand_off          asks the staff to call the customer back (handoffs.py)

The customer's name and phone number come from the conversation (identify.py),
never from the model: the tools are made for this customer, so the model
cannot book, list or cancel under anyone else's number.
"""

import logging
from typing import Any

from ..booking.appointments import get_appointments
from ..booking.tools import (
    book_appointment,
    booking_tools,
    cancel_booking,
    confirmation,
    guarded,
)
from ..lc.retriever import get_retriever
from ..models import Tool
from ..rag.errors import ModelUnavailable
from .handoffs import record_handoff

log = logging.getLogger("spadesk.agent")

# The tools that change a booking: the graph asks the customer before running them.
NEEDS_YES = {"book_appointment", "cancel_booking"}


def search_guide(args: dict[str, Any]) -> dict[str, Any]:
    """The parts of the business document that best answer the question, or
    an error the model can read when the search cannot be reached."""
    try:
        documents = get_retriever().invoke(args["question"])
    except ModelUnavailable as error:
        log.warning("guide search unavailable: %s", error)
        return {"error": "The guide can't be searched right now."}
    return {"excerpts": [{"sections": doc.metadata["sections"], "text": doc.page_content} for doc in documents]}


def agent_tools(name: str, phone: str) -> list[Tool]:
    """Every tool the agent has, for the customer with this name and number."""
    finder, booker, _ = booking_tools()  # find_free_times, and book_appointment's description of its inputs
    booking_inputs = {key: value for key, value in booker.parameters["properties"].items()
                      if key not in ("name", "phone")}
    return [
        Tool("search_guide",
             "Searches the spa's guide (services, prices, team, hours, policies, suitability, offers). Use it "
             "for any fact about the spa before you state it.",
             {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
             search_guide),
        finder,
        Tool("book_appointment",
             "Books a free time found with find_free_times, for this customer. Call it as soon as the customer "
             "has chosen the service, therapist, day and time: the system then asks them to confirm, so do "
             "not ask yourself.",
             {"type": "object", "properties": booking_inputs, "required": ["service", "therapist", "start"]},
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
