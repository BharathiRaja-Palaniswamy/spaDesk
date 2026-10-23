"""The booking tools a model can ask for, written once for any model or assistant.

    find_free_times   customer and owner   free times for a service on a day
    book_appointment  customer and owner   books one of those times
    cancel_booking    customer and owner   cancels, with the reference and phone number
    day_schedule      owner only           the day's bookings, with customers' details

Each is a neutral Tool (spadesk/models/tools.py), so the same tools serve
the chat now, and a voice assistant or an MCP server later. Every request
goes through the booking rules (appointments.py): a tool can only do what
those rules allow, whatever the model asks.

The customer's tools never show other customers' names or phone numbers;
only the owner's day_schedule does. A broken rule comes back as
{"error": "..."}, written for the customer, so the model can pass it on or
try again.

The lists of services and team members in the tools' descriptions are read
from the document each time the tools are made, so they never go stale.
"""

import logging
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from ..business.store import get_knowledge
from ..models.tools import Tool
from .appointments import get_appointments
from .calendars import Booking
from .errors import BookingError, CalendarUnavailable
from .schedule import read_schedule

log = logging.getLogger("spadesk.booking")
UNAVAILABLE = "Booking is unavailable right now. Please call the spa to book or cancel."


def _guarded(run: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
    """`run`, with a broken booking rule, or a calendar that cannot be
    reached, returned as {"error": ...} instead of raised."""
    def guarded(args: dict[str, Any]) -> Any:
        try:
            return run(args)
        except BookingError as error:
            return {"error": str(error)}
        except CalendarUnavailable as error:
            log.warning("calendar unavailable: %s", error)  # the reason, for whoever reads the server log
            return {"error": UNAVAILABLE}
        except ValueError:  # a date or time the model wrote in the wrong format
            return {"error": "Dates are YYYY-MM-DD and times YYYY-MM-DDTHH:MM, in the spa's time zone."}
    return guarded


def _when(moment: datetime) -> str:
    """A time as the tools write it: '2026-10-24T15:00'."""
    return moment.strftime("%Y-%m-%dT%H:%M")


def _confirmation(booking: Booking) -> dict[str, Any]:
    """What the customer is told about their own booking."""
    return {"reference": booking.id, "service": booking.service, "therapist": booking.therapist,
            "start": _when(booking.start), "end": _when(booking.end), "name": booking.name}


def find_free_times(args: dict[str, Any]) -> dict[str, Any]:
    """Free start times on a day, by therapist."""
    slots = get_appointments().free_slots(args["service"], date.fromisoformat(args["date"]),
                                          args.get("therapist"), args.get("minutes"))
    free: dict[str, list[str]] = {}
    for slot in slots:
        free.setdefault(slot.therapist, []).append(slot.start.strftime("%H:%M"))
    return {"service": args["service"], "date": args["date"], "free": free} if free else \
        {"service": args["service"], "date": args["date"], "free": {}, "note": "No free times that day."}


def book_appointment(args: dict[str, Any]) -> dict[str, Any]:
    """Book one free time."""
    booking = get_appointments().book(args["service"], args["therapist"], datetime.fromisoformat(args["start"]),
                                      args["name"], args["phone"], args.get("minutes"))
    return {"booked": _confirmation(booking)}


def cancel_booking(args: dict[str, Any]) -> dict[str, Any]:
    """Cancel a booking, given its reference and phone number."""
    return {"cancelled": _confirmation(get_appointments().cancel(args["reference"], args["phone"]))}


def day_schedule(args: dict[str, Any]) -> dict[str, Any]:
    """The day's bookings, with the customers' details."""
    bookings = get_appointments().day_bookings(date.fromisoformat(args["date"]), args.get("therapist"))
    return {"date": args["date"], "bookings": [{**_confirmation(b), "phone": b.phone} for b in bookings]}


def booking_tools(owner: bool = False) -> list[Tool]:
    """The customer's booking tools; with `owner`, day_schedule as well."""
    schedule = read_schedule(get_knowledge())
    services = {"type": "string", "enum": list(schedule.services)}
    therapists = {"type": "string", "enum": list(schedule.therapists)}
    day = {"type": "string", "description": "The date, YYYY-MM-DD."}
    minutes = {"type": "integer", "description": "The length, for a service that comes in more than one "
                                                 "(e.g. a 90-minute massage). Leave out for the standard length."}
    tools = [
        Tool("find_free_times",
             "Free start times for a service on one day, by therapist, in the spa's time zone. Give a "
             "therapist only if the customer asked for one. Use this before offering any time.",
             {"type": "object", "properties": {"service": services, "date": day, "therapist": therapists,
                                               "minutes": minutes},
              "required": ["service", "date"]},
             _guarded(find_free_times)),
        Tool("book_appointment",
             "Books a free time found with find_free_times. Only call this after the customer has seen "
             "the service, therapist, day and time, and has said yes, and has given a name and a phone "
             "number. Returns a booking reference to give the customer.",
             {"type": "object", "properties": {
                 "service": services, "therapist": therapists, "minutes": minutes,
                 "start": {"type": "string", "description": "The start, YYYY-MM-DDTHH:MM, as find_free_times gave it."},
                 "name": {"type": "string"}, "phone": {"type": "string"}},
              "required": ["service", "therapist", "start", "name", "phone"]},
             _guarded(book_appointment)),
        Tool("cancel_booking",
             "Cancels a booking. Needs its reference (e.g. BK-7F3A2C) and the phone number it "
             "was booked with. Only call this after the customer has said they want to cancel.",
             {"type": "object", "properties": {"reference": {"type": "string"}, "phone": {"type": "string"}},
              "required": ["reference", "phone"]},
             _guarded(cancel_booking)),
    ]
    if owner:
        tools.append(Tool(
            "day_schedule",
            "Every booking on one day, with the customers' names and phone numbers; one therapist's or everyone's.",
            {"type": "object", "properties": {"date": day, "therapist": therapists}, "required": ["date"]},
            _guarded(day_schedule)))
    return tools
