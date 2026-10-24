"""The booking engine's actions, in the shape a model's tool request takes.

    find_free_times   free times for a service on a day, by therapist
    book_appointment  books one of those times
    cancel_booking    cancels, given the reference and the phone number

Each takes the arguments of a model's request (a dict) and returns what the
model is told (something JSON can hold). Every action goes through the
booking rules (appointments.py), so it can only do what those rules allow,
whatever the model asks. A broken rule comes back as {"error": "..."},
written for the customer, so the model can pass it on or try again
(`guarded`).

How these are offered to a model, and to whom, is agent/tools.py.
"""

import logging
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from .appointments import get_appointments
from .calendars import Booking
from .errors import BookingError, CalendarUnavailable

log = logging.getLogger("spadesk.booking")
UNAVAILABLE = "Booking is unavailable right now. Please call the spa to book or cancel."


def guarded(run: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
    """`run`, with a broken booking rule, or a calendar that cannot be
    reached, returned as {"error": ...} instead of raised."""
    def wrapped(args: dict[str, Any]) -> Any:
        try:
            return run(args)
        except BookingError as error:
            return {"error": str(error)}
        except CalendarUnavailable as error:
            log.warning("calendar unavailable: %s", error)  # the reason, for whoever reads the server log
            return {"error": UNAVAILABLE}
        except ValueError:  # a date or time the model wrote in the wrong format
            return {"error": "Dates are YYYY-MM-DD and times YYYY-MM-DDTHH:MM, in the spa's time zone."}
    return wrapped


def _when(moment: datetime) -> str:
    """A time as the tools write it: '2026-10-24T15:00'."""
    return moment.strftime("%Y-%m-%dT%H:%M")


def confirmation(booking: Booking) -> dict[str, Any]:
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
    return {"booked": confirmation(booking)}


def cancel_booking(args: dict[str, Any]) -> dict[str, Any]:
    """Cancel a booking, given its reference and phone number."""
    return {"cancelled": confirmation(get_appointments().cancel(args["reference"], args["phone"]))}
