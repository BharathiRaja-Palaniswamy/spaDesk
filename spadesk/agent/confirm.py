"""The customer's yes, before a booking is made or cancelled.

This is the line between what the agent does alone and what needs the
customer: it may search and offer times on its own, but a booking is only
made, or cancelled, after the customer says yes to exactly what will happen.
The graph, not the model, enforces it: when the model asks to book or
cancel, the graph stops, asks the question written here, and runs the
request only if the customer's next message is a yes.

Before asking, the request is checked against the booking rules, so the
customer is never asked to confirm something that would then fail (a time
just taken, a booking less than 30 minutes away): the problem goes back to
the model instead.
"""

import re
from datetime import datetime
from typing import Any

from ..booking.appointments import get_appointments
from ..booking.errors import BookingError, CalendarUnavailable
from ..booking.tools import UNAVAILABLE

# A reply that starts like this is a yes; anything else is not.
YES = re.compile(r"^\s*(y|yes|yeah|yep|yup|sure|ok|okay|confirm(ed)?|please do|go ahead|do it|book it|cancel it|"
                 r"correct|that'?s right)\b", re.IGNORECASE)
CHOICES = {"book_appointment": ["Yes, book it", "No"], "cancel_booking": ["Yes, cancel it", "No, keep it"]}


def is_yes(message: str) -> bool:
    """True when the customer's reply is a yes."""
    return bool(YES.match(message))


def _when(start: datetime) -> str:
    """'Friday 30 October at 15:00'."""
    return f"{start:%A} {start.day} {start:%B} at {start:%H:%M}"


def question_for(request: dict[str, Any], name: str, phone: str) -> str:
    """What to ask the customer before running `request`. Raises BookingError
    (or CalendarUnavailable) when the request would fail anyway."""
    desk, args = get_appointments(), request["args"]
    if request["name"] == "book_appointment":
        start = datetime.fromisoformat(args["start"]).replace(tzinfo=desk.zone)
        free = desk.free_slots(args["service"], start.date(), args["therapist"], args.get("minutes"))
        slot = next((s for s in free if s.start == start), None)
        if slot is None:
            raise BookingError(f"{args['therapist']} is not free for {args['service']} at that time.")
        length = int((slot.end - slot.start).total_seconds() // 60)
        return (f"Shall I book {args['service']} ({length} minutes) with {args['therapist']} on {_when(start)}, "
                f"for {name} ({phone})?")
    booking = desk.check_cancel(args["reference"], phone)
    return (f"Shall I cancel your {booking.service} with {booking.therapist} on {_when(booking.start)} "
            f"(reference {booking.id})?")


def problem_with(request: dict[str, Any], name: str, phone: str) -> tuple[str, str]:
    """(the question to ask, "") when `request` can go ahead, or ("", what is wrong) when it cannot."""
    try:
        return question_for(request, name, phone), ""
    except BookingError as error:
        return "", str(error)
    except CalendarUnavailable:
        return "", UNAVAILABLE
    except (ValueError, KeyError):  # a time in the wrong format, or a missing input
        return "", "The request was incomplete or a time was not written YYYY-MM-DDTHH:MM."
