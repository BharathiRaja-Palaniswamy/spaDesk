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


def _when(start: datetime, spoken: bool) -> str:
    """'Friday 30 October at 15:00' to read; 'Friday the 30th of October at 3 PM' to say."""
    if not spoken:
        return f"{start:%A} {start.day} {start:%B} at {start:%H:%M}"
    suffix = "th" if 11 <= start.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(start.day % 10, "th")
    hour = f"{start:%I}".lstrip("0") + (f":{start:%M}" if start.minute else "")
    return f"{start:%A} the {start.day}{suffix} of {start:%B} at {hour} {start:%p}"


def question_for(request: dict[str, Any], name: str, phone: str, spoken: bool = False) -> str:
    """What to ask the customer before running `request`, written to be read,
    or to be said on a phone call (`spoken`). Raises BookingError (or
    CalendarUnavailable) when the request would fail anyway."""
    desk, args = get_appointments(), request["args"]
    if request["name"] == "book_appointment":
        start = datetime.fromisoformat(args["start"]).replace(tzinfo=desk.zone)
        free = desk.free_slots(args["service"], start.date(), args["therapist"], args.get("minutes"))
        slot = next((s for s in free if s.start == start), None)
        if slot is None:
            raise BookingError(f"{args['therapist']} is not free for {args['service']} at that time.")
        length = int((slot.end - slot.start).total_seconds() // 60)
        if spoken:
            return (f"Just to confirm: a {length}-minute {args['service']} with {args['therapist']} on "
                    f"{_when(start, True)}, for {name}. Shall I book it?")
        return (f"Shall I book {args['service']} ({length} minutes) with {args['therapist']} on "
                f"{_when(start, False)}, for {name} ({phone})?")
    booking = desk.check_cancel(args["reference"], phone)
    if spoken:
        return f"Just to confirm: cancel your {booking.service} on {_when(booking.start, True)}. Shall I cancel it?"
    return (f"Shall I cancel your {booking.service} with {booking.therapist} on {_when(booking.start, False)} "
            f"(reference {booking.id})?")


def problem_with(request: dict[str, Any], name: str, phone: str, spoken: bool = False) -> tuple[str, str]:
    """(the question to ask, "") when `request` can go ahead, or ("", what is wrong) when it cannot."""
    try:
        return question_for(request, name, phone, spoken), ""
    except BookingError as error:
        return "", str(error)
    except CalendarUnavailable:
        return "", UNAVAILABLE
    except (ValueError, KeyError):  # a time in the wrong format, or a missing input
        return "", "The request was incomplete or a time was not written YYYY-MM-DDTHH:MM."
