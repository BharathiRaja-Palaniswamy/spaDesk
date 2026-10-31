"""The customer's yes, before a Plan is booked or a booking cancelled.

This is the line between what the agents do alone and what needs the
customer: they may search, offer times and make a plan on their own, but a
plan is only booked, or a booking cancelled, after the customer says yes
to exactly what will happen. The graph, not a model, enforces it: it stops,
asks the question written here, and goes ahead only if the customer's
next message is a yes.

Three things wait for a yes:
  details   the details Pebby collected, read back before booking (code)
            looks for times (`details_question`), so a wrong detail is fixed
            before anything is planned
  plan      the times found, every item of the plan (`plan_question`); for a
            change, which booking is cancelled and what is booked instead
  cancel    the booking to cancel (`cancel_question`)
Booking (code) checked the plan against the calendar (booking.py) and
the cancellation against the 30-minute rule before asking, so the customer
is never asked to confirm something that would then fail.
"""

import calendar
from datetime import datetime
from typing import Any

from ..booking.appointments import get_appointments
from .booking import Plan
from .rules import BUT, YES

# The replies to tap, by what is waiting for the yes.
CHOICES = {"details": ["Yes, that's right", "Change something"], "plan": ["Yes, book it", "No"],
           "cancel": ["Yes, cancel it", "No, keep it"]}



def is_yes(message: str) -> bool:
    """True when the customer's reply is a plain yes."""
    return bool(YES.match(message)) and not BUT.search(message)


def day_words(day: datetime, spoken: bool) -> str:
    """'Friday 30 October' to read; 'Friday the 30th of October' to say."""
    if not spoken:
        return f"{day:%A} {day.day} {day:%B}"
    suffix = "th" if 11 <= day.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day.day % 10, "th")
    return f"{day:%A} the {day.day}{suffix} of {day:%B}"


def time_words(moment: datetime, spoken: bool) -> str:
    """'15:00' to read; '3 PM' to say."""
    if not spoken:
        return f"{moment:%H:%M}"
    return f"{moment:%I}".lstrip("0") + (f":{moment:%M}" if moment.minute else "") + f" {moment:%p}"


def when(start: datetime, spoken: bool) -> str:
    """'Friday 30 October at 15:00' to read; 'Friday the 30th of October at 3 PM' to say."""
    return f"{day_words(start, spoken)} at {time_words(start, spoken)}"


PARTS_OF_DAY = {"morning", "afternoon", "evening"}


def details_question(request: dict[str, Any], said: dict[str, str], spoken: bool = False) -> str:
    """What to ask before booking (code) looks for times: the details collected
    (a BookingRequest's fields), so the customer can correct them. The day and
    the time are in the customer's own words (`said`: "tomorrow", "morning"),
    with the date code worked out from them, so it can be checked; the times
    code reads from the words ("morning" is before 12:00) are only for finding
    times, never shown as if the customer had said them."""
    day = day_words(datetime.fromisoformat(request["day"]), spoken)
    def lower(words: str) -> str:  # mid-sentence: "Today" -> "today"; a name ("Friday", "October") keeps its capital
        first = words.split()[0].lower() if words.split() else ""
        names = {d.lower() for d in calendar.day_name} | {m.lower()[:3] for m in calendar.month_name if m}
        return words if first in names or first[:3] in names else words[:1].lower() + words[1:]
    their_day, their_time = lower(said.get("day", "").strip()), lower(said.get("time", "").strip())
    if their_time and their_time.lower() in their_day.lower():  # said once, in the day: "this Friday morning"
        their_time = ""
    # "tomorrow (Monday 26 October)"; "on Friday 30 October" when their words are in the date already
    if "same" in f"{their_day} {their_time}" and request.get("after"):  # "same time": say which it is
        day += f", from {request['after']}"
    on = f"{their_day} ({day})" if their_day and their_day.lower() not in day.lower() else f"on {day}"
    package = f" (the {request['package']})" if request.get("package") else ""
    extras = f" plus {' and '.join(request['add_ons']).lower()}" if request.get("add_ons") else ""
    parts = [f"{' and '.join(request['services'])}{package}{extras} {on}"]
    if their_time:  # "in the morning" for a part of the day; otherwise as they said it ("after 3pm")
        parts.append(f"in the {their_time}" if their_time.lower() in PARTS_OF_DAY else their_time)
    elif request.get("after") and "same" not in their_day:  # a change keeping its time: say which
        parts.append(f"from {request['after']}")
    parts += [f"with {request['therapist']}"] if request.get("therapist") else []
    parts += [f"instead of your booking {request['references'][0]}"] if request.get("references") else []
    notes = f" Also: {request['notes']}." if request.get("notes") else ""
    return f"Just to check before I look for times: {', '.join(parts)}.{notes} Is that right?"


def plan_question(plan: Plan, name: str, phone: str, spoken: bool = False,
                  replaces: dict[str, Any] | None = None) -> str:
    """What to ask before booking `plan`: every item, in order, written to be read
    or to be said on a phone call (`spoken`). For a change, it says first which
    booking is cancelled (`replaces`, as actions.confirmation gives it), so the
    customer knows exactly what happens to both."""
    services = get_appointments().schedule.services
    parts = []
    for n, item in enumerate(plan.items):  # the add-ons go with the first appointment
        length = item.minutes or services[item.service].minutes[0]
        extras = f" plus {' and '.join(plan.add_ons).lower()}" if plan.add_ons and n == 0 else ""
        parts.append(f"a {length}-minute {item.service}{extras} with {item.therapist} on {when(item.start, True)}"
                     if spoken else f"{item.service}{extras} ({length} minutes) with {item.therapist} on "
                                    f"{when(item.start, False)}")
    if replaces:
        old_start = datetime.fromisoformat(replaces["start"])
        old = (f"your {replaces['service']} on {when(old_start, True)}" if spoken else
               f"your current booking ({replaces['service']} with {replaces['therapist']} on "
               f"{when(old_start, False)}, {replaces['reference']})")
        new = (", then ".join(parts) if spoken or len(parts) == 1
               else "these: " + " ".join(f"{n}. {part}." for n, part in enumerate(parts, 1)))
        lead = "Just to confirm: I'll" if spoken else "I'll"
        return f"{lead} cancel {old} and book {new} instead, for {name}. Shall I go ahead?"
    if spoken:
        return (f"Just to confirm: {', then '.join(parts)}, for {name}. "
                f"Shall I book {'them' if len(parts) > 1 else 'it'}?")
    if len(parts) == 1:
        return f"Shall I book {parts[0]}, for {name} ({phone})?"
    listed = " ".join(f"{n}. {part}." for n, part in enumerate(parts, 1))
    return f"Shall I book these for {name} ({phone})? {listed}"


def cancel_question(bookings: list[dict[str, Any]], spoken: bool = False) -> str:
    """What to ask before cancelling `bookings` (as actions.confirmation gives
    them), written to be read, or to be said on a phone call (`spoken`)."""
    def one(booking: dict[str, Any], spoken: bool) -> str:
        start = datetime.fromisoformat(booking["start"])
        return (f"your {booking['service']} on {when(start, True)}" if spoken else
                f"your {booking['service']} with {booking['therapist']} on {when(start, False)} "
                f"(reference {booking['reference']})")
    if spoken:
        return (f"Just to confirm: cancel {', and '.join(one(b, True) for b in bookings)}. "
                f"Shall I cancel {'them' if len(bookings) > 1 else 'it'}?")
    if len(bookings) == 1:
        return f"Shall I cancel {one(bookings[0], False)}?"
    listed = " ".join(f"{n}. {one(b, False).removeprefix('your ')}." for n, b in enumerate(bookings, 1))
    return f"Shall I cancel these {len(bookings)} bookings? {listed}"


def done_message(result: dict[str, Any], spoken: bool = False) -> str:
    """What to tell the customer once their yes is carried out: each booking made,
    with its reference (theirs to change or cancel it later), and each one cancelled.
    "" when something was not done: then Pebby explains, from the result. Written
    by code, as the questions are, so no reference or time can be left out or changed."""
    if result.get("not_booked") or result.get("not_cancelled") or result.get("error"):
        return ""
    def said(reference: str) -> str:  # on a call, letter by letter
        return " ".join(reference) if spoken else reference
    def one(booking: dict[str, Any]) -> str:
        start = datetime.fromisoformat(booking["start"])
        return f"{booking['service']} with {booking['therapist']} on {when(start, spoken)}, reference {said(booking['reference'])}"
    parts = []
    if booked := result.get("booked"):
        parts.append("You're booked: " + "; ".join(one(b) for b in booked) + ".")
        if replaced := result.get("cancelled"):  # a change: the booking it replaces
            parts.append(f"Your previous booking, {said(replaced)}, is cancelled.")
    elif cancelled := result.get("cancelled"):
        parts.append("Cancelled: " + "; ".join(one(b) for b in cancelled) + ".")
    if not parts:
        return ""
    return " ".join(parts) + (" We look forward to seeing you!" if booked else " Is there anything else I can help with?")
