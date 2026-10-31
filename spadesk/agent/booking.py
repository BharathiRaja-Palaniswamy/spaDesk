"""Booking (code): a booking request in, what can be done out.

RESPONSIBLE FOR: answering a BookingRequest (the details Pebby collected,
checked against the customer's own words and read into dates and times by
details.py) with a BookingResult: a plan to approve, a cancellation to
approve, their bookings, the free days or times, or why nothing fits. It
never talks to the customer (Pebby words the result) and never books (a
plan or a cancellation goes to the customer's yes, yes.py, then to
executor.py).

    BookingRequest (from Pebby's booking tool)   BookingResult (to Pebby)
      intent      new | change | cancel |          status   needs_info | plan_ready |
                  my_bookings | availability                cancel_ready | bookings |
      services    exact names (a package: every             options | not_possible | handoff
                  service it includes)               missing  what to ask for
      package     its name, if one was asked         options  days or times to offer
      day         YYYY-MM-DD                         plan     the exact items, for the yes
      after/before HH:MM                             replaces the booking a change replaces
      therapist   a name, if one was asked           cancel   the bookings to cancel
      references  the bookings to change / cancel    bookings their upcoming bookings
      add_ons     extras they named                  reason   why, in plain words

It is all code: every request is complete when it gets here (the booking
tool answers "what is missing" first), so there is nothing left to decide
but exact rules. It keeps no memory: the same request gets the same answer.

    my_bookings   the bookings on their number
    cancel        the bookings, checked against the notice rule; part of a
                  package goes to the staff
    availability  the free times on a day, or with no day, the days that fit
    new / change  fit_plan: the services one after another on the day

The Plan is here too: what is approved and booked, the check that it can
still be booked (check_plan), and how services are fitted into a day.
"""

import json
import logging
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..booking.actions import confirmation
from ..booking.appointments import Appointments, Slot, get_appointments
from ..booking.calendars import Booking
from ..booking.errors import BookingError
from .debug import note, outcome
from .rules import LOOK_AHEAD_DAYS, MAX_DAYS


class PlanItem(BaseModel):
    """One appointment of a Plan."""

    service: str = Field(description="The service's exact name, e.g. 'Prenatal Massage'.")
    therapist: str = Field(description="The therapist's exact name, e.g. 'Priya Nair'.")
    start: datetime = Field(description="When it starts, in the spa's local time: YYYY-MM-DDTHH:MM.")
    minutes: int | None = Field(default=None, description="Its length, for a service that comes in more "
                                                          "than one; leave out for the standard length.")


class Plan(BaseModel):
    """What the customer is offered, and books with one yes."""

    items: list[PlanItem] = Field(description="The appointments, in the order the customer has them.")
    summary: str = Field(description="One sentence for the customer, e.g. 'A massage, then a facial, on "
                                     "Saturday afternoon.'")
    package: str = ""  # the package these items make up, if the customer asked for one: kept on each booking
    add_ons: list[str] = Field(default_factory=list)  # extras the customer named, booked with the first item


def _slot(desk: Appointments, item: PlanItem, ignoring: Booking | None = None) -> Slot | None:
    """The free slot this item asks for, or None if it is not free (or not allowed) now."""
    start = item.start.astimezone(desk.zone) if item.start.tzinfo else item.start.replace(tzinfo=desk.zone)
    free = desk.free_slots(item.service, start.date(), item.therapist, item.minutes, ignoring)
    return next((slot for slot in free if slot.start == start), None)


def _label(item: PlanItem) -> str:
    """'Prenatal Massage with Priya Nair at Fri 30 Oct 15:00'."""
    return f"{item.service} with {item.therapist} at {item.start:%a %d %b %H:%M}"


def check_plan(plan: Plan, desk: Appointments, ignoring: Booking | None = None) -> list[str]:
    """Every problem that stops `plan` being booked right now; empty when it can be.
    `ignoring`: the booking a change replaces, whose time counts as free."""
    if not plan.items:
        return ["The plan has no appointments."]
    problems, slots = [], []
    for item in plan.items:
        try:
            slot = _slot(desk, item, ignoring)
        except BookingError as error:  # e.g. a service the therapist does not offer
            problems.append(f"{_label(item)}: {error}")
            continue
        if slot is None:
            problems.append(f"{_label(item)} is not a free time.")
        else:
            slots.append(slot)
    # One customer, one appointment at a time: in order, and not overlapping.
    ordered = sorted(slots, key=lambda slot: slot.start)
    for before, after in zip(ordered, ordered[1:]):
        if after.start < before.end:
            problems.append(f"{after.start:%H:%M} starts before the appointment at {before.start:%H:%M} ends "
                            f"({before.end:%H:%M}): one customer can't have both at once.")
    return problems


def _free(desk: Appointments, service: str, day: date, therapist: str | None,
          after: time | None, before: time | None, ignoring: Booking | None) -> list[Slot]:
    """The free times for `service` on `day`, within the preferences."""
    slots = desk.free_slots(service, day, therapist, ignoring=ignoring)
    return [s for s in slots
            if (after is None or s.start.time() >= after) and (before is None or s.end.time() <= before)]


def _rest(first: Slot, later: list[list[Slot]]) -> list[Slot] | None:
    """`first`, then each later service's earliest free time once the one before it ends; None if one has none."""
    chosen = [first]
    for options in later:
        slot = next((s for s in options if s.start >= chosen[-1].end), None)
        if slot is None:
            return None
        chosen.append(slot)
    return chosen


def fit_plan(services: Sequence[str], day: date, desk: Appointments, after: time | None = None,
             before: time | None = None, therapists: dict[str, str] | None = None,
             ignoring: Booking | None = None) -> Plan | list[str]:
    """A Plan of `services` one after another on `day`, earliest first; or, when
    they cannot all fit, the problems in words. Raises BookingError for a
    service or therapist the spa does not have (e.g. a therapist who does not
    offer that service)."""
    if not services:
        return ["No service was chosen."]
    chosen_by = therapists or {}
    options = [_free(desk, name, day, chosen_by.get(name), after, before, ignoring) for name in services]
    if empty := [name for name, slots in zip(services, options) if not slots]:
        return [f"No free time for {', '.join(empty)} on {day:%A %d %B}"
                + (" within those times." if after or before else ".")]
    for first in options[0]:
        if fitted := _rest(first, options[1:]):
            items = [PlanItem(service=name, therapist=slot.therapist, start=datetime.combine(day, slot.start.time()))
                     for name, slot in zip(services, fitted)]
            return Plan(items=items, summary=f"{', then '.join(services)} on {day:%A %d %B}.")
    return [f"{', '.join(services)} cannot all fit one after another on {day:%A %d %B}."]



def days_that_fit(services: Sequence[str], desk: Appointments, after: time | None = None, before: time | None = None,
                  therapists: dict[str, str] | None = None, skip: date | None = None) -> list[str]:
    """The next days (from today, at most MAX_DAYS within LOOK_AHEAD_DAYS) on which
    every one of `services` has a free time, as "Wednesday 2026-10-28": within the
    times the customer asked for (`after`, `before`) and with the therapist they
    chose, so a day offered is one that has what they want. `skip`: a day already
    looked at. A day when no one who offers one of them works is skipped without
    reading the calendar, so a search costs few calendar reads. Each service having
    a free time does not promise they fit one after another: fit_plan settles that
    for the day the customer picks."""
    therapists = therapists or {}
    working = {name: {day for t in desk.schedule.therapists_for(name) for day in t.days} for name in services}
    today, days = desk.now().date(), []
    for offset in range(LOOK_AHEAD_DAYS):
        day = today + timedelta(days=offset)
        if day.weekday() not in desk.schedule.hours or any(day.weekday() not in working[n] for n in services):
            continue
        if day == skip:
            continue
        if all(_free(desk, name, day, therapists.get(name), after, before, None) for name in services):
            days.append(f"{day:%A} {day.isoformat()}")
            if len(days) == MAX_DAYS:
                break
    return days


log = logging.getLogger("spadesk.agent")


class BookingRequest(BaseModel):
    """What the customer wants, cleaned up by Pebby: everything said so far."""

    intent: Literal["new", "change", "cancel", "my_bookings", "availability"]
    services: list[str] = Field(default_factory=list)
    package: str = ""
    day: str = ""  # YYYY-MM-DD
    after: str = ""  # HH:MM
    before: str = ""  # HH:MM
    therapist: str = ""
    references: list[str] = Field(default_factory=list)  # the bookings to change (one) or cancel (one or more)
    add_ons: list[str] = Field(default_factory=list)  # extras the customer named, e.g. ["Scalp massage"]


class BookingResult(BaseModel):
    """What can be done about a BookingRequest."""

    status: Literal["needs_info", "plan_ready", "cancel_ready", "bookings", "options", "not_possible", "handoff"]
    missing: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    plan: Plan | None = None
    replaces: dict[str, Any] | None = None  # a change: the booking it replaces, as confirmation() gives it
    cancel: list[dict[str, Any]] | None = None  # cancel_ready: the bookings to cancel
    bookings: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""


def _clock(text: str) -> time | None:
    """'17:00' -> 17:00; '' -> None."""
    return time.fromisoformat(text) if text else None


def _therapists(request: BookingRequest, desk: Appointments) -> dict[str, str]:
    """The chosen therapist, for each service they offer (the others: whoever is free)."""
    if not request.therapist:
        return {}
    if (person := desk.schedule.therapists.get(request.therapist)) is None:
        raise BookingError(f"We have no therapist called {request.therapist}.")
    offers = person.services
    return {name: request.therapist for name in request.services if name in offers}


def _days(services: list[str], desk: Appointments, request: "BookingRequest | None" = None,
          skip: date | None = None) -> list[str]:
    """The next days on which every service has a free time: within the request's
    times and with its therapist, when there is a request."""
    if request is None:
        return days_that_fit(services, desk)
    return days_that_fit(services, desk, _clock(request.after), _clock(request.before), _therapists(request, desk), skip)


def _window(request: "BookingRequest") -> str:
    """The times asked for, as said back: "before 12:00", "from 15:00", "15:00 to 17:00"; "" for any time."""
    if request.after and request.before:
        return f"{request.after} to {request.before}"
    return f"from {request.after}" if request.after else f"before {request.before}" if request.before else ""


def _times(slots: list[Slot]) -> str:
    return ", ".join(f"{s.start:%H:%M} with {s.therapist}" for s in slots[:6])


def _availability(request: BookingRequest, day: date, desk: Appointments) -> BookingResult:
    """The free times on `day` of each service asked about, as options. When the times
    asked for have none, what is nearest is offered instead, so the customer always has
    something to book: the same day's other times, then the next days that have times
    within what they asked for."""
    options, window = [], _window(request)
    for name in request.services:
        therapist = _therapists(request, desk).get(name)
        slots = _free(desk, name, day, therapist, _clock(request.after), _clock(request.before), None)
        if slots:  # the same window a booking uses
            options.append(f"{name} on {day:%A %d %B}: {_times(slots)}")
            continue
        options.append(f"{name} on {day:%A %d %B}: no free times" + (f" {window}" if window else ""))
        if window and (other := _free(desk, name, day, therapist, None, None, None)):
            options.append(f"{name} on {day:%A %d %B} at other times: {_times(other)}")
        if later := _days([name], desk, request, skip=day):
            options.append(f"{name} on other days" + (f", {window}" if window else "") + f": {', '.join(later)}")
    return BookingResult(status="options", options=options)


def _open(day: date, desk: Appointments) -> str:
    """The day's opening hours, said with every "nothing fits", so Pebby has the fact
    and nothing to guess: 'We are open 10:00 to 20:30 on Thursdays.'"""
    if (hours := desk.schedule.hours.get(day.weekday())) is None:
        return f"We are closed on {day:%A}s."
    return f"We are open {hours[0]:%H:%M} to {hours[1]:%H:%M} on {day:%A}s."


def _plan(request: BookingRequest, desk: Appointments, replaces: Booking | None) -> BookingResult:
    """The days that fit (no day yet), the free times (availability), or a plan on the day."""
    if not request.day:  # only an availability question comes without a day
        if days := _days(request.services, desk):
            return BookingResult(status="options", options=days)
        return BookingResult(status="not_possible", reason="No day in the next two weeks has all of these free.")
    day = date.fromisoformat(request.day)
    if request.intent == "availability":
        return _availability(request, day, desk)
    fitted = fit_plan(request.services, day, desk, _clock(request.after), _clock(request.before),
                      _therapists(request, desk), replaces)
    if isinstance(fitted, Plan):
        fitted = fitted.model_copy(update={"package": request.package, "add_ons": request.add_ons})
        return BookingResult(status="plan_ready", plan=fitted, replaces=confirmation(replaces) if replaces else None)
    return BookingResult(status="not_possible", reason=f"{fitted[0]} {_open(day, desk)}",
                         options=_days(request.services, desk, request, skip=day))


def handle(request: BookingRequest, phone: str, desk: Appointments | None = None) -> BookingResult:
    """The answer to `request`, for the customer with this phone number. Raises
    CalendarUnavailable when the calendar cannot be reached."""
    result = _answer(request, phone, desk or get_appointments())
    described = describe(result)
    note("Booking", "Pebby" if result.status not in ("plan_ready", "cancel_ready") else "The yes",
         outcome(described),
         {"request": request.model_dump(exclude_defaults=True), "result": described})
    return result


def _part_of_a_package(chosen: list[Booking], upcoming: list[Booking]) -> str:
    """Why changing or cancelling `chosen` needs the staff: it leaves part of a
    package booked (the rest of it, booked the same day, is not among them),
    and what that does to the package's price is for the spa to settle. ""
    when it does not."""
    picked = {b.id for b in chosen}
    for booking in chosen:
        rest = [b for b in upcoming if b.package == booking.package and b.start.date() == booking.start.date()
                and b.id not in picked] if booking.package else []
        if rest:
            return (f"Your {booking.service} is part of your {booking.package}: our staff will help you change "
                    "part of a package.")
    return ""


def _answer(request: BookingRequest, phone: str, desk: Appointments) -> BookingResult:
    """handle's answer."""
    if request.intent == "my_bookings":
        return BookingResult(status="bookings", bookings=[confirmation(b) for b in desk.upcoming(phone)])
    replaces = None
    if request.intent in ("change", "cancel"):
        try:  # theirs, and not within the notice
            chosen = [desk.check_cancel(reference, phone) for reference in request.references]
        except BookingError as error:
            return BookingResult(status="not_possible", reason=str(error))
        if request.intent == "change" and len(chosen) != 1:
            return BookingResult(status="not_possible", reason="One booking can be changed at a time.")
        if reason := _part_of_a_package(chosen, desk.upcoming(phone)):
            return BookingResult(status="handoff", reason=reason)
        if request.intent == "cancel":
            return BookingResult(status="cancel_ready", cancel=[confirmation(b) for b in chosen])
        replaces = chosen[0]
    try:
        _therapists(request, desk)  # a therapist the spa does not have is refused here, whatever else is asked
        return _plan(request, desk, replaces)
    except BookingError as error:  # e.g. a therapist who does not offer one of the services
        return BookingResult(status="not_possible", reason=str(error))
    except ValueError:
        return BookingResult(status="not_possible", reason="Dates are YYYY-MM-DD and times HH:MM.")


def missing_result(request: BookingRequest, missing: list[str], phone: str,
                   desk: Appointments | None = None) -> BookingResult:
    """What Pebby must still ask the customer for, with what helps them answer:
    their bookings when it is which booking, the days that work when it is the day."""
    desk = desk or get_appointments()
    if "references" in missing:
        return BookingResult(status="needs_info", missing=missing,
                             bookings=[confirmation(b) for b in desk.upcoming(phone)])
    options = _days(request.services, desk) if "day" in missing and request.services else []
    return BookingResult(status="needs_info", missing=missing, options=options)


def describe(result: BookingResult) -> dict[str, Any]:
    """`result` as Pebby is told it: only the fields that say something."""
    return json.loads(result.model_dump_json(exclude_defaults=True))
