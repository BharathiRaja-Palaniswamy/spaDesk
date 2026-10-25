"""The Executor (code): books a Plan the customer has said yes to.

RESPONSIBLE FOR: turning an approved Plan into bookings, all of them or none,
and never two bookings in the same time.

It is code, not an LLM: once the customer has said yes there is nothing left
to decide, only an exact action to do. Steps:

  1. lock    take the therapists' locks, so no other booking made through
             spaDesk can take the same times meanwhile (one lock per
             therapist, for this server; several are taken in name order, so
             two plans never wait for each other forever)
  2. check   check the Plan again (plan.py): it was free when drafted, but
             the customer may have taken a while to say yes
  3. book    book the items in order (the engine checks each one once more)
  4. verify  read the calendars back: did an event appear in one of the
             times meanwhile, e.g. typed into Google Calendar by hand?
  5. replace for a change, cancel the booking it replaces (`replaces`)
  6. undo    if any step fails, cancel what this Plan booked: all or none

The answer says what was booked, or why nothing was, so Pebby can explain
and ask the Booking agent for new times.

A change books the new times first and cancels the old booking only once
they are safely booked, so a customer is never left with neither. The old
booking's time counts as free while it does (`ignoring`), so a change can
keep the same time: a different service at 17:00 in place of the 17:00 one.

The lock covers bookings made through spaDesk; the read-back covers the rest
(an event added by hand in Google Calendar at the same moment), because the
calendar lists every event on its own, even two at exactly the same times.

Once the customer said yes, `book` and `cancel` carry it out and tell
Pebby how it went: after MAX_FAILED_PLANS approved plans in a row that
could not be booked, the request goes to the staff.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ...booking.actions import confirmation
from ...booking.appointments import Appointments, get_appointments
from ...booking.calendars import Booking
from ...booking.errors import UNAVAILABLE, BookingError, CalendarUnavailable
from ..rules import MAX_FAILED_PLANS
from ..staff import record_handoff
from .plan import Plan, check_plan

_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}  # therapist -> the lock for booking their times


@dataclass(frozen=True)
class Outcome:
    """What happened to a Plan."""

    booked: list[Booking] = field(default_factory=list)  # every item, when the Plan was booked
    problems: list[str] = field(default_factory=list)  # why nothing was booked, when it was not

    @property
    def done(self) -> bool:
        """True when the whole Plan was booked."""
        return bool(self.booked) and not self.problems


@contextmanager
def _holding(therapists: set[str]) -> Iterator[None]:
    """Hold the locks of `therapists`, taken in name order and given back in reverse."""
    with _guard:
        locks = [_locks.setdefault(name, threading.Lock()) for name in sorted(therapists)]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def _undo(desk: Appointments, made: list[Booking]) -> None:
    """Remove the bookings this Plan made (straight from the calendar: no notice rule applies to undoing)."""
    for booking in reversed(made):
        desk.calendar.remove(booking.id)


def _clashes(desk: Appointments, made: list[Booking], old: Booking | None = None) -> list[str]:
    """Anything else on the calendars in the times just booked: another booking, or
    an event typed into Google Calendar by hand (not `old`, the booking a change replaces)."""
    clashes = []
    for booking in made:
        periods = desk.calendar.busy(booking.therapist, booking.start, booking.end)
        mine = (booking.start, booking.end)
        if mine in periods:
            periods.remove(mine)  # this booking itself; anything left overlaps it
        else:
            periods.append(mine)  # not read back at all: treat it as not safely booked
        if old and old.therapist == booking.therapist and (old.start, old.end) in periods:
            periods.remove((old.start, old.end))  # cancelled next
        if periods:
            clashes.append(f"{booking.service} with {booking.therapist} at {booking.start:%a %d %b %H:%M} was "
                           "taken at the same moment.")
    return clashes


def execute(plan: Plan, name: str, phone: str, desk: Appointments, replaces: str | None = None) -> Outcome:
    """Book every item of `plan` for this customer, or none of them; for a change,
    then cancel the booking `replaces` (its reference)."""
    old = desk.calendar.find(replaces) if replaces else None
    with _holding({item.therapist for item in plan.items} | ({old.therapist} if old else set())):
        if problems := check_plan(plan, desk, old):
            return Outcome(problems=problems)
        made: list[Booking] = []
        try:
            for n, item in enumerate(plan.items):  # the add-ons go with the first appointment
                made.append(desk.book(item.service, item.therapist, item.start, name, phone, item.minutes, old,
                                      plan.package, tuple(plan.add_ons) if n == 0 else ()))
            if clashes := _clashes(desk, made, old):
                _undo(desk, made)
                return Outcome(problems=clashes)
            if replaces:
                desk.cancel(replaces, phone)
        except BookingError as error:
            _undo(desk, made)
            return Outcome(problems=[str(error)])
        except CalendarUnavailable:
            _undo(desk, made)  # may fail too, if the calendar is down; the error goes up either way
            raise
        return Outcome(booked=made)


def book(plan: dict[str, Any], replaces: str, name: str, phone: str, failed: int) -> tuple[dict[str, Any], int]:
    """Book the approved `plan` (for a change, in place of the booking `replaces`).
    Returns what Pebby is told, and the failures in a row so far."""
    approved = Plan.model_validate(plan)
    try:
        outcome = execute(approved, name, phone, get_appointments(), replaces or None)
    except CalendarUnavailable:
        # The calendar failed part way: an item may be booked and not undone, so a person checks.
        record_handoff(name, phone, f"The calendar failed while booking an approved plan; check for partial "
                                    f"bookings: {approved.summary}")
        return {"error": UNAVAILABLE, "handed_off": True}, failed
    if outcome.done:
        done = {"booked": [confirmation(b) for b in outcome.booked]}
        return done | ({"cancelled": replaces} if replaces else {}), 0
    failed += 1
    if failed >= MAX_FAILED_PLANS:
        record_handoff(name, phone, "Booking failed again after the customer approved: " + " ".join(outcome.problems))
        return {"not_booked": outcome.problems, "handed_off": True,
                "next": "Nothing was booked. Tell the customer our staff will call them to arrange it."}, 0
    return {"not_booked": outcome.problems,
            "next": "Nothing was booked. Tell the customer what was taken, and call booking again for new "
                    "times."}, failed


def cancel(references: list[str], phone: str) -> dict[str, Any]:
    """Cancel the approved bookings, each on its own: one that cannot be
    cancelled (e.g. it came within the notice since the yes) does not stop
    the others, and is reported."""
    cancelled, problems = [], []
    for reference in references:
        try:
            cancelled.append(confirmation(get_appointments().cancel(reference, phone)))
        except BookingError as error:
            problems.append(f"{reference}: {error}")
        except CalendarUnavailable:
            problems.append(f"{reference}: {UNAVAILABLE}")
    return {"cancelled": cancelled} | ({"not_cancelled": problems} if problems else {})
