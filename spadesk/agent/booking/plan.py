"""The Plan, and the plan check (code).

RESPONSIBLE FOR: the one shape an approved booking takes inside the team,
and checking that a Plan can really be booked right now.

A Plan is a list of exact items (service, therapist, start, length), in the
order the customer will have them. fit_plan (fit.py) makes one, the yes
(agent/booking/yes.py) reads it to the customer, the Executor books it. Nobody
rewrites it in between, so what the customer said yes to is exactly what
gets booked.

The plan check asks the booking engine (booking/appointments.py) about every
item: is it a free time, with a therapist who offers the service, within
opening hours, with 30 minutes' notice? It also checks that the items do not
overlap each other: one customer cannot be in two treatments at once. It
returns the problems found, in words Pebby (and the customer) can read; no problems means the Plan can be booked.

Fitting (code): `days_that_fit`, the next days on which every service has
a free time, and `fit_plan`, the chosen services back to back on the
chosen day, earliest first, within after / before / therapist preferences.
"""

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta

from pydantic import BaseModel, Field

from ...booking.appointments import Appointments, Slot
from ...booking.calendars import Booking
from ...booking.errors import BookingError
from ..rules import LOOK_AHEAD_DAYS, MAX_DAYS


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



def days_that_fit(services: Sequence[str], desk: Appointments) -> list[str]:
    """The next days (from today, at most MAX_DAYS within LOOK_AHEAD_DAYS) on which
    every one of `services` has a free time, as "Wednesday 2026-10-28". A day when
    no one who offers one of them works is skipped without reading the calendar,
    so a search costs few calendar reads. Each service having a free time does
    not promise they fit one after another: fit_plan settles that for the day
    the customer picks."""
    working = {name: {day for t in desk.schedule.therapists_for(name) for day in t.days} for name in services}
    today, days = desk.now().date(), []
    for offset in range(LOOK_AHEAD_DAYS):
        day = today + timedelta(days=offset)
        if day.weekday() not in desk.schedule.hours or any(day.weekday() not in working[n] for n in services):
            continue
        if all(desk.free_slots(name, day) for name in services):
            days.append(f"{day:%A} {day.isoformat()}")
            if len(days) == MAX_DAYS:
                break
    return days
