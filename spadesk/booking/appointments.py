"""The booking rules: free times, booking, cancelling, and the day's schedule.

    free time = the spa's opening hours on that day
              - (therapist not working that day)
              - the classes they teach (an instructor's timetable)
              - whatever is on their calendar (bookings, time off)
              - times less than SPADESK_MIN_NOTICE_MINUTES from now (or past)

These rules are ours, which is why the app talks to the calendar through its
own interface (calendars/) rather than handing a model the calendar itself:
a booking can only be made in a free time, with a therapist who offers the
service, and a customer can only cancel a booking whose phone number they
know, and not once it is less than SPADESK_MIN_NOTICE_MINUTES away (they
call the spa instead). The model (and later a voice assistant) asks through tools.py, and
every request is checked here.

Times are in the spa's time zone (SPADESK_TIMEZONE). Appointments start on a
grid, every SPADESK_SLOT_MINUTES from opening, and must end by closing time.
"""

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ..business.store import get_knowledge
from ..config import get_settings
from .calendars import Booking, Calendar, get_calendar
from .errors import BookingError
from .schedule import Schedule, Service, Therapist, read_schedule


@dataclass(frozen=True)
class Slot:
    """A time a therapist is free for the whole service."""

    therapist: str
    start: datetime
    end: datetime


def _digits(phone: str) -> str:
    """'(408) 555-0188' -> '4085550188', so two ways of writing a number compare equal."""
    return re.sub(r"\D", "", phone)


class Appointments:
    """Books appointments on a calendar, following the schedule in the document."""

    def __init__(self, schedule: Schedule, calendar: Calendar, zone: ZoneInfo, step: int, notice: int = 0,
                 now: Callable[[], datetime] | None = None) -> None:
        """`step`: minutes between possible start times. `notice`: the fewest
        minutes ahead a booking can be made, changed or cancelled. `now` gives
        the current time; tests pass a fixed one."""
        self.schedule, self.calendar, self.zone = schedule, calendar, zone
        self.step, self.notice = timedelta(minutes=step), timedelta(minutes=notice)
        self._now = now or (lambda: datetime.now(zone))

    # ---- what the customer can ask -----------------------------------------

    def free_slots(self, service: str, day: date, therapist: str | None = None,
                   minutes: int | None = None) -> list[Slot]:
        """Every free time on `day` for `service` (of length `minutes`, its
        standard length if not given), with `therapist` or anyone who offers
        it. Earliest first."""
        chosen = self._service(service)
        length = timedelta(minutes=self._length(chosen, minutes))
        slots = [slot for person in self._therapists(chosen, therapist)
                 for slot in self._free_for(person, day, length)]
        return sorted(slots, key=lambda slot: (slot.start, slot.therapist))

    def book(self, service: str, therapist: str, start: datetime, name: str, phone: str,
             minutes: int | None = None) -> Booking:
        """Book `service` with `therapist` at `start`, if that time is free.
        A time with no time zone is taken as the spa's."""
        if not name.strip():
            raise BookingError("Please give the name the booking is for.")
        if len(_digits(phone)) < 7:
            raise BookingError("Please give a phone number the spa can reach you on.")
        start = start.astimezone(self.zone) if start.tzinfo else start.replace(tzinfo=self.zone)
        free = self.free_slots(service, start.date(), therapist, minutes)
        slot = next((s for s in free if s.start == start), None)
        if slot is None:
            raise BookingError(f"{therapist} is not free for {service} at that time. Please choose a free time.")
        # The check and the booking are two steps: two customers booking the
        # same time at the same moment could both pass the check. That is rare
        # for one small spa; a busier one would lock the time first.
        booking = Booking("BK-" + secrets.token_hex(3).upper(), service, therapist, slot.start, slot.end,
                          name.strip(), phone.strip())
        self.calendar.add(booking)
        return booking

    def cancel(self, booking_id: str, phone: str) -> Booking:
        """Cancel a booking, given its reference and the phone number it was
        booked with, if it is at least `notice` away. The rest of the
        cancellation policy (fees) is left to the spa."""
        booking = self.check_cancel(booking_id, phone)
        self.calendar.remove(booking.id)
        return booking

    def check_cancel(self, booking_id: str, phone: str) -> Booking:
        """The booking `cancel` would cancel, without cancelling it: raises
        BookingError if it may not be cancelled. Lets a caller show the
        booking and ask "are you sure?" first."""
        booking = self.calendar.find(booking_id.strip().upper())
        # The same answer whether the reference or the phone number is wrong,
        # so nobody can find out which bookings exist by guessing.
        if booking is None or _digits(booking.phone) != _digits(phone):
            raise BookingError("No booking matches that reference and phone number.")
        if booking.start - self._now() < self.notice:
            minutes = int(self.notice.total_seconds() // 60)
            raise BookingError(f"That appointment is less than {minutes} minutes away, so it can't be changed "
                               "or cancelled here. Please call the spa.")
        return booking

    def upcoming(self, phone: str, days: int = 60) -> list[Booking]:
        """The bookings made with this phone number in the next `days` days, earliest first."""
        now = self._now()
        return [b for b in self.calendar.bookings(now, now + timedelta(days=days)) if _digits(b.phone) == _digits(phone)]

    # ---- for the owner only ------------------------------------------------

    def day_bookings(self, day: date, therapist: str | None = None) -> list[Booking]:
        """Every booking on `day`, earliest first, with the customers' details."""
        if therapist is not None:
            self._therapist(therapist)
        start = datetime.combine(day, time(0), self.zone)
        return self.calendar.bookings(start, start + timedelta(days=1), therapist)

    # ---- checks ------------------------------------------------------------

    def _service(self, name: str) -> Service:
        """The bookable service called `name`."""
        if name not in self.schedule.services:
            raise BookingError(f"{name!r} is not a service that can be booked here. "
                               f"Choose one of: {', '.join(self.schedule.services)}.")
        return self.schedule.services[name]

    def _length(self, service: Service, minutes: int | None) -> int:
        """`minutes` if the service comes in that length, its standard length if None."""
        if minutes is None:
            return service.minutes[0]
        if minutes not in service.minutes:
            lengths = " or ".join(str(m) for m in service.minutes)
            raise BookingError(f"{service.name} is {lengths} minutes long.")
        return minutes

    def _therapist(self, name: str) -> Therapist:
        """The team member called `name`."""
        if name not in self.schedule.therapists:
            raise BookingError(f"{name!r} is not on the team. The team: {', '.join(self.schedule.therapists)}.")
        return self.schedule.therapists[name]

    def _therapists(self, service: Service, name: str | None) -> list[Therapist]:
        """`name` if they offer the service; everyone who does if `name` is None."""
        if name is None:
            return self.schedule.therapists_for(service.name)
        person = self._therapist(name)
        if service.name not in person.services:
            others = ", ".join(t.name for t in self.schedule.therapists_for(service.name))
            raise BookingError(f"{name} does not offer {service.name}. It is offered by: {others}.")
        return [person]

    # ---- free time -----------------------------------------------------------

    def _free_for(self, person: Therapist, day: date, length: timedelta) -> list[Slot]:
        """The free start times for one therapist on one day."""
        hours = self.schedule.hours.get(day.weekday())
        if hours is None or day.weekday() not in person.days:
            return []
        opens, closes = (datetime.combine(day, t, self.zone) for t in hours)
        busy = self.calendar.busy(person.name, opens, closes) + self._teaching(person, day)
        earliest, slots, start = self._now() + self.notice, [], opens
        while start + length <= closes:
            end = start + length
            if start >= earliest and all(end <= b_start or start >= b_end for b_start, b_end in busy):
                slots.append(Slot(person.name, start, end))
            start += self.step
        return slots

    def _teaching(self, person: Therapist, day: date) -> list[tuple[datetime, datetime]]:
        """The times `person` teaches a class on `day`, from the timetable."""
        periods = []
        for lesson in self.schedule.lessons:
            if lesson.teacher == person.name and lesson.day == day.weekday():
                start = datetime.combine(day, lesson.start, self.zone)
                periods.append((start, start + timedelta(minutes=lesson.minutes)))
        return periods


def get_appointments() -> Appointments:
    """The booking rules over the current document and the configured calendar."""
    settings = get_settings()
    return Appointments(read_schedule(get_knowledge()), get_calendar(), ZoneInfo(settings.timezone),
                        settings.slot_minutes, settings.min_notice_minutes)
