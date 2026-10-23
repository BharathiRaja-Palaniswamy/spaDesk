"""Tests for the booking rules (spadesk/booking/appointments.py), on the real
document's schedule, an in-memory calendar and a fixed clock.

"Now" is Thursday 22 October 2026, 2:10 PM at the spa, and a booking needs
30 minutes' notice. Priya works Fridays (the 23rd); Maya teaches Prenatal
Yoga at 10:00 on Wednesdays (the 28th).

Run with:  .venv/bin/python -m pytest tests/test_booking_appointments.py
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from spadesk.booking.appointments import Appointments
from spadesk.booking.calendars.memory import MemoryCalendar
from spadesk.booking.errors import BookingError
from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge

SPA = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 10, 22, 14, 10, tzinfo=SPA)
THURSDAY, FRIDAY, MONDAY, WEDNESDAY = date(2026, 10, 22), date(2026, 10, 23), date(2026, 10, 26), date(2026, 10, 28)


class Clock:
    """A clock the test can move forward."""

    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock(NOW)


@pytest.fixture
def desk(clock):
    return Appointments(read_schedule(get_knowledge()), MemoryCalendar(), SPA, 30, notice=30, now=clock)


def times(slots):
    return [slot.start.strftime("%H:%M") for slot in slots]


def at(day, clock):
    hour, minute = map(int, clock.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=SPA)


def test_free_times_run_from_opening_to_the_last_start_that_ends_by_closing(desk):
    slots = desk.free_slots("Prenatal Massage", FRIDAY)
    assert {slot.therapist for slot in slots} == {"Priya Nair"}  # the only one who offers it
    assert times(slots)[0] == "10:00" and times(slots)[-1] == "18:00"  # Friday: 10 AM to 7 PM


def test_nobody_is_offered_on_a_day_they_do_not_work(desk):
    assert desk.free_slots("Prenatal Massage", MONDAY) == []


def test_times_less_than_30_minutes_ahead_are_not_offered(desk):
    # At 14:10, 14:30 is only 20 minutes away; 15:00 is the first with 30 minutes' notice.
    assert times(desk.free_slots("Swedish Massage", THURSDAY))[0] == "15:00"


def test_a_time_exactly_30_minutes_ahead_can_be_booked(desk, clock):
    clock.now = at(THURSDAY, "14:30")
    assert desk.book("Swedish Massage", "Daniel Ruiz", at(THURSDAY, "15:00"), "Asha", "408 555 0101")
    with pytest.raises(BookingError, match="not free"):
        desk.book("Swedish Massage", "Daniel Ruiz", at(THURSDAY, "14:30"), "Ben", "408 555 0102")


def test_an_instructor_is_not_free_while_teaching(desk):
    offered = times(desk.free_slots("Private Yoga Session", WEDNESDAY))
    assert "10:00" not in offered and "10:30" not in offered and "11:00" in offered  # Prenatal Yoga, 10-11
    assert "18:00" not in offered  # Vinyasa Flow starts 18:30


def test_a_booked_time_is_no_longer_free(desk):
    booking = desk.book("Prenatal Massage", "Priya Nair", at(FRIDAY, "10:00"), "Asha", "(408) 555-0101")
    assert booking.id.startswith("BK-") and booking.end == at(FRIDAY, "11:00")
    offered = times(desk.free_slots("Swedish Massage", FRIDAY, "Priya Nair"))
    assert "10:00" not in offered and "10:30" not in offered and "11:00" in offered
    with pytest.raises(BookingError, match="not free"):
        desk.book("Swedish Massage", "Priya Nair", at(FRIDAY, "10:30"), "Ben", "408 555 0102")


def test_a_longer_option_needs_the_whole_time_free(desk):
    assert times(desk.free_slots("Swedish Massage", FRIDAY, minutes=90))[-1] == "17:30"
    with pytest.raises(BookingError, match="60 or 90 minutes"):
        desk.free_slots("Swedish Massage", FRIDAY, minutes=75)


def test_a_therapist_who_does_not_offer_the_service_is_refused(desk):
    with pytest.raises(BookingError, match="It is offered by: Priya Nair"):
        desk.book("Prenatal Massage", "Daniel Ruiz", at(MONDAY, "10:00"), "Asha", "408 555 0101")


def test_a_name_and_a_phone_number_are_needed(desk):
    with pytest.raises(BookingError, match="phone number"):
        desk.book("Prenatal Massage", "Priya Nair", at(FRIDAY, "10:00"), "Asha", "123")


def test_cancelling_needs_the_phone_number_it_was_booked_with(desk):
    booking = desk.book("Prenatal Massage", "Priya Nair", at(FRIDAY, "10:00"), "Asha", "(408) 555-0101")
    with pytest.raises(BookingError, match="No booking matches"):
        desk.cancel(booking.id, "408 555 9999")
    # Written another way, the same number matches; the time is then free again.
    assert desk.cancel(booking.id.lower(), "408-555-0101") == booking
    assert "10:00" in times(desk.free_slots("Prenatal Massage", FRIDAY))


def test_a_booking_less_than_30_minutes_away_cannot_be_cancelled(desk, clock):
    booking = desk.book("Prenatal Massage", "Priya Nair", at(FRIDAY, "10:00"), "Asha", "(408) 555-0101")
    clock.now = at(FRIDAY, "09:40")
    with pytest.raises(BookingError, match="less than 30 minutes away"):
        desk.cancel(booking.id, "4085550101")
    clock.now = at(FRIDAY, "09:30")  # exactly 30 minutes ahead is still allowed
    assert desk.cancel(booking.id, "4085550101") == booking


def test_the_owner_sees_the_days_bookings_for_one_therapist_or_all(desk):
    desk.book("Prenatal Massage", "Priya Nair", at(FRIDAY, "10:00"), "Asha", "408 555 0101")
    desk.book("Deep Renewal Facial", "Sofia Martinez", at(FRIDAY, "12:00"), "Ben", "408 555 0102")
    assert [b.name for b in desk.day_bookings(FRIDAY)] == ["Asha", "Ben"]
    assert [b.name for b in desk.day_bookings(FRIDAY, "Sofia Martinez")] == ["Ben"]
    assert desk.day_bookings(MONDAY) == []
