"""Tests for reading the booking schedule from the business document
(spadesk/booking/schedule.py). They read the real document.

Run with:  .venv/bin/python -m pytest tests/test_booking_schedule.py
"""

from datetime import time

from spadesk.booking.schedule import Lesson, read_schedule
from spadesk.business.store import get_knowledge

FRIDAY, MONDAY, WEDNESDAY = 4, 0, 2


def schedule():
    return read_schedule(get_knowledge())


def test_only_one_to_one_appointments_with_a_length_are_bookable():
    services = schedule().services
    assert services["Swedish Massage"].minutes == (60, 90) and services["Swedish Massage"].prices == (115, 160)
    assert "Private Yoga Session" in services
    assert services["Lip Wax"].minutes == (45,)
    # A group class and a package are booked other ways.
    assert not {"Vinyasa Flow", "Relax & Glow Package"} & set(services)


def test_each_therapist_works_their_days_and_offers_their_specialties():
    team = schedule().therapists
    priya = team["Priya Nair"]
    assert priya.days == {FRIDAY, 5, 6}
    assert set(priya.services) == {"Prenatal Massage", "Aromatherapy Massage", "Swedish Massage"}
    # A specialty can name a category ("body treatments") or part of a name ("private yoga").
    assert {"Detox Body Scrub", "Sea Mineral Body Wrap"} <= set(team["Sofia Martinez"].services)
    assert team["Maya Chen"].services == ("Private Yoga Session",)
    assert "Full Leg Wax" in team["Hannah Lee"].services  # "hair removal (waxing)"


def test_who_offers_a_service():
    assert [t.name for t in schedule().therapists_for("Swedish Massage")] == ["Daniel Ruiz", "Priya Nair"]


def test_opening_hours_and_the_timetable_are_read():
    plan = schedule()
    assert plan.hours[FRIDAY] == (time(10), time(19)) and plan.hours[MONDAY] == (time(10), time(20))
    assert Lesson(WEDNESDAY, time(10), 60, "Prenatal Yoga", "Maya Chen") in plan.lessons
