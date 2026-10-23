"""Tests for the booking tools (spadesk/booking/tools.py): what a model sends
and gets back, on the real document, an in-memory calendar and a fixed clock.

Run with:  .venv/bin/python -m pytest tests/test_booking_tools.py
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from spadesk.booking import tools
from spadesk.booking.appointments import Appointments
from spadesk.booking.calendars.memory import MemoryCalendar
from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge
from spadesk.models.tools import run_requests

SPA = ZoneInfo("America/Los_Angeles")
ASHA = {"service": "Prenatal Massage", "therapist": "Priya Nair", "start": "2026-10-23T10:00",
        "name": "Asha", "phone": "(408) 555-0101"}


@pytest.fixture(autouse=True)
def desk(monkeypatch):
    """The tools book on one in-memory calendar, with "now" fixed on Thursday 22 October 2026."""
    desk = Appointments(read_schedule(get_knowledge()), MemoryCalendar(), SPA, 30,
                        now=lambda: datetime(2026, 10, 22, 14, 10, tzinfo=SPA))
    monkeypatch.setattr(tools, "get_appointments", lambda: desk)
    return desk


def use(name, args, owner=False):
    """Run one tool as a model's request would, and read back what the model would see."""
    [message] = run_requests([{"name": name, "args": args, "id": "call-1"}], tools.booking_tools(owner))
    return json.loads(message.content)


def test_customers_get_three_tools_and_the_owner_also_sees_the_day():
    assert [t.name for t in tools.booking_tools()] == ["find_free_times", "book_appointment", "cancel_booking"]
    assert tools.booking_tools(owner=True)[-1].name == "day_schedule"


def test_the_tool_descriptions_list_the_documents_services_and_team():
    find = tools.booking_tools()[0].parameters["properties"]
    assert "Prenatal Massage" in find["service"]["enum"] and "Priya Nair" in find["therapist"]["enum"]


def test_free_times_come_back_by_therapist():
    free = use("find_free_times", {"service": "Swedish Massage", "date": "2026-10-23"})["free"]
    assert list(free) == ["Priya Nair"] and free["Priya Nair"][0] == "10:00"


def test_booking_returns_a_reference_and_the_time_is_then_taken():
    booked = use("book_appointment", ASHA)["booked"]
    assert booked["reference"].startswith("BK-") and booked["end"] == "2026-10-23T11:00"
    assert "phone" not in booked
    assert "10:00" not in use("find_free_times", {"service": "Prenatal Massage", "date": "2026-10-23"})["free"]["Priya Nair"]
    assert "not free" in use("book_appointment", {**ASHA, "name": "Ben"})["error"]


def test_a_broken_rule_or_a_bad_date_comes_back_as_an_error_the_model_can_read():
    assert "does not offer" in use("book_appointment", {**ASHA, "therapist": "Daniel Ruiz"})["error"]
    assert "YYYY-MM-DD" in use("find_free_times", {"service": "Swedish Massage", "date": "next Friday"})["error"]


def test_cancelling_needs_the_right_phone_number():
    reference = use("book_appointment", ASHA)["booked"]["reference"]
    assert "No booking matches" in use("cancel_booking", {"reference": reference, "phone": "408 555 9999"})["error"]
    assert use("cancel_booking", {"reference": reference, "phone": "4085550101"})["cancelled"]["reference"] == reference


def test_only_the_owners_day_schedule_shows_customers_phone_numbers():
    use("book_appointment", ASHA)
    [booking] = use("day_schedule", {"date": "2026-10-23"}, owner=True)["bookings"]
    assert booking["name"] == "Asha" and booking["phone"] == "(408) 555-0101"
