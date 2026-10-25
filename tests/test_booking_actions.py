"""Tests for the booking actions (spadesk/booking/actions.py), run as the
Booking agent's model would run them (spadesk/agent/booking/agent.py,
spadesk/agent/booking/agent.py): what a model sends and gets back, on the
real document, an in-memory calendar and a fixed clock.

Run with:  .venv/bin/python -m pytest tests/test_booking_actions.py
"""

import json
from datetime import datetime

import pytest
from standin_calendar import standin_desk

from spadesk.agent.booking import executor as approved
from spadesk.agent.booking.agent import free_times_tool
from spadesk.agent.booking.plan import days_that_fit
from spadesk.agent.pebby.tools import agent_tools
from spadesk.agent.rules import MAX_DAYS
from spadesk.booking import actions
from spadesk.models.tools import run_requests

FRIDAY_10 = datetime(2026, 10, 23, 10, 0)


@pytest.fixture(autouse=True)
def desk(monkeypatch):
    """The actions book on one in-memory calendar, with "now" fixed on Thursday 22 October 2026."""
    desk = standin_desk(notice=0)
    for module in (actions, approved):
        monkeypatch.setattr(module, "get_appointments", lambda: desk)
    return desk


def free(args):
    """Run find_free_times as a model's request would, and read back what the model would see."""
    [message] = run_requests([{"name": "find_free_times", "args": args, "id": "call-1"}], [free_times_tool()])
    return json.loads(message.content)


def test_the_tool_description_lists_the_documents_services_and_team():
    find = free_times_tool().parameters["properties"]
    assert "Prenatal Massage" in find["service"]["enum"] and "Priya Nair" in find["therapist"]["enum"]


def test_free_times_come_back_by_therapist():
    times = free({"service": "Swedish Massage", "date": "2026-10-23"})["free"]
    assert list(times) == ["Priya Nair"] and times["Priya Nair"][0] == "10:00"


def test_pebby_cannot_book_or_cancel_directly_only_ask_the_booking_agent():
    names = [t.name for t in agent_tools("Asha", "4085550101")]
    assert names == ["search_guide", "booking", "record_feedback", "hand_off"]


def test_a_broken_rule_or_a_bad_date_comes_back_as_an_error_the_model_can_read():
    assert "does not offer" in free({"service": "Prenatal Massage", "therapist": "Daniel Ruiz",
                                     "date": "2026-10-23"})["error"]
    assert "YYYY-MM-DD" in free({"service": "Swedish Massage", "date": "next Friday"})["error"]


def test_a_customer_can_only_cancel_their_own_booking(desk):
    reference = desk.book("Prenatal Massage", "Priya Nair", FRIDAY_10, "Asha", "(408) 555-0101").id
    assert "No booking matches" in approved.cancel([reference], "408 555 9999")["not_cancelled"][0]
    assert approved.cancel([reference], "(408) 555-0101")["cancelled"][0]["reference"] == reference


def test_days_that_fit_are_the_days_every_service_has_a_free_time(desk):
    # The Bridal Glow Prep Package: Sofia (Vitamin C) works Tue-Sat, Hannah (the rest) Mon, Wed, Thu, Sun.
    package = ["Brightening Vitamin C Facial", "Glow Express Facial", "Eyebrow Wax"]
    days = days_that_fit(package, desk)
    assert days[:2] == ["Thursday 2026-10-22", "Wednesday 2026-10-28"]
    assert all(day.split()[0] in ("Wednesday", "Thursday") for day in days) and len(days) == MAX_DAYS


def test_a_day_whose_times_are_all_taken_is_not_offered(desk):
    for hour in range(10, 20):  # Hannah's whole Wednesday is booked
        desk.book("Eyebrow Wax", "Hannah Lee", datetime(2026, 10, 28, hour, 0), "Ben", "408 555 0102")
    days = days_that_fit(["Brightening Vitamin C Facial", "Eyebrow Wax"], desk)
    assert "Wednesday 2026-10-28" not in days
