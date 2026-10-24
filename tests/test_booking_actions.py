"""Tests for the booking actions (spadesk/booking/actions.py), run as Pebby's
tools (spadesk/agent/tools.py) would run them: what a model sends and gets
back, on the real document, an in-memory calendar and a fixed clock.

Run with:  .venv/bin/python -m pytest tests/test_booking_actions.py
"""

import json

import pytest
from standin_calendar import standin_desk

from spadesk.agent.tools import agent_tools
from spadesk.booking import actions
from spadesk.models.tools import run_requests

PRENATAL = {"service": "Prenatal Massage", "therapist": "Priya Nair", "start": "2026-10-23T10:00"}


@pytest.fixture(autouse=True)
def desk(monkeypatch):
    """The actions book on one in-memory calendar, with "now" fixed on Thursday 22 October 2026."""
    desk = standin_desk(notice=0)
    monkeypatch.setattr(actions, "get_appointments", lambda: desk)
    return desk


def use(name, args, phone="(408) 555-0101"):
    """Run one of Pebby's tools for the customer with `phone`, as a model's request would,
    and read back what the model would see."""
    tools = agent_tools("Asha", phone)
    [message] = run_requests([{"name": name, "args": args, "id": "call-1"}], tools)
    return json.loads(message.content)


def test_the_tool_descriptions_list_the_documents_services_and_team():
    find = next(t for t in agent_tools("Asha", "4085550101") if t.name == "find_free_times").parameters["properties"]
    assert "Prenatal Massage" in find["service"]["enum"] and "Priya Nair" in find["therapist"]["enum"]


def test_free_times_come_back_by_therapist():
    free = use("find_free_times", {"service": "Swedish Massage", "date": "2026-10-23"})["free"]
    assert list(free) == ["Priya Nair"] and free["Priya Nair"][0] == "10:00"


def test_booking_returns_a_reference_and_the_time_is_then_taken(desk):
    booked = use("book_appointment", PRENATAL)["booked"]
    assert booked["reference"].startswith("BK-") and booked["end"] == "2026-10-23T11:00"
    assert "phone" not in booked
    # Booked under the customer the tools were made for, whatever the model sends.
    assert desk.calendar.find(booked["reference"]).phone == "(408) 555-0101"
    assert "10:00" not in use("find_free_times", {"service": "Prenatal Massage", "date": "2026-10-23"})["free"]["Priya Nair"]
    assert "not free" in use("book_appointment", PRENATAL, phone="408 555 0102")["error"]


def test_a_broken_rule_or_a_bad_date_comes_back_as_an_error_the_model_can_read():
    assert "does not offer" in use("book_appointment", {**PRENATAL, "therapist": "Daniel Ruiz"})["error"]
    assert "YYYY-MM-DD" in use("find_free_times", {"service": "Swedish Massage", "date": "next Friday"})["error"]


def test_a_customer_can_only_cancel_their_own_booking():
    reference = use("book_appointment", PRENATAL)["booked"]["reference"]
    assert "No booking matches" in use("cancel_booking", {"reference": reference}, phone="408 555 9999")["error"]
    assert use("cancel_booking", {"reference": reference})["cancelled"]["reference"] == reference
