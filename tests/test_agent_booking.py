"""Tests for booking requests (spadesk/agent/booking.py): a BookingRequest
in, a BookingResult out; code for complete requests, the model only for notes.

The model is a stand-in that plays back prepared answers (tests/standins.py),
so nothing is spent. The real document's schedule, an in-memory calendar and
a fixed clock (Thursday 22 October 2026, 2:10 PM).

Run with:  .venv/bin/python -m pytest tests/test_agent_booking.py
"""

import json
from datetime import datetime

import pytest
from standin_calendar import standin_desk
from standins import use_standin

from evals.booking import CASES, grade, stale, summary
from spadesk.agent import booking
from spadesk.agent.booking import (
    BookingRequest,
    handle,
    missing_result,
)

ASHA = "(408) 555-0101"
BRIDAL = ["Brightening Vitamin C Facial", "Glow Express Facial", "Eyebrow Wax"]


@pytest.fixture
def desk(monkeypatch):
    """One in-memory calendar with a fixed clock, for the agent and its tools."""
    desk = standin_desk()
    for module in (booking,):
        monkeypatch.setattr(module, "get_appointments", lambda: desk)
    return desk


def request(**fields) -> BookingRequest:
    return BookingRequest(**{"intent": "new", **fields})


# ---- in code: no model call ----------------------------------------------------------------

def test_what_is_missing_comes_with_the_days_that_fit(desk, monkeypatch):
    sent = use_standin(monkeypatch, [])
    result = missing_result(request(services=BRIDAL, package="Bridal Glow Prep Package"), ["day", "time"], ASHA, desk)
    assert result.status == "needs_info" and result.missing == ["day", "time"]
    assert result.options[:2] == ["Thursday 2026-10-22", "Wednesday 2026-10-28"]
    assert sent == []  # answered in code


def test_which_days_are_free_is_answered_with_the_days_that_fit(desk):
    result = handle(request(intent="availability", services=BRIDAL), ASHA, desk)
    assert result.status == "options" and "Wednesday 2026-10-28" in result.options


def test_a_package_on_a_day_gets_a_plan_of_exactly_its_services(desk):
    result = handle(request(services=BRIDAL, day="2026-10-29"), ASHA, desk)
    assert result.status == "plan_ready"
    assert [(i.service, f"{i.start:%H:%M}") for i in result.plan.items] == [
        ("Brightening Vitamin C Facial", "10:00"), ("Glow Express Facial", "11:00"), ("Eyebrow Wax", "11:30")]


def test_a_day_that_does_not_work_says_why_and_offers_other_days(desk):
    result = handle(request(services=BRIDAL, day="2026-10-27"), ASHA, desk)  # Hannah is off on Tuesdays
    assert result.status == "not_possible" and "Glow Express Facial" in result.reason
    assert result.reason.endswith("We are open 10:00 to 20:00 on Tuesdays.")  # the fact, so nothing is guessed
    assert "Wednesday 2026-10-28" in result.options


def test_a_chosen_therapist_is_kept(desk):
    result = handle(request(services=["Swedish Massage"], day="2026-10-30", therapist="Priya Nair"), ASHA, desk)
    assert result.plan.items[0].therapist == "Priya Nair"
    assert handle(request(services=["Swedish Massage"], therapist="Nobody"), ASHA, desk).status == "not_possible"


def test_availability_gives_times_not_a_plan(desk):
    result = handle(request(intent="availability", services=["Prenatal Massage"], day="2026-10-30"), ASHA, desk)
    assert result.status == "options" and result.options[0].startswith("Prenatal Massage on Friday 30 October: 10:00")


def test_their_bookings_and_a_cancellation_to_approve(desk):
    mine = desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha", ASHA)
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 16, 0), "Ben", "408 555 0102")
    assert [b["reference"] for b in handle(request(intent="my_bookings"), ASHA, desk).bookings] == [mine.id]
    result = handle(request(intent="cancel", references=[mine.id]), ASHA, desk)
    assert result.status == "cancel_ready" and [b["reference"] for b in result.cancel] == [mine.id]
    assert handle(request(intent="cancel", references=[mine.id]), "408 555 9999", desk).status == "not_possible"


def test_cancelling_part_of_a_package_goes_to_the_staff_and_all_of_it_can_be_cancelled(desk):
    package = "Bridal Glow Prep Package"
    made = [desk.book(service, therapist, datetime(2026, 10, 29, hour, minute), "Asha", ASHA, package=package)
            for service, therapist, hour, minute in [("Brightening Vitamin C Facial", "Sofia Martinez", 10, 0),
                                                      ("Glow Express Facial", "Hannah Lee", 11, 0),
                                                      ("Eyebrow Wax", "Hannah Lee", 11, 30)]]
    part = handle(request(intent="cancel", references=[made[1].id]), ASHA, desk)
    assert part.status == "handoff" and "part of your Bridal Glow Prep Package" in part.reason
    whole = handle(request(intent="cancel", references=[b.id for b in made]), ASHA, desk)
    assert whole.status == "cancel_ready" and len(whole.cancel) == 3
    # Changing one of them is changing part of it too.
    change = handle(request(intent="change", references=[made[2].id], services=["Eyebrow Wax"], day="2026-10-29"),
                    ASHA, desk)
    assert change.status == "handoff"


def test_a_change_without_a_reference_comes_with_their_bookings_to_choose_from(desk):
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha", ASHA)
    result = missing_result(request(intent="change", services=["Prenatal Massage"]), ["references"], ASHA, desk)
    assert result.missing == ["references"] and len(result.bookings) == 1


def test_a_change_names_the_booking_it_replaces(desk):
    old = desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 11, 0), "Asha", ASHA)
    result = handle(request(intent="change", references=[old.id], services=["Prenatal Massage"], day="2026-10-30",
                            after="15:00"), ASHA, desk)
    assert result.status == "plan_ready" and result.replaces["reference"] == old.id
    assert result.replaces["service"] == "Prenatal Massage"  # what is cancelled, to say so in the question


# ---- with notes: the model thinks, with the code tools ---------------------------------------

def test_the_booking_eval_cases_match_the_document_and_its_layers_grade_as_meant():
    # evals/booking.py spends real tokens; its cases and grading are checked here for free.
    assert stale(json.loads(CASES.read_text())) == []
    case = {"turns": ["a prenatal massage after 3pm"],
            "routing": {"tool": "booking", "intent": ["new"], "not_tools": ["hand_off"]},
            "extraction": {"services": ["Prenatal Massage"], "time_said_has": "3", "no_day_said": True,
                           "only_named_services": True},
            "downstream": {"booked": 1}}
    good = [{"name": "booking", "args": {"intent": "new", "services": ["Prenatal Massage"], "time_said": "after 3pm"}}]
    assert grade(case, good, None, booked=1) == {"routing": [], "extraction": [], "downstream": []}
    wrong = [{"name": "hand_off", "args": {"reason": "?"}},
             {"name": "booking", "args": {"intent": "availability", "services": ["Swedish Massage"],
                                          "day_said": "today"}}]
    graded = grade(case, wrong, None, booked=0)
    assert graded["routing"] == ["used hand_off", "intent 'availability', expected one of ['new']"]
    assert graded["extraction"] == ["chose ['Swedish Massage'] for them", "day_said 'today', but the customer said none",
                                    "services ['Swedish Massage'], expected ['Prenatal Massage']",
                                    "time_said None, expected it to mention '3'"]
    assert graded["downstream"] == ["0 bookings made, expected 1"]
    assert summary([grade(case, good, None, 1), graded]) == [
        "routing      1/2  (50%)", "extraction   1/2  (50%)", "downstream   1/2  (50%)",
        "wrong path   1/2  (50%)", "all layers   1/2"]
