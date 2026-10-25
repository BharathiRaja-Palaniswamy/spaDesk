"""Tests for fit_plan (spadesk/agent/booking/plan.py): services fitted one after
another on a day, in code.

The real document's schedule, an in-memory calendar and a fixed clock
(Thursday 22 October 2026, 2:10 PM); no model is called.

Run with:  .venv/bin/python -m pytest tests/test_agent_fit.py
"""

from datetime import date, datetime, time

from standin_calendar import standin_desk

from spadesk.agent.booking.plan import Plan, check_plan, fit_plan

THURSDAY = date(2026, 10, 29)
BRIDAL = ["Brightening Vitamin C Facial", "Glow Express Facial", "Eyebrow Wax"]


def times(plan: Plan) -> list[tuple[str, str, str]]:
    """Each item as (service, therapist, "HH:MM")."""
    return [(i.service, i.therapist, f"{i.start:%H:%M}") for i in plan.items]


def test_a_package_is_fitted_back_to_back_from_opening():
    plan = fit_plan(BRIDAL, THURSDAY, standin_desk())
    assert times(plan) == [("Brightening Vitamin C Facial", "Sofia Martinez", "10:00"),
                           ("Glow Express Facial", "Hannah Lee", "11:00"),
                           ("Eyebrow Wax", "Hannah Lee", "11:30")]


def test_what_it_fits_always_passes_the_plan_check():
    desk = standin_desk()
    desk.book("Glow Express Facial", "Hannah Lee", datetime(2026, 10, 29, 11, 0), "Ben", "408 555 0102")
    plan = fit_plan(BRIDAL, THURSDAY, desk)
    assert check_plan(plan, desk) == []
    assert times(plan)[1] == ("Glow Express Facial", "Hannah Lee", "11:30")  # the next free time instead


def test_preferences_after_before_and_a_chosen_therapist():
    desk = standin_desk()
    late = fit_plan(BRIDAL, THURSDAY, desk, after=time(17, 0))
    assert [t for _, _, t in times(late)] == ["17:00", "18:00", "18:30"]
    assert isinstance(fit_plan(BRIDAL, THURSDAY, desk, before=time(11, 0)), list)  # 2h15 does not fit by 11
    friday = date(2026, 10, 30)  # Priya works Fridays
    massage = fit_plan(["Swedish Massage"], friday, desk, therapists={"Swedish Massage": "Priya Nair"})
    assert times(massage)[0][1] == "Priya Nair"


def test_a_day_a_service_is_not_offered_says_which():
    tuesday = date(2026, 10, 27)  # Hannah does not work on Tuesdays
    problems = fit_plan(BRIDAL, tuesday, standin_desk())
    assert problems == ["No free time for Glow Express Facial, Eyebrow Wax on Tuesday 27 October."]


def test_a_later_start_is_tried_when_the_rest_cannot_follow_an_early_one():
    desk = standin_desk()
    for hour in (11, 12):  # Hannah is busy from 11:00 to 13:00 ...
        for minute in (0, 30):
            desk.book("Glow Express Facial", "Hannah Lee", datetime(2026, 10, 29, hour, minute), "Ben", "4085550102")
    plan = fit_plan(BRIDAL, THURSDAY, desk)
    assert [t for _, _, t in times(plan)] == ["10:00", "13:00", "13:30"]  # ... so the facial waits for her
