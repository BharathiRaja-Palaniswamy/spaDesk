"""Tests for the Plan, the plan check and the Executor (spadesk/agent/booking.py, executor.py):
a Plan is booked all or nothing, re-checked at the yes, never double-booked.

The real document's schedule, a stand-in calendar and a fixed clock
(Thursday 22 October 2026, 2:10 PM); no model is called.

Run with:  .venv/bin/python -m pytest tests/test_agent_executor.py
"""

import threading
from datetime import datetime

from standin_calendar import SPA, standin_desk

from spadesk.agent.booking import Plan, PlanItem, check_plan
from spadesk.agent.executor import execute
from spadesk.booking.calendars import Booking

FRIDAY = datetime(2026, 10, 30).date()


def item(service, therapist, clock, minutes=None) -> PlanItem:
    hour, minute = map(int, clock.split(":"))
    return PlanItem(service=service, therapist=therapist, start=datetime(2026, 10, 30, hour, minute),
                    minutes=minutes)


MASSAGE_THEN_FACIAL = Plan(summary="A massage, then a facial, on Friday afternoon.", items=[
    item("Prenatal Massage", "Priya Nair", "14:00"), item("Signature Radiance Facial", "Sofia Martinez", "15:30")])


def someone_else_books(desk, service, therapist, clock):
    hour, minute = map(int, clock.split(":"))
    return desk.book(service, therapist, datetime(2026, 10, 30, hour, minute), "Ben", "408 555 0102")


# ---- the plan check ----------------------------------------------------------------------

def test_a_plan_of_free_times_has_no_problems():
    assert check_plan(MASSAGE_THEN_FACIAL, standin_desk()) == []


def test_the_check_names_each_item_that_cannot_be_booked():
    plan = Plan(summary="", items=[item("Prenatal Massage", "Daniel Ruiz", "14:00"),   # he doesn't offer it
                                   item("Signature Radiance Facial", "Sofia Martinez", "19:30")])  # after closing
    problems = check_plan(plan, standin_desk())
    assert len(problems) == 2 and "does not offer" in problems[0] and "not a free time" in problems[1]


def test_one_customer_cannot_be_in_two_appointments_at_once():
    plan = Plan(summary="", items=[item("Prenatal Massage", "Priya Nair", "14:00"),
                                   item("Signature Radiance Facial", "Sofia Martinez", "14:30")])
    assert "can't have both at once" in check_plan(plan, standin_desk())[0]


# ---- booking it ----------------------------------------------------------------------------

def test_an_approved_plan_is_booked_in_full():
    desk = standin_desk()
    outcome = execute(MASSAGE_THEN_FACIAL, "Asha Rao", "(408) 555-0101", desk)
    assert outcome.done and [b.therapist for b in outcome.booked] == ["Priya Nair", "Sofia Martinez"]
    assert all(b.phone == "(408) 555-0101" for b in desk.day_bookings(FRIDAY))


def test_a_time_taken_before_the_yes_books_nothing_and_says_which():
    desk = standin_desk()
    someone_else_books(desk, "Signature Radiance Facial", "Sofia Martinez", "15:30")  # while Asha was deciding
    outcome = execute(MASSAGE_THEN_FACIAL, "Asha Rao", "(408) 555-0101", desk)
    assert not outcome.done and "Signature Radiance Facial with Sofia Martinez" in outcome.problems[0]
    assert [b.name for b in desk.day_bookings(FRIDAY)] == ["Ben"]  # the massage was not booked alone


def test_a_failure_part_way_undoes_what_was_already_booked(monkeypatch):
    desk = standin_desk()
    real_book = desk.book

    def taken_in_between(service, therapist, *args):
        # The facial is taken in the instant after the plan was checked.
        if service == "Signature Radiance Facial":
            real_book(service, therapist, datetime(2026, 10, 30, 15, 30), "Ben", "408 555 0102")
        return real_book(service, therapist, *args)
    monkeypatch.setattr(desk, "book", taken_in_between)
    outcome = execute(MASSAGE_THEN_FACIAL, "Asha Rao", "(408) 555-0101", desk)
    assert not outcome.done and [b.name for b in desk.day_bookings(FRIDAY)] == ["Ben"]


def test_an_event_typed_in_by_hand_at_the_same_moment_is_caught_and_undone(monkeypatch):
    desk = standin_desk()
    real_add = desk.calendar.add

    def owner_adds_too(booking):
        real_add(booking)
        if booking.therapist == "Priya Nair":  # the owner blocks the same hour in Google Calendar
            real_add(Booking("BY-HAND", "Time off", "Priya Nair", booking.start, booking.end, "", ""))
    monkeypatch.setattr(desk.calendar, "add", owner_adds_too)
    outcome = execute(MASSAGE_THEN_FACIAL, "Asha Rao", "(408) 555-0101", desk)
    assert not outcome.done and "taken at the same moment" in outcome.problems[0]
    assert [b.id for b in desk.day_bookings(FRIDAY)] == ["BY-HAND"]  # Asha's bookings undone


def test_two_customers_approving_the_same_time_at_once_get_it_only_once():
    desk = standin_desk()
    plan = Plan(summary="", items=[item("Prenatal Massage", "Priya Nair", "14:00")])
    results = []

    def approve(n):
        results.append(execute(plan, f"Guest {n}", f"40855501{n:02}", desk))
    customers = [threading.Thread(target=approve, args=(n,)) for n in range(5)]
    for thread in customers:
        thread.start()
    for thread in customers:
        thread.join()
    assert sum(outcome.done for outcome in results) == 1
    assert len(desk.day_bookings(FRIDAY)) == 1


def test_the_spas_time_zone_is_used_for_a_time_given_without_one():
    outcome = execute(Plan(summary="", items=[item("Prenatal Massage", "Priya Nair", "14:00")]),
                      "Asha Rao", "(408) 555-0101", standin_desk())
    assert outcome.booked[0].start == datetime(2026, 10, 30, 14, 0, tzinfo=SPA)
