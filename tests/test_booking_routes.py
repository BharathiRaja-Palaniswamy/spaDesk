"""Tests for the booking form's routes (spadesk/booking/routes.py), through the
app, on the real document, an in-memory calendar and a fixed clock.

Run with:  .venv/bin/python -m pytest tests/test_booking_routes.py
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from spadesk import app as app_module
from spadesk.booking import routes
from spadesk.booking.appointments import Appointments
from spadesk.booking.calendars.memory import MemoryCalendar
from spadesk.booking.errors import CalendarUnavailable
from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge

SPA = ZoneInfo("America/Los_Angeles")
ASHA = {"service": "Prenatal Massage", "therapist": "Priya Nair", "start": "2026-10-30T15:00",
        "name": "Asha Rao", "phone": "(408) 555-0101"}


@pytest.fixture
def client(monkeypatch):
    """The app, booking on one in-memory calendar, with "now" fixed on Thursday 22 October 2026."""
    desk = Appointments(read_schedule(get_knowledge()), MemoryCalendar(), SPA, 30,
                        now=lambda: datetime(2026, 10, 22, 14, 10, tzinfo=SPA))
    monkeypatch.setattr(routes, "get_appointments", lambda: desk)
    return TestClient(app_module.app)


def test_the_options_list_each_bookable_service_and_who_offers_it(client):
    services = {s["name"]: s for s in client.get("/api/booking/options").json()["services"]}
    assert services["Swedish Massage"]["minutes"] == [60, 90]
    assert services["Swedish Massage"]["therapists"] == ["Daniel Ruiz", "Priya Nair"]
    assert "Vinyasa Flow" not in services  # a group class is not booked here


def test_free_times_come_back_by_therapist(client):
    free = client.get("/api/booking/free", params={"service": "Prenatal Massage", "day": "2026-10-30"}).json()
    assert list(free["free"]) == ["Priya Nair"] and free["free"]["Priya Nair"][0] == "10:00"


def test_booking_returns_a_reference_and_the_time_is_then_taken(client):
    booked = client.post("/api/bookings", json=ASHA).json()
    assert booked["reference"].startswith("BK-") and booked["end"] == "2026-10-30T16:00"
    free = client.get("/api/booking/free", params={"service": "Prenatal Massage", "day": "2026-10-30"}).json()
    assert "15:00" not in free["free"]["Priya Nair"]
    taken = client.post("/api/bookings", json={**ASHA, "name": "Ben"})
    assert taken.status_code == 422 and "not free" in taken.json()["detail"]


def test_a_broken_rule_is_a_readable_422(client):
    wrong = client.post("/api/bookings", json={**ASHA, "therapist": "Daniel Ruiz"})
    assert wrong.status_code == 422 and "does not offer" in wrong.json()["detail"]
    missing = client.post("/api/bookings", json={**ASHA, "phone": ""})
    assert missing.status_code == 422  # pydantic's own check: the phone is too short


def test_a_calendar_that_cannot_be_reached_is_a_503_with_a_general_message(client, monkeypatch):
    def down():
        raise CalendarUnavailable("Google Calendar could not be reached: timed out")
    monkeypatch.setattr(routes, "get_appointments", down)
    reply = client.get("/api/booking/free", params={"service": "Prenatal Massage", "day": "2026-10-30"})
    assert reply.status_code == 503 and reply.json()["detail"] == routes.UNAVAILABLE
