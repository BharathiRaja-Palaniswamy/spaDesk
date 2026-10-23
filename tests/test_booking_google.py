"""Tests for the Google Calendar adapter (spadesk/booking/calendars/google.py),
with a stand-in for Google's Calendar API: no account, no network.

The stand-in keeps calendars and their events and answers the requests the
adapter sends (calendar list, calendar insert, sharing, free/busy, events
insert, list and delete), so the booking rules can run on it end to end,
including an event added by hand in Google Calendar (time off), which has
none of our marks.

Run with:  .venv/bin/python -m pytest tests/test_booking_google.py
"""

from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from googleapiclient.errors import HttpError

from spadesk.booking import tools
from spadesk.booking.appointments import Appointments
from spadesk.booking.calendars import google
from spadesk.booking.calendars.google import GoogleCalendar, calendar_name
from spadesk.booking.errors import CalendarUnavailable
from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge

SPA = ZoneInfo("America/Los_Angeles")
FRIDAY = date(2026, 10, 23)


def at(clock):
    hour, minute = map(int, clock.split(":"))
    return datetime(2026, 10, 23, hour, minute, tzinfo=SPA)


class Request:
    """A prepared request: runs when `execute()` is called, as Google's do."""

    def __init__(self, answer):
        self.answer = answer

    def execute(self):
        return self.answer() if callable(self.answer) else self.answer


class FakeApi:
    """Stands in for the Calendar API client, with calendars it already has."""

    def __init__(self, calendars: dict[str, str]):
        self.names = calendars  # name -> id
        self.stored: dict[str, list[dict]] = {cid: [] for cid in calendars.values()}
        self.fail = False
        self.shared: list[tuple[str, str]] = []  # (calendar id, person)

    # api.calendars().insert(body=...)
    def calendars(self):
        def insert(body):
            cid = f"cal-new-{len(self.names)}"
            self.names[body["summary"]] = cid
            self.stored[cid] = []
            return Request({"id": cid})
        return SimpleNamespace(insert=insert)

    # api.acl().insert(calendarId=..., body=...)
    def acl(self):
        def insert(calendarId, body):
            self.shared.append((calendarId, body["scope"]["value"]))
            return Request({})
        return SimpleNamespace(insert=insert)

    # api.calendarList().list(...)
    def calendarList(self):  # noqa: N802 (Google's own name)
        items = [{"summary": name, "id": cid} for name, cid in self.names.items()]
        return SimpleNamespace(list=lambda pageToken=None: Request({"items": items}))

    # api.freebusy().query(body=...)
    def freebusy(self):
        def query(body):
            start, end = (datetime.fromisoformat(body[k]) for k in ("timeMin", "timeMax"))
            return Request(lambda: {"calendars": {item["id"]: {"busy": [
                {"start": e["start"]["dateTime"], "end": e["end"]["dateTime"]}
                for e in self._overlapping(item["id"], start, end)]} for item in body["items"]}})
        return SimpleNamespace(query=query)

    # api.events().insert / list / delete
    def events(self):
        def insert(calendarId, body):
            def run():
                if self.fail:
                    raise HttpError(SimpleNamespace(status=503, reason="Unavailable"), b"")
                self.stored[calendarId].append({**body, "id": f"event-{sum(map(len, self.stored.values()))}"})
                return {}
            return Request(run)

        def list_(calendarId, singleEvents, pageToken=None, timeMin=None, timeMax=None, privateExtendedProperty=None):
            key, _, value = (privateExtendedProperty or "").partition("=")
            found = self.stored[calendarId]
            if timeMin:
                found = self._overlapping(calendarId, datetime.fromisoformat(timeMin), datetime.fromisoformat(timeMax))
            if key:
                found = [e for e in found if e.get("extendedProperties", {}).get("private", {}).get(key) == value]
            return Request({"items": found})

        def delete(calendarId, eventId):
            return Request(lambda: self.stored[calendarId].remove(
                next(e for e in self.stored[calendarId] if e["id"] == eventId)))
        return SimpleNamespace(insert=insert, list=list_, delete=delete)

    def _overlapping(self, calendar, start, end):
        return [e for e in self.stored[calendar]
                if datetime.fromisoformat(e["start"]["dateTime"]) < end
                and datetime.fromisoformat(e["end"]["dateTime"]) > start]


TEAM = list(read_schedule(get_knowledge()).therapists)


@pytest.fixture
def api():
    return FakeApi({calendar_name("Pebble Spa", name): f"cal-{n}" for n, name in enumerate(TEAM)})


@pytest.fixture
def desk(api):
    calendar = GoogleCalendar(api, {name: f"cal-{n}" for n, name in enumerate(TEAM)}, SPA)
    return Appointments(read_schedule(get_knowledge()), calendar, SPA, 30,
                        now=lambda: datetime(2026, 10, 22, 14, 10, tzinfo=SPA))


def priya(api):
    return api.stored[f"cal-{TEAM.index('Priya Nair')}"]


def test_a_booking_becomes_an_event_on_the_therapists_own_calendar(api, desk):
    booking = desk.book("Prenatal Massage", "Priya Nair", at("10:00"), "Asha", "(408) 555-0101")
    [event] = priya(api)
    assert event["summary"] == "Prenatal Massage – Asha"
    assert event["start"] == {"dateTime": "2026-10-23T10:00:00-07:00", "timeZone": "America/Los_Angeles"}
    assert event["extendedProperties"]["private"]["reference"] == booking.id


def test_bookings_read_back_as_they_were_made_and_can_be_cancelled(api, desk):
    booking = desk.book("Prenatal Massage", "Priya Nair", at("10:00"), "Asha", "(408) 555-0101")
    assert desk.day_bookings(FRIDAY) == [booking]
    assert desk.cancel(booking.id, "4085550101") == booking
    assert priya(api) == [] and desk.day_bookings(FRIDAY) == []


def test_time_off_added_in_google_calendar_blocks_those_times_but_is_not_a_booking(api, desk):
    # The owner adds "Priya – dentist" from 12:00 to 14:00 directly in Google Calendar.
    priya(api).append({"id": "by-hand", "summary": "Priya – dentist",
                       "start": {"dateTime": at("12:00").isoformat()}, "end": {"dateTime": at("14:00").isoformat()}})
    offered = [s.start.strftime("%H:%M") for s in desk.free_slots("Prenatal Massage", FRIDAY)]
    assert "11:00" in offered and "11:30" not in offered and "13:30" not in offered and "14:00" in offered
    assert desk.day_bookings(FRIDAY) == []


def test_google_failing_is_reported_as_calendar_unavailable_and_the_tool_says_so(api, desk, monkeypatch):
    api.fail = True
    with pytest.raises(CalendarUnavailable, match="could not be reached"):
        desk.book("Prenatal Massage", "Priya Nair", at("10:00"), "Asha", "(408) 555-0101")
    # Through the tool, the model gets a message it can pass on, not the reason.
    monkeypatch.setattr(tools, "get_appointments", lambda: desk)
    book = next(t for t in tools.booking_tools() if t.name == "book_appointment")
    reply = book.run({"service": "Prenatal Massage", "therapist": "Priya Nair", "start": "2026-10-23T10:00",
                      "name": "Asha", "phone": "4085550101"})
    assert reply == {"error": tools.UNAVAILABLE}


def test_someone_new_in_the_team_gets_a_calendar_shared_with_the_owner(api, monkeypatch):
    del api.names[calendar_name("Pebble Spa", "Hannah Lee")]  # as if Hannah just joined
    monkeypatch.setattr(google, "connect", lambda: api)
    monkeypatch.setenv("SPADESK_CALENDAR_OWNER", "owner@example.com")
    calendar = google.make_google()
    new = api.names[calendar_name("Pebble Spa", "Hannah Lee")]
    assert calendar.calendars["Hannah Lee"] == new and api.shared == [(new, "owner@example.com")]


def test_bookings_with_someone_who_left_the_team_are_still_found(api, monkeypatch):
    # A calendar named after the spa, for someone no longer in the document.
    api.names[calendar_name("Pebble Spa", "Ana Former")] = "cal-gone"
    api.stored["cal-gone"] = [{"id": "e1", "start": {"dateTime": at("10:00").isoformat()},
                               "end": {"dateTime": at("11:00").isoformat()},
                               "extendedProperties": {"private": {
                                   "booked_by": "spadesk", "reference": "BK-OLD001", "service": "Lip Wax",
                                   "customer": "Ben", "phone": "4085550102"}}}]
    monkeypatch.setattr(google, "connect", lambda: api)
    calendar = google.make_google()
    assert calendar.find("BK-OLD001").therapist == "Ana Former"
    assert [b.id for b in calendar.bookings(at("00:00"), at("23:59"))] == ["BK-OLD001"]


def test_without_a_key_the_error_says_what_to_do(monkeypatch):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT", "")
    with pytest.raises(CalendarUnavailable, match="GOOGLE_SERVICE_ACCOUNT is not set"):
        google.connect()
