"""The therapists' calendars in Google Calendar, through the Calendar API.

Chosen with SPADESK_CALENDAR=google. Each therapist has their own calendar,
named after the spa and the therapist ("Pebble Spa – Priya Nair"), so the
calendars are found by name and nothing else needs to be configured.

When the app starts using Google Calendar, it creates the calendar of anyone
in the team who has none yet (someone who just joined). If
SPADESK_CALENDAR_OWNER names a Google account, each new calendar is shared
with it: that person sees all of them side by side in Google Calendar and
can add time off to any of them. Without it, the calendars work just the
same, but nobody can open them in Google Calendar. Someone who has left
the team keeps their calendar: no new bookings are offered with them, but
their bookings can still be found, listed and cancelled.

The app signs in as a service account: a Google account for a program, not
a person, with its own key (GOOGLE_SERVICE_ACCOUNT, the key file's JSON
text, or the path to the file). It owns the calendars it creates, so no
person has to sign in from the server.

How the Calendar interface (base.py) maps onto Google:
  busy      Google's free/busy query: every event on the therapist's
            calendar, ours or not, so time off added by hand counts too
  add       a new event, with our reference and the customer's details in
            its private properties, which the app searches by, and in its
            description, for whoever opens it in Google Calendar
  find      a search for the event whose private "reference" is the one given
  remove    deleting that event
  bookings  the events with our private "booked_by" mark, between two times

Any failure to reach Google (no key, a refused key, no network) is raised
as CalendarUnavailable (errors.py).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from google.auth.exceptions import GoogleAuthError
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2 import HttpLib2Error

from ...business.store import get_knowledge
from ...config import get_settings
from ..errors import CalendarUnavailable
from ..schedule import read_schedule
from .base import Booking

SCOPES = ["https://www.googleapis.com/auth/calendar"]  # read and write calendars
MARK = ("booked_by", "spadesk")  # a private property on every event the app books
FAILURES = (HttpError, GoogleAuthError, HttpLib2Error, OSError, ValueError)


def calendar_name(business: str, therapist: str) -> str:
    """The calendar a therapist's bookings go on: 'Pebble Spa – Priya Nair'."""
    return f"{business} – {therapist}"


def connect() -> Any:
    """A Calendar API client signed in as the service account in GOOGLE_SERVICE_ACCOUNT."""
    key = get_settings().google_service_account.strip()
    if not key:
        raise CalendarUnavailable("SPADESK_CALENDAR is google but GOOGLE_SERVICE_ACCOUNT is not set. "
                                  "See docs/google-calendar.md.")
    try:
        # The key file's JSON text (a host's secret), or the path to the file (on a laptop).
        info = json.loads(key if key.startswith("{") else Path(key).read_text(encoding="utf-8"))
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        return build("calendar", "v3", credentials=credentials, cache_discovery=False)
    except FAILURES as error:
        raise CalendarUnavailable(f"Could not read the Google service account key: {error}") from error


def run(request: Any) -> dict[str, Any]:
    """Send one prepared API request, turning any failure into CalendarUnavailable."""
    try:
        return request.execute()
    except FAILURES as error:
        raise CalendarUnavailable(f"Google Calendar could not be reached: {error}") from error


def own_calendars(api: Any) -> dict[str, str]:
    """Every calendar the service account has, as name -> id."""
    calendars, page = {}, None
    while True:
        found = run(api.calendarList().list(pageToken=page))
        calendars |= {entry["summary"]: entry["id"] for entry in found.get("items", [])}
        if not (page := found.get("nextPageToken")):
            return calendars


class GoogleCalendar:
    """A Calendar (see base.py) backed by one Google calendar per therapist."""

    def __init__(self, api: Any, calendars: dict[str, str], zone: ZoneInfo) -> None:
        """`calendars`: therapist name -> Google calendar id."""
        self.api, self.calendars, self.zone = api, calendars, zone

    def busy(self, therapist: str, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Everything on the therapist's calendar between `start` and `end`."""
        calendar = self.calendars[therapist]
        found = run(self.api.freebusy().query(body={
            "timeMin": start.isoformat(), "timeMax": end.isoformat(), "items": [{"id": calendar}]}))
        return [(self._time(period["start"]), self._time(period["end"]))
                for period in found["calendars"][calendar]["busy"]]

    def add(self, booking: Booking) -> None:
        """A new event on the therapist's calendar."""
        zone = str(self.zone)
        run(self.api.events().insert(calendarId=self.calendars[booking.therapist], body={
            "summary": f"{booking.service} – {booking.name}",
            "description": f"Booked through the spa's site.\nReference: {booking.id}\nPhone: {booking.phone}",
            "start": {"dateTime": booking.start.isoformat(), "timeZone": zone},
            "end": {"dateTime": booking.end.isoformat(), "timeZone": zone},
            "extendedProperties": {"private": {
                MARK[0]: MARK[1], "reference": booking.id, "service": booking.service,
                "customer": booking.name, "phone": booking.phone}},
        }))

    def find(self, booking_id: str) -> Booking | None:
        """The booking with this reference, on whichever calendar it is."""
        return next(iter(self._search(privateExtendedProperty=f"reference={booking_id}")), None)

    def remove(self, booking_id: str) -> None:
        """Delete the event for this booking."""
        for therapist, event in self._events(privateExtendedProperty=f"reference={booking_id}"):
            run(self.api.events().delete(calendarId=self.calendars[therapist], eventId=event["id"]))

    def bookings(self, start: datetime, end: datetime, therapist: str | None = None) -> list[Booking]:
        """The app's bookings between `start` and `end`, earliest first."""
        found = self._search(therapist, timeMin=start.isoformat(), timeMax=end.isoformat(),
                             privateExtendedProperty=f"{MARK[0]}={MARK[1]}")
        return sorted(found, key=lambda booking: booking.start)

    # ---- reading events ----------------------------------------------------

    def _events(self, therapist: str | None = None, **query: str) -> list[tuple[str, dict[str, Any]]]:
        """(therapist, event) for every event matching `query`, on one calendar or all."""
        names = [therapist] if therapist else list(self.calendars)
        events = []
        for name in names:
            page = None
            while True:
                found = run(self.api.events().list(calendarId=self.calendars[name], singleEvents=True,
                                                   pageToken=page, **query))
                events += [(name, event) for event in found.get("items", [])]
                if not (page := found.get("nextPageToken")):
                    break
        return events

    def _search(self, therapist: str | None = None, **query: str) -> list[Booking]:
        """The bookings among the events matching `query`."""
        return [self._booking(name, event) for name, event in self._events(therapist, **query)]

    def _booking(self, therapist: str, event: dict[str, Any]) -> Booking:
        """A Booking from one of the app's events."""
        private = event["extendedProperties"]["private"]
        return Booking(private["reference"], private["service"], therapist,
                       self._time(event["start"]["dateTime"]), self._time(event["end"]["dateTime"]),
                       private["customer"], private["phone"])

    def _time(self, text: str) -> datetime:
        """A time from Google ('2026-10-23T17:00:00Z') in the spa's time zone."""
        return datetime.fromisoformat(text).astimezone(self.zone)


def share(api: Any, calendar: str, person: str) -> None:
    """Make `person` (a Google account) an owner of `calendar`: they see it,
    can add time off to it, and can share it with its therapist. Sharing
    again with the same person changes nothing."""
    run(api.acl().insert(calendarId=calendar, body={"role": "owner", "scope": {"type": "user", "value": person}}))


def ensure_calendars(api: Any) -> tuple[dict[str, str], list[str]]:
    """The spa's calendars as therapist -> id, after creating any the team is
    missing (each shared with SPADESK_CALENDAR_OWNER, if set). Also returns
    the names of the therapists whose calendar was just created.

    Every calendar named after the spa is included, the team's and those of
    people who have left, so no booking is lost when the team changes."""
    kb, settings = get_knowledge(), get_settings()
    prefix = calendar_name(kb.name, "")
    calendars = {name.removeprefix(prefix): cid for name, cid in own_calendars(api).items()
                 if name.startswith(prefix)}
    created = [person for person in read_schedule(kb).therapists if person not in calendars]
    for person in created:
        calendars[person] = run(api.calendars().insert(body={
            "summary": calendar_name(kb.name, person), "timeZone": settings.timezone}))["id"]
        if settings.calendar_owner:
            share(api, calendars[person], settings.calendar_owner)
    return calendars, created


def make_google() -> GoogleCalendar:
    """The spa's Google calendars, one per therapist, created where missing."""
    api = connect()
    calendars, _ = ensure_calendars(api)
    return GoogleCalendar(api, calendars, ZoneInfo(get_settings().timezone))
