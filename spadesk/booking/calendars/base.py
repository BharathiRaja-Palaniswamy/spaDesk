"""The interface every calendar follows, and the booking it keeps.

The spa has one calendar per therapist ("Pebble Spa – Priya Nair", ...). A
Calendar here stands for all of them together: every method names the
therapist whose calendar it reads or writes. The owner sees them side by
side in Google Calendar; the app never needs to.

A calendar holds what the document cannot: the bookings, and anything else
that makes a therapist busy (in Google Calendar, an event a therapist adds
for time off). It knows nothing about services, opening hours or booking
rules; those are in schedule.py and appointments.py, so a calendar is easy
to swap: in memory for tests (memory.py), Google Calendar on the site.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Booking:
    """One confirmed appointment."""

    id: str  # our reference, e.g. "BK-7F3A2C", told to the customer
    service: str
    therapist: str
    start: datetime  # with its time zone (the spa's)
    end: datetime
    name: str  # the customer's
    phone: str


class Calendar(Protocol):
    """Reads and writes the therapists' calendars."""

    def busy(self, therapist: str, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Every period between `start` and `end` when `therapist` is not free:
        their bookings, and any other event on their calendar."""
        ...

    def add(self, booking: Booking) -> None:
        """Put `booking` on its therapist's calendar."""
        ...

    def find(self, booking_id: str) -> Booking | None:
        """The booking with this reference, or None."""
        ...

    def remove(self, booking_id: str) -> None:
        """Take the booking with this reference off its calendar."""
        ...

    def bookings(self, start: datetime, end: datetime, therapist: str | None = None) -> list[Booking]:
        """The bookings between `start` and `end`, earliest first; one
        therapist's, or everyone's when `therapist` is None."""
        ...
