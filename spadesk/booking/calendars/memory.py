"""A calendar kept in this process's memory: for tests and for working offline.

Chosen with SPADESK_CALENDAR=memory (the default until Google Calendar is
set up). Bookings are lost when the server restarts.
"""

from datetime import datetime

from .base import Booking


class MemoryCalendar:
    """A Calendar (see base.py) that keeps bookings in a dictionary."""

    def __init__(self) -> None:
        self._bookings: dict[str, Booking] = {}  # by reference

    def busy(self, therapist: str, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """The therapist's bookings that overlap `start`..`end`."""
        return [(b.start, b.end) for b in self.bookings(start, end, therapist)]

    def add(self, booking: Booking) -> None:
        """Keep `booking`."""
        self._bookings[booking.id] = booking

    def find(self, booking_id: str) -> Booking | None:
        """The booking with this reference, or None."""
        return self._bookings.get(booking_id)

    def remove(self, booking_id: str) -> None:
        """Forget the booking with this reference."""
        self._bookings.pop(booking_id, None)

    def bookings(self, start: datetime, end: datetime, therapist: str | None = None) -> list[Booking]:
        """The bookings that overlap `start`..`end`, earliest first."""
        found = [b for b in self._bookings.values()
                 if b.start < end and b.end > start and therapist in (None, b.therapist)]
        return sorted(found, key=lambda b: b.start)
