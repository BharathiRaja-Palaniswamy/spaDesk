"""Where bookings are kept: the therapists' Google calendars.

`get_calendar()` is the only way the rest of the app gets the calendar. It is
always Google Calendar (google.py), one calendar per therapist, signed in
with GOOGLE_SERVICE_ACCOUNT. There is no second place: if Google cannot be
reached (no key, a refused key, no network), booking says so
(CalendarUnavailable) rather than quietly keeping bookings somewhere else.

The booking rules only use the methods in base.py, so another calendar
(Outlook, say) would be a new file with those methods, and tests use a
stand-in with them (tests/standin_calendar.py).
"""

from functools import lru_cache

from .base import Booking, Calendar
from .google import make_google

__all__ = ["Booking", "Calendar", "get_calendar"]


@lru_cache  # one calendar for the whole server; a failed attempt is not kept, so the next request tries again
def get_calendar() -> Calendar:
    """The spa's Google calendars."""
    return make_google()
