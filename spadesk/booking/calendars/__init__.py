"""Where bookings are kept: a pluggable calendar.

`get_calendar()` is the only way the rest of the app gets the calendar.
Which one it returns is a setting, SPADESK_CALENDAR:

  memory  in this process (memory.py); lost on restart. For tests and offline work.
  google  one Google calendar per therapist (google.py); needs GOOGLE_SERVICE_ACCOUNT

To add a calendar: write a class with the methods in base.py in its own file
here, and add one line to PROVIDERS below.
"""

from collections.abc import Callable
from functools import lru_cache

from ...config import get_settings
from .base import Booking, Calendar
from .google import make_google
from .memory import MemoryCalendar

__all__ = ["Booking", "Calendar", "get_calendar"]

# Setting value -> how to make that calendar.
PROVIDERS: dict[str, Callable[[], Calendar]] = {
    "memory": MemoryCalendar,
    "google": make_google,
}


@lru_cache  # one calendar for the whole server, so every request sees the same bookings
def get_calendar() -> Calendar:
    """The calendar named by the SPADESK_CALENDAR setting."""
    provider = get_settings().calendar
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown calendar: {provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    return PROVIDERS[provider]()
