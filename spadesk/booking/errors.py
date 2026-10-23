"""The errors booking raises."""


class BookingError(ValueError):
    """A request that is well-formed but breaks a booking rule. The message is
    written for the customer: the chat passes it on, the form shows it."""


class CalendarUnavailable(RuntimeError):
    """The calendar could not be reached (no key, a refused key, no network).
    The message says why, for the server log; the customer is told only that
    booking is unavailable right now."""
