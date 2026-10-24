"""The errors booking raises."""


class BookingError(ValueError):
    """A request that is well-formed but breaks a booking rule. The message is
    written for the customer: the chat passes it on, the form shows it."""


# What the customer is told when the calendar cannot be reached (the form, and Pebby's tools).
UNAVAILABLE = "Booking is unavailable right now. Please call the spa to book or cancel."


class CalendarUnavailable(RuntimeError):
    """The calendar could not be reached (no key, a refused key, no network).
    The message says why, for the server log; the customer is told only that
    booking is unavailable right now."""
