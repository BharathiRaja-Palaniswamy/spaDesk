"""The error a booking rule raises."""


class BookingError(ValueError):
    """A request that is well-formed but breaks a booking rule. The message is
    written for the customer: the chat passes it on, the form shows it."""
