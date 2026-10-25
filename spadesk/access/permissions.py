"""Who may do what: one table, checked everywhere.

There are two roles. Anyone using the customer site (the chat, the booking
form) or calling the phone line is a customer; the owner is whoever has
signed in on the manage page with the owner's password (session.py).

Each role has a fixed set of permissions. A route that needs one says so
(`require` in session.py), and a tool the model may be offered names the
permission it needs (agent/pebby/tools.py), so an action is allowed only when the
table below allows it, not because of which page or route happened to
offer it.

What each permission covers:

  search_guide   the business document (the same for everyone: it holds no
                 owner-only facts; the reports data is not in it)
  contact_staff  ask the staff to call back (a handoff)
  book           find free times, book, list and cancel bookings, for the
                 customer's own phone number only (booking/appointments.py)
  reports        revenue, bookings, channels and feedback figures, and the
                 owner's assistant that reads them
  voice_test     the phone-line test widget on the manage page
"""

from enum import Enum


class Role(str, Enum):
    """Who is asking."""

    CUSTOMER = "customer"
    OWNER = "owner"


PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.CUSTOMER: frozenset({"search_guide", "contact_staff", "book"}),
    Role.OWNER: frozenset({"search_guide", "contact_staff", "book", "reports", "voice_test"}),
}


def allowed(role: Role, permission: str) -> bool:
    """True when `role` has `permission`. A permission not in the table is never allowed."""
    return permission in PERMISSIONS[role]
