"""Step one of every conversation: the customer's name and phone number.

Until both are known, every message goes through this step, not to the
agent: it is a fixed step of the graph, so the model cannot skip it. A
small, fast model reads the message for a name, a phone number, and what
the customer asked for besides (`asked`): telling Pebby that is its job, so
the request is answered once they have given their details.

On a phone call the number calling (caller ID) is taken as the phone
number, so only the name is asked for, and the caller is told which number
will be kept; a number they give instead wins.

Once the name and number are known, the number's upcoming bookings are
looked up: if there are any, Pebby asks whether the customer is calling about
one of them or about something new, before anything else. The phone number is taken as given;
it only has to have at least 7 digits.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ...booking.appointments import digits, get_appointments
from ...booking.errors import BookingError, CalendarUnavailable
from ...models import structured
from ..booking.yes import when
from ..rules import READER_MODEL

READ = """You read one message a customer sent to a spa's website chat, which asked for \
their name and phone number. Take out what the message gives. Do not guess."""


class Details(BaseModel):
    """What one message says about who the customer is."""

    name: str = Field(description="The customer's name, as they wrote it, if the message gives it; otherwise "
                                  "empty.")
    phone: str = Field(description="The phone number, as written, if the message gives one; otherwise empty.")
    asked: str = Field(default="", description="What the message asks for besides a greeting, a name or a "
                                               "phone number, in the customer's own words, even a few words "
                                               "(e.g. 'silent facial', 'what time do you open?'); empty if "
                                               "it is only a greeting, a name or a number.")


def read_details(message: str, asked: str = "") -> Details:
    """The name, phone number and request in one message, read by the model.
    `asked`: the question the message answers, so a reply to "your name?" is
    read as the name, however unusual."""
    context = f"\n\nThe message answers this question: {asked}" if asked else ""
    return structured(READER_MODEL, [SystemMessage(READ + context), HumanMessage(message)], Details).reply


def valid_phone(phone: str) -> bool:
    """At least 7 digits, however written: the only check (numbers are not verified)."""
    return len(digits(phone)) >= 7


def ask_for(name: str, phone: str, caller: str = "") -> str:
    """What to ask for, given what is known so far. `caller` is the number
    calling, on a phone call."""
    if not name and caller and phone == caller:
        return (f"May I have your name, please? I'll keep the number you're calling from, ending in "
                f"{digits(caller)[-4:]}, unless you'd like to give another.")
    if not name and not phone:
        return "Before we start, may I have your name and phone number?"
    if not phone:
        return f"Thank you, {name}! And a phone number we can reach you on?"
    return "Thank you! And your name, please?"


def about_bookings(name: str, phone: str, spoken: bool) -> str:
    """The question to ask when this number has upcoming bookings ("" when it
    has none, or the calendar cannot be read: then Pebby simply carries on)."""
    try:
        upcoming = get_appointments().upcoming(phone)
    except (BookingError, CalendarUnavailable):
        return ""
    if not upcoming:
        return ""
    first, others = upcoming[0], len(upcoming) - 1
    more = f", and {others} more booking{'s' if others > 1 else ''}" if others else ""
    return (f"Thank you, {name}! I can see your {first.service} with {first.therapist} on "
            f"{when(first.start, spoken)}{more}. Is this about {'that booking' if not others else 'a booking'}, "
            "or something new?")
