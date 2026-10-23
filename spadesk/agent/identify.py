"""Step one of every conversation: the customer's name and phone number.

Until both are known, every message goes through this step, not to the
agent: it is a fixed step of the graph, so the model cannot skip it. A
small, fast model reads the message for a name, a phone number, and whether
the customer also asked for something (so that question can be answered
once they have given their details). The phone number is taken as given;
it only has to have at least 7 digits.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..models import structured

PROFILE = "cheap-router"  # a small, fast model: this is a quick reading job

READ = """You read one message a customer sent to a spa's website chat, which asked for \
their name and phone number. Take out what the message gives. Do not guess."""


class Details(BaseModel):
    """What one message says about who the customer is."""

    name: str = Field(description="The customer's name, as they wrote it, if the message gives it; otherwise "
                                  "empty.")
    phone: str = Field(description="The phone number, as written, if the message gives one; otherwise empty.")
    asks: bool = Field(description="True if the message also asks a question or asks for something (e.g. a "
                                   "booking, prices, hours), beyond giving a name or number.")


def read_details(message: str) -> Details:
    """The name, phone number and request in one message, read by the model."""
    return structured(PROFILE, [SystemMessage(READ), HumanMessage(message)], Details).reply


def valid_phone(phone: str) -> bool:
    """At least 7 digits, however written: the only check (numbers are not verified)."""
    return len(re.sub(r"\D", "", phone)) >= 7


def ask_for(name: str, phone: str) -> str:
    """What to ask for, given what is known so far."""
    if not name and not phone:
        return "Before we start, may I have your name and phone number?"
    if not phone:
        return f"Thank you, {name}! And a phone number we can reach you on?"
    return "Thank you! And your name, please?"
