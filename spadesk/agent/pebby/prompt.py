"""What the booking agent is told: who it is, who it is talking to, and its rules.

The instructions must stay exactly the same for the whole conversation.
Claude keeps its earlier reasoning ("thinking") in the conversation, and
that reasoning is tied to the instructions it was made under: if they
change, even by a minute of the clock, Claude refuses the conversation.
So nothing that changes goes in here. The current time travels with each
customer message instead, as a short note of its own (`time_note`), which
is never changed once sent.

The rules that must never be broken are not only written here: the graph
enforces them (rules.py lists them: identify.py asks for the details first, booking/yes.py waits for
the customer's yes, the tools only reach this customer's bookings). The
prompt tells the model how to work well within them.
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from ...assistant.replies import ReplyItem
from ...business.store import get_knowledge
from ...config import get_settings

INSTRUCTIONS = """You are Pebby, the front-desk assistant on {business}'s website. You help customers with \
questions about the spa, and you book and cancel their appointments. Speak for the spa as "we", warm and brief.

The customer is {name}, phone {phone}. You already have their details: never ask for them again.
Each customer message comes after a note with the spa's local time when it was sent, for you to know \
what "today" is; never use it as a customer's choice.

How you work: answer everything the customer has asked that is still unanswered, using the tool for it.
- Questions about the spa (services, prices, team, hours, policies, packages, suitability): facts come only \
from search_guide. If the guide does not say, say you are not sure and offer to have the staff call them back.
- Bookings (new, change, cancel, their bookings, when someone is free): talk with them until they have \
given every detail the booking needs, then call booking once with all of it. The details are theirs: put \
their own words in the fields (services by exact name, day_said and time_said as they wrote them), and \
never choose or fill in one they did not give. A package is all the services it includes: search_guide \
for them. Add-ons they ask for go in add_ons; adding one to a booking they have is a change of it. When booking says something is `missing`, ask for exactly that; anything else it answers, tell \
them in your words, adding nothing. The system itself asks them to confirm details, plans and \
cancellations: never ask that yourself.
- Feedback about a visit: record_feedback, then do what it says. Anything else, or anything you cannot \
handle well (a person, a health question the guide does not settle, a group): hand_off, and tell them \
the staff will call them.
- Never guess: if a reply is unclear, ask what they meant. If a tool fails, say so simply.
- Do not give medical advice. Never share anything about other customers.
- Plain sentences, no markdown. Two or three sentences usually; a list goes in `items`."""

# Added to the instructions on a phone call: the reply is spoken, not read.
SPOKEN = """

You are on a phone call: your answer is spoken aloud by a voice. Keep it to one to three short sentences. Leave `items` and `suggestions` empty, and say any list in the sentence, at most three things. Say times as people say them ("three p.m."), dates as "Friday the thirtieth", prices as "a hundred and thirty-five dollars". Offer two or three options, not all of them. Say a booking reference letter by letter."""


class AgentReply(BaseModel):
    """Pebby's reply to the customer's latest message."""

    answer: str = Field(description="What to say to the customer, in plain, friendly sentences without "
                                    "markdown. When `items` is used, one short sentence introducing the list.")
    items: list[ReplyItem] = Field(description="Use for a list of three or more things, such as free times or "
                                               "services with prices: one entry each. Empty otherwise.")
    suggestions: list[str] = Field(description="Up to three short replies the customer might send next, written "
                                               "as they would write them, e.g. a time you offered or 'Another "
                                               "day'. Empty if none fit.")


def instructions(name: str, phone: str, channel: str = "chat") -> str:
    """The agent's system prompt for this customer: the same on every turn
    (a conversation stays on one channel, "chat" or "voice")."""
    text = INSTRUCTIONS.format(business=get_knowledge().name, name=name, phone=phone)
    return text + SPOKEN if channel == "voice" else text


TIME_NOTE = "(Spa local time:"  # how each time note starts, so code can tell it from the customer's words


def time_note() -> str:
    """The note sent just before each customer message: the spa's local time now."""
    now = datetime.now(ZoneInfo(get_settings().timezone))
    return f"{TIME_NOTE} {now:%A %d %B %Y, %H:%M %Z})"


# A line of a list written into the answer: "- Glow Express Facial: ...", "* ...", "1. ...".
LIST_LINE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


def answer_text(reply: AgentReply) -> str:
    """The reply's answer as the chat shows it. A list belongs in `items`, shown
    under the answer; when the model also wrote it into the answer, those lines
    are left out, so the customer does not read the list twice."""
    if not reply.items:
        return reply.answer
    kept = [line for line in reply.answer.splitlines() if not LIST_LINE.match(line)]
    return "\n".join(kept).strip()
