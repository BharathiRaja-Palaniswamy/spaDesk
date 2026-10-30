"""Pebby, the front AI agent: what it is told, name and phone first, and its tools.

The instructions must stay exactly the same for the whole conversation.
Claude keeps its earlier reasoning ("thinking") in the conversation, and
that reasoning is tied to the instructions it was made under: if they
change, even by a minute of the clock, Claude refuses the conversation.
So nothing that changes goes in here. The current time travels with each
customer message instead, as a short note of its own (`time_note`), which
is never changed once sent.

The rules that must never be broken are not only written here: the graph
enforces them (rules.py lists them: flow.py's identify step asks for the details first, yes.py waits for
the customer's yes, the tools only reach this customer's bookings). The
prompt tells the model how to work well within them.

Also here: name and phone first (read_details: a small model reads a
message for them, for what the customer asked, and answers them warmly;
ask_for: what is still missing), and Pebby's tools (agent_tools):
search_guide, booking (details.py checks the details, booking.py answers),
record_feedback (the Feedback agent, feedback.py) and hand_off (staff.py).
"""

import logging
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..access.permissions import Role, allowed
from ..assistant.replies import ReplyItem
from ..booking.actions import guarded
from ..booking.appointments import digits, get_appointments
from ..booking.errors import BookingError, CalendarUnavailable
from ..booking.schedule import read_schedule
from ..business.store import get_knowledge
from ..config import get_settings
from ..lc.retriever import get_retriever
from ..models import Tool, structured
from ..rag.errors import ModelUnavailable
from .booking import BookingRequest, describe, handle, missing_result
from .debug import brief, note
from .details import TIME_NOTE, collect
from .feedback import give_feedback
from .rules import MAX_SERVICES, READER_MODEL, REPEATING
from .staff import record_handoff
from .yes import when

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
- Feedback about a visit, or how they felt about it: record_feedback with their words; the Feedback \
agent answers them. If it asked them something, pass their answer with record_feedback and its card. Anything else, or anything you cannot \
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




def time_note() -> str:
    """The note sent just before each customer message: the spa's local time now."""
    now = datetime.now(ZoneInfo(get_settings().timezone))
    return f"{TIME_NOTE} {now:%A %d %B %Y, %H:%M %Z})"


# A line of a list written into the answer: "- Glow Express Facial: ...", "* ...", "1. ...".
LIST_LINE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
MARKDOWN = re.compile(r"\*\*|__|`")  # bold and code marks


def answer_text(reply: AgentReply) -> str:
    """The reply's answer as the chat shows it: plain sentences. Markdown a model
    wrote anyway (**bold**, list marks) is taken out; and a list belongs in `items`,
    shown under the answer, so when the model also wrote it into the answer, those
    lines are left out and the customer does not read the list twice."""
    lines = reply.answer.splitlines()
    if reply.items:
        lines = [line for line in lines if not LIST_LINE.match(line)]
    else:
        lines = [LIST_LINE.sub("", line) for line in lines]
    return MARKDOWN.sub("", "\n".join(lines)).strip()


READ = """You read one message a customer sent to a spa's website chat, which asked for \
their name and phone number. Take out what the message gives. Do not guess."""


class Details(BaseModel):
    """What one message says about who the customer is."""

    name: str = Field(description="The customer's name, as they wrote it, if the message gives it; otherwise "
                                  "empty.")
    phone: str = Field(description="The phone number, as written, if the message gives one; otherwise empty.")
    # No sample values in these descriptions: a small model copies them into its answer.
    reply: str = Field(default="", description="A short, warm sentence answering only what this message says "
                                               "to you as a person: a greeting gets a greeting, a question "
                                               "about how you are gets an answer, thanks get thanks. Never "
                                               "answer a question about the spa or a booking, never ask for "
                                               "anything. Empty when the message says nothing of that kind.")
    asked: str = Field(default="", description="Anything this message asks for besides a greeting, a name or a "
                                               "phone number, copied in the customer's own words; empty when "
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


def ask_for(name: str, phone: str, caller: str = "") -> tuple[str, str]:
    """(an opening, the question) for what is still missing, given what is known
    so far; the opening gives way to the reader's own warm reply when there is
    one (flow.py identify). `caller` is the number calling, on a phone call."""
    if name and phone:
        return f"Thank you, {name}!", "How can I help you today?"
    if not name and caller and phone == caller:
        return "", (f"May I have your name, please? I'll keep the number you're calling from, ending in "
                    f"{digits(caller)[-4:]}, unless you'd like to give another.")
    if not name and not phone:
        return "", "Before we start, may I have your name and phone number?"
    if not phone:
        return f"Thank you, {name}!", "And a phone number we can reach you on?"
    return "Thank you!", "And your name, please?"


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


log = logging.getLogger("spadesk.agent")

# Requests the graph runs itself (flow._hold): they may end in a question for the customer's yes.
HELD = {"booking"}
# Tool -> the permission it needs. Every tool must be listed: one that is not is never offered.
NEEDS = {"search_guide": "search_guide", "booking": "book", "record_feedback": "contact_staff",
         "hand_off": "contact_staff"}


def search_guide(args: dict[str, Any]) -> dict[str, Any]:
    """The parts of the business document that best answer the question, or
    an error the model can read when the search cannot be reached."""
    try:
        documents = get_retriever().invoke(args["question"])
    except ModelUnavailable as error:
        log.warning("guide search unavailable: %s", error)
        return {"error": "The guide can't be searched right now."}
    return {"excerpts": [{"sections": doc.metadata["sections"], "text": doc.page_content} for doc in documents]}


def booking_parameters() -> dict[str, Any]:
    """The booking tool's inputs, with the document's services and team. The day and
    the time are the customer's own words; code works out the date and times."""
    schedule = read_schedule(get_knowledge())
    return {"type": "object", "required": ["intent"], "properties": {
        "intent": {"type": "string", "enum": ["new", "change", "cancel", "my_bookings", "availability"],
                   "description": "new: they want to book. availability: they only ask when someone or "
                                  "something is free, without asking to book. change, cancel: one of their "
                                  "bookings. my_bookings: they ask what they have booked."},
        "services": {"type": "array", "items": {"type": "string", "enum": list(schedule.services)},
                     "description": "Every service, by its exact name; for a package, every service it includes."},
        "package": {"type": "string", "description": "The package's name, if they asked for one."},
        "add_ons": {"type": "array", "items": {"type": "string", "enum": list(schedule.add_ons)},
                    "description": "Extras they asked for with it (e.g. a scalp massage); to add one to an "
                                   "existing booking, a change of that booking."},
        "day_said": {"type": "string", "description": "The customer's own words for the day, as they wrote them "
                                                      "(e.g. 'Thursday', 'tomorrow', '30 October'); empty if "
                                                      "they have not said one."},
        "time_said": {"type": "string", "description": "The customer's own words for the time, as they wrote "
                                                       "them (e.g. 'after 3pm', 'morning', 'any time'); empty if "
                                                       "they have not said one."},
        "therapist": {"type": "string", "enum": list(schedule.therapists)},
        "references": {"type": "array", "items": {"type": "string"},
                       "description": "The references of the bookings to change (one) or cancel (one or "
                                      "more), if you know them; the customer's words pick out which."}}}


def _beyond_one_booking(request: BookingRequest, args: dict[str, Any]) -> str:
    """Why a request is for the staff, not one booking (rules.py, 5): a bulk request,
    or one that repeats. "" when it is one booking."""
    if len(request.services) > MAX_SERVICES:
        return f"Booking {len(request.services)} services at once is a bulk request, which our staff arrange."
    if REPEATING.search(f"{args.get('day_said', '')} {args.get('time_said', '')}"):
        return "Repeating appointments are arranged by our staff."
    return ""


def _to_staff(result: dict[str, Any], name: str, phone: str) -> dict[str, Any]:
    """A result that hands the request to the staff is also written down for them."""
    if result.get("status") == "handoff":
        record_handoff(name, phone, result.get("reason", "Booking request for the staff."))
        result = result | {"handed_off": True}
    return result


def booking(args: dict[str, Any], name: str, phone: str, words: str) -> dict[str, Any]:
    """The booking tool: what is missing, the details to read back, or booking (code)'s answer.
    Anything for the staff (beyond one booking, part of a package) is written down for them."""
    desk = get_appointments()
    # Theirs only: no one else's booking can be named.
    upcoming = desk.upcoming(phone) if args.get("intent") in ("change", "cancel") else []
    request, missing, unnamed = collect(args, words, desk.now().date(), upcoming)
    to = "Pebby" if missing else "The yes" if request.intent in ("new", "change") else "Booking"
    worked_out = brief(request.model_dump(exclude_defaults=True, exclude={"intent"}))
    note("Collect", to, f"{worked_out or 'nothing yet'} · " + (f"missing {', '.join(missing)}" if missing
                                                                else "nothing missing"),
         {"said": {key: args[key] for key in ("day_said", "time_said") if args.get(key)},
          "worked out": request.model_dump(exclude_defaults=True), "missing": missing,
          "not named by the customer": unnamed + ([args["therapist"]] if args.get("therapist") and
                                                   not request.therapist else [])
          + [r for r in args.get("references", []) if r not in request.references]})
    if reason := _beyond_one_booking(request, args):
        return _to_staff({"status": "handoff", "reason": reason}, name, phone)
    if missing:
        result = describe(missing_result(request, missing, phone))
        # No service they named: the ones chosen for them are offered as choices, never used.
        return result | ({"options": unnamed} if "services" in missing and unnamed else {})
    if request.intent in ("new", "change"):  # complete: the customer confirms the details first
        # The customer's own words for the day and time go with it, for the read-back.
        said = {"day": args.get("day_said", ""), "time": args.get("time_said", "")}
        return {"status": "confirm_details", "request": request.model_dump(exclude_defaults=True), "said": said}
    return _to_staff(describe(handle(request, phone)), name, phone)


def agent_tools(name: str, phone: str, role: Role = Role.CUSTOMER, words: str = "") -> list[Tool]:
    """The tools `role` may use, made for the customer with this name and number.
    `words`: what the customer has written so far (collect.customer_words)."""
    tools = [
        Tool("search_guide",
             "Searches the spa's guide (services, prices, team, hours, policies, suitability, offers, packages). "
             "Use it for any fact about the spa before you state it.",
             {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
             search_guide),
        Tool("booking",
             "Anything about bookings: a new booking, changing or cancelling one, their bookings, when someone "
             "is free. Call it once you have every detail from the customer (for a booking: the service they "
             "named, the day and the time), not for each one. The system then reads the details back to them "
             "and shows the times found for their yes; anything else it answers, tell them in your words.",
             booking_parameters(),
             guarded(lambda args: booking(args, name, phone, words))),
        Tool("record_feedback",
             "Passes feedback about a visit (praise, a complaint, a suggestion, how they felt) to the Feedback "
             "agent, which answers the customer itself. Not for questions or requests.",
             {"type": "object", "properties": {
                 "words": {"type": "string", "description": "The customer's feedback in their own words, all of it."},
                 "card": {"type": "string", "description": "When they are answering the Feedback agent's "
                                                           "question: the card it gave (e.g. FB-1A2B3C)."}},
              "required": ["words"]},
             lambda args: give_feedback(name, phone, args["words"], args.get("card", ""))),
        Tool("hand_off",
             "Asks the spa's staff to call this customer back. Use it when they ask for a person, have a "
             "health question the guide does not settle, want something you cannot do (a class, a group), "
             "or booking keeps failing (feedback, complaints included, goes to record_feedback). Say why, in one "
             "sentence.",
             {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
             lambda args: {"handed_off": True, **record_handoff(name, phone, args["reason"])}),
    ]
    return [tool for tool in tools if allowed(role, NEEDS.get(tool.name, ""))]
