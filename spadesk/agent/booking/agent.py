"""The Booking agent: everything about bookings, behind Pebby.

RESPONSIBLE FOR: answering a BookingRequest (what the customer wants: the
details Pebby collected, checked against the customer's own words and read
into dates and times by code, agent/booking/collect.py) with a BookingResult (what
can be done: a plan to approve, a cancellation to approve, their bookings,
the options). It never talks to the customer: Pebby words the result.
It never books: a plan or a cancellation goes to the customer's yes
(yes.py), then to the Executor (executor.py).

    BookingRequest (from Pebby)            BookingResult (to Pebby)
      intent     new | change | cancel |     status   needs_info | plan_ready |
                 my_bookings | availability           cancel_ready | bookings |
      services   exact names (a package:              options | not_possible | handoff
                 every service it includes)  missing  what to ask for ("day", ...)
      package    its name, if one was asked  options  days or times to offer
      day        YYYY-MM-DD                  plan     the exact items, for the yes
      after / before  HH:MM                  replaces the booking a change replaces
      therapist  a name, if one was asked    cancel   the booking to cancel, for the yes
      references the bookings to change (one) bookings their upcoming bookings
                 or cancel (one or more)
      notes      anything else, in words     reason   why, in plain words

Each request is complete: the booking tool sends it only once nothing is
missing (`missing_result` says what is, before that), and a new booking
or a change only after the customer confirmed its details. So this agent
keeps no memory: the same request gets the same answer.

Most requests need no thinking, only code (the fast path):
    my_bookings   the bookings on their number
    cancel        the booking, checked against the 30-minute rule
    availability  the free times on a day, or with no day, the days that fit
    new / change  fit_plan (fit.py): the services one after another that day
Only a request with `notes` (something the fields do not hold, e.g. "after
my yoga class", "me and my sister") goes to the model, which reasons with
the same code tools. Whatever it decides, the plan it offers is one
fit_plan made, with exactly the services and day asked for (checked in
code); if it cannot make one, or cannot be reached, the fast path answers.

find_free_times, as the Booking agent's model is offered it, is here too
(free_times_tool).
"""

import json
import logging
from collections import Counter
from datetime import date, time
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ...booking.actions import confirmation, find_free_times, guarded
from ...booking.appointments import Appointments, get_appointments
from ...booking.calendars import Booking
from ...booking.errors import BookingError
from ...booking.schedule import read_schedule
from ...business.store import get_knowledge
from ...models import ModelError, Tool, run_requests, structured
from ..debug import brief, note, outcome
from ..rules import BOOKING_AGENT_MODEL, BOOKING_AGENT_ROUNDS
from .plan import Plan, days_that_fit, fit_plan

log = logging.getLogger("spadesk.agent")


INSTRUCTIONS = """You are the booking agent of a spa. You never talk to the customer. You get a booking \
request as JSON, cleaned up by the front desk, and you answer in the BookingAnswer shape.

The spa's local time now: {now:%A %d %B %Y, %H:%M}.

The request's fields are settled: keep its services, day and times. Its `notes` say something more: use \
them to choose (e.g. "after my 10am class" means after 11:00). Tools:
- days_that_fit: the next days on which every service has a free time.
- find_free_times: the free times of one service on one day.
- fit_plan: the services one after another on a day. To offer a plan you must call it; its plan is what \
the customer will be asked to approve.

Answer:
- plan_ready: fit_plan gave a plan that suits the request and its notes.
- needs_info: something must be asked first; say what in `missing`, and give `options` to choose from.
- not_possible: nothing fits; say why in `reason`, and give other days in `options`.
- handoff: the notes ask for something these tools cannot do (two people at the same time, a group, a \
class, a gift); say why in `reason`."""


class BookingRequest(BaseModel):
    """What the customer wants, cleaned up by Pebby: everything said so far."""

    intent: Literal["new", "change", "cancel", "my_bookings", "availability"]
    services: list[str] = Field(default_factory=list)
    package: str = ""
    day: str = ""  # YYYY-MM-DD
    after: str = ""  # HH:MM
    before: str = ""  # HH:MM
    therapist: str = ""
    references: list[str] = Field(default_factory=list)  # the bookings to change (one) or cancel (one or more)
    add_ons: list[str] = Field(default_factory=list)  # extras the customer named, e.g. ["Scalp massage"]
    notes: str = ""


class BookingResult(BaseModel):
    """What can be done about a BookingRequest."""

    status: Literal["needs_info", "plan_ready", "cancel_ready", "bookings", "options", "not_possible", "handoff"]
    missing: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    plan: Plan | None = None
    replaces: dict[str, Any] | None = None  # a change: the booking it replaces, as confirmation() gives it
    cancel: list[dict[str, Any]] | None = None  # cancel_ready: the bookings to cancel
    bookings: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""


class BookingAnswer(BaseModel):
    """The model's answer, when a request has notes. The plan is not part of it:
    it is the one fit_plan made."""

    status: Literal["needs_info", "plan_ready", "not_possible", "handoff"]
    missing: list[str] = Field(default_factory=list, description="What to ask the customer, e.g. ['day'].")
    options: list[str] = Field(default_factory=list, description="Days or times to offer.")
    reason: str = Field(default="", description="Why, in plain words, for not_possible or handoff.")


def _clock(text: str) -> time | None:
    """'17:00' -> 17:00; '' -> None."""
    return time.fromisoformat(text) if text else None


def _therapists(request: BookingRequest, desk: Appointments) -> dict[str, str]:
    """The chosen therapist, for each service they offer (the others: whoever is free)."""
    if not request.therapist:
        return {}
    if (person := desk.schedule.therapists.get(request.therapist)) is None:
        raise BookingError(f"We have no therapist called {request.therapist}.")
    offers = person.services
    return {name: request.therapist for name in request.services if name in offers}


def _days(services: list[str], desk: Appointments) -> list[str]:
    """The next days on which every service has a free time."""
    return days_that_fit(services, desk)


def _availability(request: BookingRequest, day: date, desk: Appointments) -> BookingResult:
    """The free times on `day` of each service asked about, as options."""
    options = []
    for name in request.services:
        slots = desk.free_slots(name, day, _therapists(request, desk).get(name))
        times = [f"{s.start:%H:%M} with {s.therapist}" for s in slots[:6]]
        options.append(f"{name} on {day:%A %d %B}: {', '.join(times) or 'no free times'}")
    return BookingResult(status="options", options=options)


def _fast(request: BookingRequest, desk: Appointments, replaces: Booking | None) -> BookingResult:
    """Answer a request for new times in code: the days that fit, or a plan on the day."""
    if not request.day:  # only an availability question comes without a day
        if days := _days(request.services, desk):
            return BookingResult(status="options", options=days)
        return BookingResult(status="not_possible", reason="No day in the next two weeks has all of these free.")
    day = date.fromisoformat(request.day)
    if request.intent == "availability":
        return _availability(request, day, desk)
    fitted = fit_plan(request.services, day, desk, _clock(request.after), _clock(request.before),
                      _therapists(request, desk), replaces)
    if isinstance(fitted, Plan):
        fitted = fitted.model_copy(update={"package": request.package, "add_ons": request.add_ons})
        return BookingResult(status="plan_ready", plan=fitted, replaces=confirmation(replaces) if replaces else None)
    return BookingResult(status="not_possible", reason=fitted[0], options=_days(request.services, desk))


def _tools(request: BookingRequest, desk: Appointments, made: list[Plan], replaces: Booking | None) -> list[Tool]:
    """The model's tools; each plan fit_plan makes is added to `made`. `replaces`:
    for a change, the booking it replaces, whose time counts as free."""
    services = {"type": "array", "items": {"type": "string", "enum": list(desk.schedule.services)}}

    def fit(args: dict[str, Any]) -> dict[str, Any]:
        fitted = fit_plan(args["services"], date.fromisoformat(args["day"]), desk, _clock(args.get("after", "")),
                          _clock(args.get("before", "")), _therapists(request, desk), replaces)
        if isinstance(fitted, Plan):
            made.append(fitted)
            return {"plan": fitted.model_dump(mode="json")["items"]}
        return {"problems": fitted}

    def safe(run):  # a broken rule or a badly written date goes back to the model as an error
        def wrapped(args):
            try:
                return run(args)
            except (BookingError, ValueError, KeyError) as error:
                return {"error": str(error)}
        return wrapped

    clock = {"type": "string", "description": "HH:MM, or leave out."}
    return [
        free_times_tool(),
        Tool("days_that_fit", "The next days on which every one of these services has a free time.",
             {"type": "object", "properties": {"services": services}, "required": ["services"]},
             safe(lambda args: {"days": days_that_fit(args["services"], desk)})),
        Tool("fit_plan", "The services one after another on a day, earliest first, within after/before.",
             {"type": "object", "properties": {"services": services, "after": clock, "before": clock,
                                               "day": {"type": "string", "description": "YYYY-MM-DD"}},
              "required": ["services", "day"]},
             safe(fit)),
    ]


def _fits_request(plan: Plan, request: BookingRequest) -> bool:
    """True when `plan` has exactly the request's services, on the request's day."""
    same_services = Counter(item.service for item in plan.items) == Counter(request.services)
    # No day asked for, no plan: the customer chooses the day, never a model.
    same_day = bool(request.day) and all(item.start.date().isoformat() == request.day for item in plan.items)
    return same_services and same_day


def _think(request: BookingRequest, desk: Appointments, replaces: Booking | None) -> BookingResult:
    """Answer a request with notes: the model reasons with the code tools."""
    made: list[Plan] = []
    tools = _tools(request, desk, made, replaces)
    messages = [SystemMessage(INSTRUCTIONS.format(now=desk.now())),
                HumanMessage(request.model_dump_json(exclude_defaults=True))]
    answer = None
    for round_ in range(1, BOOKING_AGENT_ROUNDS + 1):  # the last round must answer with what it has
        result = structured(BOOKING_AGENT_MODEL, messages, BookingAnswer, tools,
                            answer_now=round_ == BOOKING_AGENT_ROUNDS)
        if (answer := result.reply) is not None:
            note("Booking agent's model", "Booking agent", answer.status, answer.model_dump())
            break
        for call in result.tool_calls:
            note("Booking agent's model", call["name"], brief(call["args"]), call["args"])
        messages += [result.message, *run_requests(result.tool_calls, tools)]
    if answer is None or answer.status == "plan_ready":
        plan = next((p.model_copy(update={"package": request.package, "add_ons": request.add_ons}) for p in reversed(made)
                     if _fits_request(p, request)), None)
        # No answer, or a plan that is not what was asked: the fast path answers instead.
        return BookingResult(status="plan_ready", plan=plan, replaces=confirmation(replaces) if replaces else None) \
            if plan and answer else \
            _fast(request, desk, replaces)
    return BookingResult(status=answer.status, missing=answer.missing, options=answer.options, reason=answer.reason)


def handle(request: BookingRequest, phone: str, desk: Appointments | None = None) -> BookingResult:
    """The answer to `request`, for the customer with this phone number. Raises
    CalendarUnavailable when the calendar cannot be reached."""
    result = _answer(request, phone, desk or get_appointments())
    described = describe(result)
    note("Booking agent", "Pebby" if result.status not in ("plan_ready", "cancel_ready") else "The yes",
         outcome(described) + (" (its model thought about the notes)" if request.notes else " (in code)"),
         {"request": request.model_dump(exclude_defaults=True), "result": described})
    return result


def _part_of_a_package(chosen: list[Booking], upcoming: list[Booking]) -> str:
    """Why changing or cancelling `chosen` needs the staff: it leaves part of a
    package booked (the rest of it, booked the same day, is not among them),
    and what that does to the package's price is for the spa to settle. ""
    when it does not."""
    picked = {b.id for b in chosen}
    for booking in chosen:
        rest = [b for b in upcoming if b.package == booking.package and b.start.date() == booking.start.date()
                and b.id not in picked] if booking.package else []
        if rest:
            return (f"Your {booking.service} is part of your {booking.package}: our staff will help you change "
                    "part of a package.")
    return ""


def _answer(request: BookingRequest, phone: str, desk: Appointments) -> BookingResult:
    """handle's answer."""
    if request.intent == "my_bookings":
        return BookingResult(status="bookings", bookings=[confirmation(b) for b in desk.upcoming(phone)])
    replaces = None
    if request.intent in ("change", "cancel"):
        try:  # theirs, and not within the notice
            chosen = [desk.check_cancel(reference, phone) for reference in request.references]
        except BookingError as error:
            return BookingResult(status="not_possible", reason=str(error))
        if request.intent == "change" and len(chosen) != 1:
            return BookingResult(status="not_possible", reason="One booking can be changed at a time.")
        if reason := _part_of_a_package(chosen, desk.upcoming(phone)):
            return BookingResult(status="handoff", reason=reason)
        if request.intent == "cancel":
            return BookingResult(status="cancel_ready", cancel=[confirmation(b) for b in chosen])
        replaces = chosen[0]
    try:
        _therapists(request, desk)  # a therapist the spa does not have is refused here, whatever else is asked
        if request.notes:
            try:
                return _think(request, desk, replaces)
            except ModelError as error:  # the thinking is a help, not a must: the fast path still answers
                log.warning("booking agent model unavailable: %s", error)
        return _fast(request, desk, replaces)
    except BookingError as error:  # e.g. a therapist who does not offer one of the services
        return BookingResult(status="not_possible", reason=str(error))
    except ValueError:
        return BookingResult(status="not_possible", reason="Dates are YYYY-MM-DD and times HH:MM.")


def missing_result(request: BookingRequest, missing: list[str], phone: str,
                   desk: Appointments | None = None) -> BookingResult:
    """What Pebby must still ask the customer for, with what helps them answer:
    their bookings when it is which booking, the days that work when it is the day."""
    desk = desk or get_appointments()
    if "references" in missing:
        return BookingResult(status="needs_info", missing=missing,
                             bookings=[confirmation(b) for b in desk.upcoming(phone)])
    options = _days(request.services, desk) if "day" in missing and request.services else []
    return BookingResult(status="needs_info", missing=missing, options=options)


def describe(result: BookingResult) -> dict[str, Any]:
    """`result` as Pebby is told it: only the fields that say something."""
    return json.loads(result.model_dump_json(exclude_defaults=True))


def free_times_tool() -> Tool:
    """find_free_times, with the document's services and team as its choices."""
    schedule = read_schedule(get_knowledge())
    return Tool("find_free_times",
                "Free start times for a service on one day, by therapist, in the spa's time zone. Give a "
                "therapist only if the customer asked for one. Use this before offering any time.",
                {"type": "object", "properties": {
                    "service": {"type": "string", "enum": list(schedule.services)},
                    "therapist": {"type": "string", "enum": list(schedule.therapists)},
                    "minutes": {"type": "integer", "description": "The length, for a service that comes in more "
                                                                  "than one (e.g. a 90-minute massage). Leave out "
                                                                  "for the standard length."},
                    "date": {"type": "string", "description": "The date, YYYY-MM-DD."}},
                 "required": ["service", "date"]},
                guarded(find_free_times))
