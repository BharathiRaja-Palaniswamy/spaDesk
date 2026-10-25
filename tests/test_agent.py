"""Tests for the booking agent (spadesk/agent/): the rules the graph enforces,
whatever the model says.

The model is a stand-in that plays back prepared messages (tests/standins.py),
so these run without an API key and spend nothing. Bookings go to an
in-memory calendar with "now" fixed on Thursday 22 October 2026, 2:10 PM.

Run with:  .venv/bin/python -m pytest tests/test_agent.py
"""

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from standin_calendar import standin_desk
from standins import use_standin

from spadesk import agent
from spadesk.agent import respond
from spadesk.agent.booking import agent as booking_agent
from spadesk.agent.booking import executor as approved
from spadesk.agent.booking import yes as confirm
from spadesk.agent.booking.collect import customer_words
from spadesk.agent.pebby import identify, nodes, tools, turns
from spadesk.agent.pebby.feedback import Reading
from spadesk.agent.pebby.graph import get_graph
from spadesk.agent.pebby.identify import Details
from spadesk.agent.pebby.nodes import MAX_TOOL_ROUNDS
from spadesk.agent.pebby.prompt import AgentReply
from spadesk.agent.staff import HANDOVER
from spadesk.app import app
from spadesk.booking import actions
from spadesk.booking.errors import CalendarUnavailable
from spadesk.models import ModelError

ASHA = Details(name="Asha Rao", phone="(408) 555-0101")
# The customer asks for a prenatal massage; Pebby passes on their own words for the day and the time.
SAYS = "A prenatal massage on 30 October, after 3pm"
PRENATAL = {"intent": "new", "services": ["Prenatal Massage"], "day_said": "30 October", "time_said": "after 3pm"}
FRIDAY = datetime(2026, 10, 30).date()


def details(name="", phone="", asked="") -> AIMessage:
    """The small model reading a message for a name and number."""
    return AIMessage(content=Details(name=name, phone=phone, asked=asked).model_dump_json())


def says(answer: str) -> AIMessage:
    """The agent's reply."""
    return AIMessage(content=AgentReply(answer=answer, items=[], suggestions=[]).model_dump_json())


def asks(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    """The agent asking for one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def books(request: dict = PRENATAL, call_id: str = "call-1") -> AIMessage:
    """Pebby passing a cleaned-up request to the Booking agent (which answers it in code)."""
    return asks("booking", request, call_id)


def to_plan(thread: str, message: str):
    """Send a booking request: the customer is first asked to confirm the details;
    their yes has the Booking agent find times. Returns the turn after that yes."""
    details = respond(thread, message)
    assert details.confirming and details.answer.startswith("Just to check before I look for times")
    assert details.suggestions == ["Yes, that's right", "Change something"]
    return respond(thread, "Yes, that's right")


@pytest.fixture
def desk(monkeypatch, tmp_path):
    """One in-memory calendar with a fixed clock, for every part of the agent; a fresh memory."""
    desk = standin_desk()
    for module in (tools, confirm, identify, actions, approved, booking_agent):
        monkeypatch.setattr(module, "get_appointments", lambda: desk)
    monkeypatch.setenv("SPADESK_HANDOFFS", str(tmp_path / "handoffs.jsonl"))
    monkeypatch.setenv("SPADESK_FEEDBACK", str(tmp_path / "feedback.jsonl"))
    get_graph.cache_clear()
    yield desk
    get_graph.cache_clear()


def chat(monkeypatch, *replies: AIMessage):
    """Plays back `replies` as the models' answers (high and low tier alike), and returns the messages
    each model call was sent."""
    return use_standin(monkeypatch, list(replies))


def memory(thread: str) -> dict:
    """What the agent remembers about a conversation."""
    return get_graph().get_state({"configurable": {"thread_id": thread}}).values


def identified(monkeypatch, thread: str) -> None:
    """Gives the agent the customer's details, as a first message would."""
    chat(monkeypatch, details("Asha Rao", "(408) 555-0101"))
    assert respond(thread, "Asha Rao, (408) 555-0101").answer == "Thank you, Asha Rao! How can I help you today?"


# ---- name and phone first --------------------------------------------------------

def test_a_question_waits_until_the_customer_gives_their_details(desk, monkeypatch):
    sent = chat(monkeypatch, details(asked="What time do you open?"), details("Asha Rao", "(408) 555-0101"),
                says("We open at 10."))
    assert respond("thread-one", "What time do you open?").answer == ("Before we start, may I have your name "
                                                                       "and phone number?")
    # Once the details are given, the question asked first is answered.
    assert respond("thread-one", "Asha Rao, (408) 555-0101").answer == "We open at 10."
    assert memory("thread-one")["name"] == "Asha Rao" and memory("thread-one")["phone"] == "(408) 555-0101"
    # The agent was told who the customer is, and saw the first question.
    agent_call = sent[-1]
    assert "Asha Rao, phone (408) 555-0101" in agent_call[0].content
    assert any(m.content == "What time do you open?" for m in agent_call)


def test_the_name_reader_is_told_the_question_the_message_answers(desk, monkeypatch):
    # As seen live: "poiqwe", the reply to "your name?", was not taken as a name.
    sent = chat(monkeypatch, details(phone="9282838292"), details(name="poiqwe"))
    assert respond("thread-q", "9282838292").answer == "Thank you! And your name, please?"
    respond("thread-q", "poiqwe")
    assert sent[-1][0].content.endswith("The message answers this question: Thank you! And your name, please?")


def test_only_what_is_missing_is_asked_for(desk, monkeypatch):
    chat(monkeypatch, details(name="Asha"), details(phone="12"))
    assert respond("thread-two", "I'm Asha").answer == "Thank you, Asha! And a phone number we can reach you on?"
    # A number with fewer than 7 digits is not taken.
    assert respond("thread-two", "12").answer == "Thank you, Asha! And a phone number we can reach you on?"


# ---- booking needs the customer's yes ------------------------------------------------

def test_a_booking_is_made_only_after_the_customer_says_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-book")
    sent = chat(monkeypatch, books(), says("You're booked!"))
    details = respond("thread-book", SAYS)
    assert details.answer == ("Just to check before I look for times: Prenatal Massage on Friday 30 October, "
                              "after 3pm. Is that right?")  # the customer's own words
    turn = respond("thread-book", "Yes, that's right")
    assert turn.confirming and turn.suggestions == ["Yes, book it", "No"]
    assert turn.answer == ("Shall I book Prenatal Massage (60 minutes) with Priya Nair on Friday 30 October at "
                           "15:00, for Asha Rao ((408) 555-0101)?")
    assert desk.day_bookings(FRIDAY) == []  # nothing yet
    assert len(sent) == 1  # a complete request: the Booking agent answered in code, with no model call

    assert respond("thread-book", "Yes, book it").answer == "You're booked!"
    [booking] = desk.day_bookings(FRIDAY)
    # Booked under the details the customer gave, not anything a model wrote.
    assert (booking.name, booking.phone, booking.therapist) == ("Asha Rao", "(408) 555-0101", "Priya Nair")


def test_one_yes_books_a_whole_plan(desk, monkeypatch):
    identified(monkeypatch, "thread-two-items")
    both = {**PRENATAL, "services": ["Prenatal Massage", "Signature Radiance Facial"]}
    chat(monkeypatch, books(both), says("All booked!"))
    turn = to_plan("thread-two-items", SAYS + ", then a Signature Radiance Facial")
    assert turn.answer == ("Shall I book these for Asha Rao ((408) 555-0101)? 1. Prenatal Massage (60 minutes) "
                           "with Priya Nair on Friday 30 October at 15:00. 2. Signature Radiance Facial "
                           "(60 minutes) with Sofia Martinez on Friday 30 October at 16:00.")
    assert respond("thread-two-items", "yes").answer == "All booked!"
    assert [b.service for b in desk.day_bookings(FRIDAY)] == ["Prenatal Massage", "Signature Radiance Facial"]


def test_without_a_day_the_days_that_fit_go_back_to_pebby(desk, monkeypatch):
    identified(monkeypatch, "thread-days")
    package = {"intent": "new", "package": "Bridal Glow Prep Package",
               "services": ["Brightening Vitamin C Facial", "Glow Express Facial", "Eyebrow Wax"]}
    sent = chat(monkeypatch, books(package), says("It fits on Thursday or Wednesday. Which suits you?"))
    assert respond("thread-days", "Book the bridal package").answer == "It fits on Thursday or Wednesday. Which suits you?"
    told = json.loads(sent[-1][-1].content)
    assert told["status"] == "needs_info" and told["missing"] == ["day", "time"] and "Wednesday 2026-10-28" in told["options"]


def test_a_time_taken_before_the_yes_books_nothing_and_pebby_is_told_to_ask_again(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-race")
    sent = chat(monkeypatch, books(), says("Sorry, 3pm was just taken."))
    assert to_plan("thread-race", SAYS).confirming
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Ben", "408 555 0102")  # meanwhile
    assert respond("thread-race", "Yes").answer == "Sorry, 3pm was just taken."
    told = json.loads(next(m for m in sent[-1] if isinstance(m, ToolMessage)).content)["result"]
    assert "not a free time" in told["not_booked"][0] and "call booking again" in told["next"]
    assert [b.name for b in desk.day_bookings(FRIDAY)] == ["Ben"]
    assert memory("thread-race")["failed_plans"] == 1


def test_after_the_yes_pebby_is_told_it_is_booked_not_asked_again(desk, monkeypatch):
    # As seen live: the customer's "Yes, book it" was passed on after the booking, and
    # Pebby took it as a new request and offered the next time. Now it goes inside the result.
    identified(monkeypatch, "thread-once")
    sent = chat(monkeypatch, books(), books(call_id="call-2"), says("You're booked at 3pm!"))
    to_plan("thread-once", SAYS)
    turn = respond("thread-once", "Yes, book it")
    assert not turn.confirming and turn.answer == "You're booked at 3pm!"
    after_yes = sent[-2]  # the model call straight after the yes
    assert not isinstance(after_yes[-1], HumanMessage)
    assert json.loads(after_yes[-1].content)["customer_said"] == "Yes, book it"
    # Asking for the same booking again in that turn is refused, so nothing is offered twice.
    assert json.loads(sent[-1][-1].content)["error"] == nodes.REPEATED
    assert len(desk.day_bookings(FRIDAY)) == 1


def test_changing_the_details_asks_the_booking_agent_nothing(desk, monkeypatch):
    identified(monkeypatch, "thread-fix")
    sent = chat(monkeypatch, books(), says("Of course. Which day instead?"))
    assert respond("thread-fix", SAYS).confirming
    assert respond("thread-fix", "Change something: not Friday").answer == "Of course. Which day instead?"
    told = json.loads(next(m for m in sent[-1] if isinstance(m, ToolMessage)).content)
    assert "did not say yes" in told["not_done"] and memory("thread-fix").get("pending") is None


def test_a_request_missing_details_is_answered_in_code_without_the_booking_agents_model(desk, monkeypatch):
    identified(monkeypatch, "thread-missing")
    no_day = {"intent": "new", "services": ["Prenatal Massage"], "notes": "Customer wants a prenatal massage."}
    sent = chat(monkeypatch, books(no_day), says("Which day would you like? Sunday or Friday?"))
    turn = respond("thread-missing", "book prenatal")
    assert not turn.confirming and len(sent) == 2  # Pebby asked, then answered: no Booking agent model call
    told = json.loads(sent[-1][-1].content)
    assert told["status"] == "needs_info" and told["missing"] == ["day", "time"]


def test_a_day_or_time_the_customer_never_said_is_not_used(desk, monkeypatch):
    # As seen live: for "Book Detox Body Scrub", Pebby sent today's date and the time now.
    identified(monkeypatch, "thread-guess")
    guessed = {"intent": "new", "services": ["Detox Body Scrub"], "day_said": "2026-10-22", "time_said": "14:10"}
    sent = chat(monkeypatch, books(guessed), says("Which day and time suit you?"))
    turn = respond("thread-guess", "Book Detox Body Scrub")
    assert not turn.confirming and turn.answer == "Which day and time suit you?"
    assert json.loads(sent[-1][-1].content)["missing"] == ["day", "time"]


def test_any_time_is_an_answer_and_two_requests_at_once_run_neither(desk, monkeypatch):
    identified(monkeypatch, "thread-any")
    both = AIMessage(content="", tool_calls=[
        {"name": "booking", "args": {**PRENATAL, "time_said": "any time"}, "id": "call-1"},
        {"name": "booking", "args": {"intent": "availability", "services": ["Prenatal Massage"]}, "id": "call-2"}])
    sent = chat(monkeypatch, both, books({**PRENATAL, "time_said": "any time"}, "call-3"))
    details = respond("thread-any", "A prenatal massage on 30 October, any time")
    assert details.answer == ("Just to check before I look for times: Prenatal Massage on Friday 30 October, "
                              "any time. Is that right?")
    assert all("one booking request at a time" in json.loads(m.content)["error"]
               for m in sent[1] if isinstance(m, ToolMessage))


def test_the_details_are_read_back_in_the_customers_own_words(desk, monkeypatch):
    # As seen live: "tomorrow morning" was read back as "finishing by 12:00", which they never said.
    identified(monkeypatch, "thread-words")
    facial = {"intent": "new", "services": ["Glow Express Facial"], "day_said": "tomorrow", "time_said": "morning"}
    chat(monkeypatch, books(facial))
    details = respond("thread-words", "A Glow Express Facial tomorrow morning")
    assert details.answer == ("Just to check before I look for times: Glow Express Facial tomorrow (Friday 23 "
                              "October), in the morning. Is that right?")


def test_a_second_approved_plan_that_cannot_be_booked_goes_to_the_staff(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-race-twice")
    for attempt in range(approved.MAX_FAILED_PLANS):
        start = datetime(2026, 10, 30, 15 + attempt, 0)
        said = f"after {start:%-I}pm"
        chat(monkeypatch, books({**PRENATAL, "time_said": said}, f"call-{attempt}"), says("Sorry about that."))
        assert to_plan("thread-race-twice", f"A prenatal massage on 30 October, {said}").confirming
        desk.book("Prenatal Massage", "Priya Nair", start, "Ben", "408 555 0102")
        respond("thread-race-twice", "Yes")
    handoff = json.loads((tmp_path / "handoffs.jsonl").read_text())
    assert handoff["phone"] == "(408) 555-0101" and "failed again" in handoff["reason"]


def test_anything_but_yes_books_nothing_and_goes_back_to_the_agent(desk, monkeypatch):
    identified(monkeypatch, "thread-no")
    sent = chat(monkeypatch, books(), says("Sure, which time instead?"))
    assert to_plan("thread-no", SAYS).confirming
    assert respond("thread-no", "Actually, can we do 4pm?").answer == "Sure, which time instead?"
    assert desk.day_bookings(FRIDAY) == []
    # The model learns nothing was booked, then reads the customer's replies (after their time notes), in order.
    messages = sent[-1]
    [result] = [m for m in messages if isinstance(m, ToolMessage)]
    after = messages[messages.index(result) + 1:]
    assert "did not say yes" in json.loads(result.content)["not_done"]
    assert all(isinstance(m, HumanMessage) for m in after) and after[-1].content == "Actually, can we do 4pm?"
    assert after[-2].content.startswith("(Spa local time:")


def test_the_instructions_stay_the_same_from_turn_to_turn(desk, monkeypatch):
    # Claude refuses a conversation whose instructions changed since its earlier
    # reasoning was made, so the time goes with each message, not in the instructions.
    identified(monkeypatch, "thread-same")
    sent = chat(monkeypatch, says("Hello again."), says("Still here."))
    respond("thread-same", "Hi")
    respond("thread-same", "Are you there?")
    first, second = sent
    assert first[0].content == second[0].content
    # Each turn's messages are the earlier ones, unchanged, with the new ones after them.
    assert [m.content for m in second[1:len(first)]] == [m.content for m in first[1:]]


def test_a_change_to_another_service_at_the_same_day_and_time(desk, monkeypatch):
    # As seen live: Swedish Massage with Daniel, Monday 17:00 -> "deep tissue instead, same day and time".
    identified(monkeypatch, "thread-swap")
    monday = datetime(2026, 10, 26, 17, 0)
    old = desk.book("Swedish Massage", "Daniel Ruiz", monday, "Asha Rao", "(408) 555-0101")
    swap = {"intent": "change", "references": [old.id], "services": ["Deep Tissue Massage"],
            "day_said": "same day and time", "time_said": "same day and time"}
    chat(monkeypatch, books(swap), says("Changed!"))
    details = respond("thread-swap", "I'd like deep tissue instead, same day and time")
    assert details.answer == ("Just to check before I look for times: Deep Tissue Massage same day and time "
                              f"(Monday 26 October, from 17:00), instead of your booking {old.id}. Is that right?")
    plan = respond("thread-swap", "Yes, that's right")
    # The old booking's time counts as free: the same slot, with the same therapist, is offered.
    assert plan.answer == (f"I'll cancel your current booking (Swedish Massage with Daniel Ruiz on Monday 26 October "
                           f"at 17:00, {old.id}) and book Deep Tissue Massage (60 minutes) with Daniel Ruiz on Monday "
                           "26 October at 17:00 instead, for Asha Rao. Shall I go ahead?")
    assert respond("thread-swap", "yes").answer == "Changed!"
    [booking] = desk.day_bookings(monday.date())
    assert (booking.service, booking.therapist, f"{booking.start:%H:%M}") == ("Deep Tissue Massage", "Daniel Ruiz",
                                                                           "17:00")


def test_an_add_on_is_booked_with_the_service_and_shown_in_both_questions(desk, monkeypatch):
    identified(monkeypatch, "thread-extra")
    chat(monkeypatch, books({**PRENATAL, "add_ons": ["Scalp massage"]}), says("Booked, with the scalp massage!"))
    details = respond("thread-extra", SAYS + ", with a scalp massage")
    assert details.answer.startswith("Just to check before I look for times: Prenatal Massage plus scalp massage on ")
    plan = respond("thread-extra", "Yes, that's right")
    assert "Prenatal Massage plus scalp massage (60 minutes) with Priya Nair" in plan.answer
    respond("thread-extra", "yes")
    [booking] = desk.day_bookings(FRIDAY)
    assert booking.add_ons == ("Scalp massage",)  # on the calendar, for the staff


def test_an_add_on_is_added_to_an_existing_booking_keeping_everything_else(desk, monkeypatch):
    identified(monkeypatch, "thread-add")
    old = desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha Rao", "(408) 555-0101")
    chat(monkeypatch, books({"intent": "change", "references": [old.id], "add_ons": ["Scalp massage"]}),
         says("Added!"))
    details = respond("thread-add", "Please add a scalp massage to my prenatal massage")
    assert details.answer == ("Just to check before I look for times: Prenatal Massage plus scalp massage on Friday "
                              f"30 October, from 15:00, instead of your booking {old.id}. Is that right?")
    respond("thread-add", "Yes, that's right")
    assert respond("thread-add", "yes").answer == "Added!"
    [booking] = desk.day_bookings(FRIDAY)  # the same slot, now with the add-on; the old booking cancelled
    assert (f"{booking.start:%H:%M}", booking.service, booking.add_ons) == ("15:00", "Prenatal Massage",
                                                                          ("Scalp massage",))


def test_a_time_that_is_not_free_is_not_offered_for_a_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-taken")
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Ben", "408 555 0102")
    sent = chat(monkeypatch, books({**PRENATAL, "time_said": "between 3 and 4"}), says("3pm is taken; Thursday works."))
    turn = to_plan("thread-taken", "A prenatal massage on 30 October, between 3 and 4")
    assert not turn.confirming and turn.answer == "3pm is taken; Thursday works."
    told = json.loads(sent[-1][-1].content)
    assert told["customer_said"] == "Yes, that's right"  # their yes to the details, inside the result
    assert told["result"]["status"] == "not_possible" and told["result"]["options"]  # other days to offer


def test_changing_a_booking_books_the_new_time_then_cancels_the_old(desk, monkeypatch):
    identified(monkeypatch, "thread-move")
    old = desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 11, 0), "Asha Rao", "(408) 555-0101")
    chat(monkeypatch, books({**PRENATAL, "intent": "change", "references": [old.id]}), says("Moved!"))
    turn = to_plan("thread-move", SAYS)
    assert turn.confirming and turn.answer == (
        f"I'll cancel your current booking (Prenatal Massage with Priya Nair on Friday 30 October at 11:00, {old.id}) "
        "and book Prenatal Massage (60 minutes) with Priya Nair on Friday 30 October at 15:00 instead, for Asha Rao. "
        "Shall I go ahead?")
    assert respond("thread-move", "yes").answer == "Moved!"
    assert [f"{b.start:%H:%M}" for b in desk.day_bookings(FRIDAY)] == ["15:00"]


def test_cancelling_also_waits_for_a_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-cancel")
    booking = desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha Rao", "4085550101")
    chat(monkeypatch, books({"intent": "cancel", "references": [booking.id]}), says("Cancelled."))
    turn = respond("thread-cancel", "Cancel my Friday massage")
    assert turn.confirming and booking.id in turn.answer and turn.suggestions == ["Yes, cancel it", "No, keep it"]
    assert respond("thread-cancel", "yes").answer == "Cancelled."
    assert desk.day_bookings(datetime(2026, 10, 30).date()) == []


def test_several_bookings_are_cancelled_with_one_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-cancel-two")
    one = desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha Rao", "4085550101")
    two = desk.book("Signature Radiance Facial", "Sofia Martinez", datetime(2026, 10, 30, 16, 0), "Asha Rao",
                    "4085550101")
    chat(monkeypatch, books({"intent": "cancel", "references": [one.id, two.id]}), says("Both cancelled."))
    turn = respond("thread-cancel-two", "Please cancel both of them")
    assert turn.answer == (f"Shall I cancel these 2 bookings? 1. Prenatal Massage with Priya Nair on Friday 30 "
                           f"October at 15:00 (reference {one.id}). 2. Signature Radiance Facial with Sofia Martinez "
                           f"on Friday 30 October at 16:00 (reference {two.id}).")
    assert respond("thread-cancel-two", "yes").answer == "Both cancelled."
    assert desk.day_bookings(FRIDAY) == []


def test_a_returning_customer_is_asked_about_their_upcoming_booking_first(desk, monkeypatch):
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha Rao", "(408) 555-0101")
    sent = chat(monkeypatch, details("Asha Rao", "408-555-0101"), says("Sure, let's move it."))
    question = respond("thread-back", "Hi, it's Asha, 408-555-0101").answer  # no request yet: so they are asked
    assert question == ("Thank you, Asha Rao! I can see your Prenatal Massage with Priya Nair on Friday 30 October "
                        "at 15:00. Is this about that booking, or something new?")
    assert len(sent) == 1  # asked by the graph itself: no call to Pebby's model yet
    # Their answer goes to Pebby, with the whole conversation.
    assert respond("thread-back", "The existing one, I want to move it").answer == "Sure, let's move it."


def test_a_request_made_before_the_details_is_answered_not_the_existing_booking_question(desk, monkeypatch):
    # As seen live: "i want face wax", then the details, was met with "is this about your facial?".
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha Rao", "(408) 555-0101")
    sent = chat(monkeypatch, details(asked="face wax"), details("Asha Rao", "(408) 555-0101"),
                says("Happy to help with a Full Face Wax! Which day and time suit you?"))
    respond("thread-first", "i want face wax")
    turn = respond("thread-first", "Asha Rao, (408) 555-0101")
    assert turn.answer == "Happy to help with a Full Face Wax! Which day and time suit you?"
    assert any(m.content == "i want face wax" for m in sent[-1])  # Pebby saw the first request


def test_on_a_call_the_question_is_spoken_and_uses_the_caller_id(desk, monkeypatch):
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Raja", "+19257198037")
    chat(monkeypatch, details(), details(name="Raja"))
    respond("thread-call", "Hi", channel="voice", caller="+19257198037")  # asks for the name only
    assert respond("thread-call", "Raja", channel="voice", caller="+19257198037").answer == (
        "Thank you, Raja! I can see your Prenatal Massage with Priya Nair on Friday the 30th of October at 3 PM. "
        "Is this about that booking, or something new?")


def test_a_calendar_that_cannot_be_read_just_skips_the_question(desk, monkeypatch):
    def down():
        raise CalendarUnavailable("timed out")
    monkeypatch.setattr(identify, "get_appointments", down)
    chat(monkeypatch, details("Asha Rao", "(408) 555-0101"))
    assert respond("thread-down", "Asha Rao, (408) 555-0101").answer == "Thank you, Asha Rao! How can I help you today?"


def test_the_tools_only_reach_this_customers_bookings(desk, monkeypatch):
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Ben", "408 555 0102")
    ask = next(t for t in tools.agent_tools("Asha Rao", "(408) 555-0101") if t.name == "booking")
    assert ask.run({"intent": "my_bookings"}) == {"status": "bookings"}  # none of Ben's


# ---- when it cannot go on --------------------------------------------------------------

def test_too_many_steps_hands_the_customer_to_the_staff(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-stuck")
    looping = [asks("search_guide", {"question": "hours"}, f"call-{n}") for n in range(MAX_TOOL_ROUNDS + 1)]
    chat(monkeypatch, *looping)
    assert respond("thread-stuck", "Help me book").answer == HANDOVER
    handoff = json.loads((tmp_path / "handoffs.jsonl").read_text())
    assert handoff["name"] == "Asha Rao" and handoff["phone"] == "(408) 555-0101"


def test_a_request_repeated_unchanged_is_not_run_again_and_goes_to_the_staff(desk, monkeypatch, tmp_path):
    # As seen with an open model: a made-up reference, cancelled again and again.
    identified(monkeypatch, "thread-loop")
    made_up = [books({"intent": "cancel", "references": ["BK789XYZ"]}, f"call-{n}") for n in range(1 + nodes.MAX_REPEATS)]
    sent = chat(monkeypatch, *made_up)
    assert respond("thread-loop", "3").answer == HANDOVER
    assert len(sent) == 1 + nodes.MAX_REPEATS  # stopped before the round limit
    # The first try ran: a reference that is not one of theirs is not named, so they are asked which booking.
    assert json.loads(sent[1][-1].content)["missing"] == ["references"]
    assert json.loads(sent[2][-1].content)["error"] == nodes.REPEATED  # the repeat did not
    assert json.loads((tmp_path / "handoffs.jsonl").read_text())["phone"] == "(408) 555-0101"
    # Every request has exactly one result, so the conversation can go on.
    results = [m.tool_call_id for m in memory("thread-loop")["messages"] if isinstance(m, ToolMessage)]
    assert sorted(results) == sorted(set(results))


def test_after_a_repeated_request_pebby_must_answer(desk, monkeypatch):
    # As seen live: the same search_guide three times, then a handover, instead of an answer.
    identified(monkeypatch, "thread-answer")
    search = {"question": "facial hair wax services"}
    chat(monkeypatch, asks("search_guide", search), asks("search_guide", search, "call-2"),
         says("We offer Full Face Wax, Lip Wax and Eyebrow Wax."))
    monkeypatch.setattr(tools, "search_guide", lambda args: {"excerpts": []})
    forced = []
    real = nodes.structured

    def recording(*args, answer_now=False, **kwargs):
        forced.append(answer_now)
        return real(*args, answer_now=answer_now, **kwargs)
    monkeypatch.setattr(nodes, "structured", recording)
    assert respond("thread-answer", "i want to book facial hair wax").answer == (
        "We offer Full Face Wax, Lip Wax and Eyebrow Wax.")  # answered, not handed over
    assert forced == [False, False, True]  # the call after the repeat had to answer


def test_two_requests_repeated_at_once_are_one_repeat_and_pebby_then_answers(desk, monkeypatch):
    # As seen live: two searches, both repeated in the next round, went straight to the staff.
    identified(monkeypatch, "thread-two-repeats")
    both = [{"name": "search_guide", "args": {"question": q}, "id": f"{q}-{n}"} for n in (1, 2)
            for q in ("Lip Wax", "Scalp Massage")]
    chat(monkeypatch, AIMessage(content="", tool_calls=both[:2]), AIMessage(content="", tool_calls=both[2:]),
         says("Lip Wax is $18; a scalp massage add-on is $20."))
    monkeypatch.setattr(tools, "search_guide", lambda args: {"excerpts": []})
    assert respond("thread-two-repeats", "lip wax and scalp massage?").answer == (
        "Lip Wax is $18; a scalp massage add-on is $20.")


def test_a_handoff_asked_for_again_in_the_same_turn_is_written_once(desk, monkeypatch, tmp_path):
    # As seen with an open model: the same hand_off six times in one turn.
    identified(monkeypatch, "thread-six")
    again = asks("hand_off", {"reason": "Bridal package."}, "call-2")
    chat(monkeypatch, asks("hand_off", {"reason": "Bridal package."}), again, says("Our staff will call you."))
    assert respond("thread-six", "bridal").answer == "Our staff will call you."
    assert len((tmp_path / "handoffs.jsonl").read_text().splitlines()) == 1
    # Repeated until the staff take over: still one handoff, not a second one for giving up.
    chat(monkeypatch, *[asks("hand_off", {"reason": "Bridal again."}, f"call-{n}") for n in range(3)])
    assert respond("thread-six", "bridal?").answer == HANDOVER
    assert len((tmp_path / "handoffs.jsonl").read_text().splitlines()) == 2


def test_the_agent_can_hand_off_when_asked_for_a_person(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-person")
    chat(monkeypatch, asks("hand_off", {"reason": "Wants to talk to a person about a group booking."}),
         says("Our team will call you shortly."))
    assert respond("thread-person", "Can I talk to someone?").answer == "Our team will call you shortly."
    assert json.loads((tmp_path / "handoffs.jsonl").read_text())["reason"].startswith("Wants to talk")


def test_feedback_goes_to_the_feedback_agent_and_a_complaint_to_the_staff(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-feedback")
    reading = Reading(sentiment="negative", topic="facilities", summary="The room was cold.", complaint=True)
    sent = chat(monkeypatch, asks("record_feedback", {"words": "My massage room was freezing yesterday."}),
                AIMessage(content=reading.model_dump_json()), says("I'm sorry. Our staff will call you."))
    assert respond("thread-feedback", "My massage room was freezing yesterday.").answer == (
        "I'm sorry. Our staff will call you.")
    assert json.loads((tmp_path / "feedback.jsonl").read_text())["topic"] == "facilities"
    assert "The room was cold." in json.loads((tmp_path / "handoffs.jsonl").read_text())["reason"]
    # Pebby was told what to say.
    assert "staff will call" in json.loads(sent[-1][-1].content)["next"]


def test_a_turn_that_fails_part_way_is_forgotten_so_the_next_one_works(desk, monkeypatch):
    identified(monkeypatch, "thread-broken")
    asked = books({**PRENATAL, "notes": "after my yoga class"})
    chat(monkeypatch, asked)  # after the yes, the Booking agent's model fails: the stand-in has no more replies
    assert respond("thread-broken", SAYS + ", after my yoga class").confirming
    with pytest.raises(Exception):
        respond("thread-broken", "Yes, that's right")
    # No tool request is left without its result, so the next turn is not refused.
    sent = chat(monkeypatch, says("How can I help?"))
    assert respond("thread-broken", "Hello?").answer == "How can I help?"
    calls = [call["id"] for m in sent[0] for call in getattr(m, "tool_calls", None) or []]
    results = [m.tool_call_id for m in sent[0] if isinstance(m, ToolMessage)]
    assert sorted(calls) == sorted(results)


def test_a_yes_that_changes_the_plan_is_not_a_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-but")
    chat(monkeypatch, books(), says("Sure, let me look at 4pm."))
    assert to_plan("thread-but", SAYS).confirming
    assert respond("thread-but", "Yes, but can we do 4pm instead?").answer == "Sure, let me look at 4pm."
    assert desk.day_bookings(FRIDAY) == []


def test_a_calendar_failing_while_booking_goes_to_the_staff(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-down")
    chat(monkeypatch, books(), says("Booking is unavailable right now."))
    assert to_plan("thread-down", SAYS).confirming

    def down(*args, **kwargs):
        raise CalendarUnavailable("timed out")
    monkeypatch.setattr(desk, "book", down)
    assert respond("thread-down", "yes").answer == "Booking is unavailable right now."
    assert "check for partial bookings" in json.loads((tmp_path / "handoffs.jsonl").read_text())["reason"]


def test_a_list_written_into_the_answer_is_shown_once_under_it(desk, monkeypatch):
    # As seen live: the facials were listed in the answer and again in `items`.
    identified(monkeypatch, "thread-list")
    items = [{"name": "Glow Express Facial", "detail": "30 minutes, $80"},
             {"name": "Deep Renewal Facial", "detail": "75 minutes, $150"}]
    reply = AgentReply(answer="We offer several facials:\n\n- Glow Express Facial: 30 minutes, $80\n"
                              "- Deep Renewal Facial: 75 minutes, $150", items=items, suggestions=[])
    chat(monkeypatch, AIMessage(content=reply.model_dump_json()))
    turn = respond("thread-list", "How much is a facial?")
    assert turn.answer == "We offer several facials:" and len(turn.items) == 2


def test_markdown_never_reaches_the_customer(desk, monkeypatch):
    identified(monkeypatch, "thread-md")
    chat(monkeypatch, says("Your appointment is confirmed!\n- **Lip Wax** with **Hannah Lee**\n- **Time**: 14:00"))
    assert respond("thread-md", "thanks").answer == ("Your appointment is confirmed!\nLip Wax with Hannah Lee\n"
                                                     "Time: 14:00")


# ---- a clarifying question, on either way of replying (models/structured.py) ----------------------

def answered(answer: str) -> AIMessage:
    """An open model's reply, given as the answer tool's call."""
    reply = AgentReply(answer=answer, items=[], suggestions=[])
    return AIMessage(content="", tool_calls=[{"name": "answer", "args": reply.model_dump(), "id": "answer-1"}])


@pytest.mark.parametrize("method, reply", [("native", says("Which facial would you like?")),
                                           ("answer-tool", answered("Which facial would you like?")),
                                           ("answer-tool", AIMessage(content="Which facial would you like?"))])
def test_pebby_asks_a_clarifying_question_in_one_model_call_and_no_tool(desk, monkeypatch, tmp_path, method, reply):
    identified(monkeypatch, f"thread-ask-{method}-{bool(reply.tool_calls)}")
    sent = use_standin(monkeypatch, [reply], method=method)
    used = []
    monkeypatch.setattr(nodes, "run_requests", lambda requests, offered: used.extend(requests) or [])
    turn = respond(f"thread-ask-{method}-{bool(reply.tool_calls)}", "want to book facial")
    assert turn.answer == "Which facial would you like?" and not turn.confirming
    assert len(sent) == 1 and used == []  # one model call; no booking, no hand_off, no other tool
    assert not (tmp_path / "handoffs.jsonl").exists()
    # What is remembered is the question as text, with no tool call left without its result.
    last = memory(f"thread-ask-{method}-{bool(reply.tool_calls)}")["messages"][-1]
    assert not last.tool_calls and "Which facial would you like?" in last.text


def test_a_reply_that_comes_with_another_tool_request_ends_the_turn_without_running_it(desk, monkeypatch):
    identified(monkeypatch, "thread-both")
    both = answered("Which day suits you?")
    both.tool_calls.append({"name": "hand_off", "args": {"reason": "?"}, "id": "extra-1"})
    sent = use_standin(monkeypatch, [both], method="answer-tool")
    assert respond("thread-both", "a lip wax please").answer == "Which day suits you?"
    assert len(sent) == 1  # the reply ends the turn: no further model round
    assert not memory("thread-both")["messages"][-1].tool_calls  # and nothing is left unanswered


def test_on_an_open_model_a_yes_to_pebbys_question_counts_as_the_customers_words(desk, monkeypatch):
    identified(monkeypatch, "thread-yes-open")
    use_standin(monkeypatch, [answered("Would that be the Signature Radiance Facial?")], method="answer-tool")
    respond("thread-yes-open", "i want a facial")
    words = customer_words([*memory("thread-yes-open")["messages"], HumanMessage("yes")])
    assert "signature radiance facial" in words


# ---- the route --------------------------------------------------------------------------

def test_the_route_runs_a_turn_and_hands_over_when_anything_fails(desk, monkeypatch, tmp_path):
    client = TestClient(app)
    chat(monkeypatch, details())  # "Hi": a greeting only
    reply = client.post("/api/assistant", json={"thread": "thread-route", "message": "Hi"}).json()
    assert reply == {"answer": "Before we start, may I have your name and phone number?", "items": [],
                     "suggestions": [], "confirming": False}

    # A model that cannot be reached, or any other failure: never an error, always the handover.
    for failure in (ModelError("unavailable", "standin", "down"), KeyError("a bug")):
        def broken(*args, failure=failure, **kwargs):
            raise failure
        monkeypatch.setattr("spadesk.agent.routes.respond", broken)
        reply = client.post("/api/assistant", json={"thread": "thread-route", "message": "Book me in"})
        assert reply.status_code == 200 and reply.json()["answer"] == HANDOVER
    # Each failure was written down for the staff, with what the customer asked.
    handoffs = (tmp_path / "handoffs.jsonl").read_text().splitlines()
    assert len(handoffs) == 2 and "Book me in" in json.loads(handoffs[0])["reason"]


def test_debug_mode_shows_the_steps_between_the_agents_only_when_it_is_on(desk, monkeypatch):
    def ask(thread):
        identified(monkeypatch, thread)
        chat(monkeypatch, books(), says("unused"))
        return TestClient(app).post("/api/assistant", json={"thread": thread, "message": SAYS}).json()

    monkeypatch.setenv("SPADESK_DEBUG", "true")
    steps = ask("thread-debug-on")["debug"]
    # Pebby -> the booking tool -> Collect (code) -> the yes -> the customer: the details are read back.
    assert [(step["who"], step["to"]) for step in steps] == [
        ("Pebby", "booking"), ("Collect", "The yes"), ("The yes", "Customer")]
    assert [step["kinds"] for step in steps] == [["ai", "tool"], ["code", "code"], ["code", "person"]]
    assert steps[0]["what"] == 'intent new · Prenatal Massage · day "30 October" · time "after 3pm"'
    assert steps[1]["what"] == "Prenatal Massage · day 2026-10-30 · after 15:00 · nothing missing"
    assert steps[2]["what"].startswith("Just to check before I look for times")
    monkeypatch.setenv("SPADESK_DEBUG", "false")
    assert "debug" not in ask("thread-debug-off")


# ---- turns replaced by newer ones (agent/pebby/turns.py) ---------------------------------------

@contextmanager
def interrupted(module, name: str, thread: str):
    """Within the block, `module.name` behaves as before, but a newer turn of `thread`
    arrives while it runs, as when a caller keeps talking after a pause."""
    real = getattr(module, name)

    def with_newer_turn(*args, **kwargs):
        turns.arrive(thread)
        return real(*args, **kwargs)
    setattr(module, name, with_newer_turn)
    try:
        yield
    finally:
        setattr(module, name, real)


def test_a_replaced_turn_is_dropped_and_forgotten(desk, monkeypatch):
    identified(monkeypatch, "thread-paused")
    chat(monkeypatch, says("Shall we say today?"), says("Sunday works: 10 or 12:30."))
    monkeypatch.setattr(agent, "WAIT_FOR_NEWER", 0.05)  # the newer turn here never runs, so don't wait for it
    with interrupted(nodes, "structured", "thread-paused"):
        assert respond("thread-paused", "I'd like a prenatal with Priya") is None  # its own reply is dropped
    assert respond("thread-paused", "I'd like a prenatal with Priya on Sunday").answer == "Sunday works: 10 or 12:30."
    remembered = [m.content for m in memory("thread-paused")["messages"]]
    # Pebby remembers what the customer finally said and heard, not the replaced turn.
    assert "I'd like a prenatal with Priya on Sunday" in remembered
    assert "I'd like a prenatal with Priya" not in remembered
    assert not any("Shall we say today?" in str(m) for m in remembered)


def test_a_turn_waiting_behind_a_newer_one_never_runs_but_gets_its_reply(desk, monkeypatch):
    identified(monkeypatch, "thread-queue")
    chat(monkeypatch, says("Only the newest is answered."))
    results = {}
    with turns.lock("thread-queue"):  # a turn is running: the next two have to wait
        first = threading.Thread(target=lambda: results.update(first=respond("thread-queue", "Hi")))
        first.start()
        while turns._newest["thread-queue"] < 2:  # wait until the first has arrived and is waiting
            time.sleep(0.01)
        second = threading.Thread(target=lambda: results.update(second=respond("thread-queue", "Hi, hello")))
        second.start()
        while turns._newest["thread-queue"] < 3:
            time.sleep(0.01)
    first.join(), second.join()
    # The model answered once (the stand-in has one reply), and both requests were answered with it.
    assert results["first"].answer == results["second"].answer == "Only the newest is answered."


def test_a_replaced_turn_that_booked_is_remembered_so_it_is_not_booked_twice(desk, monkeypatch):
    identified(monkeypatch, "thread-yes")
    chat(monkeypatch, books(), says("Booked!"), says("All set, you're booked."))
    assert to_plan("thread-yes", SAYS).confirming
    monkeypatch.setattr(agent, "WAIT_FOR_NEWER", 0.05)
    with interrupted(nodes, "is_yes", "thread-yes"):
        assert respond("thread-yes", "Yes") is None  # replaced, but the booking was made
    # The newer turn is not asked to confirm again: the memory kept the booking.
    turn = respond("thread-yes", "Yes, book it please")
    assert not turn.confirming and turn.answer == "All set, you're booked."
    assert len(desk.day_bookings(FRIDAY)) == 1
