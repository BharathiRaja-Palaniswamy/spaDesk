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
from langchain_core.messages import AIMessage, ToolMessage
from standin_calendar import standin_desk
from standins import use_standin

from spadesk import agent
from spadesk.agent import confirm, identify, nodes, respond, tools, turns
from spadesk.agent.graph import get_graph
from spadesk.agent.handoffs import HANDOVER
from spadesk.agent.identify import Details
from spadesk.agent.nodes import MAX_TOOL_ROUNDS
from spadesk.agent.prompt import AgentReply
from spadesk.app import app
from spadesk.booking import actions
from spadesk.booking.errors import CalendarUnavailable
from spadesk.models import ModelError

ASHA = Details(name="Asha Rao", phone="(408) 555-0101", asks=False)
PRENATAL = {"service": "Prenatal Massage", "therapist": "Priya Nair", "start": "2026-10-30T15:00"}


def details(name="", phone="", asks=False) -> AIMessage:
    """The small model reading a message for a name and number."""
    return AIMessage(content=Details(name=name, phone=phone, asks=asks).model_dump_json())


def says(answer: str) -> AIMessage:
    """The agent's reply."""
    return AIMessage(content=AgentReply(answer=answer, items=[], suggestions=[]).model_dump_json())


def asks(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    """The agent asking for one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def desk(monkeypatch, tmp_path):
    """One in-memory calendar with a fixed clock, for every part of the agent; a fresh memory."""
    desk = standin_desk()
    for module in (tools, confirm, identify, actions):
        monkeypatch.setattr(module, "get_appointments", lambda: desk)
    monkeypatch.setenv("SPADESK_HANDOFFS", str(tmp_path / "handoffs.jsonl"))
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
    sent = chat(monkeypatch, details(asks=True), details("Asha Rao", "(408) 555-0101"), says("We open at 10."))
    assert respond("thread-one", "What time do you open?").answer == ("Before we start, may I have your name "
                                                                       "and phone number?")
    # Once the details are given, the question asked first is answered.
    assert respond("thread-one", "Asha Rao, (408) 555-0101").answer == "We open at 10."
    assert memory("thread-one")["name"] == "Asha Rao" and memory("thread-one")["phone"] == "(408) 555-0101"
    # The agent was told who the customer is, and saw the first question.
    agent_call = sent[-1]
    assert "Asha Rao, phone (408) 555-0101" in agent_call[0].content
    assert any(m.content == "What time do you open?" for m in agent_call)


def test_only_what_is_missing_is_asked_for(desk, monkeypatch):
    chat(monkeypatch, details(name="Asha"), details(phone="12"))
    assert respond("thread-two", "I'm Asha").answer == "Thank you, Asha! And a phone number we can reach you on?"
    # A number with fewer than 7 digits is not taken.
    assert respond("thread-two", "12").answer == "Thank you, Asha! And a phone number we can reach you on?"


# ---- booking needs the customer's yes ------------------------------------------------

def test_a_booking_is_made_only_after_the_customer_says_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-book")
    chat(monkeypatch, asks("find_free_times", {"service": "Prenatal Massage", "date": "2026-10-30"}),
         asks("book_appointment", PRENATAL, "call-2"), says("You're booked!"))
    turn = respond("thread-book", "A prenatal massage on Friday 30 October at 3pm, please")
    assert turn.confirming and turn.suggestions == ["Yes, book it", "No"]
    assert turn.answer == ("Shall I book Prenatal Massage (60 minutes) with Priya Nair on Friday 30 October at "
                           "15:00, for Asha Rao ((408) 555-0101)?")
    assert desk.day_bookings(datetime(2026, 10, 30).date()) == []  # nothing yet

    assert respond("thread-book", "Yes, book it").answer == "You're booked!"
    [booking] = desk.day_bookings(datetime(2026, 10, 30).date())
    # Booked under the details the customer gave, not anything the model wrote.
    assert (booking.name, booking.phone, booking.therapist) == ("Asha Rao", "(408) 555-0101", "Priya Nair")


def test_anything_but_yes_books_nothing_and_goes_back_to_the_agent(desk, monkeypatch):
    identified(monkeypatch, "thread-no")
    sent = chat(monkeypatch, asks("book_appointment", PRENATAL), says("Sure, which time instead?"))
    assert respond("thread-no", "Book Friday 3pm with Priya").confirming
    assert respond("thread-no", "Actually, can we do 4pm?").answer == "Sure, which time instead?"
    assert desk.day_bookings(datetime(2026, 10, 30).date()) == []
    # The model learns nothing was booked, then reads the customer's reply (after its time note), in that order.
    *_, result, note, reply = sent[-1]
    assert isinstance(result, ToolMessage) and "did not say yes" in json.loads(result.content)["not_done"]
    assert note.content.startswith("(Spa local time:") and reply.content == "Actually, can we do 4pm?"


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


def test_a_booking_that_would_fail_is_not_offered_for_a_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-taken")
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Ben", "408 555 0102")
    sent = chat(monkeypatch, asks("book_appointment", PRENATAL), says("That time was just taken."))
    turn = respond("thread-taken", "Book Friday 3pm with Priya")
    assert not turn.confirming and turn.answer == "That time was just taken."
    assert "not free" in json.loads(sent[-1][-1].content)["error"]


def test_cancelling_also_waits_for_a_yes(desk, monkeypatch):
    identified(monkeypatch, "thread-cancel")
    booking = desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha Rao", "4085550101")
    chat(monkeypatch, asks("cancel_booking", {"reference": booking.id}), says("Cancelled."))
    turn = respond("thread-cancel", "Cancel my Friday massage")
    assert turn.confirming and booking.id in turn.answer and turn.suggestions == ["Yes, cancel it", "No, keep it"]
    assert respond("thread-cancel", "yes").answer == "Cancelled."
    assert desk.day_bookings(datetime(2026, 10, 30).date()) == []


def test_a_returning_customer_is_asked_about_their_upcoming_booking_first(desk, monkeypatch):
    desk.book("Prenatal Massage", "Priya Nair", datetime(2026, 10, 30, 15, 0), "Asha Rao", "(408) 555-0101")
    sent = chat(monkeypatch, details("Asha Rao", "408-555-0101", asks=True), says("Sure, let's move it."))
    question = respond("thread-back", "Hi, it's Asha, 408-555-0101, I need help").answer
    assert question == ("Thank you, Asha Rao! I can see your Prenatal Massage with Priya Nair on Friday 30 October "
                        "at 15:00. Is this about that booking, or something new?")
    assert len(sent) == 1  # asked by the graph itself: no call to Pebby's model yet
    # Their answer goes to Pebby, with the whole conversation.
    assert respond("thread-back", "The existing one, I want to move it").answer == "Sure, let's move it."


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
    mine = next(t for t in tools.agent_tools("Asha Rao", "(408) 555-0101") if t.name == "my_bookings")
    assert mine.run({}) == {"bookings": []}


# ---- when it cannot go on --------------------------------------------------------------

def test_too_many_steps_hands_the_customer_to_the_staff(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-stuck")
    looping = [asks("search_guide", {"question": "hours"}, f"call-{n}") for n in range(MAX_TOOL_ROUNDS + 1)]
    chat(monkeypatch, *looping)
    assert respond("thread-stuck", "Help me book").answer == HANDOVER
    handoff = json.loads((tmp_path / "handoffs.jsonl").read_text())
    assert handoff["name"] == "Asha Rao" and handoff["phone"] == "(408) 555-0101"


def test_the_agent_can_hand_off_when_asked_for_a_person(desk, monkeypatch, tmp_path):
    identified(monkeypatch, "thread-person")
    chat(monkeypatch, asks("hand_off", {"reason": "Wants to talk to a person about a group booking."}),
         says("Our team will call you shortly."))
    assert respond("thread-person", "Can I talk to someone?").answer == "Our team will call you shortly."
    assert json.loads((tmp_path / "handoffs.jsonl").read_text())["reason"].startswith("Wants to talk")


# ---- the route --------------------------------------------------------------------------

def test_the_route_runs_a_turn_and_hands_over_when_anything_fails(desk, monkeypatch, tmp_path):
    client = TestClient(app)
    chat(monkeypatch, details(asks=True))
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


# ---- turns replaced by newer ones (agent/turns.py) ---------------------------------------

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
    chat(monkeypatch, asks("book_appointment", PRENATAL), says("Booked!"), says("All set, you're booked."))
    assert respond("thread-yes", "Book Friday 3pm with Priya").confirming
    monkeypatch.setattr(agent, "WAIT_FOR_NEWER", 0.05)
    with interrupted(nodes, "is_yes", "thread-yes"):
        assert respond("thread-yes", "Yes") is None  # replaced, but the booking was made
    # The newer turn is not asked to confirm again: the memory kept the booking.
    turn = respond("thread-yes", "Yes, book it please")
    assert not turn.confirming and turn.answer == "All set, you're booked."
    assert len(desk.day_bookings(datetime(2026, 10, 30).date())) == 1
