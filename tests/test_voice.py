"""Tests for the phone line (spadesk/voice/): the endpoint ElevenLabs calls on each turn.

Requests are built the way ElevenLabs sends them (the whole call so far, in
the OpenAI chat format), and the model is a stand-in, so these need no
ElevenLabs or Twilio account and spend nothing.

Run with:  .venv/bin/python -m pytest tests/test_voice.py
"""

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from standins import use_standin

from spadesk.agent import confirm, tools
from spadesk.agent.graph import get_graph
from spadesk.agent.handoffs import HANDOVER
from spadesk.agent.identify import Details
from spadesk.agent.prompt import AgentReply
from spadesk.app import app
from spadesk.booking import tools as booking_tools
from spadesk.booking.appointments import Appointments
from spadesk.booking.calendars.memory import MemoryCalendar
from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge
from spadesk.models import ModelError
from spadesk.voice import routes
from spadesk.voice.call import read_call
from spadesk.voice.routes import FILLER

SPA = ZoneInfo("America/Los_Angeles")
SECRET = "test-voice-secret"
PROMPT = "You are Pebby on the phone.\nCall ID: conv_123\nCaller number: +1 408 555 0101"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """The app with the phone line on, an in-memory calendar with a fixed clock, and a fresh memory."""
    monkeypatch.setenv("SPADESK_VOICE_SECRET", SECRET)
    monkeypatch.setenv("SPADESK_HANDOFFS", str(tmp_path / "handoffs.jsonl"))
    desk = Appointments(read_schedule(get_knowledge()), MemoryCalendar(), SPA, 30, notice=30,
                        now=lambda: datetime(2026, 10, 22, 14, 10, tzinfo=SPA))
    for module in (tools, confirm, booking_tools):
        monkeypatch.setattr(module, "get_appointments", lambda: desk)
    get_graph.cache_clear()
    yield TestClient(app)
    get_graph.cache_clear()


def turn(client, *lines: str, stream: bool = False, prompt: str = PROMPT, key: str = SECRET):
    """Sends a call's lines (caller and agent taking turns, caller first) as ElevenLabs would."""
    messages = [{"role": "system", "content": prompt}]
    messages += [{"role": "user" if n % 2 == 0 else "assistant", "content": line} for n, line in enumerate(lines)]
    return client.post("/v1/chat/completions", json={"model": "pebby", "messages": messages, "stream": stream},
                       headers={"Authorization": f"Bearer {key}"})


def spoken(response) -> str:
    """What the caller hears: the text of every streamed chunk, joined."""
    events = [line.removeprefix("data: ") for line in response.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])


def test_the_call_id_and_caller_are_read_from_the_prompt():
    call = read_call([{"role": "system", "content": PROMPT}, {"role": "user", "content": "Hi there"}])
    assert (call.id, call.caller, call.said) == ("conv_123", "+1 408 555 0101", "Hi there")


def test_only_elevenlabs_with_the_secret_may_call(client, monkeypatch):
    assert turn(client, "Hi", key="wrong").status_code == 401
    monkeypatch.setenv("SPADESK_VOICE_SECRET", "")
    assert turn(client, "Hi").status_code == 503  # the line is off without the setting


def test_a_caller_is_asked_only_for_their_name(client, monkeypatch):
    use_standin(monkeypatch, [AIMessage(content=Details(name="", phone="", asks=True).model_dump_json())])
    reply = turn(client, "Hi, I'd like a massage", stream=True)
    assert reply.headers["content-type"].startswith("text/event-stream")
    said = spoken(reply)
    assert said.startswith("May I have your name") and "ending in 0101" in said
    assert FILLER not in said  # a quick fixed reply needs no filler


def test_once_named_pebby_answers_after_a_filler_using_the_caller_number(client, monkeypatch):
    sent = use_standin(monkeypatch, [
        AIMessage(content=Details(name="", phone="", asks=True).model_dump_json()),
        AIMessage(content=Details(name="Asha", phone="", asks=False).model_dump_json()),
        AIMessage(content=AgentReply(answer="Thanks Asha! We have Swedish and deep tissue.", items=[],
                                     suggestions=[]).model_dump_json()),
        AIMessage(content=AgentReply(answer="Sure.", items=[], suggestions=[]).model_dump_json())])
    turn(client, "Hi, I'd like a massage", stream=True)
    assert spoken(turn(client, "Hi, I'd like a massage", "May I have your name?", "Asha", stream=True)) == \
        "Thanks Asha! We have Swedish and deep tissue."
    # Pebby was told it is on a phone call, with the caller's number as the phone number.
    assert "phone +1 408 555 0101" in sent[-1][0].content and "phone call" in sent[-1][0].content
    # A reply that takes longer than FILLER_AFTER starts with a filler while Pebby thinks.
    monkeypatch.setattr(routes, "FILLER_AFTER", 0)
    assert spoken(turn(client, "Hi", "...", "Asha", "...", "Which is best for back pain?", stream=True)) == \
        FILLER + "Sure."


def test_a_quick_reply_has_no_filler_and_silence_gets_no_reply(client, monkeypatch):
    use_standin(monkeypatch, [AIMessage(content=Details(name="", phone="", asks=False).model_dump_json())])
    assert FILLER not in spoken(turn(client, "Hi", stream=True))  # the stand-in answers at once
    assert spoken(turn(client, "Hi", "May I have your name?", "...", stream=True)) == ""


def test_a_request_without_a_call_id_is_handed_over_not_dropped(client):
    reply = turn(client, "Hi", prompt="You are Pebby.")
    assert reply.json()["choices"][0]["message"]["content"] == HANDOVER


def test_the_confirmation_is_worded_to_be_spoken(client):
    request = {"name": "book_appointment",
               "args": {"service": "Prenatal Massage", "therapist": "Priya Nair", "start": "2026-10-30T15:00"}}
    question, problem = confirm.problem_with(request, "Asha", "+1 408 555 0101", spoken=True)
    assert problem == "" and question == ("Just to confirm: a 60-minute Prenatal Massage with Priya Nair on "
                                          "Friday the 30th of October at 3 PM, for Asha. Shall I book it?")


def test_the_voice_widget_shows_only_when_an_agent_is_set(monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv("SPADESK_VOICE_AGENT_ID", "")
    assert client.get("/api/voice-widget").json() == {"agent_id": ""}
    monkeypatch.setenv("SPADESK_VOICE_AGENT_ID", "agent_abc123")
    assert client.get("/api/voice-widget").json() == {"agent_id": "agent_abc123"}


def test_a_failing_turn_is_handed_over_with_the_callers_number(client, monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise ModelError("unavailable", "standin", "down")
    monkeypatch.setattr(routes, "respond", broken)
    assert spoken(turn(client, "I'd like a massage", stream=True)) == HANDOVER
    handoff = json.loads((tmp_path / "handoffs.jsonl").read_text())
    assert handoff["phone"] == "+1 408 555 0101" and "I'd like a massage" in handoff["reason"]


def test_a_turn_that_takes_too_long_is_handed_over_before_the_line_drops(client, monkeypatch, tmp_path):
    def slow(*args, **kwargs):
        time.sleep(0.5)
    monkeypatch.setattr(routes, "respond", slow)
    monkeypatch.setattr(routes, "FILLER_AFTER", 0.05)
    monkeypatch.setattr(routes, "TURN_LIMIT", 0.2)
    assert spoken(turn(client, "Book Friday", stream=True)) == FILLER + HANDOVER
    assert "took too long" in json.loads((tmp_path / "handoffs.jsonl").read_text())["reason"]
