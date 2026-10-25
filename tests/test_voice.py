"""Tests for the phone line (spadesk/voice/): the endpoint ElevenLabs calls on each turn.

Requests are built the way ElevenLabs sends them (the whole call so far, in
the OpenAI chat format), and the model is a stand-in, so these need no
ElevenLabs or Twilio account and spend nothing.

Run with:  .venv/bin/python -m pytest tests/test_voice.py
"""

import json
import time
from datetime import datetime

import pytest
from conftest import signed_in
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from standin_calendar import standin_desk
from standins import use_standin

from spadesk.agent import booking as booking_agent
from spadesk.agent import executor as approved
from spadesk.agent import pebby as identify
from spadesk.agent import pebby as tools
from spadesk.agent import yes as confirm
from spadesk.agent.booking import Plan, PlanItem
from spadesk.agent.flow import get_graph
from spadesk.agent.pebby import AgentReply, Details
from spadesk.agent.staff import HANDOVER
from spadesk.app import app
from spadesk.models import ModelError
from spadesk.voice import routes
from spadesk.voice.call import read_call
from spadesk.voice.routes import FILLER

SECRET = "test-voice-secret"
PROMPT = "You are Pebby on the phone.\nCall ID: conv_123\nCaller number: +1 408 555 0101"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """The app with the phone line on, an in-memory calendar with a fixed clock, and a fresh memory."""
    monkeypatch.setenv("SPADESK_VOICE_SECRET", SECRET)
    monkeypatch.setenv("SPADESK_HANDOFFS", str(tmp_path / "handoffs.jsonl"))
    desk = standin_desk()
    for module in (tools, confirm, identify, approved, booking_agent):
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
    use_standin(monkeypatch, [AIMessage(content=Details(name="", phone="", asked="I'd like a massage").model_dump_json())])
    reply = turn(client, "Hi, I'd like a massage", stream=True)
    assert reply.headers["content-type"].startswith("text/event-stream")
    said = spoken(reply)
    assert said.startswith("May I have your name") and "ending in 0101" in said
    assert FILLER not in said  # a quick fixed reply needs no filler


def test_once_named_pebby_answers_after_a_filler_using_the_caller_number(client, monkeypatch):
    sent = use_standin(monkeypatch, [
        AIMessage(content=Details(name="", phone="", asked="I'd like a massage").model_dump_json()),
        AIMessage(content=Details(name="Asha", phone="").model_dump_json()),
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


def test_a_quick_reply_has_no_filler_and_silence_gets_a_nudge(client, monkeypatch):
    use_standin(monkeypatch, [AIMessage(content=Details(name="", phone="").model_dump_json())])
    assert FILLER not in spoken(turn(client, "Hi", stream=True))  # the stand-in answers at once
    # Never an empty reply: ElevenLabs would take it as an error and drop the call.
    assert spoken(turn(client, "Hi", "May I have your name?", "...", stream=True)) == routes.NUDGE


def test_a_request_without_a_call_id_is_handed_over_not_dropped(client):
    reply = turn(client, "Hi", prompt="You are Pebby.")
    assert reply.json()["choices"][0]["message"]["content"] == HANDOVER


def test_the_plan_to_confirm_is_worded_to_be_spoken(client):
    plan = Plan(summary="", items=[
        PlanItem(service="Prenatal Massage", therapist="Priya Nair", start=datetime(2026, 10, 30, 15, 0)),
        PlanItem(service="Signature Radiance Facial", therapist="Sofia Martinez", start=datetime(2026, 10, 30, 16, 30))])
    assert confirm.plan_question(plan, "Asha", "+1 408 555 0101", spoken=True) == (
        "Just to confirm: a 60-minute Prenatal Massage with Priya Nair on Friday the 30th of October at 3 PM, "
        "then a 60-minute Signature Radiance Facial with Sofia Martinez on Friday the 30th of October at 4:30 PM, "
        "for Asha. Shall I book them?")


def test_the_voice_widget_is_the_owners_and_shows_only_when_an_agent_is_set(monkeypatch):
    assert TestClient(app).get("/api/voice-widget").status_code == 401  # not signed in
    client = signed_in(app)
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


def test_a_change_is_said_as_what_is_cancelled_and_what_is_booked(client):
    plan = Plan(summary="", items=[PlanItem(service="Deep Tissue Massage", therapist="Daniel Ruiz",
                                            start=datetime(2026, 10, 26, 17, 0))])
    old = {"reference": "BK-7B1DA6", "service": "Swedish Massage", "therapist": "Daniel Ruiz",
           "start": "2026-10-26T17:00"}
    assert confirm.plan_question(plan, "Asha", "+1 408 555 0101", spoken=True, replaces=old) == (
        "Just to confirm: I'll cancel your Swedish Massage on Monday the 26th of October at 5 PM and book a "
        "60-minute Deep Tissue Massage with Daniel Ruiz on Monday the 26th of October at 5 PM instead, for Asha. "
        "Shall I go ahead?")
