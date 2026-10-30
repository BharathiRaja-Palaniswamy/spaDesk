"""Tests for the Feedback agent (spadesk/agent/feedback.py): it reads feedback,
answers the customer, and files a card the owner can close.

The model is a stand-in that plays back prepared readings (tests/standins.py),
so nothing is spent; the cards go to a temporary folder.

Run with:  .venv/bin/python -m pytest tests/test_agent_feedback.py
"""

import json

import pytest
from conftest import signed_in
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from standins import use_standin

from spadesk.agent.feedback import THANKS, Reading, cards, close, give_feedback
from spadesk.app import app
from spadesk.models import ModelError, gateway

PRAISE = Reading(sentiment="positive", topic="staff", summary="Loved Priya's massage.", complaint=False,
                 reply="Thank you so much! We'll pass your kind words to Priya.")
COLD_ROOM = Reading(sentiment="mixed", topic="facilities", summary="Great massage; the room was cold.",
                    complaint=True, reply="So glad you enjoyed it, and sorry the room was cold.")


def reads(reading: Reading) -> AIMessage:
    """The Feedback agent's model reading the feedback."""
    return AIMessage(content=reading.model_dump_json())


@pytest.fixture
def lines(monkeypatch, tmp_path):
    """Feedback and handoffs in a temporary folder; returns a reader for a file's lines."""
    monkeypatch.setenv("SPADESK_FEEDBACK", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("SPADESK_HANDOFFS", str(tmp_path / "handoffs.jsonl"))

    def read(name: str) -> list[dict]:
        path = tmp_path / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return read


def test_feedback_is_read_answered_and_filed_as_an_open_card(lines, monkeypatch):
    sent = use_standin(monkeypatch, [reads(PRAISE)])
    told = give_feedback("Asha Rao", "(408) 555-0101", "Priya was wonderful, best massage ever!")
    assert told["reply_to_customer"] == PRAISE.reply and not told["complaint"]
    [card] = cards()
    assert (card["id"], card["status"], card["sentiment"], card["name"]) == (told["card"], "open", "positive",
                                                                             "Asha Rao")
    assert card["words"] == "Priya was wonderful, best massage ever!"
    assert sent[0][-1].content == "Priya was wonderful, best massage ever!"  # the model read their own words
    assert lines("handoffs") == []


def test_a_complaint_is_a_flagged_card_not_a_call_back(lines, monkeypatch):
    use_standin(monkeypatch, [reads(COLD_ROOM)])
    told = give_feedback("Asha Rao", "(408) 555-0101", "Massage was great but the room was freezing.")
    assert told["complaint"] and cards()[0]["complaint"] is True
    assert lines("handoffs") == []


def test_feedback_is_never_lost_when_the_model_cannot_be_reached(lines, monkeypatch):
    def down(*args, **kwargs):
        raise ModelError("unavailable", "standin", "down")
    monkeypatch.setattr(gateway, "_first_that_works", down)
    told = give_feedback("Asha Rao", "(408) 555-0101", "The front desk was rude.")
    assert told["reply_to_customer"] == THANKS
    [card] = cards()
    assert card["words"] == "The front desk was rude." and card["unread"]


def test_the_owner_closes_a_card_with_a_note_and_open_cards_come_first(lines, monkeypatch):
    use_standin(monkeypatch, [reads(PRAISE), reads(COLD_ROOM)])
    first = give_feedback("Asha", "4085550101", "Lovely massage")["card"]
    second = give_feedback("Ben", "4085550102", "Room was cold")["card"]
    closed = close(first, "Thanked Priya")
    assert (closed["status"], closed["owner_note"]) == ("closed", "Thanked Priya")
    assert [c["id"] for c in cards()] == [second, first]  # open first
    assert close(first) is None  # already closed
    assert len(lines("feedback")) == 3  # two cards and a closing: nothing rewritten


def test_only_the_signed_in_owner_sees_and_closes_cards(lines, monkeypatch):
    use_standin(monkeypatch, [reads(PRAISE)])
    card = give_feedback("Asha", "4085550101", "Lovely massage")["card"]
    assert TestClient(app).get("/api/feedback").status_code == 401
    assert TestClient(app).post(f"/api/feedback/{card}/close", json={}).status_code == 401
    owner = signed_in(app)
    assert [c["id"] for c in owner.get("/api/feedback").json()] == [card]
    assert owner.post(f"/api/feedback/{card}/close", json={"note": "Called her"}).json()["status"] == "closed"
    assert owner.post(f"/api/feedback/{card}/close", json={}).status_code == 404


def test_brief_feedback_always_asks_its_question(lines, monkeypatch):
    # The agent meant to ask, but its reply asked nothing: the question is added.
    brief = Reading(sentiment="negative", topic="other", summary="Unhappy.", complaint=True,
                    reply="We're sorry to hear about your experience.", needs_more=True)
    use_standin(monkeypatch, [reads(brief)])
    said = give_feedback("Asha", "4085550101", "bad service")
    assert said["asked_for_more"] and said["reply_to_customer"].endswith("and what happened?")


def test_brief_feedback_gets_one_question_and_the_answer_updates_the_same_card(lines, monkeypatch):
    # As asked: "Your glowing facial spa is worst" says neither which visit nor what happened.
    brief = Reading(sentiment="negative", topic="treatment", summary="Unhappy with a facial.", complaint=True,
                    reply="I'm so sorry to hear that. Which day was your facial, and what went wrong?",
                    needs_more=True)
    fuller = Reading(sentiment="negative", topic="treatment", summary="Thursday's facial left her skin red.",
                     complaint=True, reply="Thank you for telling us. Our team will look into it.", needs_more=True)
    sent = use_standin(monkeypatch, [reads(brief), reads(fuller)])
    first = give_feedback("Raja", "92822929", "Your glowing facial spa is worst")
    assert first["asked_for_more"] and first["reply_to_customer"].endswith("what went wrong?")
    assert cards()[0]["status"] == "open"  # filed straight away
    second = give_feedback("Raja", "92822929", "Thursday, my skin went red", first["card"])
    assert second["card"] == first["card"] and not second["asked_for_more"]  # one question only
    [card] = cards()
    assert card["summary"] == "Thursday's facial left her skin red."
    assert card["words"] == "Your glowing facial spa is worst\nThursday, my skin went red"
    assert "do not ask anything" in sent[1][0].content  # told not to ask again


def test_an_answer_cannot_be_added_to_someone_elses_card(lines, monkeypatch):
    use_standin(monkeypatch, [reads(COLD_ROOM), reads(PRAISE)])
    theirs = give_feedback("Ben", "4085550102", "Room was cold")["card"]
    mine = give_feedback("Asha", "4085550101", "Lovely massage", theirs)["card"]
    assert mine != theirs and len(cards()) == 2
