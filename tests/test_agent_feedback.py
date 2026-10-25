"""Tests for Pebby's feedback tool (spadesk/agent/pebby/feedback.py): feedback is
read, recorded, and a complaint is passed to the staff.

The model is a stand-in that plays back prepared readings (tests/standins.py),
so nothing is spent; the records go to a temporary folder.

Run with:  .venv/bin/python -m pytest tests/test_agent_feedback.py
"""

import json

import pytest
from langchain_core.messages import AIMessage
from standins import use_standin

from spadesk.agent.pebby.feedback import Reading, record_feedback
from spadesk.models import ModelError, gateway

PRAISE = Reading(sentiment="positive", topic="staff", summary="Loved Priya's massage.", complaint=False)
COLD_ROOM = Reading(sentiment="mixed", topic="facilities", summary="Great massage; the room was cold.",
                    complaint=True)


def reads(reading: Reading) -> AIMessage:
    """The small model's Reading."""
    return AIMessage(content=reading.model_dump_json())


@pytest.fixture
def files(monkeypatch, tmp_path):
    """Feedback and handoffs written to a temporary folder; returns a reader for each file's lines."""
    monkeypatch.setenv("SPADESK_FEEDBACK", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("SPADESK_HANDOFFS", str(tmp_path / "handoffs.jsonl"))

    def lines(name: str) -> list[dict]:
        path = tmp_path / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return lines


def test_praise_is_recorded_as_read_with_who_said_it(files, monkeypatch):
    sent = use_standin(monkeypatch, [reads(PRAISE)])
    told = record_feedback("Asha Rao", "(408) 555-0101", "Priya was wonderful, best massage ever!")
    assert told == {"recorded": True, "next": "Thank the customer for their feedback."}
    [entry] = files("feedback")
    assert (entry["name"], entry["sentiment"], entry["topic"]) == ("Asha Rao", "positive", "staff")
    assert entry["words"] == "Priya was wonderful, best massage ever!"
    assert files("handoffs") == []  # nothing for the staff to follow up
    assert sent[0][-1].content == "Priya was wonderful, best massage ever!"  # the model read their own words


def test_a_complaint_is_also_passed_to_the_staff(files, monkeypatch):
    use_standin(monkeypatch, [reads(COLD_ROOM)])
    told = record_feedback("Asha Rao", "(408) 555-0101", "Massage was great but the room was freezing.")
    assert told["handed_off"] and "staff will call" in told["next"]
    assert files("feedback")[0]["complaint"] is True
    [handoff] = files("handoffs")
    assert handoff["phone"] == "(408) 555-0101" and "the room was cold" in handoff["reason"]


def test_feedback_is_never_lost_when_the_model_cannot_be_reached(files, monkeypatch):
    def down(*args, **kwargs):
        raise ModelError("unavailable", "standin", "down")
    monkeypatch.setattr(gateway, "_first_that_works", down)
    told = record_feedback("Asha Rao", "(408) 555-0101", "The front desk was rude.")
    assert told["recorded"] and told["handed_off"]
    [entry] = files("feedback")
    assert entry["words"] == "The front desk was rude." and entry["unread"]
    assert "The front desk was rude." in files("handoffs")[0]["reason"]

