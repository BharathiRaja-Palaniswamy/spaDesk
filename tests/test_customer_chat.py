"""Tests for the customer assistant in spadesk/assistant/customer.py.

The model is replaced by a stand-in, so these run without an API key.

Run with:  .venv/bin/python -m pytest tests/test_customer_chat.py
"""

from types import SimpleNamespace

import anthropic
import pytest
from fastapi.testclient import TestClient

from spadesk.app import app
from spadesk.assistant.customer import CustomerReply
from spadesk.assistant.replies import ReplyItem
from spadesk.rag.stores import get_store


@pytest.fixture
def client():
    """A test browser for the app: makes requests without starting a server."""
    return TestClient(app)


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: records the request, returns a fixed reply."""

    last_request = None
    # What the "model" answers. A test can replace it; None means no structured reply.
    reply = CustomerReply(answer="We have three massages.", covered=True,
                          items=[ReplyItem(name="Swedish Massage", detail="60 minutes, $115")],
                          sections=["Massage Therapy", "Made-up Section"],
                          suggestions=["One?", "Two?", "Three?", "Four?"])

    def __init__(self):
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **request):
        FakeAnthropic.last_request = request
        return SimpleNamespace(parsed_output=FakeAnthropic.reply and FakeAnthropic.reply.model_copy(),
                               stop_reason="end_turn", content=[])


def test_chat_answers_from_the_profile_with_no_tools(client, monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    reply = client.post("/api/chat", json={"question": "Which massages do you offer?"})
    # The made-up section is dropped and at most three suggestions are passed on.
    assert reply.json() == {"answer": "We have three massages.", "covered": True,
                            "items": [{"name": "Swedish Massage", "detail": "60 minutes, $115"}],
                            "sections": ["Massage Therapy"], "suggestions": ["One?", "Two?", "Three?"],
                            "rag": False, "chunks": 0}

    sent = FakeAnthropic.last_request
    assert sent["output_format"] is CustomerReply
    assert "front-desk assistant" in sent["system"] and "<profile>" in sent["system"]
    # The customer assistant is offered no tools, so it cannot reach report figures.
    assert "tools" not in sent


def test_chat_with_rag_sends_only_the_retrieved_chunks(client, monkeypatch, tmp_path):
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))  # an empty store for this test
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "3")  # a known number, whatever `.env` says
    get_store.cache_clear()
    reply = client.post("/api/chat", json={"question": "Which massages do you offer?", "rag": True})
    get_store.cache_clear()
    assert (reply.json()["rag"], reply.json()["chunks"]) == (True, 3)
    assert FakeAnthropic.last_request["system"].count("<excerpt>\n") == 3


def test_chat_reports_when_the_model_is_unavailable(client, monkeypatch):
    def no_key():
        raise TypeError("no API key")

    monkeypatch.setattr(anthropic, "Anthropic", no_key)
    reply = client.post("/api/chat", json={"question": "Are you open today?"})
    assert reply.status_code == 503
