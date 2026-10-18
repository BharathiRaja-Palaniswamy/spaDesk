"""Tests for the LangChain chat (spadesk/lc/prompt.py, lc/chat.py) behind the
customer route, with SPADESK_RAG_ENGINE=langchain.

The model is replaced by a stand-in, so these run without an API key and
spend nothing. The point of the LangChain pipeline is to answer the same way
by different code, so one test checks that it builds exactly the prompt the
hand-built pipeline builds, with RAG off and on.

Run with:  .venv/bin/python -m pytest tests/test_lc_chat.py
"""

import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from spadesk.app import app
from spadesk.assistant.customer import SYSTEM, CustomerReply
from spadesk.assistant.prompt import profile_prompt
from spadesk.assistant.replies import ReplyItem
from spadesk.lc import chat
from spadesk.lc.prompt import PROMPT, prompt_values
from spadesk.lc.retriever import get_retriever
from spadesk.rag.stores import get_store

REPLY = CustomerReply(answer="We have three massages.", covered=True,
                      items=[ReplyItem(name="Swedish Massage", detail="60 minutes, $115")],
                      sections=["Massage Therapy", "Made-up Section"],
                      suggestions=["One?", "Two?", "Three?", "Four?"])


@pytest.fixture(autouse=True)
def langchain_engine(tmp_path, monkeypatch):
    """The LangChain engine, empty store folders and known settings for every test."""
    monkeypatch.setenv("SPADESK_RAG_ENGINE", "langchain")
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "3")
    get_store.cache_clear()
    yield
    get_store.cache_clear()


@pytest.fixture
def sent(monkeypatch):
    """Replaces the chat model with a stand-in that records the prompt it is given."""
    record = {}

    def fake_model(schema):
        def reply(prompt_value):
            record["schema"], record["messages"] = schema, prompt_value.to_messages()
            return REPLY.model_copy()
        return RunnableLambda(reply)

    monkeypatch.setattr(chat, "structured_model", fake_model)
    return record


def without_time(text: str) -> str:
    """The prompt with its current time removed: the two prompts are built moments apart."""
    return re.sub(r"<currentTime>.*?</currentTime>", "", text)


def test_the_route_answers_through_the_langchain_chain(sent):
    reply = TestClient(app).post("/api/chat", json={"question": "Which massages do you offer?"})
    # The same checks as the hand-built engine: no made-up section, at most three suggestions.
    assert reply.json() == {"answer": "We have three massages.", "covered": True,
                            "items": [{"name": "Swedish Massage", "detail": "60 minutes, $115"}],
                            "sections": ["Massage Therapy"], "suggestions": ["One?", "Two?", "Three?"],
                            "rag": False, "chunks": 0}
    system, human = sent["messages"]
    assert sent["schema"] is CustomerReply
    assert "front-desk assistant" in system.content and "<profile>" in system.content
    assert human.content == "Which massages do you offer?"


def test_with_rag_only_the_retrieved_chunks_are_sent(sent):
    reply = TestClient(app).post("/api/chat", json={"question": "Which massages do you offer?", "rag": True})
    assert (reply.json()["rag"], reply.json()["chunks"]) == (True, 3)
    assert sent["messages"][0].content.count("<excerpt>\n") == 3


@pytest.mark.parametrize("rag", [False, True])
def test_the_prompt_is_the_same_as_the_hand_built_one(rag):
    question = "Can I have the Sea Mineral Body Wrap while pregnant?"
    documents = get_retriever().invoke(question) if rag else None
    ours = PROMPT.invoke(prompt_values(SYSTEM, question, documents)).to_messages()[0].content
    theirs = profile_prompt(SYSTEM, question, rag).system
    assert without_time(ours) == without_time(theirs)


def test_the_route_reports_when_the_model_is_unavailable(monkeypatch):
    def unavailable(schema):
        def fail(prompt_value):
            raise TypeError("no API key")
        return RunnableLambda(fail)

    monkeypatch.setattr(chat, "structured_model", unavailable)
    reply = TestClient(app).post("/api/chat", json={"question": "Are you open today?"})
    assert reply.status_code == 503


def test_an_unknown_chat_provider_is_reported(monkeypatch):
    monkeypatch.setenv("SPADESK_CHAT_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Unknown chat provider"):
        chat.get_chat_model()
