"""Tests for the owner chat as a LangGraph graph (spadesk/lc/owner/), behind
POST /api/report/chat with SPADESK_RAG_ENGINE=langchain.

The chat model is replaced by a stand-in that plays back prepared messages,
so these run without an API key and spend nothing.

Run with:  .venv/bin/python -m pytest tests/test_lc_owner_graph.py
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from spadesk.app import app
from spadesk.assistant.replies import ReplyItem
from spadesk.lc import owner
from spadesk.lc.owner import graph
from spadesk.reports.routes import SYSTEM, ChatReply, get_dataset
from spadesk.reports.tools import MAX_TOOL_ROUNDS, TOOL_NAME

ANSWER = ChatReply(answer="Facials in September 2026.", covered=True,
                   items=[ReplyItem(name="Revenue", detail="$7,530")],
                   insights=["Facials were the second-highest earning category."],
                   sections=["Made-up Section"], suggestions=["One?", "Two?", "Three?", "Four?"])
SEPTEMBER_FACIALS = {"start": "2026-09-01", "end": "2026-09-30", "category": "Facials"}


def asks_for_report(call_id: str = "call-1") -> AIMessage:
    """A model message that asks for the September facials report."""
    return AIMessage(content="", tool_calls=[{"name": TOOL_NAME, "args": SEPTEMBER_FACIALS, "id": call_id}])


def answers() -> AIMessage:
    """A model message that gives the structured answer, as JSON text."""
    return AIMessage(content=ANSWER.model_dump_json())


@pytest.fixture
def run(monkeypatch):
    """Runs the graph with a stand-in model that plays back `replies` in order.
    Returns the final state."""
    def run_graph(*replies: AIMessage, rag: bool = False) -> dict:
        model = GenericFakeChatModel(messages=iter(replies))
        return graph.build_graph(SYSTEM, model, get_dataset()).invoke(
            {"question": "How did facials do in September?", "rag": rag, "messages": [], "reports": []})
    return run_graph


def test_the_graph_runs_the_tool_then_answers(run):
    state = run(asks_for_report(), answers())
    system, question, request, result, reply = state["messages"]
    assert isinstance(system, SystemMessage) and "owner of a spa" in system.content
    assert question.content == "How did facials do in September?"
    assert request.tool_calls[0]["args"] == SEPTEMBER_FACIALS
    # The tool's result is tied to the request by id, and holds the real figures.
    assert isinstance(result, ToolMessage) and result.tool_call_id == "call-1"
    report = json.loads(result.content)
    assert report["filters"]["category"] == "Facials" and report["summary"]["revenue"] == 7530
    assert ChatReply.model_validate_json(reply.text) == ANSWER
    assert state["reports"] == ["2026-09-01 to 2026-09-30, Facials"] and state["rounds"] == 1


def test_a_profile_question_needs_no_tool(run):
    state = run(answers())
    assert len(state["messages"]) == 3 and state["rounds"] == 0 and state["reports"] == []


def test_with_rag_the_prompt_holds_only_the_retrieved_chunks(run, tmp_path, monkeypatch):
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "3")
    state = run(answers(), rag=True)
    assert state["chunks"] == 3 and state["messages"][0].content.count("<excerpt>\n") == 3


def test_the_graph_stops_when_the_model_keeps_asking(run):
    state = run(*[asks_for_report(f"call-{n}") for n in range(MAX_TOOL_ROUNDS + 1)])
    assert state["rounds"] == MAX_TOOL_ROUNDS
    assert state["messages"][-1].tool_calls  # it ended still asking: no answer


def test_an_unknown_tool_gets_an_error_result(run):
    unknown = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {}, "id": "call-1"}])
    state = run(unknown, answers())
    assert "no tool called get_weather" in json.loads(state["messages"][3].content)["error"]


def test_the_model_is_offered_the_tool_and_the_reply_shape():
    bound = graph.owner_model(get_dataset(), ChatReply)
    assert bound.kwargs["tools"][0]["name"] == TOOL_NAME
    assert bound.kwargs["output_config"]["format"]["type"] == "json_schema"


@pytest.fixture
def graph_engine(monkeypatch):
    """The LangChain engine, with the graph's model replaced by a stand-in."""
    monkeypatch.setenv("SPADESK_RAG_ENGINE", "langchain")

    def stand_in(dataset, schema):
        return GenericFakeChatModel(messages=iter([asks_for_report(), answers()]))

    monkeypatch.setattr(owner, "owner_model", stand_in)


def test_the_route_answers_through_the_graph(graph_engine):
    reply = TestClient(app).post("/api/report/chat", json={"question": "How did facials do in September?"}).json()
    # The same checks as the hand-built loop: no made-up section, at most three suggestions.
    assert reply == {"answer": "Facials in September 2026.", "covered": True,
                     "items": [{"name": "Revenue", "detail": "$7,530"}],
                     "insights": ["Facials were the second-highest earning category."],
                     "sections": [], "suggestions": ["One?", "Two?", "Three?"],
                     "rag": False, "chunks": 0, "reports": ["2026-09-01 to 2026-09-30, Facials"]}


def test_the_route_reports_when_the_graph_has_no_answer(monkeypatch):
    monkeypatch.setenv("SPADESK_RAG_ENGINE", "langchain")
    monkeypatch.setattr(owner, "owner_model",
                        lambda dataset, schema: GenericFakeChatModel(messages=iter([AIMessage(content="Not JSON")])))
    reply = TestClient(app).post("/api/report/chat", json={"question": "Hello"})
    assert reply.status_code == 503 and "could not answer" in reply.json()["detail"]
