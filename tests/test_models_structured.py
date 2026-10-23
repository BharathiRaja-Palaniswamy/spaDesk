"""Tests for tools and fixed-shape replies in the model gateway
(spadesk/models/tools.py, structured.py, gateway.structured).

The checks of what each provider is sent build the real request payloads
without sending them; the rest use stand-in models. No key, nothing spent.

Run with:  .venv/bin/python -m pytest tests/test_models_structured.py
"""

import json

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from spadesk.models import ModelError, Tool, run_requests, structured
from spadesk.models import providers
from spadesk.models.structured import ANSWER, prepare
from spadesk.reports.routes import get_dataset
from spadesk.reports.tools import report_tool


class Shape(BaseModel):
    """A small reply shape for the tests."""

    answer: str
    covered: bool


OPENING = Tool(name="opening_hours", description="The hours for one day.",
               parameters={"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]},
               run=lambda args: {"day": args["day"], "hours": "9:00 AM – 7:00 PM"})


def request(name: str, args: dict, call_id: str = "call-1") -> dict:
    """One tool request, as LangChain gives it for every provider."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class StandIn(GenericFakeChatModel):
    """A stand-in model that accepts tools (and ignores them), playing back its messages."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def plays(monkeypatch):
    """Registers stand-in providers: "claude-like" uses the native method,
    "open-like" the answer-tool method. Returns a function that sets what each
    one answers, and which is the main model and which the fallback."""
    monkeypatch.setitem(providers.STRUCTURED, "claude-like", "native")
    monkeypatch.setitem(providers.STRUCTURED, "open-like", "answer-tool")

    def use(main: str, replies: dict[str, AIMessage], fallback: str = "") -> None:
        for provider, message in replies.items():
            monkeypatch.setitem(providers.PROVIDERS, provider, lambda model, m=message: StandIn(messages=iter([m])))
        monkeypatch.setenv("SPADESK_CLAUDE_HIGH", main)
        monkeypatch.setenv("SPADESK_MODEL_FALLBACK", fallback)
    return use


# ---- One tool, every provider ----

def test_a_tool_is_written_once_and_each_provider_gets_its_own_format(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-one")
    claude = ChatAnthropic(model="claude-opus-5-5", max_tokens=100).bind_tools([OPENING.description_for_model()])
    open_model = ChatOpenAI(model="Qwen/x", api_key="test", base_url="https://nebius.example/v1/") \
        .bind_tools([OPENING.description_for_model()])
    to_claude = claude.bound._get_request_payload([HumanMessage("hi")], **claude.kwargs)["tools"][0]
    to_open = open_model.bound._get_request_payload([HumanMessage("hi")], **open_model.kwargs)["tools"][0]
    assert to_claude == {"name": "opening_hours", "description": "The hours for one day.",
                         "input_schema": OPENING.parameters}
    assert to_open == {"type": "function", "function": {"name": "opening_hours",
                       "description": "The hours for one day.", "parameters": OPENING.parameters}}


def test_the_report_tool_is_a_neutral_tool_that_runs_the_report():
    tool = report_tool(get_dataset())
    assert tool.name == "get_report" and tool.parameters["required"] == ["start", "end"]
    assert tool.for_anthropic_sdk()["input_schema"] == tool.parameters  # the hand-built loop's format
    assert tool.run({"start": "2026-09-01", "end": "2026-09-30"})["summary"]["revenue"] > 0


def test_requests_are_run_and_answered_by_id():
    first, unknown = run_requests([request("opening_hours", {"day": "Sunday"}, "call-1"),
                                   request("get_weather", {}, "call-2")], [OPENING])
    assert first.tool_call_id == "call-1" and json.loads(first.content)["hours"] == "9:00 AM – 7:00 PM"
    assert unknown.tool_call_id == "call-2" and "no tool called get_weather" in json.loads(unknown.content)["error"]


def test_the_answer_tool_method_offers_the_shape_as_a_required_tool():
    open_model = ChatOpenAI(model="Qwen/x", api_key="test", base_url="https://nebius.example/v1/")
    runnable = prepare(open_model, "answer-tool", Shape, [OPENING])
    payload = runnable.bound._get_request_payload([HumanMessage("hi")], **runnable.kwargs)
    assert [tool["function"]["name"] for tool in payload["tools"]] == ["opening_hours", ANSWER]
    assert payload["tool_choice"] == "required"


# ---- Fixed-shape replies ----

def test_native_method_reads_the_reply_from_the_text(plays):
    plays("claude-like:m", {"claude-like": AIMessage(content='{"answer": "We open at 9.", "covered": true}')})
    result = structured("chat-default", "When do you open?", Shape, [OPENING])
    assert result.reply == Shape(answer="We open at 9.", covered=True) and result.tool_calls == []


def test_native_method_passes_tool_requests_back(plays):
    plays("claude-like:m", {"claude-like": AIMessage(content="", tool_calls=[request("opening_hours", {"day": "Sun"})])})
    result = structured("chat-default", "Open Sunday?", Shape, [OPENING])
    assert result.reply is None and result.tool_calls[0]["name"] == "opening_hours"


def test_answer_tool_method_reads_the_reply_from_the_answer_call(plays):
    plays("open-like:m", {"open-like": AIMessage(content="", tool_calls=[
        request(ANSWER, {"answer": "We open at 9.", "covered": True})])})
    result = structured("chat-default", "When do you open?", Shape, [OPENING])
    assert result.reply == Shape(answer="We open at 9.", covered=True) and result.tool_calls == []


def test_answer_tool_method_passes_other_tool_requests_back(plays):
    plays("open-like:m", {"open-like": AIMessage(content="", tool_calls=[request("opening_hours", {"day": "Sun"})])})
    result = structured("chat-default", "Open Sunday?", Shape, [OPENING])
    assert result.reply is None and [call["name"] for call in result.tool_calls] == ["opening_hours"]


def test_answer_tool_method_accepts_the_shape_written_as_text(plays):
    plays("open-like:m", {"open-like": AIMessage(content='{"answer": "We open at 9.", "covered": true}')})
    assert structured("chat-default", "When?", Shape).reply.answer == "We open at 9."


def test_a_reply_in_the_wrong_shape_falls_back_to_the_second_model(plays):
    plays("open-like:m", {"open-like": AIMessage(content="Sure! We open at 9."),
                          "claude-like": AIMessage(content='{"answer": "We open at 9.", "covered": true}')},
          fallback="claude-like:m")
    result = structured("chat-default", "When?", Shape)
    assert result.reply.covered is True and result.usage.model == "claude-like:m" and result.usage.fallback


def test_a_reply_in_the_wrong_shape_without_a_fallback_is_a_bad_output_error(plays):
    plays("open-like:m", {"open-like": AIMessage(content="Sure! We open at 9.")})
    with pytest.raises(ModelError) as raised:
        structured("chat-default", "When?", Shape)
    assert raised.value.kind == "bad_output"
