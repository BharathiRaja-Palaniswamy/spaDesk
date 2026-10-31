"""Tests for tools and fixed-shape replies in the model gateway
(spadesk/models/tools.py, structured.py, gateway.structured).

The checks of what each provider is sent build the real request payloads
without sending them; the rest use stand-in models. No key, nothing spent.

Run with:  .venv/bin/python -m pytest tests/test_models_structured.py
"""

import json

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from spadesk.models import ModelError, Tool, providers, run_requests, structured
from spadesk.models.structured import ANSWER, ask
from spadesk.reports.routes import get_dataset
from spadesk.reports.tools import report_tool
from tests.standins import StandIn


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


@pytest.fixture
def plays(monkeypatch):
    """Registers stand-in providers ("claude-like", "open-like"). Returns a
    function that sets what each one answers, and which is the main model and
    which the fallback."""

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


class Sent(Exception):
    """Raised in place of sending a request; holds the request."""


@pytest.fixture
def sent(monkeypatch):
    """A function that gives the request Claude and an open model would each be
    sent for a reply in Shape: built for real, never sent."""
    for chat_model in (ChatAnthropic, ChatOpenAI):
        def capture(self, messages, stop=None, run_manager=None, **kwargs):
            raise Sent(self._get_request_payload(messages, stop=stop, **kwargs))
        monkeypatch.setattr(chat_model, "_generate", capture)

    def requests(tools: list[Tool], answer_now: bool = False) -> list[dict]:
        payloads = []
        for model in (ChatAnthropic(model="claude-opus-5-5", api_key="test"),
                      ChatOpenAI(model="Qwen/x", api_key="test", base_url="https://nebius.example/v1/")):
            with pytest.raises(Sent) as raised:
                ask(model, [HumanMessage("hi")], Shape, tools, answer_now)
            payloads.append(raised.value.args[0])
        return payloads
    return requests


def tool_names(payload: dict) -> list[str]:
    return [tool.get("name") or tool["function"]["name"] for tool in payload.get("tools", [])]


def test_while_a_tool_may_be_called_every_provider_is_offered_the_shape_as_a_tool(sent):
    claude, open_model = sent([OPENING])
    assert tool_names(claude) == tool_names(open_model) == ["opening_hours", ANSWER]
    # Never forced: Claude Opus refuses a forced tool choice, and a forced model repeats calls.
    assert claude["tool_choice"] == {"type": "auto"} and open_model["tool_choice"] == "auto"
    # And no reply format: on an OpenAI-compatible host it forces every reply into the shape, so the
    # model could not call a tool.
    assert "output_config" not in claude and "response_format" not in open_model


@pytest.mark.parametrize("tools, answer_now", [([OPENING], True), ([], False)])
def test_when_no_tool_may_be_called_the_provider_makes_the_reply_fit_the_shape(sent, tools, answer_now):
    # A model that keeps asking for tools is made to answer (answer_now); and some jobs have no tools.
    claude, open_model = sent(tools, answer_now)
    assert claude["output_config"]["format"]["type"] == "json_schema" and open_model["response_format"] is Shape
    assert tool_names(claude) == tool_names(open_model) == []


# ---- Fixed-shape replies ----

def test_the_shape_written_as_text_is_read_as_the_reply(plays):
    plays("claude-like:m", {"claude-like": AIMessage(content='{"answer": "We open at 9.", "covered": true}')})
    result = structured("chat-default", "When do you open?", Shape, [OPENING])
    assert result.reply == Shape(answer="We open at 9.", covered=True) and result.tool_calls == []


def test_a_claude_tool_request_is_passed_back(plays):
    plays("claude-like:m", {"claude-like": AIMessage(content="", tool_calls=[request("opening_hours", {"day": "Sun"})])})
    result = structured("chat-default", "Open Sunday?", Shape, [OPENING])
    assert result.reply is None and result.tool_calls[0]["name"] == "opening_hours"


def test_the_reply_is_read_from_the_answer_call(plays):
    plays("open-like:m", {"open-like": AIMessage(content="", tool_calls=[
        request(ANSWER, {"answer": "We open at 9.", "covered": True})])})
    result = structured("chat-default", "When do you open?", Shape, [OPENING])
    assert result.reply == Shape(answer="We open at 9.", covered=True) and result.tool_calls == []


def test_plain_sentences_are_taken_as_the_reply(plays, monkeypatch):
    # Not required to call a tool, a model may simply answer in text.
    class Reply(BaseModel):
        answer: str
        items: list[str]
    plays("open-like:m", {"open-like": AIMessage(content="We open at 9 on Sundays.")})
    result = structured("chat-default", "Open Sunday?", Reply, [OPENING])
    assert result.reply == Reply(answer="We open at 9 on Sundays.", items=[])
    # A shape plain sentences cannot fill (a required yes/no) is asked for again, and is a bad reply
    # when the second answer does not fit either.
    unfit = AIMessage(content="We open at 9.")
    monkeypatch.setitem(providers.PROVIDERS, "open-like", lambda model: StandIn(messages=iter([unfit, unfit])))
    with pytest.raises(ModelError):
        structured("chat-default", "Open Sunday?", Shape, [OPENING])


def test_a_tool_call_written_as_text_is_read_as_the_call(plays):
    # As seen live: the small model wrote <tool_call>{...}</tool_call> as text; the turn failed.
    written = '<tool_call>\n{"name": "answer", "arguments": {"answer": "We open at 9.", "covered": true}}\n</tool_call>'
    plays("open-like:m", {"open-like": AIMessage(content=written)})
    assert structured("chat-default", "When?", Shape, [OPENING]).reply == Shape(answer="We open at 9.", covered=True)
    other = '<tool_call>{"name": "opening_hours", "arguments": {"day": "Sun"}}</tool_call>'
    plays("open-like:m", {"open-like": AIMessage(content=other)})
    result = structured("chat-default", "Sunday?", Shape, [OPENING])
    assert result.reply is None and result.tool_calls[0]["args"] == {"day": "Sun"}


def test_other_tool_requests_are_passed_back(plays):
    plays("open-like:m", {"open-like": AIMessage(content="", tool_calls=[request("opening_hours", {"day": "Sun"})])})
    result = structured("chat-default", "Open Sunday?", Shape, [OPENING])
    assert result.reply is None and [call["name"] for call in result.tool_calls] == ["opening_hours"]


def test_without_tools_the_reply_comes_in_the_shape(plays):
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


def test_a_reply_missing_a_field_is_asked_for_again_with_the_shape_enforced(monkeypatch):
    # As seen with Qwen: the answer tool's call left out "covered", and the owner saw "unavailable".
    missing = AIMessage(content="", tool_calls=[request(ANSWER, {"answer": "We open at 9."})])
    fitted = AIMessage(content='{"answer": "We open at 9.", "covered": true}')
    monkeypatch.setitem(providers.PROVIDERS, "open-like", lambda model: StandIn(messages=iter([missing, fitted])))
    monkeypatch.setenv("SPADESK_CLAUDE_HIGH", "open-like:m")
    monkeypatch.setenv("SPADESK_MODEL_FALLBACK", "")
    result = structured("chat-default", "When do you open?", Shape, [OPENING])
    assert result.reply == Shape(answer="We open at 9.", covered=True) and result.tool_calls == []
