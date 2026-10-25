"""Replies in a fixed shape (a pydantic class), from any provider.

Pebby's replies have a shape (answer, items, sections, ...), so the page can
lay them out. Providers support that in different ways, so each provider
names one of two methods (providers.STRUCTURED):

  native       the shape is sent as the reply format, and the model's text
               is JSON in that shape. Claude does this, also while offering
               tools: one call ends either in a tool request or in the reply.
  answer-tool  the shape is offered as one more tool, "answer": the model calls
               another tool when it needs one, and "answer" when it is ready.
               Tool calling is the most widely supported way to get exact JSON
               from OpenAI-compatible providers, so this works where a reply
               format may not. With other tools on offer, a call is not required
               (tool_choice "auto"): made to call one every time, a model with
               nothing left to look up tends to repeat a call instead of
               answering. So it may also reply in text: JSON in the shape, or
               plain sentences, which become the shape's text (_from_text). With
               no other tools, the reply's call is required: there is nothing
               else to call. A call some models write as text,
               "<tool_call>{...}</tool_call>", is read as the call it is.

Either way the caller gets the same thing back: the reply, or the model's
requests for other tools. A reply that does not fit the shape raises
ModelError("bad_output"), which the gateway treats like any failure: it
tries the fallback model.
"""

import json
import re
from typing import Any

from anthropic import transform_schema
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from .errors import ModelError
from .tools import Tool

ANSWER = "answer"  # the tool that carries the reply, in the answer-tool method


def _answer_tool(schema: type[BaseModel]) -> dict[str, Any]:
    """The reply's shape, offered as a tool."""
    return {"type": "function", "function": {
        "name": ANSWER,
        "description": "Give your final reply to the user in this shape. Call it once, when you have "
                       "everything you need; call the other tools first if you need them.",
        "parameters": schema.model_json_schema()}}


def prepare(model: BaseChatModel, method: str, schema: type[BaseModel], tools: list[Tool],
            answer_now: bool = False) -> Runnable:
    """`model` set up to reply in `schema`'s shape, with `tools` on offer. With
    `answer_now`, it must reply this time and cannot ask for another tool: for a
    model that keeps asking for tools (repeating one it already had)."""
    offered = [tool.description_for_model() for tool in tools]
    if method == "native":
        # Claude's reply format, the same setting the Anthropic library's messages.parse uses.
        # The tools stay described (the conversation holds earlier tool calls), but none may be called.
        reply_format = {"type": "json_schema", "schema": transform_schema(schema)}
        bound = model.bind_tools(offered, tool_choice={"type": "none"} if answer_now else None) if offered else model
        return bound.bind(output_config={"format": reply_format})
    if method == "answer-tool":
        # The reply's call is required when it is the only choice (no other tools: nothing
        # to repeat) or with answer_now; otherwise the model chooses.
        only_answer = answer_now or not offered
        return model.bind_tools([*offered, _answer_tool(schema)], tool_choice=ANSWER if only_answer else "auto")
    raise ValueError(f"Unknown structured-output method: {method!r}.")


WRITTEN_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _written_calls(text: str) -> list[dict]:
    """Tool calls a model wrote as text, '<tool_call>{"name": ..., "arguments": {...}}</tool_call>',
    as the calls they are ({"name", "args", "id"}); none when it wrote none."""
    calls = []
    for n, block in enumerate(WRITTEN_CALL.findall(text or "")):
        try:
            call = json.loads(block)
        except ValueError:
            continue
        if isinstance(call, dict) and "name" in call:
            calls.append({"name": call["name"], "args": call.get("arguments") or call.get("args") or {},
                          "id": f"written-{n}"})
    return calls


def _from_text(schema: type[BaseModel], text: str) -> BaseModel:
    """A reply written as text: JSON in `schema`'s shape, or else plain sentences,
    which become its first text field, with every list in it empty. Raises
    ValidationError when the text is neither."""
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        fields = schema.model_fields
        first = next((name for name, f in fields.items() if f.annotation is str), None)
        if first is None or not text.strip():
            raise
        lists = {name: [] for name, f in fields.items() if getattr(f.annotation, "__origin__", None) is list}
        return schema.model_validate({**lists, first: text.strip()})


def read(message: AIMessage, method: str, schema: type[BaseModel], model: str) -> tuple[BaseModel | None, list[dict]]:
    """The reply in `message` (None while the model is asking for tools) and
    its requests for other tools. Raises ModelError("bad_output") when the
    reply does not fit the shape."""
    try:
        if method == "answer-tool":
            calls = message.tool_calls or _written_calls(message.text)
            answers = [call for call in calls if call["name"] == ANSWER]
            requests = [call for call in calls if call["name"] != ANSWER]
            if answers:
                return schema.model_validate(answers[0]["args"]), requests
            if requests:
                return None, requests
            # A reply written as text instead of the tool: JSON in the shape, or plain sentences.
            return _from_text(schema, message.text), []
        if message.tool_calls:
            return None, list(message.tool_calls)
        return schema.model_validate_json(message.text), []
    except ValidationError as error:
        raise ModelError("bad_output", model, f"the reply does not fit {schema.__name__}: {error}") from error
