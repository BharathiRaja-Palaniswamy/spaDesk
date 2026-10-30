"""Replies in a fixed shape (a pydantic class), from any provider.

Pebby's replies have a shape (answer, items, sections, ...), so the page can
lay them out. Every provider is asked the same way, by one of two rules,
depending only on whether the model may still call a tool:

  a tool may be called   the shape is offered as one more tool, "answer". The
                         model calls another tool when it needs one, and
                         "answer" when it is ready; its inputs are the reply.
  no tool may be called  (no tools, or answer_now) the shape is sent as the
                         reply format, and the provider makes the reply fit it.
                         LangChain writes the format each provider's way.

Why not the reply format every time? With tools on offer, an OpenAI-compatible
host like Nebius forces every reply into the shape, so the model can no longer
call a tool, and makes up what it should look up. And why not make the model
call "answer" when it must answer? Some models (Claude Opus 5.5) refuse a
forced tool choice. So a call is never forced: tool_choice is always "auto".
That also suits the models: made to call a tool every time, a model with
nothing left to look up tends to repeat a call instead of answering.

Free to choose, a model may reply in text instead of calling "answer": JSON in
the shape, or plain sentences, which become the shape's text (_from_text). A
call some models write as text, "<tool_call>{...}</tool_call>", is read as the
call it is.

The caller gets the reply, or the model's requests for other tools. A reply
that does not fit the shape raises ModelError("bad_output"), which the
gateway treats like any failure: it tries the fallback model.
"""

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

from .errors import ModelError
from .tools import Tool

ANSWER = "answer"  # the tool that carries the reply


def _answer_tool(schema: type[BaseModel]) -> dict[str, Any]:
    """The reply's shape, offered as a tool."""
    return {"type": "function", "function": {
        "name": ANSWER,
        "description": "Give your final reply to the user in this shape. Call it once, when you have "
                       "everything you need; call the other tools first if you need them.",
        "parameters": schema.model_json_schema()}}


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


def _read(message: AIMessage, schema: type[BaseModel], model: str) -> tuple[BaseModel | None, list[dict]]:
    """The reply in `message`, which had tools on offer (None while the model is
    asking for tools), and its requests for other tools. Raises
    ModelError("bad_output") when the reply does not fit the shape."""
    try:
        calls = message.tool_calls or _written_calls(message.text)
        answers = [call for call in calls if call["name"] == ANSWER]
        requests = [call for call in calls if call["name"] != ANSWER]
        if answers:
            return schema.model_validate(answers[0]["args"]), requests
        if requests:
            return None, requests
        # A reply written as text instead of the tool: JSON in the shape, or plain sentences.
        return _from_text(schema, message.text), []
    except ValidationError as error:
        raise ModelError("bad_output", model, f"the reply does not fit {schema.__name__}: {error}") from error


def ask(model: BaseChatModel, messages: LanguageModelInput, schema: type[BaseModel], tools: list[Tool],
        answer_now: bool = False, config: RunnableConfig | None = None, name: str = ""
        ) -> tuple[AIMessage, BaseModel | None, list[dict]]:
    """Ask `model` for a reply in `schema`'s shape, with `tools` on offer. With
    `answer_now`, it must reply this time and cannot ask for another tool: for a
    model that keeps asking for tools (repeating one it already had). Returns
    the model's message, the reply (None while it asks for tools) and its
    requests for tools. `name` ("provider:model") is for the error message."""
    if tools and not answer_now:
        offered = [tool.description_for_model() for tool in tools]
        message = model.bind_tools([*offered, _answer_tool(schema)], tool_choice="auto").invoke(messages, config=config)
        return message, *_read(message, schema, name)
    # No tool may be called: the provider makes the reply fit the shape. The conversation may still
    # hold earlier tool calls and their results; the model reads them, and calls nothing.
    out = model.with_structured_output(schema, method="json_schema", include_raw=True).invoke(messages, config=config)
    if out["parsed"] is None:
        raise ModelError("bad_output", name, f"the reply does not fit {schema.__name__}: {out['parsing_error']}")
    return out["raw"], out["parsed"], []
