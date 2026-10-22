"""The model gateway: the one way the app talks to a language model.

    chat("chat-default", messages)                      -> Reply: the answer as text
    structured("chat-default", messages, Shape, tools)  -> Structured: the answer in Shape, or tool requests
    stream("voice-ready", messages)                     -> the answer's text, piece by piece

The caller names a job (a profile), never a provider. For each call the
gateway:
  1. looks up the profile's model, and its fallback (profiles.py)
  2. makes that provider's chat model (providers.py)
  3. sends the messages, timing the call
  4. turns a provider's error into a ModelError (errors.py) and, if the
     profile has a fallback, tries that model instead
  5. records what the call used (usage.py)

Messages are LangChain messages, or (role, text) pairs such as
("system", "...") and ("human", "..."), the same for every provider.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel

from . import structured as shapes
from .errors import ModelError, normalize
from .profiles import ModelSpec, Profile, get_profile
from .providers import STRUCTURED, build
from .tools import Tool
from .usage import Usage, record, usage_of

Result = TypeVar("Result")


@dataclass(frozen=True)
class Reply:
    """A model's answer, the same shape whichever provider gave it."""

    text: str  # the answer's text ("" when the model only asked for tools)
    tool_calls: list[dict]  # requests to run a tool: {"name", "args", "id"}, as LangChain normalizes them
    usage: Usage
    message: AIMessage  # the full message, for code that needs the rest (e.g. a LangGraph conversation)


@dataclass(frozen=True)
class Structured:
    """A model's answer in a fixed shape, or its requests for tools first."""

    reply: BaseModel | None  # an instance of the shape asked for; None while the model asks for tools
    tool_calls: list[dict]  # requests for the tools offered: {"name", "args", "id"}
    usage: Usage
    message: AIMessage  # the full message, to add to the conversation before running the tools


def _candidates(profile: Profile) -> list[tuple[ModelSpec, bool]]:
    """The models to try, in order, each marked True if it is the fallback."""
    return [(profile.primary, False)] + ([(profile.fallback, True)] if profile.fallback else [])


def _model_error(error: Exception, spec: ModelSpec) -> ModelError:
    """`error` as a ModelError; anything that is not a provider error is raised as it is."""
    if isinstance(error, ModelError):
        return error
    if (normalized := normalize(error, str(spec))) is None:
        raise error
    return normalized


def _first_that_works(profile: str, attempt: Callable[[ModelSpec], tuple[AIMessage, Result]]) -> tuple[Result, Usage]:
    """Run `attempt` with the profile's model, then with its fallback if the
    first failed. Returns what `attempt` made of the answer, and the usage of
    the call that worked. Raises the last ModelError when every model failed."""
    failure = None
    for spec, is_fallback in _candidates(get_profile(profile)):
        started = time.perf_counter()
        try:
            message, result = attempt(spec)
        except Exception as error:
            failure = _model_error(error, spec)
            continue
        usage = usage_of(message, profile, spec, time.perf_counter() - started, is_fallback)
        record(usage)
        return result, usage
    raise failure


def chat(profile: str, messages: LanguageModelInput) -> Reply:
    """Send `messages` to the profile's model, and return its answer as text."""
    def attempt(spec: ModelSpec) -> tuple[AIMessage, AIMessage]:
        message = build(spec).invoke(messages)
        return message, message

    message, usage = _first_that_works(profile, attempt)
    return Reply(message.text, list(message.tool_calls), usage, message)


def structured(profile: str, messages: LanguageModelInput, shape: type[BaseModel],
               tools: list[Tool] | None = None) -> Structured:
    """Ask for an answer in `shape`, with `tools` on offer. One call ends
    either in the answer or in requests for tools; the caller runs those
    (tools.run_requests), adds them to the conversation and asks again.
    A reply that does not fit `shape` counts as a failure, so the fallback
    model is tried."""
    offered = tools or []

    def attempt(spec: ModelSpec) -> tuple[AIMessage, tuple[BaseModel | None, list[dict], AIMessage]]:
        method = STRUCTURED.get(spec.provider, "answer-tool")
        message = shapes.prepare(build(spec), method, shape, offered).invoke(messages)
        reply, requests = shapes.read(message, method, shape, str(spec))
        return message, (reply, requests, message)

    (reply, requests, message), usage = _first_that_works(profile, attempt)
    return Structured(reply, requests, usage, message)


def stream(profile: str, messages: LanguageModelInput) -> Iterator[str]:
    """The answer's text, piece by piece, as the model writes it: what a voice
    reply needs to start speaking early. The fallback is tried only if the
    first model fails before sending anything; once text has gone out, a
    failure is raised, since the listener has already heard part of it."""
    chosen, failure = get_profile(profile), None
    for spec, is_fallback in _candidates(chosen):
        started, whole, sent = time.perf_counter(), None, False
        try:
            for chunk in build(spec).stream(messages):
                whole = chunk if whole is None else whole + chunk  # chunks add up to the whole message
                if chunk.text:
                    sent = True
                    yield chunk.text
        except Exception as error:
            failure = _model_error(error, spec)
            if sent:
                raise failure from error
            continue
        message = whole if whole is not None else AIMessageChunk(content="")
        record(usage_of(message, profile, spec, time.perf_counter() - started, is_fallback))
        return
    raise failure
