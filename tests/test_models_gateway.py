"""Tests for the model gateway (spadesk/models/): choosing a model, the shape
of a reply, errors, and falling back to a second model.

Stand-in models play back prepared answers or raise prepared errors, so these
need no API key and spend nothing.

Run with:  .venv/bin/python -m pytest tests/test_models_gateway.py
"""

import anthropic
import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tracers.context import collect_runs
from langchain_openai import ChatOpenAI

from spadesk.models import ModelError, chat, stream
from spadesk.models import providers, usage
from spadesk.models.errors import normalize
from spadesk.models.profiles import ModelSpec, get_profile, parse_spec

REQUEST = httpx.Request("POST", "https://api.example.com/v1/messages")


def anthropic_error(cls, status: int) -> Exception:
    """An error as the Anthropic library raises it."""
    return cls("refused", response=httpx.Response(status, request=REQUEST), body=None)


def answers(text: str = "We open at 9:00 AM.") -> AIMessage:
    """A model's answer, with token counts as every provider reports them through LangChain."""
    return AIMessage(content=text, usage_metadata={"input_tokens": 120, "output_tokens": 8, "total_tokens": 128})


class Failing(GenericFakeChatModel):
    """A stand-in model whose every call fails with `error`."""

    error: Exception

    def _generate(self, *args, **kwargs):
        raise self.error

    def _stream(self, *args, **kwargs):
        raise self.error


@pytest.fixture
def models(monkeypatch):
    """Registers stand-in providers: "good" answers, "busy" is rate-limited.
    Returns a function that sets the main model and the fallback."""
    monkeypatch.setitem(providers.PROVIDERS, "good", lambda model: GenericFakeChatModel(messages=iter([answers()])))
    monkeypatch.setitem(providers.PROVIDERS, "busy", lambda model: Failing(
        messages=iter([]), error=anthropic_error(anthropic.RateLimitError, 429)))

    def use(main: str, fallback: str = "") -> None:
        monkeypatch.setenv("SPADESK_CLAUDE_HIGH", main)
        monkeypatch.setenv("SPADESK_MODEL_FALLBACK", fallback)
    return use


# ---- Choosing a model ----

def test_a_model_is_written_provider_colon_model():
    assert parse_spec("nebius:Qwen/Qwen3-32B") == ModelSpec("nebius", "Qwen/Qwen3-32B")
    assert parse_spec("claude-opus-5-5") == ModelSpec("anthropic", "claude-opus-5-5")  # a bare name is Claude


def test_claude_is_the_default(monkeypatch):
    for name in ("SPADESK_MODELS", "SPADESK_CLAUDE_HIGH", "SPADESK_CLAUDE_LOW", "SPADESK_MODEL_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    profile = get_profile("chat-default")
    assert str(profile.primary) == "anthropic:claude-opus-5-5" and profile.fallback is None
    assert str(get_profile("cheap-router").primary) == "anthropic:claude-haiku-4-5-20251001"


def test_one_setting_moves_every_job_to_the_open_set_each_keeping_its_tier(monkeypatch):
    for name in ("SPADESK_OPENSOURCE_HIGH", "SPADESK_OPENSOURCE_LOW"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPADESK_MODELS", "opensource")
    high = "nebius:Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert str(get_profile("booking-agent").primary) == str(get_profile("chat-default").primary) == high
    assert str(get_profile("cheap-router").primary) == "nebius:Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_an_unknown_model_set_is_named(monkeypatch):
    monkeypatch.setenv("SPADESK_MODELS", "gpt")
    with pytest.raises(ValueError, match="Unknown model set"):
        get_profile("chat-default")


def test_one_profile_can_be_moved_to_another_provider(monkeypatch):
    monkeypatch.setenv("SPADESK_PROFILE_CHEAP_ROUTER", "nebius:Qwen/Qwen3-32B")
    assert str(get_profile("cheap-router").primary) == "nebius:Qwen/Qwen3-32B"
    assert get_profile("chat-default").primary.provider != "nebius"  # the others are unchanged


def test_the_judge_stays_on_claude_whatever_the_app_uses(monkeypatch):
    monkeypatch.setenv("SPADESK_MODELS", "opensource")
    assert get_profile("judge").primary.provider == "anthropic"


def test_an_unknown_profile_or_provider_is_named():
    with pytest.raises(ValueError, match="Unknown model profile"):
        get_profile("chat-defualt")
    with pytest.raises(ValueError, match="Unknown model provider"):
        providers.build(ModelSpec("nebious", "x"))


def test_each_provider_gives_a_langchain_chat_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-real-one")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key-not-a-real-one")
    monkeypatch.setenv("SPADESK_NEBIUS_BASE_URL", "https://nebius.example/v1/")
    assert isinstance(providers.build(ModelSpec("anthropic", "claude-opus-5-5")), ChatAnthropic)
    nebius = providers.build(ModelSpec("nebius", "Qwen/Qwen3-32B"))
    assert isinstance(nebius, ChatOpenAI) and nebius.openai_api_base == "https://nebius.example/v1/"


def test_nebius_without_a_key_is_an_auth_error(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    with pytest.raises(ModelError) as raised:
        providers.build(ModelSpec("nebius", "Qwen/Qwen3-32B"))
    assert raised.value.kind == "auth"


# ---- The reply ----

def test_a_reply_has_the_same_shape_whichever_provider_answered(models):
    models("good:any-model")
    reply = chat("chat-default", [("human", "When do you open?")])
    assert reply.text == "We open at 9:00 AM." and reply.tool_calls == []
    assert (reply.usage.model, reply.usage.input_tokens, reply.usage.output_tokens) == ("good:any-model", 120, 8)
    assert reply.usage.seconds >= 0 and reply.usage.fallback is False
    assert reply.usage.cost is None  # no price known for this model


def test_cost_is_estimated_when_the_model_has_a_price(models, monkeypatch):
    models("good:any-model")
    monkeypatch.setitem(usage.PRICES, "good:any-model", (3.0, 15.0))  # $ per million tokens in, out
    assert chat("chat-default", "hi").usage.cost == pytest.approx((120 * 3 + 8 * 15) / 1_000_000)


def test_stream_sends_the_answer_in_pieces(models):
    models("good:any-model")
    assert "".join(stream("chat-default", "hi")) == "We open at 9:00 AM."


# ---- Errors and fallback ----

def test_provider_errors_become_one_kind_of_error():
    assert normalize(anthropic_error(anthropic.AuthenticationError, 401), "m").kind == "auth"
    assert normalize(openai.APITimeoutError(request=REQUEST), "m").kind == "timeout"
    assert normalize(anthropic_error(anthropic.RateLimitError, 429), "m").kind == "rate_limit"
    assert normalize(ValueError("a bug in our code"), "m") is None  # not hidden behind a fallback


def test_a_failing_model_falls_back_to_the_second(models):
    models("busy:big-model", fallback="good:small-model")
    reply = chat("chat-default", "hi")
    assert reply.text == "We open at 9:00 AM."
    assert reply.usage.model == "good:small-model" and reply.usage.fallback is True


def test_without_a_fallback_the_error_reaches_the_caller(models):
    models("busy:big-model")
    with pytest.raises(ModelError) as raised:
        chat("chat-default", "hi")
    assert (raised.value.kind, raised.value.model) == ("rate_limit", "busy:big-model")


def test_stream_falls_back_before_anything_is_sent(models):
    models("busy:big-model", fallback="good:small-model")
    assert "".join(stream("chat-default", "hi")) == "We open at 9:00 AM."


def test_a_bug_is_not_hidden_behind_the_fallback(models, monkeypatch):
    monkeypatch.setitem(providers.PROVIDERS, "broken", lambda model: Failing(messages=iter([]), error=KeyError("x")))
    models("broken:any", fallback="good:small-model")
    with pytest.raises(KeyError):
        chat("chat-default", "hi")


# ---- Labels for LangSmith ----

def test_every_call_is_labelled_with_its_job_and_model(models, monkeypatch):
    # collect_runs records each step as LangSmith would see it, without sending anything.
    monkeypatch.setenv("SPADESK_CLAUDE_LOW", "good:x")
    models("busy:y")  # the high tier, not used here
    with collect_runs() as runs:
        chat("cheap-router", [("human", "Hi")])
    [run] = [r for r in runs.traced_runs if r.run_type == "llm"]
    assert run.name == "cheap-router"
    assert {"profile:cheap-router", "provider:good", "model:x", "tier:low", "set:claude"} <= set(run.tags)
    labels = {"profile": "cheap-router", "tier": "low", "model_set": "claude", "provider": "good", "model": "x",
              "fallback": False}
    assert labels.items() <= run.extra["metadata"].items()
