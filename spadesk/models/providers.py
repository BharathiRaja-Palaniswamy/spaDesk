"""Providers: how to make a chat model for each one.

This is the only place in the app that names a provider's library. Every
provider gives back a LangChain chat model (BaseChatModel), so the rest of
the app works with one interface: the same messages in, the same AIMessage
out, tool calls and token counts in the same fields, whichever provider ran.

  anthropic  Claude, through Anthropic's API (ANTHROPIC_API_KEY)
  nebius     open models (Qwen, Llama, DeepSeek...) hosted by Nebius. Its
             API copies OpenAI's, so LangChain's OpenAI chat model talks to
             it, pointed at Nebius's address (NEBIUS_API_KEY)

To add a provider, write one function that returns its chat model and add a
line to PROVIDERS; it needs tool calling, which is also how replies in a fixed
shape are asked for (structured.py). Another OpenAI-compatible service
(OpenRouter, Together, a local Ollama server) needs only its own address and
key, like _nebius.
"""

from collections.abc import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import get_settings
from .errors import ModelError
from .profiles import ModelSpec

# The most tokens an answer may use. Claude allows long answers; many open
# models are served with smaller limits, and a request above them is refused.
CLAUDE_MAX_TOKENS = 16000
OPEN_MODEL_MAX_TOKENS = 4000
# The longest one model call may take, in seconds, and how often it is tried again after
# a failure. Without a limit, a provider that stops answering keeps the request waiting
# forever; with it, the call fails as a ModelError("timeout"), and the app answers anyway
# (a handover, or "the assistant is unavailable"), so no request is left hanging.
TIMEOUT = 60
RETRIES = 1


def _anthropic(model: str) -> BaseChatModel:
    """Claude. The key is read from ANTHROPIC_API_KEY by the library itself."""
    return ChatAnthropic(model=model, max_tokens=CLAUDE_MAX_TOKENS, default_request_timeout=TIMEOUT,
                         max_retries=RETRIES)


def _nebius(model: str) -> BaseChatModel:
    """An open model hosted by Nebius, through its OpenAI-compatible API."""
    settings = get_settings()
    if not settings.nebius_api_key:
        raise ModelError("auth", f"nebius:{model}", "NEBIUS_API_KEY is not set.")
    return ChatOpenAI(model=model, base_url=settings.nebius_base_url, api_key=settings.nebius_api_key,
                      max_tokens=OPEN_MODEL_MAX_TOKENS, timeout=TIMEOUT, max_retries=RETRIES)


# Provider name (the part before ":" in "provider:model") -> how to make its chat model.
PROVIDERS: dict[str, Callable[[str], BaseChatModel]] = {
    "anthropic": _anthropic,
    "nebius": _nebius,
}

# Provider -> the name shown to people ("Powered by Claude" under each chat).
NAMES: dict[str, str] = {
    "anthropic": "Claude",
    "nebius": "Nebius",
}


def build(spec: ModelSpec) -> BaseChatModel:
    """The chat model for `spec`."""
    if spec.provider not in PROVIDERS:
        raise ValueError(f"Unknown model provider: {spec.provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    return PROVIDERS[spec.provider](spec.model)
