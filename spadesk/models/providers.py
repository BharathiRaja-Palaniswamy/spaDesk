"""Providers: how to make a chat model for each one.

This is the only place in the app that names a provider's library. Every
provider gives back a LangChain chat model (BaseChatModel), so the rest of
the app works with one interface: the same messages in, the same AIMessage
out, tool calls and token counts in the same fields, whichever provider ran.

  anthropic  Claude, through Anthropic's API (ANTHROPIC_API_KEY)
  nebius     open models (Qwen, Llama, DeepSeek...) hosted by Nebius. Its
             API copies OpenAI's, so LangChain's OpenAI chat model talks to
             it, pointed at Nebius's address (NEBIUS_API_KEY)

To add a provider, write one function that returns its chat model, add a
line to PROVIDERS, and, if it supports a JSON reply format alongside tools
as Claude does, a line to STRUCTURED. Another OpenAI-compatible service
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


def _anthropic(model: str) -> BaseChatModel:
    """Claude. The key is read from ANTHROPIC_API_KEY by the library itself."""
    return ChatAnthropic(model=model, max_tokens=CLAUDE_MAX_TOKENS)


def _nebius(model: str) -> BaseChatModel:
    """An open model hosted by Nebius, through its OpenAI-compatible API."""
    settings = get_settings()
    if not settings.nebius_api_key:
        raise ModelError("auth", f"nebius:{model}", "NEBIUS_API_KEY is not set.")
    return ChatOpenAI(model=model, base_url=settings.nebius_base_url, api_key=settings.nebius_api_key,
                      max_tokens=OPEN_MODEL_MAX_TOKENS)


# Provider name (the part before ":" in "provider:model") -> how to make its chat model.
PROVIDERS: dict[str, Callable[[str], BaseChatModel]] = {
    "anthropic": _anthropic,
    "nebius": _nebius,
}

# Provider -> how it is asked for a reply in a fixed shape (see structured.py).
# A provider missing here uses "answer-tool", which any model with tool calling supports.
STRUCTURED: dict[str, str] = {
    "anthropic": "native",
    "nebius": "answer-tool",
}


def build(spec: ModelSpec) -> BaseChatModel:
    """The chat model for `spec`."""
    if spec.provider not in PROVIDERS:
        raise ValueError(f"Unknown model provider: {spec.provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    return PROVIDERS[spec.provider](spec.model)
