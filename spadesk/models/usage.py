"""What one model call used: the same record whichever provider answered.

LangChain puts the token counts of every provider in the same place
(`AIMessage.usage_metadata`), so one function reads them all. The cost is an
estimate from PRICES; a model missing from it gets no cost rather than a
wrong one.

Each record is written to the "spadesk.models" log. The full picture,
prompts included, stays in LangSmith when tracing is on.
"""

import logging
from dataclasses import dataclass

from langchain_core.messages import AIMessage

from .profiles import ModelSpec

log = logging.getLogger("spadesk.models")

# "provider:model" -> (US dollars per million input tokens, per million output tokens).
# Copy the figures from each provider's pricing page; they change, so they are
# kept here, in one place, rather than guessed.
PRICES: dict[str, tuple[float, float]] = {}


@dataclass(frozen=True)
class Usage:
    """One model call, as the logs and the caller see it."""

    profile: str  # the job, e.g. "chat-default"
    model: str  # "provider:model" that answered
    input_tokens: int
    output_tokens: int
    seconds: float  # from sending the request to the whole answer
    cost: float | None  # estimated US dollars; None when the model has no price in PRICES
    fallback: bool  # True when the profile's own model failed and this one answered


def usage_of(message: AIMessage, profile: str, spec: ModelSpec, seconds: float, fallback: bool) -> Usage:
    """The usage record for one answer."""
    counts = message.usage_metadata or {}
    tokens_in, tokens_out = counts.get("input_tokens", 0), counts.get("output_tokens", 0)
    price = PRICES.get(str(spec))
    cost = (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000 if price else None
    return Usage(profile, str(spec), tokens_in, tokens_out, round(seconds, 3), cost, fallback)


def record(usage: Usage) -> None:
    """Write one line to the log, e.g.
    chat-default anthropic:claude-opus-5-5 in=4065 out=212 3.10s cost=n/a"""
    cost = f"${usage.cost:.4f}" if usage.cost is not None else "n/a"
    log.info("%s %s in=%d out=%d %.2fs cost=%s%s", usage.profile, usage.model, usage.input_tokens,
             usage.output_tokens, usage.seconds, cost, " (fallback)" if usage.fallback else "")
