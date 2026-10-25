"""Model profiles: which model each kind of work uses.

A profile names a job, not a model ("chat-default", "judge"), so code asks
for the job and the settings decide the model.

The models come in sets, each with a high-end and a low-end model:

    claude      high: Claude Opus           low: Claude Haiku
    opensource  high: Qwen3 235B (Nebius)   low: Qwen3 30B (Nebius)

One setting, SPADESK_MODELS, picks the set the whole app uses; each job asks
for a tier, never a model:

    high  jobs that need judgement: the chats, the booking agent
    low   quick, simple jobs: reading a name and number, short spoken answers

So moving the app from Claude to open models is one line in `.env`, and
every job keeps a model of the right size. The models in each set are
settings too (SPADESK_CLAUDE_HIGH, ... in config.py).

The judge that grades eval answers is outside the sets: it stays on Claude
Opus, so a change in a score means the answers changed, not the grader.

A single job can still be given a model of its own, which wins over its
tier, e.g. to try one model without moving the rest:

    SPADESK_PROFILE_BOOKING_AGENT=nebius:moonshotai/Kimi-K3

A model is written "provider:model". Each profile can fall back to a second
model when its own fails (SPADESK_MODEL_FALLBACK, one for all; off by default).
"""

from dataclasses import dataclass

from ..config import get_settings

# The judge grades eval answers, always with the same model (see above).
JUDGE_MODEL = "anthropic:claude-opus-5-5"
# A bare model name, with no "provider:", is a Claude model.
DEFAULT_PROVIDER = "anthropic"

# Profile -> the tier of its model in the chosen set, or a fixed model.
PROFILES: dict[str, str] = {
    "chat-default": "high",  # the owner chat, and the question-and-answer chat the evals use
    "booking-agent": "high",  # Pebby, the front agent that talks to customers (spadesk/agent/)
    "booking-team": "low",  # the Booking agent behind Pebby: small, fast reasoning (spadesk/agent/booking/agent.py)
    "cheap-router": "low",  # small reading jobs, e.g. a name and number from a message
    "voice-ready": "low",  # short answers that must start fast, for speech
    "judge": JUDGE_MODEL,  # grades eval answers (evals/judge.py)
}
TIERS = ("high", "low")


@dataclass(frozen=True)
class ModelSpec:
    """One model at one provider."""

    provider: str  # a key of providers.PROVIDERS, e.g. "anthropic"
    model: str  # the provider's own name for it, e.g. "claude-opus-5-5"

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class Profile:
    """A job, the model that does it, and the model to try if that one fails."""

    name: str
    primary: ModelSpec
    fallback: ModelSpec | None
    tier: str  # "high" or "low"; "" for a job with a fixed model (the judge)
    model_set: str  # the set the model came from, e.g. "claude"; "" when not from a set (fixed, or overridden)


def parse_spec(text: str) -> ModelSpec:
    """"nebius:Qwen/Qwen3-32B" -> ModelSpec("nebius", "Qwen/Qwen3-32B"). Only the first
    colon separates the two, so model names may contain colons or slashes. A bare
    name is a Claude model."""
    provider, colon, model = text.partition(":")
    return ModelSpec(provider, model) if colon else ModelSpec(DEFAULT_PROVIDER, text)


def tier_model(tier: str) -> str:
    """The model of `tier` ("high" or "low") in the set SPADESK_MODELS names."""
    settings = get_settings()
    if settings.model_set not in settings.model_sets:
        raise ValueError(f"Unknown model set: {settings.model_set!r}. "
                         f"Choose one of: {', '.join(settings.model_sets)}.")
    return settings.model_sets[settings.model_set][tier]


def get_profile(name: str) -> Profile:
    """The profile called `name`, with its models read from the settings."""
    if name not in PROFILES:
        raise ValueError(f"Unknown model profile: {name!r}. Choose one of: {', '.join(PROFILES)}.")
    settings = get_settings()
    default = PROFILES[name]
    tier = default if default in TIERS else ""
    override = settings.profile_models.get(name)
    chosen = override or (tier_model(tier) if tier else default)
    fallback = parse_spec(settings.model_fallback) if settings.model_fallback else None
    primary = parse_spec(chosen)
    from_set = settings.model_set if tier and not override else ""
    return Profile(name, primary, fallback if fallback != primary else None, tier, from_set)
