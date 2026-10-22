"""Model profiles: which model each kind of work uses.

A profile names a job, not a model ("chat-default", "judge"), so code asks
for the job and the settings decide the model. Changing the model for one
job is then a line in `.env`, e.g.

    SPADESK_PROFILE_CHEAP_ROUTER=nebius:Qwen/Qwen3-32B

A model is written "provider:model". A bare model name uses the default
provider, SPADESK_MODEL_PROVIDER. Each profile can fall back to a second
model when its own fails (SPADESK_MODEL_FALLBACK, one for all profiles).
"""

from dataclasses import dataclass

from ..config import get_settings

# The judge grades eval answers. It stays on Claude whatever the app uses,
# so scores stay comparable from run to run.
JUDGE_MODEL = "anthropic:claude-opus-5-5"
# A smaller, faster Claude for quick or spoken answers.
FAST_MODEL = "anthropic:claude-haiku-4-5-20251001"

# Profile -> its default model. None means the app's main model:
# SPADESK_MODEL from SPADESK_MODEL_PROVIDER, which is Claude Opus by default.
PROFILES: dict[str, str | None] = {
    "chat-default": None,  # the customer and owner chats
    "booking-agent": None,  # the booking flow, once it has tools (see docs/models.md)
    "cheap-router": FAST_MODEL,  # small decisions, e.g. "which tool does this need?"
    "voice-ready": FAST_MODEL,  # short answers that must start fast, for speech
    "judge": JUDGE_MODEL,  # grades eval answers (evals/judge.py)
}


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


def parse_spec(text: str, default_provider: str) -> ModelSpec:
    """"nebius:Qwen/Qwen3-32B" -> ModelSpec("nebius", "Qwen/Qwen3-32B"). Only the first
    colon separates the two, so model names may contain colons or slashes."""
    provider, colon, model = text.partition(":")
    return ModelSpec(provider, model) if colon else ModelSpec(default_provider, text)


def get_profile(name: str) -> Profile:
    """The profile called `name`, with its models read from the settings."""
    if name not in PROFILES:
        raise ValueError(f"Unknown model profile: {name!r}. Choose one of: {', '.join(PROFILES)}.")
    settings = get_settings()
    chosen = settings.profile_models.get(name) or PROFILES[name] or settings.model
    fallback = parse_spec(settings.model_fallback, settings.model_provider) if settings.model_fallback else None
    primary = parse_spec(chosen, settings.model_provider)
    return Profile(name, primary, fallback if fallback != primary else None)
