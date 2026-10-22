"""Errors from any provider, turned into one kind the app understands.

Each provider's library raises its own exceptions: anthropic.RateLimitError,
openai.RateLimitError, and so on. The gateway turns them into ModelError
with a `kind`, so the app (and the fallback) can react without knowing which
provider failed:

  auth         the key is missing, wrong, or not allowed to use the model
  rate_limit   too many requests; trying again later or elsewhere may work
  timeout      no answer in time
  unavailable  the provider could not be reached, or failed on its side
  bad_request  the provider refused the request itself, e.g. a model name it
               does not know, a feature that model lacks, or no credit left
  bad_output   the model answered, but not in the shape asked for (structured.py)
"""

import anthropic
import openai
from pydantic import ValidationError

# (exception classes, kind), checked in order: a timeout is also a connection error.
KINDS: list[tuple[tuple[type[Exception], ...], str]] = [
    ((anthropic.AuthenticationError, anthropic.PermissionDeniedError,
      openai.AuthenticationError, openai.PermissionDeniedError), "auth"),
    ((anthropic.RateLimitError, openai.RateLimitError), "rate_limit"),
    ((anthropic.APITimeoutError, openai.APITimeoutError), "timeout"),
    ((anthropic.APIConnectionError, anthropic.InternalServerError,
      openai.APIConnectionError, openai.InternalServerError), "unavailable"),
    ((anthropic.BadRequestError, anthropic.NotFoundError, anthropic.UnprocessableEntityError,
      openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError), "bad_request"),
    # How a missing key shows up: the Anthropic SDK raises TypeError when it
    # cannot find one; LangChain's chat models raise a pydantic ValidationError.
    ((TypeError, ValidationError), "auth"),
]


class ModelError(Exception):
    """A model call that failed, in provider-neutral terms."""

    def __init__(self, kind: str, model: str, detail: str) -> None:
        super().__init__(f"{model}: {kind}: {detail}")
        self.kind = kind  # one of the kinds listed above
        self.model = model  # "provider:model", the one that failed
        self.detail = detail  # the provider's own message, for the logs


def normalize(error: Exception, model: str) -> ModelError | None:
    """`error` as a ModelError, or None when it is not a provider error (a bug
    in our code, say), which should not be hidden behind a fallback."""
    for classes, kind in KINDS:
        if isinstance(error, classes):
            return ModelError(kind, model, str(error))
    return None
