"""A stand-in model provider for tests that go through the model gateway.

`use_standin(monkeypatch, replies)` registers a provider called "standin" and
makes it the main model, so any code that asks the gateway for a model gets
one that plays back `replies` in order, records the messages it was sent, and
never calls a real API.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from spadesk.models import providers


class StandIn(GenericFakeChatModel):
    """Plays back its messages, and accepts tools (the replies are prepared in advance)."""

    def bind_tools(self, tools, **kwargs):
        return self


def use_standin(monkeypatch, replies: list[AIMessage], method: str = "native") -> list:
    """Make the gateway's main model a stand-in that answers with `replies`.
    `method` is how it gives fixed-shape replies ("native" like Claude, or
    "answer-tool" like an open model). Returns the list each call's messages
    are added to."""
    sent: list = []
    queue = iter(replies)  # shared, so each new model picks up where the last one stopped

    # Defined here so it can append to `sent` itself: a list passed in as a
    # field would be copied by pydantic, and the test would never see it fill.
    class Recording(StandIn):
        def _generate(self, messages, *args, **kwargs):
            sent.append(messages)
            return super()._generate(messages, *args, **kwargs)

    monkeypatch.setitem(providers.PROVIDERS, "standin", lambda model: Recording(messages=queue))
    monkeypatch.setitem(providers.STRUCTURED, "standin", method)
    monkeypatch.setenv("SPADESK_MODEL_PROVIDER", "standin")
    monkeypatch.setenv("SPADESK_MODEL", "any")
    return sent
