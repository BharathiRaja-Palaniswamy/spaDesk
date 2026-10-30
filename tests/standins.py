"""A stand-in model provider for tests that go through the model gateway.

`use_standin(monkeypatch, replies)` registers a provider called "standin" and
makes it both models of the Claude set, high and low, so any code that asks
the gateway for a model gets one that plays back `replies` in order, records
the messages it was sent, and never calls a real API.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from spadesk.models import providers


class StandIn(GenericFakeChatModel):
    """Plays back its messages, and accepts tools (the replies are prepared in advance)."""

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        """Like a provider's reply format: the next reply read into `schema`, from its
        text (JSON) or from an "answer" call, the way the replies are prepared."""
        def parse(messages):
            raw = self.invoke(messages)
            calls = [call["args"] for call in raw.tool_calls if call["name"] == "answer"]
            try:
                parsed = schema.model_validate(calls[0]) if calls else schema.model_validate_json(raw.text)
            except ValidationError as error:
                return {"raw": raw, "parsed": None, "parsing_error": error}
            return {"raw": raw, "parsed": parsed, "parsing_error": None}
        return RunnableLambda(parse)


def use_standin(monkeypatch, replies: list[AIMessage]) -> list:
    """Make the gateway's models a stand-in that answers with `replies`.
    Returns the list each call's messages are added to."""
    sent: list = []
    queue = iter(replies)  # shared, so each new model picks up where the last one stopped

    # Defined here so it can append to `sent` itself: a list passed in as a
    # field would be copied by pydantic, and the test would never see it fill.
    class Recording(StandIn):
        def _generate(self, messages, *args, **kwargs):
            sent.append(messages)
            return super()._generate(messages, *args, **kwargs)

    monkeypatch.setitem(providers.PROVIDERS, "standin", lambda model: Recording(messages=queue))
    monkeypatch.setenv("SPADESK_CLAUDE_HIGH", "standin:any")
    monkeypatch.setenv("SPADESK_CLAUDE_LOW", "standin:any")
    return sent
