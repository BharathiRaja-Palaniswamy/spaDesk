"""The booking agent: the customer chat that answers, books and cancels.

    state.py     what it remembers
    identify.py  name and phone first
    tools.py     what it can do: search the guide, find times, book, cancel, list, hand off
    confirm.py   the customer's yes before a booking is made or cancelled
    handoffs.py  passing a conversation to the staff
    prompt.py    its instructions and the shape of its replies
    nodes.py     its steps
    graph.py     how the steps are joined, and its memory
    routes.py    POST /api/assistant

`respond` runs one turn: the customer's message in, Pebby's reply out. The
same agent answers the website chat and phone calls (spadesk/voice/); the
channel only changes how replies are phrased, and on a call the caller's
number stands in for the phone number.
"""

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from ..assistant.replies import MAX_SUGGESTIONS, ReplyItem
from .confirm import CHOICES
from .graph import get_graph
from .prompt import time_note


@dataclass(frozen=True)
class Turn:
    """Pebby's reply to one message."""

    answer: str
    items: list[ReplyItem] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confirming: bool = False  # Pebby asked the customer to confirm a booking or cancellation


def _memory(thread: str) -> dict:
    """What the agent remembers about the conversation `thread` (empty if new)."""
    return get_graph().get_state({"configurable": {"thread_id": thread}}).values


def will_think(thread: str) -> bool:
    """True when this conversation's next message goes to the model (the
    customer's details are known), rather than to a quick fixed reply."""
    memory = _memory(thread)
    return bool(memory.get("name") and memory.get("phone"))


def respond(thread: str, message: str, channel: str = "chat", caller: str = "") -> Turn:
    """Run one turn of the conversation `thread` with the customer's `message`.
    `channel` is "chat" or "voice"; `caller` is the number calling, on a phone call."""
    # The per-turn fields start empty; the rest of the state comes from memory.
    # The message goes in after a note of the time it was sent (see prompt.py).
    state = get_graph().invoke(
        {"messages": [HumanMessage(time_note()), HumanMessage(message)], "channel": channel, "caller": caller,
         "rounds": 0, "say": "", "reply": None, "confirming": False},
        # thread_id twice: "configurable" picks the conversation's memory; "metadata"
        # labels the trace, so LangSmith shows every turn of one chat as one thread.
        config={"configurable": {"thread_id": thread}, "metadata": {"thread_id": thread, "channel": channel}})
    if state["confirming"]:
        return Turn(state["say"], suggestions=CHOICES[state["pending"]["name"]], confirming=True)
    if state["say"]:
        return Turn(state["say"])
    reply = state["reply"]
    return Turn(reply.answer, reply.items, reply.suggestions[:MAX_SUGGESTIONS])
