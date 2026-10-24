"""The booking agent: the customer chat that answers, books and cancels.

    state.py     what it remembers
    identify.py  name and phone first
    tools.py     what it can do: search the guide, find times, book, cancel, list, hand off
    confirm.py   the customer's yes before a booking is made or cancelled
    handoffs.py  passing a conversation to the staff
    prompt.py    its instructions and the shape of its replies
    nodes.py     its steps
    graph.py     how the steps are joined, and its memory
    turns.py     one turn at a time per conversation, and undoing replaced turns
    routes.py    POST /api/assistant

`respond` runs one turn: the customer's message in, Pebby's reply out.
`hand_over` is what a route says instead when a turn fails: the request goes
to the staff (handoffs.py), so the conversation never just stops. The
same agent answers the website chat and phone calls (spadesk/voice/); the
channel only changes how replies are phrased, and on a call the caller's
number stands in for the phone number.
"""

import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from ..assistant.replies import MAX_SUGGESTIONS, ReplyItem
from .confirm import CHOICES
from .graph import get_graph
from .handoffs import HANDOVER, record_handoff
from .prompt import time_note
from .turns import (
    KEEP,
    answered,
    arrive,
    lock,
    newer_reply,
    replaced,
    take_restore,
    undo_later,
)

log = logging.getLogger("spadesk.agent")
WAIT_FOR_NEWER = 10.0  # seconds a replaced turn waits for the newer turn's reply


@dataclass(frozen=True)
class Turn:
    """Pebby's reply to one message."""

    answer: str
    items: list[ReplyItem] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confirming: bool = False  # Pebby asked the customer to confirm a booking or cancellation


def hand_over(thread: str, reason: str, said: str = "", caller: str = "") -> Turn:
    """Hand the conversation `thread` to the staff after something went wrong:
    write down who to call back (the name and number Pebby has, or the number
    calling) and why, and return what to tell the customer."""
    try:
        memory = get_graph().get_state({"configurable": {"thread_id": thread}}).values
        record_handoff(memory.get("name", ""), memory.get("phone") or caller,
                       f"{reason} Their last message: {said!r}" if said else reason)
    except Exception:  # the customer must still hear HANDOVER, whatever failed here
        log.exception("could not record the handoff for %s", thread)
    return Turn(HANDOVER)


def _run(thread: str, number: int, message: str, channel: str, caller: str) -> Turn | None:
    """Run turn `number` of `thread`, one at a time; None if a newer turn replaced it."""
    with lock(thread):  # one turn at a time per conversation
        if replaced(thread, number):
            return None
        graph = get_graph()
        # thread_id twice: "configurable" picks the conversation's memory; "metadata"
        # labels the trace, so LangSmith shows every turn of one chat as one thread.
        config = {"configurable": {"thread_id": thread}, "metadata": {"thread_id": thread, "channel": channel}}
        restore = take_restore(thread)
        if restore is None:
            graph.checkpointer.delete_thread(thread)  # the replaced turn was the first: start afresh
        elif restore is not KEEP:
            config["configurable"]["checkpoint_id"] = restore  # carry on from before the replaced turn
        before = graph.get_state(config).config["configurable"].get("checkpoint_id")
        # The per-turn fields start empty; the rest of the state comes from memory.
        # The message goes in after a note of the time it was sent (see prompt.py).
        state = graph.invoke(
            {"messages": [HumanMessage(time_note()), HumanMessage(message)], "channel": channel, "caller": caller,
             "rounds": 0, "say": "", "reply": None, "confirming": False, "acted": False},
            config=config)
        if replaced(thread, number):
            # Forget this turn, unless it made or cancelled a booking: that happened.
            if not state.get("acted"):
                undo_later(thread, before)
            return None
    if state["confirming"]:
        return Turn(state["say"], suggestions=CHOICES[state["pending"]["name"]], confirming=True)
    if state["say"]:
        return Turn(state["say"])
    reply = state["reply"]
    return Turn(reply.answer, reply.items, reply.suggestions[:MAX_SUGGESTIONS])


def respond(thread: str, message: str, channel: str = "chat", caller: str = "") -> Turn | None:
    """Run one turn of the conversation `thread` with the customer's `message`.
    `channel` is "chat" or "voice"; `caller` is the number calling, on a phone call.

    If a newer turn of the same conversation replaced this one (turns.py),
    this one waits for the newer turn's reply and returns it, so no request
    is left without an answer; None only if that reply does not come in time."""
    number = arrive(thread)
    turn = _run(thread, number, message, channel, caller)
    if turn is None:
        return newer_reply(thread, number, WAIT_FOR_NEWER)
    answered(thread, number, turn)
    return turn
