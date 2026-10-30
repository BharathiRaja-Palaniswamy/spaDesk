"""Pebby, the customer chat (and phone line): two agents, and booking in code.

Read in this order; each file is one step of the flow:

    rules.py     how it behaves: six principles, and their settings
    flow.py      a turn, step by step: identify -> Pebby -> tools -> the yes
    pebby.py     Pebby, the AI: its instructions, its tools, name and phone first
    details.py   a booking's details from the customer's own words; what is missing
    yes.py       the questions for the customer's yes: details, plan, cancellation
    booking.py   what can be booked (code): the request, the plan, fitting times
    executor.py  booking the approved plan, all or none; cancelling
    feedback.py  the Feedback agent: reads feedback, answers it, files a card for the owner
    staff.py     what is written down for the staff: handoffs, and the files they go in
    turns.py     one turn at a time per conversation
    debug.py     debug mode: every step of a turn, under the reply
    routes.py    POST /api/assistant

An AI only where thinking is needed: Pebby talks and decides; the Feedback
agent reads and answers feedback; a small model reads a name. Everything
exact is code.

`respond` runs one turn: the customer's message in, Pebby's reply out.
`hand_over` is what a route says instead when a turn fails: the request goes
to the staff (staff.py), so the conversation never just stops. The same
agent answers the website chat and phone calls (spadesk/voice/); the channel
only changes how replies are phrased, and on a call the caller's number
stands in for the phone number.
"""

import logging
from dataclasses import dataclass, field, replace

from langchain_core.messages import HumanMessage

from ..assistant.replies import MAX_SUGGESTIONS, ReplyItem
from .debug import steps
from .flow import get_graph
from .pebby import answer_text, time_note
from .rules import WAIT_FOR_NEWER
from .staff import HANDOVER, record_handoff
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
from .yes import CHOICES

log = logging.getLogger("spadesk.agent")


@dataclass(frozen=True)
class Turn:
    """Pebby's reply to one message."""

    answer: str
    items: list[ReplyItem] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confirming: bool = False  # Pebby asked the customer to confirm a booking or cancellation
    debug: list[dict] = field(default_factory=list)  # what happened between the agents (debug.py)


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
        # The message goes in after a note of the time it was sent (pebby.time_note).
        try:
            state = graph.invoke(
                {"messages": [HumanMessage(time_note()), HumanMessage(message)], "channel": channel,
                 "caller": caller, "rounds": 0, "asked": [], "repeats": 0, "say": "", "suggestions": [],
                 "reply": None,
                 "confirming": False, "acted": False},
                config=config)
        except Exception:
            # A step failed part way (e.g. a model could not be reached),
            # which can leave a tool request without its result in the memory, and
            # every later turn would then be refused. So the failed turn is forgotten,
            # as a replaced one is, unless it made or cancelled a booking.
            if not graph.get_state(config).values.get("acted"):
                undo_later(thread, before)
            raise
        if replaced(thread, number):
            # Forget this turn, unless it made or cancelled a booking: that happened.
            if not state.get("acted"):
                undo_later(thread, before)
            return None
    if state["confirming"]:
        return Turn(state["say"], suggestions=CHOICES[state["pending"]["kind"]], confirming=True)
    if state["say"]:
        return Turn(state["say"], suggestions=state.get("suggestions", []))
    reply = state["reply"]
    return Turn(answer_text(reply), reply.items, reply.suggestions[:MAX_SUGGESTIONS])


def respond(thread: str, message: str, channel: str = "chat", caller: str = "") -> Turn | None:
    """Run one turn of the conversation `thread` with the customer's `message`.
    `channel` is "chat" or "voice"; `caller` is the number calling, on a phone call.

    If a newer turn of the same conversation replaced this one (turns.py),
    this one waits for the newer turn's reply and returns it, so no request
    is left without an answer; None only if that reply does not come in time."""
    number = arrive(thread)
    with steps() as notes:
        turn = _run(thread, number, message, channel, caller)
    if turn is not None:
        turn = replace(turn, debug=notes)
    if turn is None:
        return newer_reply(thread, number, WAIT_FOR_NEWER)
    answered(thread, number, turn)
    return turn
