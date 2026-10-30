"""Pebby's route (the customer site's chat), and the owner's feedback cards.

  POST /api/assistant   {"thread": "...", "message": "..."} -> Pebby's reply

The browser makes up a thread id when the chat opens and sends it with every
message; the agent keeps each thread's conversation (flow.py). The id is
not a secret: it only joins the messages of one chat.

In debug mode (SPADESK_DEBUG=true), every reply also carries `debug`: the
steps between the agents in that turn (debug.py), shown under the reply.
It is for testing: anyone using the chat sees it, so keep it off on a
site customers use.

The owner's feedback cards (the Feedback agent files one for each piece of
feedback; feedback.py), signed in only:

  GET  /api/feedback                  every card, open ones first
  POST /api/feedback/{card}/close     close one, with an optional note

Every request is answered: a turn that fails, or runs past CHAT_TURN_LIMIT,
ends in the handover. When a turn fails (a model cannot be reached, an unexpected error), the
customer gets Pebby's handover message, and the staff a handoff to act on
(agent.hand_over), rather than an error.

(POST /api/chat, the question-and-answer assistant without memory or tools,
stays for the evals, so their scores remain comparable.)
"""

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel, Field

from ..access.session import require
from ..assistant.replies import ReplyItem
from ..config import get_settings
from ..models import ModelError
from ..rag.errors import ModelUnavailable
from ..tracing import current_run_id
from . import hand_over, respond
from .feedback import cards, close
from .rules import CHAT_TURN_LIMIT

router = APIRouter(prefix="/api")
_turns = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chat-turn")  # so a turn can be timed
log = logging.getLogger("spadesk.agent")


class Message(BaseModel):
    """One customer message in one chat."""

    thread: str = Field(min_length=8, max_length=64)  # the browser's id for this chat
    message: str = Field(min_length=1, max_length=2000)


class Reply(BaseModel):
    """What the chat shows: the answer, an optional list, and replies to tap."""

    answer: str
    items: list[ReplyItem]
    suggestions: list[str]
    confirming: bool  # the customer is being asked to confirm a booking or cancellation
    trace_id: str | None = None  # this turn's trace, for its 👍 / 👎 (ratings.py); none when tracing is off
    debug: list[dict] | None = None  # debug mode only: the steps between the agents


@router.post("/assistant", response_model_exclude_none=True)  # no `debug` at all when it is not shown
@traceable  # the whole turn is the top step of the trace
def assistant(body: Message) -> Reply:
    """Run one turn of the conversation."""
    # The turn's top step carries the thread id too, which is what LangSmith
    # groups a conversation's turns by (when tracing is on).
    if run := get_current_run_tree():
        run.add_metadata({"thread_id": body.thread})
    work = _turns.submit(contextvars.copy_context().run, respond, body.thread, body.message)
    try:
        turn = work.result(timeout=CHAT_TURN_LIMIT)
    except TimeoutError:  # never left waiting: the staff take it from here
        log.warning("assistant turn took longer than %s s", CHAT_TURN_LIMIT)
        turn = hand_over(body.thread, "The assistant took too long to answer.", body.message)
    except (ModelError, ModelUnavailable) as error:
        log.warning("assistant unavailable: %s", error)  # the reason, for whoever reads the server log
        turn = hand_over(body.thread, "The assistant could not be reached.", body.message)
    except Exception:  # any failure ends in a handover, never an error page
        log.exception("assistant turn failed")
        turn = hand_over(body.thread, "The assistant hit an unexpected error.", body.message)
    if turn is None:  # replaced by a newer message whose reply did not come in time: still answered
        turn = hand_over(body.thread, "A message was replaced and its reply did not come in time.", body.message)
    return Reply(answer=turn.answer, items=turn.items, suggestions=turn.suggestions, confirming=turn.confirming,
                 trace_id=current_run_id(), debug=turn.debug if get_settings().debug else None)


owner = APIRouter(prefix="/api/feedback", dependencies=[Depends(require("feedback"))])


class Closing(BaseModel):
    """The owner's note when closing a card (it may be empty)."""

    note: str = Field(default="", max_length=500)


@owner.get("")
def feedback_cards() -> list[dict]:
    """Every feedback card, open ones first, newest first."""
    return cards()


@owner.post("/{card_id}/close")
def close_card(card_id: str, body: Closing) -> dict:
    """Close an open card, with the owner's note."""
    if (card := close(card_id, body.note)) is None:
        raise HTTPException(status_code=404, detail="No open card with that reference.")
    return card
