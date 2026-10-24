"""The booking agent's route: the customer site's chat.

  POST /api/assistant   {"thread": "...", "message": "..."} -> Pebby's reply

The browser makes up a thread id when the chat opens and sends it with every
message; the agent keeps each thread's conversation (graph.py). The id is
not a secret: it only joins the messages of one chat.

(POST /api/chat, the question-and-answer assistant without memory or tools,
stays for the evals, so their scores remain comparable.)
"""

import logging

from fastapi import APIRouter, HTTPException
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel, Field

from ..assistant.replies import ReplyItem
from ..models import ModelError
from ..rag.errors import ModelUnavailable
from . import respond

router = APIRouter(prefix="/api")
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


@router.post("/assistant")
@traceable  # the whole turn is the top step of the trace
def assistant(body: Message) -> Reply:
    """Run one turn of the conversation."""
    # The turn's top step carries the thread id too, which is what LangSmith
    # groups a conversation's turns by (when tracing is on).
    if run := get_current_run_tree():
        run.add_metadata({"thread_id": body.thread})
    try:
        turn = respond(body.thread, body.message)
    except (ModelError, ModelUnavailable) as error:
        log.warning("assistant unavailable: %s", error)  # the reason, for whoever reads the server log
        raise HTTPException(status_code=503, detail="The assistant is unavailable right now.")
    if turn is None:  # a newer message of the same chat arrived while this one ran
        raise HTTPException(status_code=409, detail="A newer message replaced this one.")
    return Reply(answer=turn.answer, items=turn.items, suggestions=turn.suggestions, confirming=turn.confirming)
