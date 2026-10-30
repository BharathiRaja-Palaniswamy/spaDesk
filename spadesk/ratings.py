"""The 👍 / 👎 under a chat reply, saved in LangSmith as feedback on its trace.

Each chat reply carries `trace_id`: the id of the LangSmith run that made it
(the turn's top step), or none when tracing is off. When the visitor taps
👍 or 👎, the page sends that id here, and it is saved on the run as feedback
"user_rating": 1 for 👍, 0 for 👎. In LangSmith the rating then sits on the
trace, next to what was said: a 👎 can be filtered, read, and sent to the
review queue (evals/annotation.py).

Only the rating is saved, never what the visitor wrote: that is in the trace
already, masked (privacy.py). Anyone may rate, as anyone may chat; a rating
on an id that is not a run is refused by LangSmith and ignored here.
"""

import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from langsmith import Client
from langsmith.run_trees import get_cached_client
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()
KEY = "user_rating"


class Rating(BaseModel):
    """One tap on 👍 or 👎."""

    trace_id: UUID  # the reply's run, as the reply gave it
    score: Literal[0, 1]  # 1: 👍, 0: 👎
    chat: Literal["customer", "owner"]  # which chat the reply was in


@router.post("/api/rating")
def rate(rating: Rating) -> dict[str, bool]:
    """Saves the rating on the reply's run. `saved` is False when it could not
    be, which the page does not need to show: the chat goes on either way."""
    client: Client = get_cached_client()  # the app's one client (tracing.py)
    try:
        client.create_feedback(rating.trace_id, KEY, score=rating.score, source_info={"chat": rating.chat})
    except Exception as error:  # LangSmith unreachable, tracing off, or not a run: never an error page
        log.warning("rating not saved: %s", error)
        return {"saved": False}
    return {"saved": True}
