"""The phone line's brain: Pebby, behind an OpenAI-style endpoint ElevenLabs calls.

  POST /v1/chat/completions   one turn of a call -> Pebby's spoken reply, streamed

A call goes: the caller dials the spa's Twilio number, ElevenLabs answers,
turns their speech into text and, on each turn, sends the call so far here
("Custom LLM" in the ElevenLabs agent settings). Pebby (spadesk/agent/)
answers with the same rules, tools and calendar as the website chat, and
ElevenLabs speaks the reply.

Only ElevenLabs may call this: it sends the shared secret SPADESK_VOICE_SECRET
as its API key ("Authorization: Bearer <secret>"). Without the setting the
line is off and every request is refused.

When something fails, the caller hears an apology instead of silence, and
the reason goes to the server log.
"""

import hmac
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel

from ..agent import respond, will_think
from ..config import get_settings
from ..models import ModelError
from ..rag.errors import ModelUnavailable
from .call import read_call
from .stream import completion, sse

router = APIRouter()
log = logging.getLogger("spadesk.voice")

FILLER = "Let me check... "  # said while Pebby thinks; ElevenLabs suggests ending a filler with "... "
SORRY = "Sorry, I'm having trouble right now. Please try again in a moment, or call back later."
NO_CALL_ID = "Sorry, this line isn't set up correctly yet. Please call back later."


class Completion(BaseModel):
    """The parts of an OpenAI chat-completion request this endpoint reads; the rest is ignored."""

    messages: list[dict]
    stream: bool = False


def _check(authorization: str | None) -> None:
    """Refuse the request unless it carries the shared secret."""
    secret = get_settings().voice_secret
    if not secret:
        raise HTTPException(status_code=503, detail="The phone line is not set up (SPADESK_VOICE_SECRET).")
    given = (authorization or "").removeprefix("Bearer ").strip()
    # compare_digest takes the same time whatever the secret, so it cannot be guessed letter by letter.
    if not hmac.compare_digest(given.encode(), secret.encode()):
        raise HTTPException(status_code=401, detail="Wrong or missing API key.")


def _spoken(answer: str, items: list) -> str:
    """The reply as one thing to say: a list, if the model gave one, said in the sentence."""
    if not items:
        return answer
    return answer + " " + "; ".join(f"{item.name}, {item.detail}" for item in items) + "."


def _reply(call_id: str, caller: str, said: str) -> Iterator[str]:
    """Pebby's reply to one turn, in pieces: a filler first when Pebby has to think."""
    if not call_id:
        log.warning("voice request without a call id: add 'Call ID: {{system__conversation_id}}' to the prompt")
        yield NO_CALL_ID
        return
    thread = f"call-{call_id}"
    if will_think(thread):
        yield FILLER
    try:
        turn = respond(thread, said, channel="voice", caller=caller)
    except (ModelError, ModelUnavailable) as error:
        log.warning("voice turn failed: %s", error)
        yield SORRY
        return
    yield _spoken(turn.answer, turn.items)


@router.post("/v1/chat/completions")
@traceable(name="phone_call_turn")  # the whole turn is the top step of the trace
def chat_completions(body: Completion, authorization: str | None = Header(default=None)):
    """One turn of a phone call: the caller's latest words in, Pebby's reply out."""
    _check(authorization)
    call = read_call(body.messages)
    # Labelled like the chat's turns, so LangSmith shows each call as one thread.
    if call.id and (run := get_current_run_tree()):
        run.add_metadata({"thread_id": f"call-{call.id}", "channel": "voice"})
    pieces = _reply(call.id, call.caller, call.said)
    if body.stream:
        return StreamingResponse(sse(pieces), media_type="text/event-stream")
    return completion("".join(pieces))
