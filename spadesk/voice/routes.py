"""The phone line's brain: Pebby, behind an OpenAI-style endpoint ElevenLabs calls.

  POST /v1/chat/completions   one turn of a call -> Pebby's spoken reply, streamed
  GET  /api/voice-widget      the agent the manage page's phone-line test connects to

A call goes: the caller dials the spa's Twilio number, ElevenLabs answers,
turns their speech into text and, on each turn, sends the call so far here
("Custom LLM" in the ElevenLabs agent settings). Pebby (spadesk/agent/)
answers with the same rules, tools and calendar as the website chat, and
ElevenLabs speaks the reply.

Only ElevenLabs may call this: it sends the shared secret SPADESK_VOICE_SECRET
as its API key ("Authorization: Bearer <secret>"). Without the setting the
line is off and every request is refused.

The call never just ends: when something fails (a model or the search
cannot be reached, a turn runs past TURN_LIMIT, an unexpected error), the
caller hears Pebby's handover message and the staff get a handoff to act on
(agent.hand_over); the reason goes to the server log.

A reply that takes a while (Pebby searching the guide, looking up times) is
preceded by a short filler, said as soon as the wait passes FILLER_AFTER, so
the caller is not left in silence; a quick reply ("You're welcome") is said
straight away, without it.
"""

import hmac
import logging
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel

from ..agent import hand_over, respond
from ..agent.handoffs import HANDOVER
from ..config import get_settings
from ..models import ModelError
from ..rag.errors import ModelUnavailable
from .call import read_call
from .stream import completion, sse

router = APIRouter()
log = logging.getLogger("spadesk.voice")

FILLER = "Let me check... "  # said while Pebby thinks; ElevenLabs suggests ending a filler with "... "
FILLER_AFTER = 1.5  # seconds: a reply ready sooner is said straight away, with no filler
# Seconds a turn may take before the caller is handed over instead. Under ElevenLabs'
# own wait for a reply (set to 10 s in the agent), so the caller hears the handover
# rather than the line going dead.
TURN_LIMIT = 8.0
# Turns run here, so the reply can be waited for with a time limit while the stream is open.
_turns = ThreadPoolExecutor(max_workers=8, thread_name_prefix="voice-turn")


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
    """Pebby's reply to one turn, in pieces: a filler first when the reply takes a while."""
    if not call_id:
        # Nothing to remember the call by, so nothing to write down: the log says why.
        log.warning("voice request without a call id: add 'Call ID: {{system__conversation_id}}' to the prompt")
        yield HANDOVER
        return
    if not re.search(r"\w", said):  # no words (ElevenLabs sends "..." for silence): nothing to answer
        return
    thread = f"call-{call_id}"
    work = _turns.submit(respond, thread, said, channel="voice", caller=caller)
    if not wait([work], timeout=FILLER_AFTER).done:
        yield FILLER  # the caller hears something while Pebby thinks
    if not wait([work], timeout=TURN_LIMIT - FILLER_AFTER).done:
        log.warning("voice turn took longer than %s s", TURN_LIMIT)
        yield hand_over(thread, "The phone call's turn took too long.", said, caller).answer
        return
    try:
        turn = work.result()
    except (ModelError, ModelUnavailable) as error:
        log.warning("voice turn failed: %s", error)
        turn = hand_over(thread, "The assistant could not be reached on a call.", said, caller)
    except Exception:  # any failure ends in a handover, never a dropped call
        log.exception("voice turn failed")
        turn = hand_over(thread, "The assistant hit an unexpected error on a call.", said, caller)
    if turn is not None:  # None: ElevenLabs sent a newer version of this turn, and nobody hears this one
        yield _spoken(turn.answer, turn.items)


@router.get("/api/voice-widget")
def voice_widget() -> dict:
    """The ElevenLabs agent for the manage page's phone-line test, or "" to show none.
    An agent id is not a secret: the widget needs it in the page anyway."""
    return {"agent_id": get_settings().voice_agent_id}


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
