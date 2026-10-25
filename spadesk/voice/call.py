"""What a phone call's request says: which call it is, who is calling, and what they just said.

ElevenLabs sends the whole call so far on every turn, in the OpenAI chat
format: a system message (the agent's prompt, as set in the ElevenLabs
dashboard), then the caller's and the agent's lines. The prompt set there
ends with two lines that ElevenLabs fills in for each call:

    Call ID: {{system__conversation_id}}
    Caller number: {{system__caller_id}}

so this module can read them back. The call ID keeps one conversation per
call in Pebby's memory; the caller's number stands in for the phone number
(spadesk/agent/pebby/identify.py). Pebby's own instructions and memory do the
rest: of the lines sent, only the caller's latest one is used.
"""

import re
from dataclasses import dataclass

CALL_ID = re.compile(r"^Call ID:\s*(\S+)\s*$", re.MULTILINE)
CALLER = re.compile(r"^Caller number:\s*(\+?[\d\s()-]{7,})\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Call:
    """One turn of a phone call."""

    id: str  # ElevenLabs' id for the call; "" if the prompt does not carry it
    caller: str  # the number calling; "" if unknown (a withheld number, or a test call)
    said: str  # the caller's latest words, as transcribed


def _text(content: str | list | None) -> str:
    """A message's text, whether sent as a string or as a list of parts."""
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def read_call(messages: list[dict]) -> Call:
    """The call's id, the caller's number and their latest words, from the messages sent."""
    prompt = "\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    said = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
    call_id, caller = CALL_ID.search(prompt), CALLER.search(prompt)
    return Call(call_id[1] if call_id else "", caller[1].strip() if caller else "", said.strip())
