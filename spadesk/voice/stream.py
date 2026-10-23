"""Replies in the shape ElevenLabs reads: OpenAI chat-completion chunks, streamed.

The answer is sent as Server-Sent Events: each piece of text is one line
"data: {json}" followed by a blank line, and the stream ends with
"data: [DONE]". ElevenLabs starts speaking as soon as the first piece
arrives, which is why a short filler ("Let me check… ") can be sent before
the real answer is ready: the caller hears something while Pebby thinks.
"""

import json
import secrets
import time
from collections.abc import Iterable, Iterator

MODEL = "pebby"  # the name the endpoint gives itself in its replies


def _chunk(reply_id: str, delta: dict, finish: str | None = None) -> str:
    """One Server-Sent Event carrying one chat-completion chunk."""
    body = {"id": reply_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(body)}\n\n"


def sse(pieces: Iterable[str]) -> Iterator[str]:
    """The pieces of text, as a stream ElevenLabs can speak, piece by piece."""
    reply_id = "chatcmpl-" + secrets.token_hex(8)
    yield _chunk(reply_id, {"role": "assistant"})
    for piece in pieces:
        if piece:
            yield _chunk(reply_id, {"content": piece})
    yield _chunk(reply_id, {}, finish="stop")
    yield "data: [DONE]\n\n"


def completion(text: str) -> dict:
    """The whole reply at once, as a plain chat completion: for a request that did not ask to stream."""
    return {"id": "chatcmpl-" + secrets.token_hex(8), "object": "chat.completion", "created": int(time.time()),
            "model": MODEL, "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": text}}]}
