"""One turn at a time per conversation, and turns that were replaced.

On a phone call, ElevenLabs sends the caller's words as soon as they pause.
If they go on talking ("I'd like a prenatal, uh, ... on Sunday"), it sends
the longer version too, and drops the reply to the first: the caller never
hears it. Run side by side on the same conversation, the two turns would
both write to Pebby's memory, and Pebby would remember a reply the caller
never heard (in testing, an "offer for today" that led to a booking on the
wrong day).

So, per conversation:
  - turns run one at a time: a newer one waits for the one running;
  - every turn gets a number when it arrives; a turn that is no longer the
    newest when it starts, or when it finishes, was replaced, and its own
    reply is dropped;
  - a replaced turn's changes to the memory are undone: the next turn
    starts from the memory as it was before it (LangGraph keeps every saved
    step, so the conversation can carry on from an earlier one);
  - except when the replaced turn made or cancelled a booking: that really
    happened, so the memory keeps it, and Pebby does not do it a second time;
  - a replaced turn's request is still answered, never left empty: it waits
    for the newer turn's reply and gives the same one (whoever is still
    listening on it hears what the newest turn says).

The numbers and locks are kept in this server's memory, as the
conversations are (flow.py).
"""

import threading

_guard = threading.Lock()  # protects the three dictionaries below
_locks: dict[str, threading.Lock] = {}  # conversation -> its one-at-a-time lock
_newest: dict[str, int] = {}  # conversation -> the number of the newest turn that arrived
# conversation -> the saved step the next turn starts from (None: start the conversation afresh)
_restore: dict[str, str | None] = {}
KEEP = object()  # take_restore()'s answer when nothing needs undoing
_answered = threading.Condition()  # signalled whenever a turn's reply is ready
_answers: dict[str, tuple[int, object]] = {}  # conversation -> (turn number, its reply), the newest so far


def arrive(thread: str) -> int:
    """Number a newly arrived turn of `thread`; the highest number is the newest."""
    with _guard:
        _newest[thread] = _newest.get(thread, 0) + 1
        _locks.setdefault(thread, threading.Lock())
        return _newest[thread]


def lock(thread: str) -> threading.Lock:
    """The lock that lets one turn of `thread` run at a time."""
    with _guard:
        return _locks.setdefault(thread, threading.Lock())


def replaced(thread: str, number: int) -> bool:
    """True when a newer turn of `thread` has arrived since turn `number` did."""
    with _guard:
        return _newest.get(thread, number) != number


def undo_later(thread: str, checkpoint: str | None) -> None:
    """Have the next turn of `thread` start from `checkpoint`, the memory as
    it was before a replaced turn. The earliest one wins, so several replaced
    turns in a row are all undone."""
    with _guard:
        _restore.setdefault(thread, checkpoint)


def take_restore(thread: str) -> object:
    """The step the next turn should start from, or KEEP when nothing needs undoing."""
    with _guard:
        return _restore.pop(thread, KEEP)


def answered(thread: str, number: int, reply: object) -> None:
    """Keep turn `number`'s reply, for requests of older turns that are waiting for it."""
    with _answered:
        if number > _answers.get(thread, (0, None))[0]:
            _answers[thread] = (number, reply)
        _answered.notify_all()


def newer_reply(thread: str, number: int, timeout: float) -> object | None:
    """The reply of a turn newer than `number`, once there is one, or None after `timeout` seconds."""
    with _answered:
        _answered.wait_for(lambda: _answers.get(thread, (0, None))[0] > number, timeout)
        newest = _answers.get(thread)
        return newest[1] if newest and newest[0] > number else None
