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
    newest when it starts, or when it finishes, was replaced, and its reply
    is dropped;
  - a replaced turn's changes to the memory are undone: the next turn
    starts from the memory as it was before it (LangGraph keeps every saved
    step, so the conversation can carry on from an earlier one);
  - except when the replaced turn made or cancelled a booking: that really
    happened, so the memory keeps it, and Pebby does not do it a second time.

The numbers and locks are kept in this server's memory, as the
conversations are (graph.py).
"""

import threading

_guard = threading.Lock()  # protects the three dictionaries below
_locks: dict[str, threading.Lock] = {}  # conversation -> its one-at-a-time lock
_newest: dict[str, int] = {}  # conversation -> the number of the newest turn that arrived
# conversation -> the saved step the next turn starts from (None: start the conversation afresh)
_restore: dict[str, str | None] = {}
KEEP = object()  # take_restore()'s answer when nothing needs undoing


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
