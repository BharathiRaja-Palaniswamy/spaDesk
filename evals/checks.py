"""The answer checks: the reply against the `answer` part of a case's label.

Every target ends in a reply, so these are shared. Each check gets the reply,
in one form whichever target gave it:

    {"question": what was asked (for Pebby, the conversation so far),
     "text":     everything the reply shows: the answer, its list, the insights,
     "sections": the guide's sections behind it (for Pebby, the ones its search found),
     "covered":  whether the guide answered it (the chats only)}

and the expected value from golden.json, and returns a problem as text, or
None when the reply passes. Code checks are cheap, exact and repeatable; use
them wherever code can tell. `judge` is the one exception: a second model
call, for what code cannot test (see judge.py).

To add a check: write a function here and add it to CHECKS under the key that
golden.json uses in "answer".
"""

from collections.abc import Callable
from typing import Any

from .figures import expected_texts
from .judge import judge_answer

Reply = dict[str, Any]


def covered(reply: Reply, expected: bool) -> str | None:
    """Whether the assistant found the answer in the guide."""
    if reply.get("covered") != expected:
        return f"covered was {reply.get('covered')}, expected {expected}"
    return None


def cites(reply: Reply, expected: list[str]) -> str | None:
    """Every expected section is among the sections behind the answer.

    This is how we test that the answer linked the right facts: for the body
    wrap question, the rule is under Pregnancy and the reason (heat) is under
    Heat-Based Treatments. Extra sections are allowed. An empty list expects
    none (a question the guide must not answer).
    """
    found = reply.get("sections", [])
    if not expected:
        return f"cited {found}, expected none" if found else None
    missing = [section for section in expected if not any(section in name for name in found)]
    if missing:
        return f"did not cite {missing}; cited {found}"
    return None


def mentions(reply: Reply, expected: list[str]) -> str | None:
    """Every expected text appears somewhere in the reply, ignoring case.

    Used for facts with one right wording: a figure such as "22,614" (written
    without the "$" so "$22,614" also passes) or a name such as "Facials".
    """
    text = reply["text"].lower()
    missing = [word for word in expected if word.lower() not in text]
    return f"did not mention {missing}" if missing else None


def lacks(reply: Reply, expected: list[str]) -> str | None:
    """No expected text appears in the reply: what must not be said, such as
    another customer's service or name."""
    text = reply["text"].lower()
    said = [word for word in expected if word.lower() in text]
    return f"mentioned {said}" if said else None


def mentions_report(reply: Reply, specs: list[dict[str, Any]]) -> str | None:
    """Like `mentions`, but the expected figures are read from the bookings
    data when the eval runs (see figures.py), so they never go stale."""
    return mentions(reply, [text for spec in specs for text in expected_texts(spec)])


def judge(reply: Reply, criteria: list[str]) -> str | None:
    """Every criterion passes, as graded by a second model call (see judge.py).
    For what code cannot test: yes or no, safety, tone, declining properly."""
    verdicts = judge_answer(reply["question"], reply["text"], criteria)
    failed = [f"judge: {v.criterion} ({v.reasoning})" for v in verdicts if not v.passed]
    return "; ".join(failed) or None


# The key in a case's "answer" -> the check that tests it.
CHECKS: dict[str, Callable[[Reply, Any], str | None]] = {
    "covered": covered,
    "cites": cites,
    "mentions": mentions,
    "lacks": lacks,
    "mentions_report": mentions_report,
    "judge": judge,
}


def grade_answer(reply: Reply, expected: dict[str, Any]) -> list[str]:
    """Runs every answer check the case asks for. An empty list means it passed.
    The judge runs even when a free check has failed: its verdicts are compared
    with a person's (judge calibration), so every judged reply needs one."""
    problems = [CHECKS[name](reply, value) for name, value in expected.items()]
    return [problem for problem in problems if problem]
