"""The checks that grade one reply against what a case expects.

Each check is a plain function: it gets the chat's reply (the JSON the route
returns, plus the question it answers) and the expected value from
cases.json, and returns a problem as text, or None when the reply passes.
Code checks are cheap, exact and repeatable; use them wherever the right
answer can be tested by code. `judge` is the one exception: it asks a second
model call, for what code cannot test (see judge.py).

To add a check: write a function here and add it to CHECKS under the key that
cases.json uses in "expect".
"""

from collections.abc import Callable
from typing import Any

from .figures import expected_texts
from .judge import judge_answer

Reply = dict[str, Any]


def covered(reply: Reply, expected: bool) -> str | None:
    """Whether the assistant found the answer in the business document."""
    if reply["covered"] != expected:
        return f"covered was {reply['covered']}, expected {expected}"
    return None


def cites(reply: Reply, expected: list[str]) -> str | None:
    """Every expected section is among the sections the answer cites.

    This is how we test that the answer linked the right facts: for the body
    wrap question, the rule is under Pregnancy and the reason (heat) is under
    Heat-Based Treatments. Extra sections are allowed.
    """
    missing = [section for section in expected if section not in reply["sections"]]
    if missing:
        return f"did not cite {missing}; cited {reply['sections']}"
    return None


def reply_text(reply: Reply) -> str:
    """All the text a reply shows: the answer, each list entry and, for the
    owner, the insights. A figure may sit in any of them."""
    items = [f"{item['name']} {item['detail']}" for item in reply.get("items", [])]
    return " ".join([reply["answer"], *items, *reply.get("insights", [])])


def mentions(reply: Reply, expected: list[str]) -> str | None:
    """Every expected text appears somewhere in the reply, ignoring case.

    Used for facts with one right wording: a figure such as "22,614" (written
    without the "$" so "$22,614" also passes) or a name such as "Facials".
    """
    text = reply_text(reply).lower()
    missing = [word for word in expected if word.lower() not in text]
    if missing:
        return f"did not mention {missing}"
    return None


def mentions_report(reply: Reply, specs: list[dict[str, Any]]) -> str | None:
    """Like `mentions`, but the expected figures are read from the bookings
    data when the eval runs (see figures.py), so they never go stale."""
    return mentions(reply, [text for spec in specs for text in expected_texts(spec)])


def uses_report(reply: Reply, expected: bool) -> str | None:
    """Owner chat only: whether the model called the get_report tool.

    `reports` lists the periods the model asked for (see ChatAnswer in
    spadesk/reports/routes.py), so it is empty when no tool was called. A
    figure in an answer with no report behind it was made up.
    """
    called = bool(reply.get("reports"))
    if called != expected:
        return f"get_report called: {called}, expected {expected} (reports: {reply.get('reports')})"
    return None


def judge(reply: Reply, criteria: list[str]) -> str | None:
    """Every criterion passes, as graded by a second model call (see judge.py).
    For what code cannot test: yes or no, safety, tone, declining properly."""
    verdicts = judge_answer(reply["question"], reply_text(reply), criteria)
    failed = [f"judge: {v.criterion} ({v.reasoning})" for v in verdicts if not v.passed]
    return "; ".join(failed) or None


# The key in a case's "expect" -> the check that tests it.
CHECKS: dict[str, Callable[[Reply, Any], str | None]] = {
    "covered": covered,
    "cites": cites,
    "mentions": mentions,
    "mentions_report": mentions_report,
    "uses_report": uses_report,
    "judge": judge,
}


def grade(reply: Reply, expect: dict[str, Any]) -> list[str]:
    """Runs every check the case asks for. An empty list means the case passed."""
    problems = [CHECKS[name](reply, value) for name, value in expect.items()]
    return [problem for problem in problems if problem]
