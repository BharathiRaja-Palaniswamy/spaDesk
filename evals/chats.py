"""Asks one rag_qa or owner_chat case through the app's own route (`observe`), and
grades the reply against its label (`grade`).

  rag_qa      the customer chat (/api/chat): routing is not checked (it has no
              tools); the answer is
  owner_chat  the owner's chat (/api/report/chat), signed in as the owner:
              routing is whether it read the bookings report (get_report), the
              only tool it has; then the answer

Both are asked with RAG on (the retrieved parts of the guide, as customers
get it). The route is called in this process (FastAPI's TestClient), so
nothing needs to be running.
"""

from typing import Any

from fastapi.testclient import TestClient

from spadesk.access.session import COOKIE, new_session

from .checks import grade_answer
from .golden import Case

ROUTES = {"rag_qa": "/api/chat", "owner_chat": "/api/report/chat"}
REPORT = "get_report"  # the owner chat's tool


def client() -> TestClient:
    """A client of the app, signed in as the owner (the customer chat does not mind)."""
    from spadesk.app import app  # imported here: the app reads its settings when first imported
    signed_in = TestClient(app)
    signed_in.cookies.set(COOKIE, new_session()[0])
    return signed_in


def routing(want: dict, reports: list[str]) -> list[str]:
    """The owner chat used the report when it had to, and only then."""
    used, problems = [REPORT] if reports else [], []
    if (tool := want.get("tool")) and tool not in used:
        problems.append(f"did not use {tool} (used {used or 'no tools'})")
    return problems + [f"used {name}" for name in want.get("not_tools", []) if name in used]


def observe(case: Case, chat: TestClient) -> dict[str, Any]:
    """Asks the case's question and records the reply, as plain data (the run's
    outputs in LangSmith)."""
    question = case["turns"][0]
    response = chat.post(ROUTES[case["target"]], json={"question": question, "rag": True})
    response.raise_for_status()
    body = response.json()
    items = [f"{item['name']} {item['detail']}" for item in body.get("items", [])]
    text = " ".join([body["answer"], *items, *body.get("insights", [])])
    return {"conversation": [f"Customer: {question}", f"Assistant: {text}"], "reports": body.get("reports", []),
            "reply": {"question": question, "text": text, "sections": body.get("sections", []),
                      "covered": body.get("covered")}}


def grade(case: Case, seen: dict[str, Any]) -> dict[str, list[str]]:
    """Each layer's problems, from what `observe` recorded, against the case's label."""
    want, graded = case["expected"], {}
    if "routing" in want:
        graded["routing"] = routing(want["routing"], seen["reports"])
    if "answer" in want:
        graded["answer"] = grade_answer(seen["reply"], want["answer"])
    return graded
