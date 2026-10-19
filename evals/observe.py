"""Makes each eval question easy to find in LangSmith.

The chat routes are already traced (spadesk/tracing.py), so an eval run sends
traces whenever LANGSMITH_TRACING=true. Without this file they would land in
the app's own project, mixed with real chats and with nothing to say which
case they were. Here each question is asked inside a parent run:

    eval: pregnancy-body-wrap (rag)        <- this file: case, mode, pass/fail
      └─ chat                              <- the route's own trace
           ├─ retrieve                        chunks and scores
           └─ ask_model                       prompt, reply, tokens

and every run goes to its own project, LANGSMITH_EVAL_PROJECT (default
"spadesk-evals"), tagged "eval", the mode and the chat. In LangSmith a failed
case can then be filtered by tag or metadata and opened to see why it failed.
"""

import os
import time
from typing import Any

from fastapi.testclient import TestClient
from langsmith import get_current_run_tree, traceable, tracing_context

from spadesk.config import get_settings

from .checks import grade

EVAL_PROJECT = os.getenv("LANGSMITH_EVAL_PROJECT", "spadesk-evals")

# Which route answers each kind of chat ("chat" in a case).
ROUTES = {"customer": "/api/chat", "owner": "/api/report/chat"}


@traceable
def _ask_and_grade(client: TestClient, case: dict[str, Any], mode: str) -> dict[str, Any]:
    """Asks one case in one mode, grades the reply and times the request.

    Its inputs and return value are what LangSmith shows for the parent run.
    """
    # Latency is measured around the whole request: retrieval, every model
    # call and every tool call, as a visitor would wait for it.
    started = time.perf_counter()
    response = client.post(ROUTES[case["chat"]], json={"question": case["question"], "rag": mode == "rag"})
    seconds = time.perf_counter() - started
    response.raise_for_status()
    # The judge needs the question as well as the answer, so it rides along with the reply.
    reply = {**response.json(), "question": case["question"]}
    problems = grade(reply, case["expect"])
    # Added to the run's metadata, so failed cases can be filtered in LangSmith.
    if run := get_current_run_tree():
        run.metadata.update({"passed": not problems, "seconds": round(seconds, 2)})
    return {"answer": reply["answer"], "passed": not problems, "problems": problems, "seconds": seconds}


def ask_case(client: TestClient, case: dict[str, Any], mode: str, label: str) -> dict[str, Any]:
    """`_ask_and_grade`, recorded in the eval project under the case's name.

    `label` names the run (e.g. "rerank"), so two runs can be told apart and
    compared in LangSmith: filter or group by the `run:<label>` tag, or by the
    `run`, `engine` and `reranker` metadata. `engine` is the pipeline that
    answered both chats (SPADESK_RAG_ENGINE).
    """
    settings = get_settings()
    with tracing_context(project_name=EVAL_PROJECT, tags=["eval", mode, case["chat"], f"run:{label}"],
                         metadata={"case": case["id"], "mode": mode, "run": label, "engine": settings.rag_engine,
                                   "reranker": settings.reranker, "top_k": settings.retrieval_top_k}):
        return _ask_and_grade(client, case, mode, langsmith_extra={"name": f"eval: {case['id']} ({mode})"})
