"""The owner chat as a LangGraph graph, next to the hand-built loop in reports/routes.py.

SPADESK_RAG_ENGINE=langchain makes POST /api/report/chat answer through it.
It does the same job the same way: the same prompt, the same report tool,
the same reply shape, at most MAX_TOOL_ROUNDS rounds of tools. Only the code
differs:

  state.py   what the graph carries from step to step
  nodes.py   the steps: prepare, agent, tools, and the choice after agent
  graph.py   how the steps are joined; the model is reached through spadesk/models/

Each node is a step of its own in a LangSmith trace when tracing is on.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from ...reports.dataset import Dataset
from .graph import build_graph


@dataclass(frozen=True)
class OwnerAnswer:
    """The graph's reply, and what the route needs to check and label it."""

    reply: BaseModel | None  # an instance of the schema, or None when there was no usable reply
    headings: list[str]  # the headings the model was shown: all a reply may cite
    chunks: int  # how many chunks were retrieved; 0 without RAG
    reports: list[str]  # the reports the model asked for


def answer(instructions: str, question: str, rag: bool, schema: type[BaseModel],
           dataset: Dataset) -> OwnerAnswer:
    """Answer one owner question by running the graph."""
    state = build_graph(instructions, schema, dataset).invoke(
        {"question": question, "rag": rag, "messages": [], "reports": []})
    # No reply when the graph stopped while the model was still asking for
    # tools (it ran out of rounds); the route reports that.
    return OwnerAnswer(reply=state["reply"], headings=state["headings"], chunks=state["chunks"],
                       reports=state["reports"])
