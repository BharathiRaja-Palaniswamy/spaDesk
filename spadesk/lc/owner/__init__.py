"""The owner chat as a LangGraph graph, next to the hand-built loop in reports/routes.py.

SPADESK_RAG_ENGINE=langchain makes POST /api/report/chat answer through it.
It does the same job the same way: the same prompt, the same report tool,
the same reply shape, at most MAX_TOOL_ROUNDS rounds of tools. Only the code
differs:

  state.py   what the graph carries from step to step
  nodes.py   the steps: prepare, agent, tools, and the choice after agent
  graph.py   the model with its tool, and how the steps are joined

Each node is a step of its own in a LangSmith trace when tracing is on.
"""

from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from ...reports.dataset import Dataset
from .graph import build_graph, owner_model


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
    graph = build_graph(instructions, owner_model(dataset, schema), dataset)
    state = graph.invoke({"question": question, "rag": rag, "messages": [], "reports": []})
    last = state["messages"][-1]
    reply = None
    # The last message is the answer unless the graph stopped while the model
    # was still asking for tools (it ran out of rounds).
    if not last.tool_calls:
        try:
            reply = schema.model_validate_json(last.text)
        except ValidationError:
            pass  # not a reply in the expected shape; the route reports it
    return OwnerAnswer(reply=reply, headings=state["headings"], chunks=state["chunks"],
                       reports=state["reports"])
