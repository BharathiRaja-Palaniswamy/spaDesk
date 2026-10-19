"""LangGraph step 3: the model the graph calls, and the graph itself.

    START -> prepare -> agent --(asked for a tool?)--> tools
                          ^                              |
                          +------------------------------+
                        agent --(answered, or out of rounds)--> END

The loop is what a chain cannot express: how many times the model asks for
the report tool is decided by the model while the question runs (none for a
profile question, one for "last month", two for a comparison).
"""

from anthropic import transform_schema
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from ...reports.dataset import Dataset
from ...reports.tools import report_tool
from ..chat import get_chat_model
from .nodes import agent, next_step, prepare, tools
from .state import OwnerState


def owner_model(dataset: Dataset, schema: type[BaseModel]) -> Runnable:
    """The chat model (SPADESK_CHAT_PROVIDER), offered the report tool and
    asked to answer in the shape of `schema`.

    One call can then end either way, as in the hand-built loop: a tool
    request, or the structured reply. The tool is the same description the
    hand-built loop sends (reports/tools.py). The reply's shape is sent as
    Claude's `output_config` format, the setting `messages.parse` uses; it is
    specific to Anthropic, so another chat provider would need its own here.
    """
    return get_chat_model().bind_tools([report_tool(dataset)]).bind(
        output_config={"format": {"type": "json_schema", "schema": transform_schema(schema)}})


def build_graph(instructions: str, model: Runnable, dataset: Dataset) -> CompiledStateGraph:
    """The owner's graph, ready to run. The nodes are given what they need
    (the instructions, the model, the bookings) here, so the state carries
    only the question's own data."""
    graph = StateGraph(OwnerState)
    graph.add_node("prepare", lambda state: prepare(state, instructions))
    graph.add_node("agent", lambda state: agent(state, model))
    graph.add_node("tools", lambda state: tools(state, dataset))
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", next_step, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(name="owner_graph")
