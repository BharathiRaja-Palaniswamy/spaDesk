"""LangGraph step 3: the graph itself.

    START -> prepare -> agent --(asked for a tool?)--> tools
                          ^                              |
                          +------------------------------+
                        agent --(answered, or out of rounds)--> END

The loop is what a chain cannot express: how many times the model asks for
the report tool is decided by the model while the question runs (none for a
profile question, one for "last month", two for a comparison).

The model is reached through the gateway (spadesk/models/), by profile, so
the graph works the same with Claude or an open model: the gateway offers the
tools and the reply's shape in whatever way that provider supports.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from ...reports.dataset import Dataset
from ...reports.tools import report_tool
from .nodes import agent, next_step, prepare, tools
from .state import OwnerState

PROFILE = "chat-default"  # the job, in spadesk/models/profiles.py


def build_graph(instructions: str, shape: type[BaseModel], dataset: Dataset) -> CompiledStateGraph:
    """The owner's graph, ready to run. The nodes are given what they need
    (the instructions, the reply's shape, the tools) here, so the state
    carries only the question's own data."""
    offered = [report_tool(dataset)]
    graph = StateGraph(OwnerState)
    graph.add_node("prepare", lambda state: prepare(state, instructions))
    graph.add_node("agent", lambda state: agent(state, PROFILE, shape, offered))
    graph.add_node("tools", lambda state: tools(state, offered))
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", next_step, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(name="owner_graph")
