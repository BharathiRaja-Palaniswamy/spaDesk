"""The booking agent as a LangGraph graph.

                 +--> identify --(details complete and a question waiting)--> agent
                 |        +--(otherwise: ask, or thank them)--> END
    START -------+--> decide (a yes or no was waited for) ---------------> agent
                 |        +--(yes to the details, and a plan to approve)--> END
                 +--> agent <------------------------------------+
                        |                                         |
                        +--(asked for tools)--> tools --(ran them)+
                        |                         +--(a booking to confirm)--> END
                        +--(too many rounds, or a request repeated)--> give_up --> END
                        +--(replied)--> END

It runs once per customer message. The checkpointer keeps each conversation's
state between messages, by thread id: that is the agent's memory. It is kept
in this server's memory, so a restart starts every conversation afresh.
"""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    AgentState,
    after_agent,
    after_decide,
    after_identify,
    after_tools,
    agent,
    decide,
    first_step,
    give_up,
    identify,
    tools,
)


@lru_cache  # one graph, and so one memory, for the whole server
def get_graph() -> CompiledStateGraph:
    """The booking agent, ready to run, with its memory."""
    graph = StateGraph(AgentState)
    for step in (identify, agent, tools, decide, give_up):
        graph.add_node(step.__name__, step)
    graph.add_conditional_edges(START, first_step, ["identify", "decide", "agent"])
    graph.add_conditional_edges("identify", after_identify, ["agent", END])
    graph.add_conditional_edges("agent", after_agent, ["tools", "give_up", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", "give_up", END])
    graph.add_conditional_edges("decide", after_decide, ["agent", END])
    graph.add_edge("give_up", END)
    return graph.compile(checkpointer=InMemorySaver(), name="booking_agent")
