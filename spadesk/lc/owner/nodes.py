"""LangGraph step 2: the nodes, and the choice of where to go after the model.

Each node is a plain function: it takes the state and returns the fields it
changed. They do the same jobs as the hand-built loop in reports/routes.py,
one job per node, so a trace shows each step on its own:

  prepare    retrieve (RAG on) and build the prompt, once per question
  agent      call the model: it answers, or asks for the report tool
  tools      run the report tool for each request and send the figures back
  next_step  not a node: after `agent`, decides between `tools` and the end
"""

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END

from ...reports.dataset import Dataset
from ...reports.tools import MAX_TOOL_ROUNDS, TOOL_NAME, report_label, run_report
from ..prompt import PROMPT, prompt_values
from ..retriever import get_retriever
from .state import OwnerState


def prepare(state: OwnerState, instructions: str) -> dict:
    """The system prompt and the question, as the first two messages.

    The prompt is the customer chain's (lc/prompt.py) with the owner's
    instructions: the rules, the current time, the headings, and the whole
    document or only the retrieved chunks.
    """
    documents = get_retriever().invoke(state["question"]) if state["rag"] else None
    values = prompt_values(instructions, state["question"], documents)
    return {"messages": PROMPT.invoke(values).to_messages(),
            "headings": values["sections"].split("\n"),
            "chunks": len(documents) if documents else 0,
            "rounds": 0}


def agent(state: OwnerState, model: Runnable) -> dict:
    """One model call with the whole conversation so far. The reply is either
    the structured answer, as JSON text, or a request to run a tool."""
    return {"messages": [model.invoke(state["messages"])]}


def tools(state: OwnerState, dataset: Dataset) -> dict:
    """Run every tool request in the model's last message.

    Each result goes back as a ToolMessage tied to its request by id, which
    is how the model knows which figures answer which request.
    """
    request = state["messages"][-1]
    results, labels = [], []
    for call in request.tool_calls:
        if call["name"] == TOOL_NAME:
            labels.append(report_label(call["args"]))
            content = run_report(dataset, call["args"])
        else:
            # Every request needs a result, or the next model call is rejected.
            content = {"error": f"There is no tool called {call['name']}."}
        results.append(ToolMessage(json.dumps(content), tool_call_id=call["id"]))
    return {"messages": results, "reports": labels, "rounds": state["rounds"] + 1}


def next_step(state: OwnerState) -> str:
    """Where to go after the model: to `tools` when it asked for one and has
    rounds left, otherwise to the end of the graph."""
    last = state["messages"][-1]
    asked = isinstance(last, AIMessage) and bool(last.tool_calls)
    return "tools" if asked and state["rounds"] < MAX_TOOL_ROUNDS else END
