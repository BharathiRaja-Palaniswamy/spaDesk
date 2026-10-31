"""LangGraph step 2: the nodes, and the choice of where to go after the model.

Each node is a plain function: it takes the state and returns the fields it
changed. They do the same jobs as the hand-built loop in reports/routes.py,
one job per node, so a trace shows each step on its own:

  prepare    retrieve (RAG on) and build the prompt, once per question
  agent      ask the model gateway: the model answers, or asks for the report tool
  tools      run the tools it asked for and send the results back
  next_step  not a node: after `agent`, decides between `tools` and the end
"""

from langgraph.graph import END
from pydantic import BaseModel

from ...models import Tool, run_requests, structured
from ...reports.tools import MAX_TOOL_ROUNDS, TOOL_NAME, report_label
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
            "rounds": 0, "requests": [], "reply": None}


def agent(state: OwnerState, profile: str, shape: type[BaseModel], offered: list[Tool]) -> dict:
    """One model call, through the gateway, with the whole conversation so far.
    It ends in the reply (in `shape`) or in requests for the tools offered;
    the gateway sorts out which, whatever provider answered. Once the tool
    rounds are used up, it must answer with what it has: the owner always gets
    a reply, never "the assistant could not answer"."""
    answer = structured(profile, state["messages"], shape, offered, answer_now=state["rounds"] >= MAX_TOOL_ROUNDS)
    return {"messages": [answer.message], "requests": answer.tool_calls, "reply": answer.reply}


def tools(state: OwnerState, offered: list[Tool]) -> dict:
    """Run every tool the model asked for. Each result goes back tied to its
    request by id, which is how the model knows which result answers which
    request."""
    labels = [report_label(request["args"]) for request in state["requests"] if request["name"] == TOOL_NAME]
    return {"messages": run_requests(state["requests"], offered), "reports": labels,
            "rounds": state["rounds"] + 1}


def next_step(state: OwnerState) -> str:
    """Where to go after the model: to `tools` when it asked for some, has not
    answered yet, and has rounds left; otherwise to the end of the graph."""
    asked = state["reply"] is None and bool(state["requests"])
    return "tools" if asked and state["rounds"] < MAX_TOOL_ROUNDS else END
