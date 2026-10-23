"""The booking agent's steps, and the choices between them.

Each step is a plain function: it takes the state and returns what it
changed. The choices (`first_step`, `after_identify`, ...) are not steps:
they read the state and name the step to go to next.

  identify  name and phone, before anything else (identify.py)
  agent     one model call: it replies, or asks for tools
  tools     runs the tools it asked for; a booking or cancellation is held
            back and the customer is asked to confirm it (confirm.py)
  decide    the customer's answer to that question: yes runs it, anything
            else goes back to the agent with their reply
  give_up   too many tool rounds in one turn: hand off to the staff
"""

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END

from ..models import run_requests, structured
from .confirm import is_yes, problem_with
from .handoffs import record_handoff
from .identify import ask_for, read_details, valid_phone
from .prompt import AgentReply, instructions
from .state import AgentState
from .tools import NEEDS_YES, agent_tools

PROFILE = "booking-agent"  # the job, in spadesk/models/profiles.py
MAX_TOOL_ROUNDS = 5  # per customer message: enough to search, find times and book, with a retry


def _latest(state: AgentState) -> HumanMessage:
    """The customer's message this turn: the last message the route added."""
    return state["messages"][-1]


# ---- where each turn starts ----------------------------------------------

def first_step(state: AgentState) -> str:
    """Details first; then a waiting yes-or-no; otherwise the agent."""
    if not (state.get("name") and state.get("phone")):
        return "identify"
    return "decide" if state.get("pending") else "agent"


# ---- identify ----------------------------------------------------------------

def identify(state: AgentState) -> dict:
    """Read the name and phone number from the customer's message, and ask
    for whatever is still missing. Once both are known, an earlier question
    goes on to the agent; otherwise the customer is thanked."""
    details = read_details(_latest(state).text)
    caller = state.get("caller", "")
    name = state.get("name") or details.name.strip()
    # A number they give wins; on a call, the number calling is used until they do.
    given = details.phone.strip() if valid_phone(details.phone) else ""
    phone = given or state.get("phone") or caller
    held = state.get("held", False) or details.asks
    known = {"name": name, "phone": phone}
    if name and phone and held:
        return {**known, "held": False}  # on to the agent, which answers what they asked
    say = f"Thank you, {name}! How can I help you today?" if name and phone else ask_for(name, phone, caller)
    return {**known, "held": held, "say": say, "messages": [AIMessage(say)]}


def after_identify(state: AgentState) -> str:
    """To the agent when the details are complete and a question is waiting; else the turn ends."""
    return END if state.get("say") else "agent"


# ---- agent and tools ------------------------------------------------------------

def agent(state: AgentState) -> dict:
    """One model call with the whole conversation: it ends in a reply or in tool requests."""
    offered = agent_tools(state["name"], state["phone"])
    rules = instructions(state["name"], state["phone"], state.get("channel", "chat"))
    messages = [SystemMessage(rules), *state["messages"]]
    answer = structured(PROFILE, messages, AgentReply, offered)
    return {"messages": [answer.message], "requests": answer.tool_calls, "reply": answer.reply}


def after_agent(state: AgentState) -> str:
    """The end when the model replied; the tools when it asked for some and has rounds left."""
    if state.get("reply") is not None or not state.get("requests"):
        return END
    return "tools" if state.get("rounds", 0) < MAX_TOOL_ROUNDS else "give_up"


def tools(state: AgentState) -> dict:
    """Run what the model asked for, except a booking or cancellation: that one
    is checked, then held until the customer says yes. A request that would
    fail goes straight back to the model as an error instead."""
    offered = agent_tools(state["name"], state["phone"])
    now = [r for r in state["requests"] if r["name"] not in NEEDS_YES]
    later = [r for r in state["requests"] if r["name"] in NEEDS_YES]
    results = run_requests(now, offered)
    update: dict = {"rounds": state.get("rounds", 0) + 1}
    if later:
        held, *extra = later
        results += [ToolMessage(json.dumps({"error": "One change at a time. Ask again after this one."}),
                                tool_call_id=r["id"]) for r in extra]
        question, problem = problem_with(held, state["name"], state["phone"], state.get("channel") == "voice")
        if problem:
            results.append(ToolMessage(json.dumps({"error": problem}), tool_call_id=held["id"]))
        else:
            # The question is shown to the customer but not added to the conversation
            # yet: the request's result must come straight after the request.
            update |= {"pending": {**held, "question": question}, "say": question, "confirming": True}
    return {**update, "messages": results}


def after_tools(state: AgentState) -> str:
    """The turn ends while waiting for the customer's yes; otherwise back to the model."""
    return END if state.get("confirming") else "agent"


# ---- the customer's yes or no ------------------------------------------------------

def decide(state: AgentState) -> dict:
    """Run the held request if the customer said yes; either way, tell the
    model what happened, then pass on the customer's message."""
    pending, said = state["pending"], _latest(state)
    # This turn's messages (the time note and the customer's message) came in
    # after the request; its result must come straight after the request, so
    # they are taken out and put back after it.
    turn = []
    for message in reversed(state["messages"]):
        if not isinstance(message, HumanMessage):
            break
        turn.insert(0, message)
    if is_yes(said.text):
        tool = next(t for t in agent_tools(state["name"], state["phone"]) if t.name == pending["name"])
        outcome = {"customer_said_yes_to": pending["question"], "result": tool.run(pending["args"])}
    else:
        outcome = {"not_done": f"The customer was asked {pending['question']!r} and did not say yes, so "
                               "nothing was changed. Their reply follows."}
    return {"pending": None, "messages": [
        *[RemoveMessage(id=message.id) for message in turn],
        ToolMessage(json.dumps(outcome), tool_call_id=pending["id"]),
        *[HumanMessage(message.content) for message in turn]]}


# ---- too many rounds -----------------------------------------------------------------

def give_up(state: AgentState) -> dict:
    """Stop, hand the request to the staff, and tell the customer."""
    record_handoff(state["name"], state["phone"], "The assistant could not finish the customer's request.")
    say = f"I'm sorry, I couldn't finish that here. I've asked our team to call you on {state['phone']}."
    # Every request gets a result, so the conversation can carry on next turn.
    stopped = [ToolMessage(json.dumps({"error": "Stopped: too many steps."}), tool_call_id=r["id"])
               for r in state["requests"]]
    return {"say": say, "messages": [*stopped, AIMessage(say)]}
