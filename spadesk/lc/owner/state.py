"""LangGraph step 1: the state, what the graph carries from node to node.

A LangGraph graph is a set of nodes (functions) joined by edges. Every node
receives the current state and returns only the fields it changed; LangGraph
merges those changes into the state before the next node runs.

How a change is merged is set per field. A plain field is replaced. A field
annotated with a reducer is combined instead:
  messages  add_messages appends the node's new messages to the conversation
  reports   operator.add joins the node's labels onto the list so far
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class OwnerState(TypedDict):
    """Everything one owner question needs, from the question to the reply."""

    question: str  # the owner's question
    rag: bool  # True: answer from retrieved chunks; False: from the whole document
    # The conversation sent to the model: the system prompt and the question,
    # then each tool request and its result. Every model call gets all of it.
    messages: Annotated[list[AnyMessage], add_messages]
    headings: list[str]  # the headings the model was shown: all a reply may cite
    chunks: int  # how many chunks were retrieved; 0 without RAG
    rounds: int  # how many times the tools node has run
    requests: list[dict]  # the model's latest tool requests: {"name", "args", "id"}
    reply: BaseModel | None  # the answer in the reply's shape, once the model gives it
    reports: Annotated[list[str], operator.add]  # the reports asked for, e.g. "2026-09-01 to 2026-09-30"
