"""What the booking agent remembers, within one turn and from turn to turn.

The graph runs once per customer message. Between messages, LangGraph keeps
the state in a checkpointer, one per conversation (its thread id), so the
next message continues where the last one stopped. That is the agent's
memory: the conversation, who the customer is, and a booking waiting for
their yes.

Fields set from the route at the start of every turn (`rounds`, `say`,
`reply`) are per turn; the others last for the whole conversation.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(TypedDict, total=False):
    """One customer conversation."""

    # The conversation as the model sees it: the customer's messages, the
    # agent's replies, and each tool request with its result. add_messages
    # appends new messages (and removes one when given a RemoveMessage).
    messages: Annotated[list[AnyMessage], add_messages]
    name: str  # the customer's name, once given
    phone: str  # their phone number, once given (not verified)
    held: bool  # they asked something before giving their details: answer it once they have
    pending: dict | None  # a booking or cancellation waiting for the customer's yes: the model's request
    rounds: int  # tool rounds this turn
    requests: list[dict]  # the model's latest tool requests: {"name", "args", "id"}
    reply: BaseModel | None  # the model's reply this turn, once it gives one
    say: str  # a reply written by the graph itself this turn (asking for details or for a yes)
    confirming: bool  # this turn ends by asking the customer to confirm
