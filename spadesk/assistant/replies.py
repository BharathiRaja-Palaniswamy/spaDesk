"""What both assistants share about replies: the question in, one list entry
out, and how many follow-up questions to show.

Each assistant defines its own reply class (its fields are described for its
own reader, owner or customer), and both use ReplyItem for lists.
"""

from pydantic import BaseModel, Field

# How many follow-up questions a chat shows under an answer. The schema cannot
# limit a list's length, so each route cuts the list to this many.
MAX_SUGGESTIONS = 3


class ChatQuestion(BaseModel):
    """The JSON body the chat sends: the question and how to answer it."""

    question: str
    # How the model is given the business document. False: all of it. True:
    # only the chunks retrieved for this question (RAG). Set by the chat's toggle.
    rag: bool = False


# Part of the structured output (see ChatReply in reports/routes.py). The
# description of each field is sent to the model as an instruction.
class ReplyItem(BaseModel):
    """One entry of a list in the assistant's reply, for example one service."""

    name: str = Field(description="What the entry is, e.g. 'Swedish Massage', 'Late cancellations' or 'Revenue'.")
    detail: str = Field(description="The facts about it, kept short, e.g. '60 / 90 minutes, $115 / $160', or for "
                                    "a figure its value and change: '$19,818, down 2.1% from $20,235'.")
