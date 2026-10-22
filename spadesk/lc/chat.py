"""LangChain step 6: answer a question.

    question -> retriever (RAG on) -> prompt template -> model gateway -> structured reply

The model is whichever one the "chat-default" profile names (spadesk/models/):
Claude unless the settings say otherwise, with its fallback if it fails. The
gateway also gets the reply in the right shape from whichever provider
answers, so this file does not know or care which one it was.

Each step is traced in LangSmith on its own (retriever, prompt, model) when
tracing is on: LangChain does that without any @traceable.
"""

from dataclasses import dataclass

from pydantic import BaseModel

from ..models import structured
from .prompt import PROMPT, prompt_values
from .retriever import get_retriever

PROFILE = "chat-default"  # the job, in spadesk/models/profiles.py


@dataclass(frozen=True)
class Answer:
    """A structured reply, and what is needed to check it."""

    reply: BaseModel | None  # an instance of the schema asked for
    headings: list[str]  # the headings the model was shown: all a reply may cite
    chunks: int  # how many chunks were retrieved; 0 without RAG


def answer(instructions: str, question: str, rag: bool, schema: type[BaseModel]) -> Answer:
    """Answer `question` from the retrieved chunks (rag) or the whole document."""
    documents = get_retriever().invoke(question) if rag else None
    values = prompt_values(instructions, question, documents)
    reply = structured(PROFILE, PROMPT.invoke(values).to_messages(), schema).reply
    return Answer(reply=reply, headings=values["sections"].split("\n"),
                  chunks=len(documents) if documents else 0)
