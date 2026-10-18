"""LangChain step 6: answer a question with a chain.

    question -> retriever (RAG on) -> prompt template -> chat model -> structured reply

The chat model is pluggable, like every other provider: SPADESK_CHAT_PROVIDER
picks it, and "anthropic" (ChatAnthropic, the default) is the one built in.
Any LangChain chat model can be added to PROVIDERS; the rest of the chain
does not change, because they all share one interface (BaseChatModel).

`with_structured_output(Schema)` makes the model reply in the shape of a
pydantic class, as `messages.parse(output_format=...)` does in the hand-built
pipeline (assistant/model.py). method="json_schema" asks Claude for that shape
directly, the same mechanism the hand-built call uses.

Each step is traced in LangSmith on its own (retriever, prompt, model) when
tracing is on: LangChain does that without any @traceable.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from ..config import get_settings
from .prompt import PROMPT, prompt_values
from .retriever import get_retriever

# Setting value -> how to make that chat model.
PROVIDERS: dict[str, Callable[[], BaseChatModel]] = {
    "anthropic": lambda: ChatAnthropic(model=get_settings().model, max_tokens=16000),
}


def get_chat_model() -> BaseChatModel:
    """The chat model named by SPADESK_CHAT_PROVIDER."""
    provider = get_settings().chat_provider
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown chat provider: {provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    return PROVIDERS[provider]()


def structured_model(schema: type[BaseModel]) -> Runnable:
    """The chat model, replying as an instance of `schema`."""
    return get_chat_model().with_structured_output(schema, method="json_schema")


@dataclass(frozen=True)
class Answer:
    """A structured reply, and what is needed to check it."""

    reply: BaseModel  # an instance of the schema asked for
    headings: list[str]  # the headings the model was shown: all a reply may cite
    chunks: int  # how many chunks were retrieved; 0 without RAG


def answer(instructions: str, question: str, rag: bool, schema: type[BaseModel]) -> Answer:
    """Answer `question` with the chain, from the retrieved chunks (rag) or the whole document."""
    documents = get_retriever().invoke(question) if rag else None
    values = prompt_values(instructions, question, documents)
    chain = (PROMPT | structured_model(schema)).with_config(run_name="answer")
    return Answer(reply=chain.invoke(values), headings=values["sections"].split("\n"),
                  chunks=len(documents) if documents else 0)
