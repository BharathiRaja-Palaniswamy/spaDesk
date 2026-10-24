"""The customer assistant: the chat on the customer site.

  POST /api/chat   one customer question -> a structured reply

It answers from the business profile only. Unlike the owner assistant it is
given no tools, so it has no way to reach bookings or revenue figures: what
a customer can see is limited by what the model is given, not only by what
it is told.
"""

import logging

import anthropic
from fastapi import APIRouter, HTTPException
from langsmith import traceable
from pydantic import BaseModel, Field

from ..lc.chat import answer as lc_answer
from ..models import ModelError
from ..rag.errors import ModelUnavailable
from .engine import rag_engine
from .model import ask_model
from .prompt import checked_sections, profile_prompt
from .replies import MAX_SUGGESTIONS, ChatQuestion, ReplyItem

router = APIRouter(prefix="/api")
log = logging.getLogger("spadesk.assistant")

# Who the assistant is talking to, and what it must not do. The rules for
# using the profile are added after these (prompt.py).
SYSTEM = """You are the front-desk assistant on a spa's website. You answer \
questions from customers about the spa: its services, prices, opening hours, \
policies and how to visit. Speak for the spa as "we", and keep a warm, calm tone.

You cannot book, change or cancel appointments here. When a customer wants to \
book, tell them to use the Book now button on this page, which shows the free \
times and books one.

Do not give medical advice. For a health question, share what the profile \
says and suggest they tell the spa before their visit.

You know nothing about other customers or how the business is doing. If \
asked, say that you can only help with questions about the spa's services \
and visits."""


# Structured output, as for the owner (see ChatReply in reports/routes.py),
# with each field described for a customer.
class CustomerReply(BaseModel):
    """The assistant's reply to one question from a customer on the spa's website."""

    answer: str = Field(description="The reply to show the customer, in plain, friendly sentences without "
                                    "markdown. Two to four sentences normally. When `items` is used, one short "
                                    "sentence that introduces the list and does not repeat its entries.")
    items: list[ReplyItem] = Field(description="Use when the answer is a list of three or more things, such as "
                                               "services with their prices: one entry per thing, in the "
                                               "profile's order. Empty for any other answer.")
    covered: bool = Field(description="True if the business profile contains the answer. False if it does not, "
                                      "or if the question is not about the spa.")
    sections: list[str] = Field(description="The profile sections the answer was taken from, by their exact "
                                            "heading, most specific first. Empty when covered is false or no "
                                            "section was needed.")
    suggestions: list[str] = Field(description="Up to three short questions the customer might naturally ask "
                                               "next. Each must be answerable from the profile, be written as a "
                                               "customer would ask it, and not repeat the question just asked.")


class CustomerAnswer(CustomerReply):
    """What the route sends to the browser: the reply, plus how it was produced."""

    rag: bool  # True when the answer was made from retrieved chunks
    chunks: int  # how many chunks were retrieved; 0 without RAG


def _reply(question: str, rag: bool) -> tuple[CustomerReply | None, list[str], int]:
    """The model's reply, the headings it may cite, and how many chunks it was given.

    SPADESK_RAG_ENGINE picks the pipeline (engine.py). Both build the same
    prompt and ask for the same reply; they differ only in the code that does it:
      handbuilt  rag/ for retrieval, assistant/model.py for the call
      langchain  lc/: a LangChain retriever, prompt template and chat model
    """
    if rag_engine() == "langchain":
        result = lc_answer(SYSTEM, question, rag, CustomerReply)
        return result.reply, result.headings, result.chunks
    prompt = profile_prompt(SYSTEM, question, rag)
    # No tools: one call, and the reply is the answer.
    response = ask_model(prompt.system, [{"role": "user", "content": question}], [], CustomerReply)
    return response.parsed_output, prompt.headings, prompt.chunks


@router.post("/chat")
@traceable  # the whole request is the top step of the trace
def chat(body: ChatQuestion) -> CustomerAnswer:
    """Answer one customer question from the business profile."""
    try:
        reply, headings, chunks = _reply(body.question, body.rag)
    except (ModelError, ModelUnavailable, anthropic.AnthropicError, TypeError) as error:
        # ModelError: the gateway's error, whichever provider failed (LangChain engine).
        # ModelUnavailable: a search model could not be loaded or reached (rag/errors.py).
        # The others: the hand-built engine calls the Anthropic library itself,
        # which raises TypeError when no API key is set.
        log.warning("assistant unavailable: %s", error)  # the reason, for whoever reads the server log
        raise HTTPException(status_code=503, detail="The assistant is unavailable right now.")

    if reply is None:
        # No structured reply came back, for example because the model declined.
        raise HTTPException(status_code=503, detail="The assistant could not answer that.")

    reply.sections = checked_sections(reply.sections, headings)
    reply.suggestions = reply.suggestions[:MAX_SUGGESTIONS]
    return CustomerAnswer(**reply.model_dump(), rag=body.rag, chunks=chunks)
