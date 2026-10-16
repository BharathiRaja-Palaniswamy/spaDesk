"""The customer assistant: the chat on the customer site.

  POST /api/chat   one customer question -> a structured reply

It answers from the business profile only. Unlike the owner assistant it is
given no tools, so it has no way to reach bookings or revenue figures: what
a customer can see is limited by what the model is given, not only by what
it is told.
"""

import anthropic
from fastapi import APIRouter, HTTPException
from langsmith import traceable
from pydantic import BaseModel, Field

from .model import ask_model
from .prompt import checked_sections, profile_prompt
from .replies import MAX_SUGGESTIONS, ChatQuestion, ReplyItem

router = APIRouter(prefix="/api")

# Who the assistant is talking to, and what it must not do. The rules for
# using the profile are added after these (prompt.py).
SYSTEM = """You are the front-desk assistant on a spa's website. You answer \
questions from customers about the spa: its services, prices, opening hours, \
policies and how to visit. Speak for the spa as "we", and keep a warm, calm tone.

You cannot book, change or cancel appointments. When a customer wants to book, \
tell them to use the Request a booking button on this page, and that the spa \
will contact them to confirm a time.

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


@router.post("/chat")
@traceable  # the whole request is the top step of the trace
def chat(body: ChatQuestion) -> CustomerAnswer:
    """Answer one customer question from the business profile."""
    prompt = profile_prompt(SYSTEM, body.question, body.rag)
    messages = [{"role": "user", "content": body.question}]
    try:
        # No tools: one call, and the reply is the answer.
        response = ask_model(prompt.system, messages, [], CustomerReply)
    except (anthropic.AnthropicError, TypeError):
        # TypeError is what the SDK raises when no API key is set.
        raise HTTPException(status_code=503, detail="The assistant is unavailable right now.")

    reply = response.parsed_output
    if reply is None:
        # No structured reply came back, for example because the model declined.
        raise HTTPException(status_code=503, detail="The assistant could not answer that.")

    reply.sections = checked_sections(reply.sections, prompt.headings)
    reply.suggestions = reply.suggestions[:MAX_SUGGESTIONS]
    return CustomerAnswer(**reply.model_dump(), rag=body.rag, chunks=prompt.chunks)
