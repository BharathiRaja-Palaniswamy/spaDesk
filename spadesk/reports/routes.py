"""HTTP routes for the manage page and its reports.

  GET /manage               the manage page (static/dashboard/index.html)
  GET /api/report/options   what the filter row can offer (date range, dropdown values)
  GET /api/report           the full report for the chosen filters
  POST /api/report/chat     one assistant question about the business -> a structured reply.
                            The body says whether to answer with RAG or the whole document.
                            The model may call the report tool (tools.py) to get figures.
                            SPADESK_RAG_ENGINE picks the code that answers: the loop
                            below, or the LangGraph graph in lc/owner/.

Example:  /api/report?start=2026-09-01&end=2026-09-30&category=Facials

The routes live in an APIRouter, a group of routes that app.py plugs in with
one line. That keeps these routes out of app.py.

All of them are the owner's: the data routes need the "reports" permission
(spadesk/access/), so a request without the owner's sign-in gets 401, and
the page sends a visitor who has not signed in to /login.
"""

import json
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from langsmith import traceable
from pydantic import BaseModel, Field

from ..access.permissions import allowed
from ..access.session import require, role_of
from ..assistant.engine import rag_engine
from ..assistant.model import ask_model
from ..assistant.prompt import checked_sections, profile_prompt
from ..assistant.replies import MAX_SUGGESTIONS, UNAVAILABLE, ChatQuestion, ReplyItem
from ..business.store import get_knowledge
from ..config import get_settings
from ..lc.owner import answer as graph_answer
from ..models import ModelError
from ..rag.errors import ModelUnavailable
from .dataset import Dataset, load
from .report import build_report
from .tools import MAX_TOOL_ROUNDS, TOOL_NAME, report_label, report_tool, run_report

PAGE = Path(__file__).resolve().parent.parent / "static" / "dashboard" / "index.html"

router = APIRouter(prefix="/api/report", dependencies=[Depends(require("reports"))])  # the data routes
page_router = APIRouter()  # the page itself, at /manage
log = logging.getLogger("spadesk.assistant")


@page_router.get("/manage", response_model=None)
def manage(request: Request) -> FileResponse | RedirectResponse:
    """The manage page: business profile and reports, as two tabs. Only for
    the signed-in owner; anyone else is sent to the sign-in page."""
    if not allowed(role_of(request), "reports"):
        return RedirectResponse("/login?next=/manage")
    # The page's CSS and JavaScript are served from /static/dashboard/ by app.py.
    return FileResponse(PAGE)


@lru_cache  # the JSON file is read once, on the first request
def get_dataset() -> Dataset:
    """The bookings, joined to the services in the business document. Loaded once."""
    return load(get_settings().report_data_path)


@router.get("/options")
def options() -> dict[str, Any]:
    """What the filter row can offer: business name, the date range the data
    covers, and the category and channel lists.
    """
    dataset = get_dataset()
    # The business name is a business fact, so it comes from the business document.
    return {"business": get_knowledge().name, "note": dataset.meta["description"], **dataset.filter_options()}


# FastAPI reads these arguments from the query string and converts them: `start`
# arrives as text and is turned into a date, or rejected with a 422 if it is not one.
@router.get("")
def report(
    start: date | None = None,
    end: date | None = None,
    category: str | None = None,
    channel: str | None = None,
) -> dict[str, Any]:
    """The full report for the chosen filters. Every filter is optional.
    Answers 422 when a filter is not valid.
    """
    dataset = get_dataset()
    available = dataset.filter_options()
    # A missing date means "from the first day" or "to the last day" of the data.
    start = start or date.fromisoformat(available["min_date"])
    end = end or date.fromisoformat(available["max_date"])

    if problem := dataset.filter_problem(start, end, category, channel):
        raise HTTPException(status_code=422, detail=problem)

    return build_report(dataset, start, end, category, channel)


# Structured output. The model is not asked for free text: it is given this
# class as a schema and must fill in every field, so the page can rely on them
# (show the answer, lay a list out as a list, say where it came from, style a
# "not covered" reply differently). The model returns data; the page decides
# how it looks. The docstring and each field's `description` are sent to the
# model as part of the schema; they are how it learns what each field is for.
class ChatReply(BaseModel):
    """The assistant's reply to one question from the spa's owner."""

    answer: str = Field(description="The reply to show the owner, in plain sentences without markdown. Two to "
                                    "four sentences normally. When `items` is used, one short sentence that "
                                    "introduces the list and does not repeat its entries. For an answer from "
                                    "report figures, one sentence naming the period and the headline result.")
    items: list[ReplyItem] = Field(description="Use when the answer is a list of three or more things, such as "
                                               "services with their prices: one entry per thing, in the "
                                               "profile's order. Always use it for report figures: one entry "
                                               "per figure, the most important first, never written out as a "
                                               "paragraph. Empty for any other answer.")
    insights: list[str] = Field(description="Only for answers from report figures: two or three short "
                                            "observations about what the figures mean for the owner, each one "
                                            "sentence, each resting on figures in the report. Say what stands "
                                            "out, what explains a change, or what deserves attention. Do not "
                                            "repeat a figure without adding a point. Empty for any other answer.")
    covered: bool = Field(description="True if the business profile or the report figures contain the answer. "
                                      "False if they do not, or if the question is not about the business.")
    sections: list[str] = Field(description="The profile sections the answer was taken from, by their exact "
                                            "heading, most specific first. Empty when covered is false, when "
                                            "the answer came from report figures, or no section was needed.")
    suggestions: list[str] = Field(description="Up to three short questions the owner might naturally ask next. "
                                               "Each must be answerable from the profile or the reports, be written "
                                               "as the owner would ask it, and not repeat the question just "
                                               "asked.")


class ChatAnswer(ChatReply):
    """What the chat route sends to the browser: the model's reply, plus how
    it was produced. Kept apart from ChatReply so the model is not asked to
    fill these in; the server knows them.
    """

    rag: bool  # True when the answer was made from retrieved chunks
    chunks: int  # how many chunks were retrieved; 0 without RAG
    reports: list[str]  # the reports the model asked for, e.g. "2026-09-01 to 2026-09-30"


# The owner assistant's own instructions. The rules for using the profile, the
# current time and the profile itself are added after them (assistant/prompt.py).
SYSTEM = """You are the assistant for the owner of a spa. You answer the owner's \
questions about their own business.

For questions about how the business is doing (revenue, bookings, \
cancellations, channels, customers, feedback), call the get_report tool and \
answer from the figures it returns. Never estimate or invent a figure. Work \
out the dates to ask for from the current time; "last month" means the last \
full calendar month. Say which period your figures cover. If the tool \
reports that there is no data for a period, tell the owner what the data covers."""


def _loop_reply(question: str, rag: bool, dataset: Dataset) -> tuple[ChatReply | None, list[str], int, list[str]]:
    """The hand-built engine: the model call in a plain loop, run until the
    model answers instead of asking for a tool.

    Returns the reply (None when there is none), the headings it may cite,
    how many chunks it was given, and the reports it asked for.
    """
    prompt = profile_prompt(SYSTEM, question, rag)
    # This loop calls the Anthropic library directly, so the tool goes in that library's format.
    tools = [report_tool(dataset).for_anthropic_sdk()]
    # The conversation with the model. It starts as the one question and grows
    # by two messages each time the model asks for a tool: its request, then
    # the result. The model has no memory between calls, so all of it is sent
    # again every time.
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    reports: list[str] = []
    response = ask_model(prompt.system, messages, tools, ChatReply)
    rounds = 0
    # stop_reason says why the model stopped: "tool_use" means it is
    # waiting for a tool result rather than giving its answer.
    while response.stop_reason == "tool_use" and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                reports.append(report_label(block.input))
                # A result is tied to its request by id, and sent as text.
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(run_report(dataset, block.input))})
        messages.append({"role": "user", "content": results})
        response = ask_model(prompt.system, messages, tools, ChatReply)
    return response.parsed_output, prompt.headings, prompt.chunks, reports


@router.post("/chat")
@traceable  # the whole request is the top step of the trace
def chat(body: ChatQuestion) -> ChatAnswer:
    """Answer one question about the business, using the business document."""
    dataset = get_dataset()
    try:
        if rag_engine() == "langchain":
            result = graph_answer(SYSTEM, body.question, body.rag, ChatReply, dataset)
            reply, headings, chunks, reports = result.reply, result.headings, result.chunks, result.reports
        else:
            reply, headings, chunks, reports = _loop_reply(body.question, body.rag, dataset)
    except (ModelError, ModelUnavailable, anthropic.AnthropicError, TypeError) as error:
        # ModelError: the gateway's error, whichever provider failed (LangChain engine).
        # ModelUnavailable: a search model could not be loaded or reached (rag/errors.py).
        # The others: the hand-built engine calls the Anthropic library itself,
        # which raises TypeError when no API key is set.
        log.warning("assistant unavailable: %s", error)  # the reason, for whoever reads the server log
        raise HTTPException(status_code=503, detail=UNAVAILABLE)

    if reply is None:
        # No structured reply came back, for example because the model declined
        # to answer, or was still asking for tools when the rounds ran out.
        raise HTTPException(status_code=503, detail="The assistant could not answer that.")

    reply.sections = checked_sections(reply.sections, headings)
    # The schema cannot limit how many suggestions come back, so the limit is applied here.
    reply.suggestions = reply.suggestions[:MAX_SUGGESTIONS]
    return ChatAnswer(**reply.model_dump(), rag=body.rag, chunks=chunks, reports=reports)
