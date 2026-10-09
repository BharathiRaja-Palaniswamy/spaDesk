"""HTTP routes for the manage page and its reports.

  GET /manage               the manage page (static/dashboard/index.html)
  GET /api/report/options   what the filter row can offer (date range, dropdown values)
  GET /api/report           the full report for the chosen filters
  POST /api/report/chat     one assistant question about the business -> a structured reply

Example:  /api/report?start=2026-09-01&end=2026-09-30&category=Facials

The routes live in an APIRouter, a group of routes that app.py plugs in with
one line. That keeps these routes out of app.py.
"""

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..business.store import get_knowledge
from ..config import get_settings
from .dataset import Dataset, load
from .report import build_report

PAGE = Path(__file__).resolve().parent.parent / "static" / "dashboard" / "index.html"

router = APIRouter(prefix="/api/report")  # the data routes
page_router = APIRouter()  # the page itself, at /manage


@page_router.get("/manage")
def manage() -> FileResponse:
    """The manage page: business profile and reports, as two tabs."""
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

    if start > end:
        raise HTTPException(status_code=422, detail="The start date must be on or before the end date.")
    if category is not None and category not in available["categories"]:
        raise HTTPException(status_code=422, detail=f"Unknown category: {category}")
    if channel is not None and channel not in available["channels"]:
        raise HTTPException(status_code=422, detail=f"Unknown channel: {channel}")

    return build_report(dataset, start, end, category, channel)


class ChatQuestion(BaseModel):
    """The JSON body of POST /api/report/chat."""

    question: str


# Structured output. The model is not asked for free text: it is given this
# class as a schema and must fill in every field, so the page can rely on them
# (show the answer, lay a list out as a list, say where it came from, style a
# "not covered" reply differently). The model returns data; the page decides
# how it looks. The docstring and each field's `description` are sent to the
# model as part of the schema; they are how it learns what each field is for.
class ReplyItem(BaseModel):
    """One entry of a list in the assistant's reply, for example one service."""

    name: str = Field(description="What the entry is, e.g. 'Swedish Massage' or 'Late cancellations'.")
    detail: str = Field(description="The facts about it, kept short, e.g. '60 / 90 minutes, $115 / $160'.")


class ChatReply(BaseModel):
    """The assistant's reply to one question from the spa's owner."""

    answer: str = Field(description="The reply to show the owner, in plain sentences without markdown. Two to "
                                    "four sentences normally. When `items` is used, one short sentence that "
                                    "introduces the list and does not repeat its entries.")
    items: list[ReplyItem] = Field(description="Use when the answer is a list of three or more things, such as "
                                               "services with their prices: one entry per thing, in the "
                                               "profile's order. Empty for any other answer.")
    covered: bool = Field(description="True if the business profile contains the answer. False if it does not, "
                                      "or if the question is not about the business.")
    sections: list[str] = Field(description="The profile sections the answer was taken from, by their exact "
                                            "heading, most specific first. Empty when covered is false or no "
                                            "section was needed.")


# The standing instructions. The current time and the business document are
# added below them on every request, between tags, so the model can tell
# instructions apart from data.
SYSTEM = """You are the assistant for the owner of a spa. You answer the owner's \
questions about their own business.

The business profile is given below between <profile> tags. It is the only \
source of facts about the business. Quote prices, hours and policies exactly \
as written there. If the profile does not cover the question, say so plainly; \
do not guess or fill in what spas usually do.

The current date and time at the spa are given below between <currentTime> \
tags. Use them to work out dates, days of the week and how much time is left \
when a question depends on them, then answer from the profile as usual.

The profile's headings and sub-headings are listed between <sections> tags. \
When you name the sections an answer came from, use those names exactly, and \
prefer the most specific one: the sub-heading the answer sits under, where \
there is one."""


@router.post("/chat")
def chat(body: ChatQuestion) -> ChatReply:
    """Answer one question about the business, using the business document."""
    settings = get_settings()
    # The whole business document, as text. It is small, so it is sent in full.
    profile = settings.data_path.read_text(encoding="utf-8")
    # Its headings and sub-headings, e.g. "Policies" and "Cancellation Policy".
    # The reply names its sources from this list. (dict.fromkeys removes
    # repeats while keeping the document's order.)
    outline = get_knowledge().outline
    headings = list(dict.fromkeys(name for pair in outline for name in pair if name))
    # The model has no clock, so it is told the current date and time in the
    # spa's own time zone, e.g. "Friday, 09 October 2026, 3:42 PM (PDT)".
    current_time = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %d %B %Y, %-I:%M %p (%Z)")

    try:
        # `parse` works like `create`, with one addition: `output_format` is a
        # pydantic class, the model must reply with JSON matching it, and the
        # SDK turns that JSON into an instance of the class (`parsed_output`).
        response = anthropic.Anthropic().messages.parse(
            model=settings.model,
            max_tokens=16000,
            system=(f"{SYSTEM}\n\n<currentTime>{current_time}</currentTime>\n\n"
                    f"<sections>\n{chr(10).join(headings)}\n</sections>\n\n<profile>\n{profile}\n</profile>"),
            messages=[{"role": "user", "content": body.question}],
            output_format=ChatReply,
        )
    except (anthropic.AnthropicError, TypeError):
        # TypeError is what the SDK raises when no API key is set.
        raise HTTPException(status_code=503, detail="The assistant is unavailable right now.")

    reply = response.parsed_output
    if reply is None:
        # No structured reply came back, for example because the model declined to answer.
        raise HTTPException(status_code=503, detail="The assistant could not answer that.")

    # Structured output guarantees the shape of the reply, not that its content
    # is right. Section names are checked against the real headings, so the
    # page never shows a source that does not exist.
    cited = [name for name in reply.sections if name in headings]
    # When a sub-heading is cited, its parent heading adds nothing: "Cancellation
    # Policy" already says where the answer is, so "Policies" is dropped.
    parents = {heading for heading, subheading in outline if subheading in cited}
    reply.sections = [name for name in cited if name not in parents]
    return reply
