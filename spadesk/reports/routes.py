"""HTTP routes for the manage page and its reports.

  GET /manage               the manage page (static/dashboard/index.html)
  GET /api/report/options   what the filter row can offer (date range, dropdown values)
  GET /api/report           the full report for the chosen filters
  POST /api/report/chat     one assistant question about the business -> the model's reply

Example:  /api/report?start=2026-09-01&end=2026-09-30&category=Facials

The routes live in an APIRouter, a group of routes that app.py plugs in with
one line. That keeps these routes out of app.py.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import get_settings
from .dataset import Dataset
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
    """The bookings data, loaded from the JSON file once."""
    return Dataset(get_settings().report_data_path)


@router.get("/options")
def options() -> dict[str, Any]:
    """What the filter row can offer: business name, the date range the data
    covers, and the category and channel lists.
    """
    dataset = get_dataset()
    return {"business": dataset.business["name"], "note": dataset.meta["description"], **dataset.filter_options()}


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


# The standing instructions. The business document is added below them on
# every request, between <profile> tags, so the model can tell instructions
# apart from data.
SYSTEM = """You are the assistant for the owner of a spa. You answer the owner's \
questions about their own business.

The business profile is given below between <profile> tags. It is the only \
source of facts about the business. Quote prices, hours and policies exactly \
as written there. If the profile does not cover the question, say so plainly; \
do not guess or fill in what spas usually do.

Reply in two to four plain sentences, without markdown."""


@router.post("/chat")
def chat(body: ChatQuestion) -> dict[str, str]:
    """Answer one question about the business, using the business document."""
    # The whole business document, as text. It is small, so it is sent in full.
    profile = get_settings().data_path.read_text(encoding="utf-8")

    try:
        response = anthropic.Anthropic().messages.create(
            model=get_settings().model,
            max_tokens=16000,
            system=f"{SYSTEM}\n\n<profile>\n{profile}\n</profile>",
            messages=[{"role": "user", "content": body.question}],
        )
    except (anthropic.AnthropicError, TypeError):
        # TypeError is what the SDK raises when no API key is set.
        raise HTTPException(status_code=503, detail="The assistant is unavailable right now.")

    return {"answer": "".join(block.text for block in response.content if block.type == "text")}
