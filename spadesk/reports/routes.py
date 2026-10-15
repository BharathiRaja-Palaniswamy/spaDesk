"""HTTP routes for the manage page and its reports.

  GET /manage               the manage page (static/dashboard/index.html)
  GET /api/report/options   what the filter row can offer (date range, dropdown values)
  GET /api/report           the full report for the chosen filters
  POST /api/report/chat     one assistant question about the business -> a structured reply.
                            The body says whether to answer with RAG or the whole document.
                            The model may call the report tool (tools.py) to get figures.

Example:  /api/report?start=2026-09-01&end=2026-09-30&category=Facials

The routes live in an APIRouter, a group of routes that app.py plugs in with
one line. That keeps these routes out of app.py.
"""

import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from langsmith import traceable
from pydantic import BaseModel, Field

from ..business.store import get_knowledge
from ..config import get_settings
from ..rag.context import retrieved_context
from ..tracing import model_messages, model_reply
from .dataset import Dataset, load
from .report import build_report
from .tools import TOOL_NAME, report_label, report_tool, run_report

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

    if problem := dataset.filter_problem(start, end, category, channel):
        raise HTTPException(status_code=422, detail=problem)

    return build_report(dataset, start, end, category, channel)


class ChatQuestion(BaseModel):
    """The JSON body of POST /api/report/chat."""

    question: str
    # How the model is given the business document. False: all of it. True:
    # only the chunks retrieved for this question (RAG). Set by the chat's toggle.
    rag: bool = False


# Structured output. The model is not asked for free text: it is given this
# class as a schema and must fill in every field, so the page can rely on them
# (show the answer, lay a list out as a list, say where it came from, style a
# "not covered" reply differently). The model returns data; the page decides
# how it looks. The docstring and each field's `description` are sent to the
# model as part of the schema; they are how it learns what each field is for.
class ReplyItem(BaseModel):
    """One entry of a list in the assistant's reply, for example one service."""

    name: str = Field(description="What the entry is, e.g. 'Swedish Massage', 'Late cancellations' or 'Revenue'.")
    detail: str = Field(description="The facts about it, kept short, e.g. '60 / 90 minutes, $115 / $160', or for "
                                    "a figure its value and change: '$19,818, down 2.1% from $20,235'.")


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


# How many follow-up questions the chat shows under an answer.
MAX_SUGGESTIONS = 3
# How many times in one question the model may ask for tools before it must
# answer. A comparison needs two reports; this leaves room and still ends.
MAX_TOOL_ROUNDS = 4

# The standing instructions. The current time and the business document are
# added below them on every request, between tags, so the model can tell
# instructions apart from data.
SYSTEM = """You are the assistant for the owner of a spa. You answer the owner's \
questions about their own business.

The business profile is given below between <profile> tags. It is the only \
source of facts about what the business offers. Quote prices, hours and policies exactly \
as written there. If the profile does not cover the question, say so plainly; \
do not guess or fill in what spas usually do.

The current date and time at the spa are given below between <currentTime> \
tags. Use them to work out dates, days of the week and how much time is left \
when a question depends on them, then answer from the profile as usual.

The profile's headings and sub-headings are listed between <sections> tags. \
When you name the sections an answer came from, use those names exactly, and \
prefer the most specific one: the sub-heading the answer sits under, where \
there is one.

For questions about how the business is doing (revenue, bookings, \
cancellations, channels, customers, feedback), call the get_report tool and \
answer from the figures it returns. Never estimate or invent a figure. Work \
out the dates to ask for from the current time; "last month" means the last \
full calendar month. Say which period your figures cover. If the tool \
reports that there is no data for a period, tell the owner what the data covers."""

# Added to the instructions when RAG is on, so the model knows it is looking
# at part of the profile and does not mistake a cut for a mistake.
RAG_NOTE = """Only the parts of the profile that best match the question are \
given, each between <excerpt> tags. They were cut by length, so an excerpt \
may begin or end mid-sentence. The rest of the profile is not shown: if the \
excerpts do not contain the answer, say you could not find it in the profile."""


# The model call, as its own function so it shows up as one step of the trace
# (see spadesk/tracing.py): `ls_provider` and `ls_model_name` tell LangSmith
# which model ran, so it can price the tokens.
@traceable(run_type="llm", process_inputs=model_messages, process_outputs=model_reply,
           metadata={"ls_provider": "anthropic", "ls_model_name": get_settings().model})
def ask_model(system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
    """Send the conversation so far to the model and return the SDK's response.
    The response is either the structured reply or a request to run a tool.
    """
    # `parse` works like `create`, with one addition: `output_format` is a
    # pydantic class, the model must reply with JSON matching it, and the
    # SDK turns that JSON into an instance of the class (`parsed_output`).
    return anthropic.Anthropic().messages.parse(
        model=get_settings().model,
        max_tokens=16000,
        system=system,
        messages=messages,
        tools=tools,
        output_format=ChatReply,
    )


@router.post("/chat")
@traceable  # the whole request is the top step of the trace
def chat(body: ChatQuestion) -> ChatAnswer:
    """Answer one question about the business, using the business document."""
    settings = get_settings()
    # The document's headings and sub-headings, e.g. "Policies" and
    # "Cancellation Policy". The reply names its sources from this list.
    # (dict.fromkeys removes repeats while keeping the document's order.)
    outline = get_knowledge().outline
    headings = list(dict.fromkeys(name for pair in outline for name in pair if name))
    instructions, chunks = SYSTEM, 0
    if body.rag:
        # RAG: only the chunks that match the question, and only their headings
        # to cite from, so a source the model was never shown cannot be named.
        context = retrieved_context(body.question)
        profile, chunks = context.text, context.chunks
        headings = [name for name in headings if name in context.sections]
        instructions = f"{SYSTEM}\n\n{RAG_NOTE}"
    else:
        # The whole business document, as text. It is small, so it fits.
        profile = settings.data_path.read_text(encoding="utf-8")
    # The model has no clock, so it is told the current date and time in the
    # spa's own time zone, e.g. "Friday, 09 October 2026, 3:42 PM (PDT)".
    current_time = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %d %B %Y, %-I:%M %p (%Z)")

    system = (f"{instructions}\n\n<currentTime>{current_time}</currentTime>\n\n"
              f"<sections>\n{chr(10).join(headings)}\n</sections>\n\n<profile>\n{profile}\n</profile>")
    dataset = get_dataset()
    tools = [report_tool(dataset)]
    # The conversation with the model. It starts as the one question and grows
    # by two messages each time the model asks for a tool: its request, then
    # the result. The model has no memory between calls, so all of it is sent
    # again every time.
    messages: list[dict[str, Any]] = [{"role": "user", "content": body.question}]
    reports: list[str] = []
    try:
        response = ask_model(system, messages, tools)
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
            response = ask_model(system, messages, tools)
    except (anthropic.AnthropicError, TypeError):
        # TypeError is what the SDK raises when no API key is set.
        raise HTTPException(status_code=503, detail="The assistant is unavailable right now.")

    reply = response.parsed_output
    if reply is None:
        # No structured reply came back, for example because the model declined
        # to answer, or was still asking for tools when the rounds ran out.
        raise HTTPException(status_code=503, detail="The assistant could not answer that.")

    # Structured output guarantees the shape of the reply, not that its content
    # is right. Section names are checked against the real headings, so the
    # page never shows a source that does not exist.
    cited = [name for name in reply.sections if name in headings]
    # When a sub-heading is cited, its parent heading adds nothing: "Cancellation
    # Policy" already says where the answer is, so "Policies" is dropped.
    parents = {heading for heading, subheading in outline if subheading in cited}
    reply.sections = [name for name in cited if name not in parents]
    # The schema cannot limit how many suggestions come back, so the limit is applied here.
    reply.suggestions = reply.suggestions[:MAX_SUGGESTIONS]
    return ChatAnswer(**reply.model_dump(), rag=body.rag, chunks=chunks, reports=reports)
