"""The system prompt both assistants build: their own instructions, then the
rules for using the profile, the current time, the headings and the profile.

The profile is either the whole business document or, with RAG on, only the
chunks retrieved for the question. Everything after the instructions sits
between tags, so the model can tell instructions apart from data.
"""

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..business.store import get_knowledge
from ..config import get_settings
from ..rag.context import retrieved_context

# How to treat the profile. Both assistants add this after their own instructions.
PROFILE_RULES = """The business profile is given below between <profile> tags. It is the only \
source of facts about what the business offers. Quote prices, hours and policies exactly \
as written there. If the profile does not cover the question, say so plainly; \
do not guess or fill in what spas usually do.

The current date and time at the spa are given below between <currentTime> \
tags. Use them to work out dates, days of the week and how much time is left \
when a question depends on them, then answer from the profile as usual.

The profile's headings and sub-headings are listed between <sections> tags. \
When you name the sections an answer came from, use those names exactly, and \
prefer the most specific one: the sub-heading the answer sits under, where \
there is one."""

# Added when RAG is on, so the model knows it is looking at part of the
# profile and does not mistake a cut for a mistake.
RAG_NOTE = """Only the parts of the profile that best match the question are \
given, each between <excerpt> tags. They were cut by length, so an excerpt \
may begin or end mid-sentence. The rest of the profile is not shown: if the \
excerpts do not contain the answer, say you could not find it in the profile."""


@dataclass(frozen=True)
class Prompt:
    """A system prompt, and what the reply is checked against afterwards."""

    system: str  # the whole system prompt
    headings: list[str]  # the headings the model was shown: all a reply may cite
    chunks: int  # how many chunks were retrieved; 0 without RAG


def profile_prompt(instructions: str, question: str, rag: bool) -> Prompt:
    """Build the system prompt: `instructions` (the assistant's own), then the
    profile, whole or retrieved for `question` when `rag` is on.
    """
    settings = get_settings()
    # The document's headings and sub-headings, e.g. "Policies" and
    # "Cancellation Policy". The reply names its sources from this list.
    # (dict.fromkeys removes repeats while keeping the document's order.)
    headings = list(dict.fromkeys(name for pair in get_knowledge().outline for name in pair if name))
    rules, chunks = PROFILE_RULES, 0
    if rag:
        # RAG: only the chunks that match the question, and only their headings
        # to cite from, so a source the model was never shown cannot be named.
        context = retrieved_context(question)
        profile, chunks = context.text, context.chunks
        headings = [name for name in headings if name in context.sections]
        rules = f"{PROFILE_RULES}\n\n{RAG_NOTE}"
    else:
        # The whole business document, as text. It is small, so it fits.
        profile = settings.data_path.read_text(encoding="utf-8")
    # The model has no clock, so it is told the current date and time in the
    # spa's own time zone, e.g. "Friday, 09 October 2026, 3:42 PM (PDT)".
    current_time = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %d %B %Y, %-I:%M %p (%Z)")

    system = (f"{instructions}\n\n{rules}\n\n<currentTime>{current_time}</currentTime>\n\n"
              f"<sections>\n{chr(10).join(headings)}\n</sections>\n\n<profile>\n{profile}\n</profile>")
    return Prompt(system=system, headings=headings, chunks=chunks)


def checked_sections(cited: list[str], headings: list[str]) -> list[str]:
    """The sections a reply cited, keeping only real ones, most specific first.

    Structured output guarantees the shape of a reply, not that its content is
    right, so the model's section names are checked against the real headings:
    the page never shows a source that does not exist.
    """
    cited = [name for name in cited if name in headings]
    # When a sub-heading is cited, its parent heading adds nothing: "Cancellation
    # Policy" already says where the answer is, so "Policies" is dropped.
    parents = {heading for heading, subheading in get_knowledge().outline if subheading in cited}
    return [name for name in cited if name not in parents]
