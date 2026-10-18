"""LangChain step 5: the prompt, as a template.

A `ChatPromptTemplate` is a list of messages with {placeholders}; invoking it
with values fills them in. It is a Runnable, so it chains straight into a
model: `prompt | model`.

The text is the same as the hand-built prompt's (assistant/prompt.py): the
assistant's own instructions, the rules for using the profile, the current
time, the headings it may cite, and the profile itself, whole or as the
excerpts the retriever found. Only the way it is put together differs, so
the two pipelines can be compared on retrieval and code alone.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ..assistant.prompt import PROFILE_RULES, RAG_NOTE
from ..business.store import get_knowledge
from ..config import get_settings

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{instructions}\n\n{rules}\n\n<currentTime>{current_time}</currentTime>\n\n"
               "<sections>\n{sections}\n</sections>\n\n<profile>\n{profile}\n</profile>"),
    ("human", "{question}"),
])


def all_headings() -> list[str]:
    """Every heading and sub-heading of the document, in order."""
    # dict.fromkeys removes repeats while keeping the document's order.
    return list(dict.fromkeys(name for pair in get_knowledge().outline for name in pair if name))


def excerpts(documents: list[Document]) -> str:
    """The retrieved chunks, each between <excerpt> tags, as rag/context.py lays them out."""
    return "\n".join(f"<excerpt>\n{doc.page_content}\n</excerpt>" for doc in documents)


def prompt_values(instructions: str, question: str, documents: list[Document] | None) -> dict[str, str]:
    """The values that fill PROMPT. `documents` are the retrieved chunks, or
    None to give the model the whole document (RAG off)."""
    settings = get_settings()
    headings, rules = all_headings(), PROFILE_RULES
    if documents is None:
        profile = settings.data_path.read_text(encoding="utf-8")
    else:
        # Only the headings of the chunks found can be cited, as in the hand-built pipeline.
        found = {name for doc in documents for name in doc.metadata["sections"]}
        headings = [name for name in headings if name in found]
        profile, rules = excerpts(documents), f"{PROFILE_RULES}\n\n{RAG_NOTE}"
    current_time = datetime.now(ZoneInfo(settings.timezone)).strftime("%A, %d %B %Y, %-I:%M %p (%Z)")
    return {"instructions": instructions, "rules": rules, "current_time": current_time,
            "sections": "\n".join(headings), "profile": profile, "question": question}
