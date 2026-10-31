"""Tracing with LangSmith: a record of what happened inside each chat question.

A trace is a tree of steps ("runs"). For one question it looks like this:

    chat                 the whole request: the question in, the answer out
      retrieve           (RAG on only) the chunks that were found, with scores
      ask_model          the exact prompt sent to the model, its reply, the tokens used
      get_report         (owner only, when the model asks) the dates it asked for and the figures returned
      ask_model          the model again, now with those figures, writing its answer

Each step is an ordinary function with `@traceable` above it (see the chat
routes, assistant/model.py and rag/retriever.py). The decorator records the function's
arguments, its result, how long it took and any error, and sends them to
LangSmith in the background, where they can be read at smith.langchain.com.

Tracing is switched on by settings in `.env`, read by the langsmith library
itself:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=spadesk     the project the traces are filed under

Without them `@traceable` does nothing and the app runs as before.

This file holds the small functions that shape what a step records, so
LangSmith can show it well. They are passed to `@traceable`.

And `setup`, run once when the app starts: every trace, from any step, goes
through one LangSmith client, which masks personal details before sending
(privacy.py) and labels the trace with the code's commit, so a trace can be
traced back to the code that made it. Which project the traces go to is a
setting (LANGSMITH_PROJECT): spadesk-local on a laptop, spadesk-prod in
production, spadesk-evals for the evals (evals/observe.py).
"""

import os
import subprocess
from typing import Any

import langsmith
from langsmith.anonymizer import create_anonymizer

from .privacy import mask


def _commit() -> str:
    """The code's git commit, or SPADESK_COMMIT where there is no git (some hosts)."""
    try:
        found = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return found.stdout.strip() or os.getenv("SPADESK_COMMIT", "unknown")
    except (OSError, subprocess.SubprocessError):
        return os.getenv("SPADESK_COMMIT", "unknown")


def setup() -> None:
    """Send every trace through one client that masks personal details, with the
    commit in each trace's metadata. Nothing is sent when tracing is off."""
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        return
    langsmith.configure(client=langsmith.Client(anonymizer=create_anonymizer(mask)),
                        metadata={"commit": _commit()})


def retrieved_documents(matches: list) -> dict[str, Any]:
    """What the "retrieve" step records as its result: one document per chunk.
    This is the layout LangSmith shows as a list of retrieved documents.
    """
    return {"documents": [
        {"page_content": match.chunk.text,
         "type": "Document",
         "metadata": {"id": match.chunk.id, "score": round(match.score, 3),
                      **({"rerank_score": round(match.rerank_score, 3)} if match.rerank_score is not None else {}),
                      "sections": list(match.chunk.sections), "tokens": match.chunk.tokens,
                      **({"header": match.chunk.header} if match.chunk.header else {})}}
        for match in matches
    ]}


def model_messages(inputs: dict[str, Any]) -> dict[str, Any]:
    """What the "ask_model" step records as its input: the system prompt and
    the conversation so far as chat messages, which LangSmith shows as a
    conversation, and the tools the model was offered.
    """
    return {"messages": [{"role": "system", "content": inputs["system"]}, *inputs["messages"]],
            "tools": inputs["tools"]}


def model_reply(response: Any) -> dict[str, Any]:
    """What the "ask_model" step records as its result: the structured reply
    or the tools the model asked for, and the token counts, from which
    LangSmith works out the cost.
    """
    reply = response.parsed_output
    usage = response.usage
    return {
        "reply": reply.model_dump() if reply else None,
        "tool_requests": [{"name": block.name, "input": block.input}
                          for block in response.content if block.type == "tool_use"],
        "usage_metadata": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                           "total_tokens": usage.input_tokens + usage.output_tokens},
    }
