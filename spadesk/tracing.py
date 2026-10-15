"""Tracing with LangSmith: a record of what happened inside each chat question.

A trace is a tree of steps ("runs"). For one question it looks like this:

    chat                 the whole request: the question in, the answer out
      retrieve           (RAG on only) the chunks that were found, with scores
      ask_model          the exact prompt sent to the model, its reply, the tokens used
      get_report         (when the model asks) the dates it asked for and the figures returned
      ask_model          the model again, now with those figures, writing its answer

Each step is an ordinary function with `@traceable` above it (see
reports/routes.py and rag/retriever.py). The decorator records the function's
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
"""

from typing import Any


def retrieved_documents(matches: list) -> dict[str, Any]:
    """What the "retrieve" step records as its result: one document per chunk.
    This is the layout LangSmith shows as a list of retrieved documents.
    """
    return {"documents": [
        {"page_content": match.chunk.text,
         "type": "Document",
         "metadata": {"id": match.chunk.id, "score": round(match.score, 3),
                      "sections": list(match.chunk.sections), "tokens": match.chunk.tokens}}
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
