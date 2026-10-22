"""Tools in one form, whichever provider the model comes from.

A tool is a function the model may ask the app to run (the model never runs
code itself). Each provider wants tools described in its own format and
returns its requests in its own format. Here a tool is written once, as a
Tool: a name, a description, its inputs as a JSON schema, and the function
that runs it. LangChain converts that description for each provider, and
gives every provider's requests back the same way, as `{"name", "args",
"id"}`, so `run_requests` below works for any of them.

The same Tool objects can serve the chat now and a voice assistant later:
neither the tool nor the code that runs it knows which one asked.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage


@dataclass(frozen=True)
class Tool:
    """A function the model can ask for, described so any provider understands it."""

    name: str  # e.g. "get_report"
    description: str  # when to use it; this is how the model decides to call it
    parameters: dict[str, Any]  # its inputs, as a JSON schema of type "object"
    run: Callable[[dict[str, Any]], Any]  # takes the model's inputs, returns something JSON can hold

    def description_for_model(self) -> dict[str, Any]:
        """The tool as sent to a model: the OpenAI "function" format, which
        LangChain converts for every other provider (Anthropic included)."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def for_anthropic_sdk(self) -> dict[str, Any]:
        """The tool in the Anthropic library's own format, for code that calls
        that library directly (the hand-built pipeline, assistant/model.py)."""
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


def run_requests(requests: list[dict[str, Any]], tools: list[Tool]) -> list[ToolMessage]:
    """Run each tool request (`{"name", "args", "id"}`) and return its result
    as a ToolMessage, tied to the request by id. Every request gets a result,
    because a model's next call is refused when one is missing; a request for
    a tool that does not exist gets an error result the model can read."""
    by_name = {tool.name: tool for tool in tools}
    results = []
    for request in requests:
        tool = by_name.get(request["name"])
        output = tool.run(request["args"]) if tool else {"error": f"There is no tool called {request['name']}."}
        results.append(ToolMessage(json.dumps(output), tool_call_id=request["id"]))
    return results
