"""The call to the model, shared by both assistants.

It is its own function so it shows up as one step of the trace (see
spadesk/tracing.py): `ls_provider` and `ls_model_name` tell LangSmith which
model ran, so it can price the tokens.
"""

from typing import Any

import anthropic
from langsmith import traceable
from pydantic import BaseModel

from ..config import get_settings
from ..tracing import model_messages, model_reply


@traceable(run_type="llm", process_inputs=model_messages, process_outputs=model_reply,
           metadata={"ls_provider": "anthropic", "ls_model_name": get_settings().model})
def ask_model(system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
              output_format: type[BaseModel]) -> Any:
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
        # Tools are left out of the request altogether when there are none.
        **({"tools": tools} if tools else {}),
        output_format=output_format,
    )
