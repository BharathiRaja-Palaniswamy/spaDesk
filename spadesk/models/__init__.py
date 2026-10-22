"""The model gateway: every model call in the app goes through here.

Why it exists: so the app can use Claude, or a cheaper open model hosted by
Nebius, or another provider later, by changing settings, not code. The rest
of the app asks for a job ("chat-default", "judge"), never for a provider.

    profiles.py   job -> model ("provider:model") and a fallback model
    providers.py  provider -> its LangChain chat model; the only file that names a provider's library
    errors.py     every provider's errors -> one ModelError with a kind
    usage.py      tokens, time and estimated cost of each call, logged the same way
    tools.py      a tool written once (Tool), and running the model's requests for it
    structured.py replies in a fixed shape, by each provider's own method
    gateway.py    chat(), structured() and stream(): the calls the app makes

See docs/models.md for how to add a provider and how a voice assistant
would share this with the chat.
"""

from .errors import ModelError
from .gateway import Reply, Structured, chat, stream, structured
from .tools import Tool, run_requests
from .usage import Usage

__all__ = ["ModelError", "Reply", "Structured", "Tool", "Usage", "chat", "run_requests", "stream", "structured"]
