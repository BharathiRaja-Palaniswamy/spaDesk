"""The one shared copy of the parsed business document.

Any part of the app that needs business facts calls `get_knowledge()`: the
customer site's routes, the manage page's profile, and the assistant chat.
It lives here, not in app.py, so that route modules can import it without
importing app.py (which imports them, and would make a circular import).
"""

from functools import lru_cache

from ..config import get_settings
from .knowledge import KnowledgeBase


@lru_cache  # runs once; every later call returns the same object
def get_knowledge() -> KnowledgeBase:
    """The business document, parsed. Read from disk on the first call only."""
    return KnowledgeBase(get_settings().data_path)
