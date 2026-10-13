"""Step 4 of RAG: find the chunks that match a question.

    question -> its vector -> the closest stored vectors -> their chunks

The question is embedded with the same model as the chunks, so the two can
be compared. Nothing here calls the chat model: retrieval is a search, and
it can be tried and tested on its own.
"""

from ..config import get_settings
from .embeddings import get_embedder
from .index import ensure_index
from .stores import get_store
from .stores.base import Match


def retrieve(question: str) -> list[Match]:
    """The chunks of the business document closest in meaning to `question`,
    best first. How many come back is the SPADESK_RETRIEVAL_TOP_K setting.
    """
    # Checked on every question, so an edit to the document is picked up
    # without a restart. When nothing changed this costs almost nothing.
    ensure_index()
    vector = get_embedder().embed([question])[0]  # one text in, so one vector out
    return get_store().search(vector, get_settings().retrieval_top_k)
