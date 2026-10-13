"""Step 3 of RAG: keep the chunks and their vectors where they can be searched.

`get_store()` is the only way the rest of the app gets a vector store. Which
one it returns is a setting, SPADESK_VECTOR_STORE.

To add a store: write a class with the three methods in base.py in its own
file here, and add one line to STORES below.
"""

from functools import lru_cache

from ...config import get_settings
from .base import VectorStore
from .chroma import ChromaStore

# Setting value -> how to make that store from the settings.
STORES = {
    "chroma": lambda settings: ChromaStore(settings.vector_store_path),
}


@lru_cache  # created once: a store holds an open database
def get_store() -> VectorStore:
    """The vector store named by the SPADESK_VECTOR_STORE setting."""
    settings = get_settings()
    if settings.vector_store not in STORES:
        raise ValueError(f"Unknown vector store: {settings.vector_store!r}. Choose one of: {', '.join(STORES)}.")
    return STORES[settings.vector_store](settings)
