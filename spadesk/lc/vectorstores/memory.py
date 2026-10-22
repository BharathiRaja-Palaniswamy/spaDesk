"""LangChain's in-memory vector store, for the LangChain pipeline.

`InMemoryVectorStore` ships with langchain-core: no database, no extra
package, no key. It lives in this process, so it starts empty when the app
starts and is filled then; with about 33 chunks that takes seconds. Used by
the tests (so they never touch a hosted store) and for working offline:
SPADESK_VECTOR_STORE=memory.

One store is kept per signature and store folder setting: a changed document
gets a new, empty store, and each test, which sets its own folder, its own.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from ...config import get_settings


class MemoryVectorStore(InMemoryVectorStore):
    """InMemoryVectorStore, with its relevance scores defined: its search
    already scores by cosine similarity, the same number the hand-built search
    reports, so it is used as it is."""

    def _select_relevance_score_fn(self):
        return lambda similarity: similarity


@lru_cache  # the same store for every question, or it would start empty each time
def _store(folder: str, signature: str, embeddings: Embeddings) -> MemoryVectorStore:
    return MemoryVectorStore(embedding=embeddings)


def make_memory(embeddings: Embeddings, signature: str) -> tuple[VectorStore, bool]:
    """The store for `signature`, and whether it already holds the chunks."""
    store = _store(str(get_settings().vector_store_path), signature, embeddings)
    return store, bool(store.store)  # `store.store` holds the chunks, by id
