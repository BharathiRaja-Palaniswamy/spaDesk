"""LangChain step 2: turn text into vectors, with a pluggable provider.

LangChain's `Embeddings` interface has two methods: `embed_documents` for the
chunks (when the index is built) and `embed_query` for a question (each time
one is asked). Any class with those two methods can be plugged in, which is
how LangChain supports many providers behind one interface.

This one adapts the hand-built pipeline's embedder (rag/embeddings/) to that
interface, so both pipelines use the same provider, chosen by the same
setting, SPADESK_EMBEDDING_PROVIDER ("fastembed" by default), and find the
same chunks. To add a provider, add it there; this adapter picks it up.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from ..rag.embeddings import get_embedder


class SpaEmbeddings(Embeddings):
    """The configured embedder, as a LangChain Embeddings."""

    def __init__(self) -> None:
        self._embedder = get_embedder()
        # Which model made the vectors, and how much of a text it reads: the
        # vector store uses both (rebuilding when `name` changes, checking sizes).
        self.name = self._embedder.name
        self.max_tokens = self._embedder.max_tokens

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """The vector for one question."""
        return self._embedder.embed([text])[0]


@lru_cache  # created once: loading a model is slow
def get_embeddings() -> Embeddings:
    """The embeddings named by the SPADESK_EMBEDDING_PROVIDER setting."""
    return SpaEmbeddings()
