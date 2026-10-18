"""LangChain step 2: turn text into vectors, with a pluggable provider.

LangChain's `Embeddings` interface has two methods: `embed_documents` for the
chunks (when the index is built) and `embed_query` for a question (each time
one is asked). Any class with those two methods can be plugged in, which is
how LangChain supports many providers behind one interface.

Which provider is used is the same setting as the hand-built pipeline's,
SPADESK_EMBEDDING_PROVIDER. "chroma" is all-MiniLM-L6-v2, Chroma's built-in
model on this machine, the same one the hand-built pipeline uses, so the two
pipelines find the same chunks.

To add a provider (e.g. OpenAI or Voyage): wrap or import its LangChain
Embeddings class, give it a `name` and `max_tokens`, and add one line to
PROVIDERS. Changing provider changes `name`, which rebuilds the index (see
vectorstores/__init__.py): vectors from different models cannot be compared.
"""

from collections.abc import Callable
from functools import lru_cache

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings

from ..config import get_settings


class MiniLMEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 as a LangChain Embeddings: 384 numbers per text."""

    name = "chroma/all-MiniLM-L6-v2"
    max_tokens = 256  # it silently ignores the rest of a longer text

    def __init__(self) -> None:
        """Set up the model. Its files are loaded on first use (~/.cache/chroma)."""
        self._model = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        return [vector.tolist() for vector in self._model(texts)]

    def embed_query(self, text: str) -> list[float]:
        """The vector for one question."""
        return self.embed_documents([text])[0]


# Setting value -> how to make that provider's Embeddings. Each must also have
# `name` (which model made the vectors) and `max_tokens` (how much of a text it reads).
PROVIDERS: dict[str, Callable[[], Embeddings]] = {
    "chroma": MiniLMEmbeddings,
}


@lru_cache  # created once: loading a model is slow
def get_embeddings() -> Embeddings:
    """The Embeddings named by the SPADESK_EMBEDDING_PROVIDER setting."""
    provider = get_settings().embedding_provider
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    return PROVIDERS[provider]()
