"""The interface every vector store follows.

A vector store keeps each chunk together with its vector, and answers one
question quickly: "which stored vectors are closest to this one?"

Like Embedder, this is a Protocol: any class with these methods counts as a
VectorStore, and the rest of the app depends only on this interface.
"""

from dataclasses import dataclass
from typing import Protocol

from ..chunking import Chunk


@dataclass(frozen=True)
class Match:
    """One search result."""

    chunk: Chunk
    # How close the chunk's vector is to the searched one (cosine similarity):
    # near 1 means the same meaning, near 0 unrelated.
    score: float
    # The reranker's score, when the matches were reranked (see rag/rerankers/);
    # None otherwise. Higher is better; it only compares chunks for one question.
    rerank_score: float | None = None


class VectorStore(Protocol):
    """Keeps chunks with their vectors and finds the closest ones."""

    def signature(self) -> str | None:
        """What the stored index was built from, or None when there is no
        index yet. See index.py, which decides when a rebuild is needed.
        """
        ...

    def replace(self, chunks: list[Chunk], vectors: list[list[float]], signature: str) -> None:
        """Throw away whatever is stored and keep these chunks, one vector
        each (same order), remembering `signature`.
        """
        ...

    def search(self, vector: list[float], k: int) -> list[Match]:
        """The `k` stored chunks closest to `vector`, best first."""
        ...
