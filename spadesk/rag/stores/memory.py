"""A vector store kept in memory: the chunks and their vectors in two lists.

The business document is about 33 chunks, so comparing a question with every
chunk takes microseconds, and the index is rebuilt in seconds whenever the
app starts (index.py notices the store is empty). No database, no extra
package: what the store does is plain to see in a few lines of numpy.
"""

import numpy as np

from ..chunking import Chunk
from .base import Match


class MemoryStore:
    """A VectorStore (see base.py) that lives in this process's memory."""

    def __init__(self) -> None:
        """Start empty: no index until replace() is called."""
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0))
        self._signature: str | None = None

    def signature(self) -> str | None:
        """What the stored index was built from, or None when there is none."""
        return self._signature

    def replace(self, chunks: list[Chunk], vectors: list[list[float]], signature: str) -> None:
        """Keep these chunks and their vectors, in place of whatever was here."""
        matrix = np.array(vectors, dtype=float)
        # Each vector is scaled to length 1 once, here, so a search is one matrix
        # product: the dot product of two length-1 vectors is their cosine similarity.
        self._vectors = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self._chunks, self._signature = list(chunks), signature

    def search(self, vector: list[float], k: int) -> list[Match]:
        """The `k` stored chunks closest to `vector` by cosine, best first."""
        if not self._chunks:
            return []
        query = np.array(vector, dtype=float)
        scores = self._vectors @ (query / np.linalg.norm(query))
        best = np.argsort(-scores)[:k]
        return [Match(self._chunks[i], score=float(scores[i])) for i in best]
