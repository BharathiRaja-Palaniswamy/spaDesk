"""Embeddings from all-MiniLM-L6-v2, run on this machine by the fastembed library.

The same model Chroma's built-in embedder uses, run by fastembed instead (the
library that already runs our reranker), so the app does not need Chroma.
No API key, no network call per text. It is downloaded once, on first use,
to ~/.cache/fastembed (about 90 MB). Each vector has 384 numbers.

It reads at most 256 tokens of a text and silently ignores the rest, so a
chunk and its header must stay shorter than that (see rag/headers.py).
"""

from fastembed import TextEmbedding

from ..rerankers.cross_encoder import CACHE

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FastEmbedEmbedder:
    """An Embedder (see base.py) backed by fastembed."""

    name = "fastembed/all-MiniLM-L6-v2"
    max_tokens = 256

    def __init__(self) -> None:
        """Load the model, downloading it the first time."""
        self._model = TextEmbedding(MODEL, cache_dir=str(CACHE))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        # fastembed yields numpy arrays; plain lists keep the interface free of any one library's types.
        return [vector.tolist() for vector in self._model.embed(texts)]
