"""Embeddings from the small model that ships with Chroma.

The model (all-MiniLM-L6-v2) runs on this machine: no API key, no network
call per text. It is downloaded once, on first use, to ~/.cache/chroma
(about 80 MB). Each vector has 384 numbers.

It reads at most 256 tokens of a text and silently ignores the rest, so
chunks must stay shorter than that (see SPADESK_CHUNK_TOKENS).
"""

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class ChromaEmbedder:
    """An Embedder (see base.py) backed by Chroma's built-in model."""

    name = "chroma/all-MiniLM-L6-v2"
    max_tokens = 256

    def __init__(self) -> None:
        """Set up the model. Its files are loaded on the first embed() call."""
        self._model = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        # The model returns numpy arrays; plain lists keep the interface free
        # of any one library's types.
        return [vector.tolist() for vector in self._model(texts)]
