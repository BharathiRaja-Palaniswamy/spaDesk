"""The interface every embedding provider follows.

An embedding is a list of numbers (a vector) that stands for the meaning of
a piece of text. Texts that mean similar things get vectors that point in
similar directions, which is what lets a question be matched to the chunks
that answer it, even when they share no words.

A Protocol lists the attributes and methods a class must have, with no code
of its own. Any class that has them counts as an Embedder; it does not need
to inherit from this one. The rest of the app depends only on this
interface, so a provider can be swapped without touching that code.
"""

from typing import Protocol


class Embedder(Protocol):
    """Turns text into vectors."""

    # Which model made the vectors, e.g. "fastembed/all-MiniLM-L6-v2". Vectors
    # from different models cannot be compared, so the index remembers this
    # name and is rebuilt when it changes.
    name: str
    # The most tokens of one text the model reads. Building the index checks
    # the chunk size against this, because some models silently ignore the rest.
    max_tokens: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        ...
