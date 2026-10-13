"""Step 5 of RAG: turn the retrieved chunks into what the model is given.

This is the "augmented" part of retrieval-augmented generation: the prompt
is augmented with the chunks that were found, in place of the whole document.
"""

from dataclasses import dataclass

from .retriever import retrieve


@dataclass(frozen=True)
class Context:
    """What the model is given to answer one question."""

    text: str  # the retrieved chunks, each between <excerpt> tags
    sections: set[str]  # every heading those chunks include: all an answer can cite
    chunks: int  # how many chunks were retrieved


def retrieved_context(question: str) -> Context:
    """Retrieve the chunks that match `question` and lay them out for the prompt."""
    chunks = [match.chunk for match in retrieve(question)]
    return Context(
        # Tags mark where one excerpt ends and the next begins. The chunks are
        # not neighbours in the document, so running them together would read
        # as one garbled passage.
        text="\n".join(f"<excerpt>\n{chunk.text}\n</excerpt>" for chunk in chunks),
        sections={name for chunk in chunks for name in chunk.sections},
        chunks=len(chunks),
    )
