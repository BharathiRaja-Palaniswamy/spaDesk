"""The interface every reranker follows.

The first search (retriever.py) compares the question's vector with each
chunk's vector. Those vectors were made separately, so the search never reads
the question and a chunk together: it is fast, but it can rank the chunk that
holds the answer too low. A reranker reads the question and each candidate
chunk side by side and scores how well the chunk answers the question. That
is more accurate and slower, so it only reorders the first search's top
candidates (SPADESK_RERANK_CANDIDATES), and the best SPADESK_RETRIEVAL_TOP_K
of them are kept.

Like Embedder, this is a Protocol: any class with these attributes counts as
a Reranker.
"""

from typing import Protocol


class Reranker(Protocol):
    """Scores how well each text answers a question."""

    # Which model scores, e.g. "onnx/ms-marco-MiniLM-L-6-v2". Shown in traces.
    name: str

    def scores(self, question: str, texts: list[str]) -> list[float]:
        """One score per text, in the same order: higher means a better answer.
        Scores only compare texts for the same question; they are not percentages.
        """
        ...
