"""LangChain step 4b: reorder the retrieved chunks, with a pluggable reranker.

LangChain calls a reranker a "document compressor": anything that takes the
retrieved documents and the question and returns fewer, better ones
(`BaseDocumentCompressor.compress_documents`). Providers such as Cohere
Rerank come as compressors; this one wraps the same small local
cross-encoder the hand-built pipeline uses (rag/rerankers/cross_encoder.py),
so both pipelines rerank alike.

Which reranker is used is the same setting as the hand-built pipeline's,
SPADESK_RERANKER: "cross-encoder" (the default), or "none" for no reranking.
To add one: write or import a BaseDocumentCompressor, and add one line to
PROVIDERS.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from ..config import get_settings
from ..rag.headers import search_text
from ..rag.rerankers.cross_encoder import CrossEncoderReranker


@lru_cache  # created once: loading a model is slow
def _cross_encoder() -> CrossEncoderReranker:
    """The local cross-encoder model."""
    return CrossEncoderReranker()


class CrossEncoderCompressor(BaseDocumentCompressor):
    """Scores each document against the question with the local cross-encoder,
    and returns them best first, each with its score in metadata["rerank_score"].
    It reads each document with its header, as the embedder did. The retriever
    keeps the top few."""

    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Callbacks | None = None) -> Sequence[Document]:
        """`documents` reordered by how well each answers `query`."""
        scores = _cross_encoder().scores(
            query, [search_text(doc.page_content, doc.metadata.get("header", "")) for doc in documents])
        scored = [doc.model_copy(update={"metadata": {**doc.metadata, "rerank_score": score}})
                  for doc, score in zip(documents, scores)]
        return sorted(scored, key=lambda doc: doc.metadata["rerank_score"], reverse=True)


# Setting value -> how to make that reranker.
PROVIDERS: dict[str, Callable[[], BaseDocumentCompressor]] = {
    "cross-encoder": CrossEncoderCompressor,
}
OFF = "none"


def get_compressor() -> BaseDocumentCompressor | None:
    """The reranker named by SPADESK_RERANKER, or None when it is "none"."""
    provider = get_settings().reranker
    if provider == OFF:
        return None
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown reranker: {provider!r}. Choose one of: {', '.join([*PROVIDERS, OFF])}.")
    return PROVIDERS[provider]()
