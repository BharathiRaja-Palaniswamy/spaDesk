"""LangChain step 4b: reorder the retrieved chunks, with a pluggable reranker.

LangChain calls a reranker a "document compressor": anything that takes the
retrieved documents and the question and returns fewer, better ones
(`BaseDocumentCompressor.compress_documents`). Providers such as Cohere
Rerank come as compressors; this one adapts the hand-built pipeline's
reranker (rag/rerankers/) to that interface, so both pipelines rerank alike,
chosen by the same setting, SPADESK_RERANKER: "pinecone" (the default),
"cross-encoder", or "none" for no reranking. To add a reranker, add it
there; this adapter picks it up.
"""

from collections.abc import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from ..rag.headers import search_text
from ..rag.rerankers import get_reranker


class SpaCompressor(BaseDocumentCompressor):
    """Scores each document against the question with the configured reranker,
    and returns them best first, each with its score in metadata["rerank_score"].
    It reads each document with its header, as the embedder did. The retriever
    keeps the top few."""

    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Callbacks | None = None) -> Sequence[Document]:
        """`documents` reordered by how well each answers `query`."""
        scores = get_reranker().scores(
            query, [search_text(doc.page_content, doc.metadata.get("header", "")) for doc in documents])
        scored = [doc.model_copy(update={"metadata": {**doc.metadata, "rerank_score": score}})
                  for doc, score in zip(documents, scores)]
        return sorted(scored, key=lambda doc: doc.metadata["rerank_score"], reverse=True)


def get_compressor() -> BaseDocumentCompressor | None:
    """The reranker named by SPADESK_RERANKER as a compressor, or None when it is "none"."""
    return SpaCompressor() if get_reranker() else None
