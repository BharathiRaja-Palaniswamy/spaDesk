"""LangChain step 4: find the chunks that best answer a question.

    question -> the store's closest chunks -> (reranker reorders them) -> the best few

A LangChain retriever is anything that turns a question into documents
(`BaseRetriever`). It is also a "Runnable", so it can be chained with a
prompt and a model (`retriever | prompt | model`), and each call is traced
in LangSmith as a retriever step without any extra code.

This one does what the hand-built rag/retriever.py does, with LangChain
parts: it searches whichever vector store is configured (vectorstores/), and
with a reranker on (rerankers.py) it fetches SPADESK_RERANK_CANDIDATES
chunks, reorders them and keeps the best SPADESK_RETRIEVAL_TOP_K. The model
is given the same number of chunks either way.

(LangChain also offers ContextualCompressionRetriever for "search, then
compress", but it now lives in the legacy langchain-classic package, and it
cannot fetch more candidates than it keeps, which reranking needs.)
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from ..config import get_settings
from .rerankers import get_compressor
from .vectorstores import get_vectorstore


class SpaRetriever(BaseRetriever):
    """Searches the business document, reranks, and keeps the best chunks.

    Each document returned has in its metadata: id, sections (a list of
    headings), score (similarity, near 1 is the same meaning) and, when
    reranked, rerank_score.
    """

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """The chunks that best answer `query`, best first.

        The search and the reranking each run as a step of their own (a
        RunnableLambda), so a LangSmith trace shows both lists: the candidates
        in similarity order, then the same candidates reordered by the
        reranker. A compressor is not traced on its own; passing
        `run_manager.get_child()` as the callbacks is how a step is recorded
        as a child of this retriever's run.
        """
        settings = get_settings()
        compressor = get_compressor()
        k = max(settings.rerank_candidates, settings.retrieval_top_k) if compressor else settings.retrieval_top_k
        documents = RunnableLambda(_search, name="search").invoke(
            {"query": query, "k": k}, config={"callbacks": run_manager.get_child()})
        if compressor:
            documents = RunnableLambda(
                lambda step: list(compressor.compress_documents(step["documents"], step["query"])),
                name="rerank",
            ).invoke({"query": query, "documents": documents}, config={"callbacks": run_manager.get_child()})
        return documents[:settings.retrieval_top_k]


def _search(step: dict) -> list[Document]:
    """The `k` chunks closest to `query` in the vector store, best first, each
    with its sections as a list and its similarity score in metadata."""
    found = get_vectorstore().similarity_search_with_relevance_scores(step["query"], k=step["k"])
    return [doc.model_copy(update={"metadata": {
        **doc.metadata,
        "sections": doc.metadata["sections"].split("\n"),  # stored as one string, a heading per line
        "score": score,
    }}) for doc, score in found]


def get_retriever() -> SpaRetriever:
    """The retriever, named so its steps are easy to find in a LangSmith trace."""
    return SpaRetriever(name="spa_retriever")
