"""Step 4 of RAG: find the chunks that match a question.

    question -> its vector -> the closest stored vectors -> their chunks
             -> (reranker reorders them) -> the best SPADESK_RETRIEVAL_TOP_K

The question is embedded with the same model as the chunks, so the two can
be compared. With a reranker on (SPADESK_RERANKER, see rag/rerankers/), the
first search fetches more candidates than are needed
(SPADESK_RERANK_CANDIDATES), the reranker scores each one against the
question, and the best are kept. The model is given the same number of
chunks either way, so the prompt does not grow.

Nothing here calls the chat model: retrieval is a search, and it can be
tried and tested on its own (see evals/retrieval.py).
"""

from dataclasses import replace

from langsmith import traceable

from ..config import get_settings
from ..tracing import retrieved_documents
from .embeddings import get_embedder
from .headers import search_text
from .index import ensure_index
from .rerankers import get_reranker
from .stores import get_store
from .stores.base import Match


# Recorded as a step of the trace when tracing is on (see spadesk/tracing.py).
@traceable(run_type="retriever", process_outputs=retrieved_documents)
def retrieve(question: str) -> list[Match]:
    """The chunks of the business document that best answer `question`, best
    first. How many come back is the SPADESK_RETRIEVAL_TOP_K setting.
    """
    settings = get_settings()
    # Checked on every question, so an edit to the document is picked up
    # without a restart. When nothing changed this costs almost nothing.
    ensure_index()
    vector = get_embedder().embed([question])[0]  # one text in, so one vector out
    if get_reranker() is None:
        return get_store().search(vector, settings.retrieval_top_k)
    # Fetch more than needed, so the reranker can lift a chunk the first search ranked too low.
    candidates = get_store().search(vector, max(settings.rerank_candidates, settings.retrieval_top_k))
    return rerank(question, candidates)[:settings.retrieval_top_k]


# Its own step of the trace, so the order before and after reranking can be compared.
@traceable(name="rerank", process_outputs=retrieved_documents)
def rerank(question: str, candidates: list[Match]) -> list[Match]:
    """`candidates` reordered by the reranker's score, best first. Each match
    keeps its similarity score and gains its rerank score. The reranker reads
    each chunk with its header, as the embedder did."""
    scores = get_reranker().scores(question, [search_text(m.chunk.text, m.chunk.header) for m in candidates])
    scored = [replace(match, rerank_score=score) for match, score in zip(candidates, scores)]
    return sorted(scored, key=lambda match: match.rerank_score, reverse=True)
