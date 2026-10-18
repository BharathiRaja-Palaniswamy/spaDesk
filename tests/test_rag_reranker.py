"""Tests for step 4b of RAG: reordering the retrieved chunks with a reranker.

Most use a fake reranker, so the rule being tested is easy to see: fetch the
candidates, reorder them by the reranker's score, keep the best few. The last
test uses the real local model on the real document, because the point of
reranking is that it finds an answer the first search ranks too low.

Run with:  .venv/bin/python -m pytest tests/test_rag_reranker.py
"""

import pytest

from spadesk.rag import rerankers
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.rerankers import get_reranker
from spadesk.rag.retriever import retrieve
from spadesk.rag.stores import get_store

QUESTION = "What time is Restorative Yoga on Sunday?"
ANSWER = "10:00 AM – Restorative Yoga"  # the line of the document that answers it


class LongestFirst:
    """A fake reranker: the longer the chunk, the higher its score."""

    name = "fake/longest-first"

    def scores(self, question, texts):
        return [float(len(text)) for text in texts]


@pytest.fixture(autouse=True)
def empty_store(tmp_path, monkeypatch):
    """Give each test its own store folder and known settings."""
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "7")
    monkeypatch.setenv("SPADESK_RERANK_CANDIDATES", "20")
    caches = (get_embedder, get_store, get_reranker)
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


def test_candidates_are_reordered_by_the_reranker_and_trimmed(monkeypatch):
    monkeypatch.setitem(rerankers.PROVIDERS, "fake", LongestFirst)
    monkeypatch.setenv("SPADESK_RERANKER", "fake")
    matches = retrieve(QUESTION)

    assert len(matches) == 7  # SPADESK_RETRIEVAL_TOP_K, not the 20 candidates
    # Best first by the reranker's score, which each match now carries.
    assert [m.rerank_score for m in matches] == sorted((m.rerank_score for m in matches), reverse=True)
    assert all(m.rerank_score == len(m.chunk.text) for m in matches)
    # Each match keeps the first search's similarity too.
    assert all(0 < m.score <= 1 for m in matches)


def test_none_switches_reranking_off(monkeypatch):
    monkeypatch.setenv("SPADESK_RERANKER", "none")
    matches = retrieve(QUESTION)
    assert all(m.rerank_score is None for m in matches)
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)


def test_an_unknown_reranker_is_reported(monkeypatch):
    monkeypatch.setenv("SPADESK_RERANKER", "nope")
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker()


def test_the_real_reranker_lifts_an_answer_the_first_search_ranks_too_low(monkeypatch):
    # Without reranking, the chunk with Sunday's timetable is not in the top 7:
    # it starts mid-timetable and is mostly about other things (see
    # docs/eval-report.md). The reranker reads the question and each chunk
    # together, and brings it in.
    monkeypatch.setenv("SPADESK_RERANKER", "none")
    assert not any(ANSWER in m.chunk.text for m in retrieve(QUESTION))

    get_reranker.cache_clear()
    monkeypatch.setenv("SPADESK_RERANKER", "cross-encoder")
    assert any(ANSWER in m.chunk.text for m in retrieve(QUESTION))
