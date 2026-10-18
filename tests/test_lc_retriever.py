"""Tests for LangChain step 4 (spadesk/lc/retriever.py, lc/rerankers.py).

The LangChain retriever must return what the hand-built one returns, with
and without reranking, so the two pipelines can be compared fairly. The
reranker is pluggable: one test plugs in a fake one to show the rule (fetch
the candidates, reorder, keep the best few).

Run with:  .venv/bin/python -m pytest tests/test_lc_retriever.py
"""

from collections.abc import Sequence

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import BaseDocumentCompressor, Document

from spadesk.lc import rerankers
from spadesk.lc.rerankers import get_compressor
from spadesk.lc.retriever import get_retriever
from spadesk.rag.rerankers import get_reranker
from spadesk.rag.retriever import retrieve
from spadesk.rag.stores import get_store

QUESTIONS = ["What time is Restorative Yoga on Sunday?", "Is my first yoga class free, and how do I book it?",
             "How much is a Swedish massage?"]


class LongestFirst(BaseDocumentCompressor):
    """A fake reranker: the longer the chunk, the better its score."""

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks=None) -> Sequence[Document]:
        scored = [d.model_copy(update={"metadata": {**d.metadata, "rerank_score": len(d.page_content)}})
                  for d in documents]
        return sorted(scored, key=lambda d: d.metadata["rerank_score"], reverse=True)


@pytest.fixture(autouse=True)
def empty_store(tmp_path, monkeypatch):
    """Each test gets its own store folders and known settings."""
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "7")
    monkeypatch.setenv("SPADESK_RERANK_CANDIDATES", "20")
    get_store.cache_clear(), get_reranker.cache_clear()
    yield
    get_store.cache_clear(), get_reranker.cache_clear()


@pytest.mark.parametrize("reranker", ["none", "cross-encoder"])
@pytest.mark.parametrize("question", QUESTIONS)
def test_it_returns_the_same_chunks_as_the_hand_built_retriever(question, reranker, monkeypatch):
    monkeypatch.setenv("SPADESK_RERANKER", reranker)
    ours = [doc.metadata["id"] for doc in get_retriever().invoke(question)]
    assert ours == [match.chunk.id for match in retrieve(question)]


def test_documents_carry_their_sections_and_scores(monkeypatch):
    monkeypatch.setenv("SPADESK_RERANKER", "cross-encoder")
    documents = get_retriever().invoke("How much is a Swedish massage?")
    assert len(documents) == 7
    assert all(isinstance(doc.metadata["sections"], list) for doc in documents)
    # Among the results, not necessarily first: the reranker may put a package
    # that includes a Swedish massage above it.
    assert any("Massage Therapy" in doc.metadata["sections"] for doc in documents)
    assert all("score" in doc.metadata and "rerank_score" in doc.metadata for doc in documents)


def test_any_compressor_can_be_plugged_in_as_the_reranker(monkeypatch):
    monkeypatch.setitem(rerankers.PROVIDERS, "fake", LongestFirst)
    monkeypatch.setenv("SPADESK_RERANKER", "fake")
    documents = get_retriever().invoke("What time is Restorative Yoga on Sunday?")
    assert len(documents) == 7  # SPADESK_RETRIEVAL_TOP_K, not the 20 candidates
    lengths = [len(doc.page_content) for doc in documents]
    assert lengths == sorted(lengths, reverse=True)


class StepNames(BaseCallbackHandler):
    """Records each step LangChain runs, with what it returned, as a trace would."""

    def __init__(self):
        self.steps = {}

    def on_chain_start(self, serialized, inputs, *, run_id, name=None, **kwargs):
        self.steps[run_id] = {"name": name}

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.steps[run_id]["output"] = outputs

    def outputs(self, name):
        return next(step["output"] for step in self.steps.values() if step["name"] == name)


def test_the_search_and_the_reranking_are_traced_as_steps(monkeypatch):
    monkeypatch.setenv("SPADESK_RERANKER", "cross-encoder")
    recorder = StepNames()
    get_retriever().invoke("What time is Restorative Yoga on Sunday?", config={"callbacks": [recorder]})
    searched, reranked = recorder.outputs("search"), recorder.outputs("rerank")
    # Both steps see all 20 candidates; the reranker changes their order.
    assert len(searched) == len(reranked) == 20
    assert {d.metadata["id"] for d in searched} == {d.metadata["id"] for d in reranked}
    assert [d.metadata["id"] for d in searched] != [d.metadata["id"] for d in reranked]


def test_none_switches_reranking_off_and_an_unknown_name_is_reported(monkeypatch):
    monkeypatch.setenv("SPADESK_RERANKER", "none")
    assert get_compressor() is None
    monkeypatch.setenv("SPADESK_RERANKER", "nope")
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_compressor()
