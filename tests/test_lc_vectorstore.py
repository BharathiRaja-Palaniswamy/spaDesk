"""Tests for LangChain steps 2 and 3 (spadesk/lc/embeddings.py, lc/vectorstores/).

The LangChain store must find what the hand-built one finds, so the two can
be compared fairly: same model, same chunks, same search. Those tests use the
real local model and the real document, each test in its own empty folder.

The store is pluggable, so keeping the index up to date must work through
LangChain's common VectorStore interface alone: one test plugs in LangChain's
in-memory store to show that.

Run with:  .venv/bin/python -m pytest tests/test_lc_vectorstore.py
"""

import pytest
from langchain_core.vectorstores import InMemoryVectorStore

from spadesk.lc import vectorstores
from spadesk.lc.embeddings import get_embeddings
from spadesk.lc.vectorstores import get_vectorstore
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.index import ensure_index
from spadesk.rag.stores import get_store

QUESTIONS = ["What is the cancellation policy?", "How much is a Swedish massage?",
             "What time is Restorative Yoga on Sunday?"]


@pytest.fixture(autouse=True)
def empty_store(tmp_path, monkeypatch):
    """Each test gets its own store folders, away from the app's real ones."""
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "vectors"))
    get_store.cache_clear()
    yield
    get_store.cache_clear()


def test_the_embeddings_match_the_hand_built_model():
    text = "Cancellations should be made at least 24 hours in advance."
    ours, theirs = get_embeddings().embed_query(text), get_embedder().embed([text])[0]
    assert len(ours) == 384
    assert ours == pytest.approx(theirs)


def test_the_index_is_built_once_and_then_reused():
    store = get_vectorstore()
    assert len(store.store) == 34  # one entry per chunk
    assert get_vectorstore() is store  # nothing changed: the same filled store, not rebuilt


def test_an_edit_to_the_document_rebuilds_the_index(tmp_path, monkeypatch):
    document = tmp_path / "business_doc.md"
    document.write_text("## Parking\n\nThere is free bicycle parking behind the building.\n", encoding="utf-8")
    monkeypatch.setenv("SPADESK_DATA", str(document))
    assert len(get_vectorstore().store) == 1

    document.write_text(document.read_text(encoding="utf-8") + "\n## Bikes\n\nBring a lock.\n", encoding="utf-8")
    best = get_vectorstore().similarity_search("Do I need a bike lock?", k=1)[0]
    assert "lock" in best.page_content and "Bikes" in best.metadata["sections"]


def test_any_langchain_store_can_be_plugged_in(monkeypatch):
    # A store provider returns the store for a signature and whether it is
    # already filled. This one is shared across calls, as a real database would be.
    shared = {}

    def plugged(embeddings, signature):
        store = shared.setdefault(signature, InMemoryVectorStore(embedding=embeddings))
        return store, bool(store.store)

    monkeypatch.setitem(vectorstores.PROVIDERS, "plugged", plugged)
    monkeypatch.setenv("SPADESK_VECTOR_STORE", "plugged")
    store = get_vectorstore()
    assert len(store.store) == 34
    best = store.similarity_search("How much is a Swedish massage?", k=1)[0]
    assert "Massage Therapy" in best.metadata["sections"]

    # The store lost its chunks (say it was wiped): the next call fills it again.
    store.delete(ids=list(store.store))
    assert len(get_vectorstore().store) == 34


def test_an_unknown_store_is_reported(monkeypatch):
    monkeypatch.setenv("SPADESK_VECTOR_STORE", "nope")
    with pytest.raises(ValueError, match="Unknown vector store"):
        get_vectorstore()


@pytest.mark.parametrize("question", QUESTIONS)
def test_search_finds_the_same_chunks_as_the_hand_built_store(question):
    ensure_index()
    ours = [doc.metadata["id"] for doc in get_vectorstore().similarity_search(question, k=7)]
    theirs = [match.chunk.id for match in get_store().search(get_embedder().embed([question])[0], 7)]
    assert ours == theirs
