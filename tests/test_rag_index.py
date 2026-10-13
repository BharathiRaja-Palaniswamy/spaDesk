"""Tests for step 3 of RAG: the vector store, and building the index.

A stand-in embedder is used, so these tests never load the real model. Each
test gets its own empty folder for the store.

Run with:  .venv/bin/pytest tests/test_rag_index.py
"""

import pytest

from spadesk.rag import embeddings
from spadesk.rag.chunking import Chunk
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.index import ensure_index
from spadesk.rag.stores import get_store
from spadesk.rag.stores.chroma import ChromaStore


class FakeEmbedder:
    """Stands in for a real model: counts how often it is used, and gives
    every text the same simple vector.
    """

    name = "fake"
    max_tokens = 200
    calls = 0

    def embed(self, texts):
        FakeEmbedder.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def rag(tmp_path, monkeypatch):
    """Point RAG at the stand-in embedder and an empty store folder."""
    monkeypatch.setitem(embeddings.PROVIDERS, "fake", FakeEmbedder)
    monkeypatch.setenv("SPADESK_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    # Known chunk settings, whatever the developer's `.env` says.
    monkeypatch.setenv("SPADESK_CHUNK_TOKENS", "150")
    monkeypatch.setenv("SPADESK_CHUNK_OVERLAP", "30")
    monkeypatch.setattr(FakeEmbedder, "calls", 0)
    # get_embedder() and get_store() remember their result; forget it so the
    # settings above are read, and again afterwards so later tests are not affected.
    get_embedder.cache_clear(), get_store.cache_clear()
    yield
    get_embedder.cache_clear(), get_store.cache_clear()


# ---- The store ----
def test_search_returns_the_closest_chunks_first(tmp_path):
    store = ChromaStore(tmp_path)
    hours = Chunk("chunk-0", "Sunday: 10:00 AM – 5:00 PM", 9, ("Location and Hours",))
    waxing = Chunk("chunk-1", "Lip Wax — $18", 5, ("Hair Removal", "Kids Services"))
    store.replace([hours, waxing], [[1.0, 0.0], [0.0, 1.0]], signature="v1")

    best, other = store.search([0.9, 0.1], k=2)
    assert best.chunk == hours and other.chunk == waxing  # every field comes back as stored
    assert best.score > 0.9 > other.score
    assert len(store.search([0.9, 0.1], k=1)) == 1


def test_the_index_survives_a_restart(tmp_path):
    ChromaStore(tmp_path).replace([Chunk("chunk-0", "text", 1, ("About",))], [[1.0, 0.0]], signature="v1")
    reopened = ChromaStore(tmp_path)  # what a restarted server would see
    assert reopened.signature() == "v1" and len(reopened.search([1.0, 0.0], k=5)) == 1


def test_an_empty_store_has_no_index(tmp_path):
    store = ChromaStore(tmp_path)
    assert store.signature() is None and store.search([1.0, 0.0], k=3) == []


def test_replace_discards_what_was_there(tmp_path):
    store = ChromaStore(tmp_path)
    store.replace([Chunk("chunk-0", "old", 1, ("About",)), Chunk("chunk-1", "old too", 2, ("About",))],
                  [[1.0, 0.0], [0.0, 1.0]], signature="v1")
    store.replace([Chunk("chunk-0", "new", 1, ("Mission",))], [[1.0, 0.0]], signature="v2")
    assert store.signature() == "v2"
    assert [match.chunk.text for match in store.search([1.0, 0.0], k=5)] == ["new"]


# ---- Building the index ----
def test_the_index_is_built_once_and_then_reused(rag):
    assert ensure_index() is True  # nothing stored yet: built
    assert ensure_index() is False  # nothing changed: reused
    assert FakeEmbedder.calls == 1  # the document was embedded only once
    assert len(get_store().search([1.0, 0.0, 0.0], k=100)) == 15  # one entry per chunk


def test_changing_the_chunk_size_rebuilds_the_index(rag, monkeypatch):
    ensure_index()
    monkeypatch.setenv("SPADESK_CHUNK_TOKENS", "100")
    assert ensure_index() is True
    assert len(get_store().search([1.0, 0.0, 0.0], k=100)) > 15  # smaller chunks, so more of them


def test_chunks_must_fit_the_embedding_model(rag, monkeypatch):
    monkeypatch.setenv("SPADESK_CHUNK_TOKENS", "500")  # the stand-in model reads 200
    with pytest.raises(ValueError, match="reads at most 200 tokens"):
        ensure_index()
