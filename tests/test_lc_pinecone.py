"""Tests for the Pinecone store (spadesk/lc/vectorstores/pinecone.py), with a
stand-in for the Pinecone client: no account, no network, nothing spent.

What is checked is the part that is ours: the index is created when missing,
the chunks live in a namespace named by the signature, a filled namespace is
reused, and older ones are removed when a new one is filled.

Run with:  .venv/bin/python -m pytest tests/test_lc_pinecone.py
"""

from types import SimpleNamespace

import pytest

from spadesk.lc.embeddings import get_embeddings
from spadesk.lc.vectorstores import pinecone as store_module

SIGNATURE = "3f9a1c0e2b7d4a65" + "0" * 48


class FakeIndex:
    """Stands in for a Pinecone index: namespaces with a vector count each."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts, self.deleted = counts, []

    def describe_index_stats(self):
        return SimpleNamespace(namespaces={name: SimpleNamespace(vector_count=n) for name, n in self.counts.items()})

    def delete(self, delete_all: bool, namespace: str) -> None:
        self.deleted.append(namespace)


class FakeClient:
    """Stands in for the Pinecone client: one index, which may not exist yet."""

    def __init__(self, exists: bool, index: FakeIndex) -> None:
        self.exists, self.index, self.created = exists, index, None

    def has_index(self, name: str) -> bool:
        return self.exists

    def create_index(self, name, dimension, metric, spec) -> None:
        self.created = {"name": name, "dimension": dimension, "metric": metric}

    def Index(self, name: str) -> FakeIndex:  # noqa: N802 (the Pinecone library's own name)
        return self.index


@pytest.fixture
def pinecone(monkeypatch):
    """Points the store at a fake client. Returns a function that sets it up."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-key-not-a-real-one")

    def connect(exists: bool, counts: dict[str, int]) -> FakeClient:
        client = FakeClient(exists, FakeIndex(counts))
        monkeypatch.setattr(store_module, "Pinecone", lambda api_key: client)
        monkeypatch.setattr(store_module, "PineconeVectorStore",
                            lambda index, embedding, namespace: SimpleNamespace(namespace=namespace))
        return client
    return connect


def test_a_missing_index_is_created_to_fit_the_embeddings(pinecone):
    client = pinecone(exists=False, counts={})
    store, filled = store_module.make_pinecone(get_embeddings(), SIGNATURE)
    assert client.created == {"name": "spadesk", "dimension": 384, "metric": "cosine"}
    assert store.namespace == "business_doc-3f9a1c0e2b7d4a65" and filled is False


def test_a_filled_namespace_for_this_signature_is_reused(pinecone):
    client = pinecone(exists=True, counts={"business_doc-3f9a1c0e2b7d4a65": 33})
    _, filled = store_module.make_pinecone(get_embeddings(), SIGNATURE)
    assert filled is True and client.created is None and client.index.deleted == []


def test_a_new_signature_removes_the_old_namespaces_but_nobody_elses(pinecone):
    client = pinecone(exists=True, counts={"business_doc-0000000000000000": 33, "someone-else": 5})
    _, filled = store_module.make_pinecone(get_embeddings(), SIGNATURE)
    assert filled is False and client.index.deleted == ["business_doc-0000000000000000"]


def test_without_a_key_the_error_says_what_to_do(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "")
    with pytest.raises(ValueError, match="PINECONE_API_KEY is not set"):
        store_module.make_pinecone(get_embeddings(), SIGNATURE)
