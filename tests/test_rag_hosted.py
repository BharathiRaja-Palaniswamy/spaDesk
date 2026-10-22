"""Tests for the models run on Pinecone's servers (spadesk/rag/hosted.py, the
Pinecone embedder and reranker), with a stand-in for the Pinecone client: no
account, no network, nothing spent.

What is checked is the part that is ours: questions and chunks are sent as
such, the vectors have the size the index expects, long lists are sent in
batches, the reranker's scores come back in the order the chunks went in,
and every failure becomes the error the chat routes turn into "unavailable".

Run with:  .venv/bin/python -m pytest tests/test_rag_hosted.py
"""

from types import SimpleNamespace

import pytest
from pinecone.exceptions import PineconeApiException

from spadesk.rag import hosted
from spadesk.rag.embeddings.pinecone import BATCH, DIMENSION, PineconeEmbedder
from spadesk.rag.errors import ModelUnavailable
from spadesk.rag.rerankers.pinecone import PineconeReranker


class FakeInference:
    """Stands in for `Pinecone().inference`: records each request and answers
    it as Pinecone would, or fails when told to."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.embedded, self.fail = [], fail

    def embed(self, model, inputs, parameters):
        if self.fail:
            raise self.fail
        self.embedded.append({"inputs": inputs, **parameters})
        return [{"values": [0.5] * parameters["dimension"]} for _ in inputs]

    def rerank(self, model, query, documents, return_documents):
        # Pinecone answers best first, each with its position in `documents`;
        # here the longest text is best.
        order = sorted(range(len(documents)), key=lambda i: len(documents[i]), reverse=True)
        return SimpleNamespace(data=[{"index": i, "score": float(len(documents[i]))} for i in order])


@pytest.fixture
def inference(monkeypatch):
    """Points the hosted models at a fake client; returns its inference part."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-key-not-a-real-one")
    fake = FakeInference()
    monkeypatch.setattr(hosted, "Pinecone", lambda api_key: SimpleNamespace(inference=fake))
    return fake


def test_chunks_and_questions_are_embedded_as_such(inference):
    embedder = PineconeEmbedder()
    chunks = embedder.embed(["Swedish Massage: $115", "Lip Wax: $18"])
    question = embedder.embed(["How much is a massage?"], query=True)
    assert len(chunks) == 2 and len(question) == 1 and len(question[0]) == DIMENSION == 384
    assert [request["input_type"] for request in inference.embedded] == ["passage", "query"]


def test_many_texts_are_sent_in_batches(inference):
    vectors = PineconeEmbedder().embed([f"text {n}" for n in range(BATCH + 4)])
    assert len(vectors) == BATCH + 4
    assert [len(request["inputs"]) for request in inference.embedded] == [BATCH, 4]


def test_rerank_scores_come_back_in_the_order_the_chunks_went_in(inference):
    assert PineconeReranker().scores("question", ["bb", "a", "ccc"]) == [2.0, 1.0, 3.0]


def test_a_pinecone_failure_becomes_model_unavailable(inference):
    inference.fail = PineconeApiException(status=429, reason="Too Many Requests")
    with pytest.raises(ModelUnavailable, match="llama-text-embed-v2 could not be reached"):
        PineconeEmbedder().embed(["Swedish Massage"])


def test_without_a_key_the_error_says_what_to_do(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "")
    with pytest.raises(ModelUnavailable, match="PINECONE_API_KEY is not set"):
        PineconeReranker().scores("question", ["a chunk"])
