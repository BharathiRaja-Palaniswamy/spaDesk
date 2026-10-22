"""Tests for step 2 of RAG: turning text into vectors.

These run the real local model, which is downloaded once on first use.

Run with:  .venv/bin/pytest tests/test_rag_embeddings.py
"""

import pytest

from spadesk.rag import embeddings
from spadesk.rag.embeddings import get_embedder


@pytest.fixture(autouse=True)
def fresh_embedder():
    """get_embedder() remembers its result; forget it so each test reads the setting."""
    get_embedder.cache_clear()
    yield
    get_embedder.cache_clear()


def similarity(a: list[float], b: list[float]) -> float:
    """How alike two vectors are: the sum of their numbers multiplied pairwise.
    These vectors all have length 1, which makes this the cosine similarity:
    near 1 for the same meaning, near 0 for unrelated text.
    """
    return sum(x * y for x, y in zip(a, b))


def test_one_vector_comes_back_per_text():
    embedder = get_embedder()
    vectors = embedder.embed(["Swedish Massage", "Cancellation Policy"])
    assert len(vectors) == 2 and len(vectors[0]) == len(vectors[1]) == 384
    assert isinstance(vectors[0][0], float)
    assert embedder.name == "onnx/all-MiniLM-L6-v2"


def test_text_with_a_similar_meaning_gets_a_similar_vector():
    question, hours, waxing = get_embedder().embed([
        "What time do you open on Sunday?",
        "Sunday: 10:00 AM – 5:00 PM",
        "Lip Wax costs $18",
    ])
    # The question shares almost no words with the hours line, yet it is far
    # closer to it than to an unrelated line. This is what retrieval relies on.
    assert similarity(question, hours) > similarity(question, waxing) + 0.3


def test_the_provider_is_chosen_by_a_setting(monkeypatch):
    class FakeEmbedder:
        name = "fake"
        max_tokens = 100

        def embed(self, texts):
            return [[0.0] for _ in texts]

    monkeypatch.setitem(embeddings.PROVIDERS, "fake", FakeEmbedder)
    monkeypatch.setenv("SPADESK_EMBEDDING_PROVIDER", "fake")
    assert get_embedder().name == "fake"


def test_an_unknown_provider_is_reported(monkeypatch):
    monkeypatch.setenv("SPADESK_EMBEDDING_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder()
