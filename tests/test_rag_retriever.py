"""Tests for step 4 of RAG: finding the chunks that match a question.

These use the real local embedding model and the real business document,
because the point is to check that the right parts of it are found. Each
test gets its own empty folder for the store.

Run with:  .venv/bin/pytest tests/test_rag_retriever.py
"""

import pytest

from spadesk.config import get_settings
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.retriever import retrieve
from spadesk.rag.stores import get_store


@pytest.fixture(autouse=True)
def empty_store(tmp_path, monkeypatch):
    """Give each test its own store folder, away from the app's real one, and
    a known number of chunks to fetch, whatever the developer's `.env` says.
    """
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "3")
    get_embedder.cache_clear(), get_store.cache_clear()
    yield
    get_embedder.cache_clear(), get_store.cache_clear()


@pytest.mark.parametrize("question, section", [
    ("What is the cancellation policy?", "Cancellation Policy"),
    ("What time do you open on Sunday?", "Location and Hours"),
    ("How much is a Swedish massage?", "Massage Therapy"),
])
def test_the_best_match_comes_from_the_right_section(question, section):
    best = retrieve(question)[0]
    assert section in best.chunk.sections


def test_an_answer_can_sit_in_a_section_one_would_not_guess():
    # The kids' services have their own section, but the direct answer to
    # this question is an FAQ entry, and that is what is found. The scores
    # are low and close together: a sign the match is a weak one.
    matches = retrieve("Can I bring my child for a treatment?")
    assert any("Do you offer services for children?" in match.chunk.text for match in matches)
    assert matches[0].score < 0.5


def test_matches_come_back_best_first_and_as_many_as_asked(monkeypatch):
    matches = retrieve("What is the cancellation policy?")
    assert len(matches) == 3  # SPADESK_RETRIEVAL_TOP_K, set to 3 for these tests
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)

    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "5")
    assert len(retrieve("What is the cancellation policy?")) == 5


def test_an_edit_to_the_document_is_found_without_a_restart(tmp_path, monkeypatch):
    document = tmp_path / "business_doc.md"
    original = get_settings().data_path.read_text(encoding="utf-8")
    document.write_text(original, encoding="utf-8")
    monkeypatch.setenv("SPADESK_DATA", str(document))
    question = "Do you have parking for bicycles?"
    assert all("bicycle" not in match.chunk.text for match in retrieve(question))

    document.write_text(original + "\n## Parking\n\nThere is free bicycle parking behind the building.\n",
                        encoding="utf-8")
    best = retrieve(question)[0]
    assert "bicycle parking" in best.chunk.text and "Parking" in best.chunk.sections
