"""Tests for step 1 of RAG: cutting the document into fixed-size chunks.

Run with:  .venv/bin/pytest tests/test_rag_chunking.py
"""

import pytest

from spadesk.config import get_settings
from spadesk.rag.chunking import split

SIZE, OVERLAP = 150, 30


@pytest.fixture(scope="module")
def text():
    """The real business document, as text."""
    return get_settings().data_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chunks(text):
    return split(text, SIZE, OVERLAP)


def test_chunks_are_a_fixed_size(chunks):
    # Every chunk is full except the last, which holds whatever is left.
    assert all(chunk.tokens == SIZE for chunk in chunks[:-1])
    assert 0 < chunks[-1].tokens <= SIZE
    assert [chunk.id for chunk in chunks[:3]] == ["chunk-0", "chunk-1", "chunk-2"]


def test_chunks_cover_the_whole_document(text, chunks):
    assert text.startswith(chunks[0].text) and text.endswith(chunks[-1].text)
    # Nothing is skipped between two chunks: each begins inside the one before.
    for before, after in zip(chunks, chunks[1:]):
        assert text.index(after.text) < text.index(before.text) + len(before.text)


def test_neighbouring_chunks_overlap(chunks):
    # The end of one chunk is repeated at the start of the next, so text
    # around a cut is whole in at least one of them.
    for before, after in zip(chunks, chunks[1:]):
        shared = next(n for n in range(len(before.text), 0, -1) if after.text.startswith(before.text[-n:]))
        assert shared > 20  # characters; 30 tokens is far more than that


def test_no_overlap_means_no_repeated_text(text):
    assert "".join(chunk.text for chunk in split(text, SIZE, 0)) == text


def test_chunks_know_which_sections_they_hold(chunks):
    swedish = next(chunk for chunk in chunks if "**Swedish Massage**" in chunk.text)
    assert "Massage Therapy" in swedish.sections
    # A chunk cut by size usually runs across more than one heading.
    assert any(len(chunk.sections) > 1 for chunk in chunks)
    assert all(chunk.sections for chunk in chunks)


def test_overlap_must_be_smaller_than_the_size(text):
    with pytest.raises(ValueError):
        split(text, 100, 100)
