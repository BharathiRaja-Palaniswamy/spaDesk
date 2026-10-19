"""Tests for chunk headers (spadesk/rag/headers.py): where each chunk sits in
the document, written above it for search only.

Run with:  .venv/bin/python -m pytest tests/test_rag_headers.py
"""

import pytest

from spadesk.config import get_settings
from spadesk.rag.chunking import split
from spadesk.rag.context import retrieved_context
from spadesk.rag.embeddings import get_embedder
from spadesk.rag.headers import (MORE, check_budget, check_fits, count_tokens, document_lines, header_for,
                                 search_text, with_headers, without_header)
from spadesk.rag.stores import get_store

DOCUMENT = """# Pebble Spa

## Location and Hours

**Yoga Studio Hours**
Saturday: 8:00 AM – 6:00 PM
Sunday: 8:00 AM – 4:00 PM

## Core Services

### Facials

**Glow Express Facial**
**Duration:** 30 minutes

**Signature Radiance Facial**
**Duration:** 60 minutes
"""


def header_of(text: str, budget: int = 50) -> str:
    """The header of a chunk spanning from `text`'s first line to the end of the document."""
    begin = DOCUMENT.index(text)
    return header_for(document_lines(DOCUMENT), begin, len(DOCUMENT), budget)


def test_a_chunk_starting_mid_block_still_carries_its_place():
    # The chunk begins at Sunday's hours: its heading and block title are in the text before it.
    assert header_of("Sunday: 8:00 AM") == ("Location and Hours > Yoga Studio Hours | "
                                             "Core Services > Facials > Glow Express Facial, Signature Radiance Facial")


def test_labels_inside_a_block_are_not_titles():
    assert "Duration" not in header_of("Sunday: 8:00 AM")


def test_a_long_header_is_shortened_least_useful_part_first():
    full = header_of("Sunday: 8:00 AM")
    shorter = header_of("Sunday: 8:00 AM", budget=count_tokens(full) - 1)
    # First the parent of a subsection goes ("Core Services"); a top-level section has none to drop.
    assert shorter == "Location and Hours > Yoga Studio Hours | Facials > Glow Express Facial, Signature Radiance Facial"
    shortest = header_of("Sunday: 8:00 AM", budget=12)
    assert count_tokens(shortest) <= 12 and shortest.startswith("Location and Hours")
    assert header_of("Sunday: 8:00 AM", budget=4).endswith(MORE)  # cut to the budget as a last resort


def test_a_budget_of_zero_turns_headers_off():
    assert header_of("Sunday: 8:00 AM", budget=0) == ""


def test_search_text_puts_the_header_above_and_can_be_taken_off_again():
    assert search_text("Sunday: 8:00 AM", "Location and Hours") == "Location and Hours\nSunday: 8:00 AM"
    assert search_text("Sunday: 8:00 AM", "") == "Sunday: 8:00 AM"
    assert without_header(search_text("Sunday", "Location and Hours"), "Location and Hours") == "Sunday"


def test_every_chunk_of_the_real_document_gets_a_header_within_budget():
    settings = get_settings()
    text = settings.data_path.read_text(encoding="utf-8")
    chunks = with_headers(text, split(text, settings.chunk_tokens, settings.chunk_overlap), settings.header_tokens)
    assert all(chunk.header and count_tokens(chunk.header) <= settings.header_tokens for chunk in chunks)
    # The chunk that runs from the studio's hours into the facials, which ranked 21st without headers.
    sunday = next(chunk for chunk in chunks if "Sunday: 8:00 AM – 4:00 PM" in chunk.text and "Facials" in chunk.sections)
    assert sunday.header.startswith("Location and Hours > Yoga Studio Hours | Core Services > Facials")


def test_chunk_and_header_must_fit_the_embedding_model_together():
    check_budget(150, 50, 256, "model")  # fits: no error
    with pytest.raises(ValueError, match="Lower one of them"):
        check_budget(220, 50, 256, "model")


def test_a_text_too_long_to_embed_is_named_before_anything_is_embedded():
    check_fits({"chunk-0": "short"}, 256, "model")  # fits: no error
    with pytest.raises(ValueError, match=r"chunk-7 \(\d+ tokens\)"):
        check_fits({"chunk-0": "short", "chunk-7": "word " * 300}, 256, "model")


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """An empty store folder, so the index is built for the test."""
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "7")
    for cached in (get_embedder, get_store):
        cached.cache_clear()
    yield
    for cached in (get_embedder, get_store):
        cached.cache_clear()


def test_the_model_is_given_the_chunks_without_their_headers(fresh_store):
    context = retrieved_context("What are the yoga studio hours on Sunday?")
    assert "Sunday: 8:00 AM – 4:00 PM" in context.text  # the chunk that ranked 21st without headers
    assert " > " not in context.text  # no header reached the prompt
