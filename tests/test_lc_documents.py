"""Tests for LangChain step 1 (spadesk/lc/documents.py): loading and splitting.

The LangChain pipeline must cut the document exactly as the hand-built one
does, so the two can be compared fairly with the same evals. These check that
every chunk has the same text and the same sections as rag/chunking.py makes.

Run with:  .venv/bin/python -m pytest tests/test_lc_documents.py
"""

from spadesk.config import get_settings
from spadesk.lc.documents import load_document, split_document
from spadesk.rag.chunking import split


def hand_built_chunks():
    settings = get_settings()
    return split(settings.data_path.read_text(encoding="utf-8"), settings.chunk_tokens, settings.chunk_overlap)


def test_the_document_loads_as_one_document_with_its_source():
    document = load_document()
    assert document.page_content.startswith("# Pebble Spa")
    assert document.metadata["source"].endswith("business_doc.md")


def test_chunks_match_the_hand_built_ones():
    ours, theirs = split_document(load_document()), hand_built_chunks()
    assert len(ours) == len(theirs)
    for lc_chunk, chunk in zip(ours, theirs):
        assert lc_chunk.page_content == chunk.text
        assert lc_chunk.metadata["sections"] == list(chunk.sections)
        assert lc_chunk.metadata["id"] == chunk.id


def test_each_chunk_knows_where_it_starts():
    document = load_document()
    for chunk in split_document(document):
        start = chunk.metadata["start"]
        assert document.page_content[start:start + len(chunk.page_content)] == chunk.page_content
