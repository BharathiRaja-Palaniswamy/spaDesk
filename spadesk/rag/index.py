"""Builds the index: every chunk of the business document, embedded and stored.

    document -> split into chunks -> a header for each -> one vector per chunk -> vector store

Each vector is made from the chunk's header and text together (headers.py),
so a chunk is found by where it sits in the document as well as by what it says.

The index is only as good as what it was built from. If the document is
edited, the chunk size changes, or a different embedding model is chosen,
the stored vectors no longer match and the index must be rebuilt. To notice
that, the index is saved with a signature: a short fingerprint of everything
it was built from. `ensure_index()` compares the signature of the stored
index with the one it would build now, and rebuilds only when they differ.
"""

import hashlib

from ..config import get_settings
from .chunking import split
from .embeddings import get_embedder
from .headers import FORMAT, check_budget, check_fits, search_text, with_headers
from .stores import get_store


def ensure_index() -> bool:
    """Make sure the stored index matches the current document and settings.
    Returns True when it had to be rebuilt, False when it was already right.
    """
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    check_budget(settings.chunk_tokens, settings.header_tokens, embedder.max_tokens, embedder.name)

    text = settings.data_path.read_text(encoding="utf-8")
    # A hash turns any amount of text into a short fixed-length code that
    # changes completely if even one character of the input changes.
    recipe = (f"{embedder.name}|{settings.chunk_tokens}|{settings.chunk_overlap}|"
              f"{settings.header_tokens}|{FORMAT}|{text}")
    signature = hashlib.sha256(recipe.encode("utf-8")).hexdigest()
    if store.signature() == signature:
        return False

    chunks = with_headers(text, split(text, settings.chunk_tokens, settings.chunk_overlap), settings.header_tokens)
    texts = {chunk.id: search_text(chunk.text, chunk.header) for chunk in chunks}
    check_fits(texts, embedder.max_tokens, embedder.name)
    vectors = embedder.embed(list(texts.values()))
    store.replace(chunks, vectors, signature)
    return True
