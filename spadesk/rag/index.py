"""Builds the index: every chunk of the business document, embedded and stored.

    document -> split into chunks -> one vector per chunk -> vector store

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
from .stores import get_store


def ensure_index() -> bool:
    """Make sure the stored index matches the current document and settings.
    Returns True when it had to be rebuilt, False when it was already right.
    """
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    if settings.chunk_tokens > embedder.max_tokens:
        raise ValueError(
            f"SPADESK_CHUNK_TOKENS is {settings.chunk_tokens}, but {embedder.name} reads at most "
            f"{embedder.max_tokens} tokens of a text. Use a smaller chunk size.")

    text = settings.data_path.read_text(encoding="utf-8")
    # A hash turns any amount of text into a short fixed-length code that
    # changes completely if even one character of the input changes.
    recipe = f"{embedder.name}|{settings.chunk_tokens}|{settings.chunk_overlap}|{text}"
    signature = hashlib.sha256(recipe.encode("utf-8")).hexdigest()
    if store.signature() == signature:
        return False

    chunks = split(text, settings.chunk_tokens, settings.chunk_overlap)
    vectors = embedder.embed([chunk.text for chunk in chunks])
    store.replace(chunks, vectors, signature)
    return True
