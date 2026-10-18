"""LangChain step 3: keep the chunks and their vectors, with a pluggable store.

Every LangChain vector store (Chroma, Pinecone, Qdrant, pgvector, ...) has the
same interface, `VectorStore`: add documents, delete them by id, get them by
id, search by meaning. The rest of the pipeline only uses that interface, so
the store is a setting, SPADESK_VECTOR_STORE, the same one the hand-built
pipeline reads:

    chroma    a local folder (vectorstores/chroma.py), the default
    pinecone  a hosted index (vectorstores/pinecone.py), needs an API key

To add a store: write a function that returns its LangChain VectorStore in a
file here, and add one line to PROVIDERS.

Keeping the index up to date is done here too, in a way that works for any
store, because LangChain does not do it. As in the hand-built pipeline
(rag/index.py), the index has a signature, a fingerprint of the document,
the chunk settings, the embedding model and the store. It is saved in a
small local file with the ids of the chunks stored. When the signature would
change, those ids are deleted and the new chunks added; when the store has
lost its chunks (say its folder was deleted), it is filled again.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ...config import get_settings
from ..documents import load_document, split_document
from ..embeddings import get_embeddings
from .chroma import make_chroma
from .pinecone import make_pinecone

# Setting value -> how to make that store, given the embeddings to use.
PROVIDERS: dict[str, Callable[[Embeddings], VectorStore]] = {
    "chroma": make_chroma,
    "pinecone": make_pinecone,
}


def _record_path(provider: str) -> Path:
    """Where the signature and chunk ids of `provider`'s index are kept."""
    return get_settings().vector_store_path.parent / f"langchain-index-{provider}.json"


def _signature(text: str, provider: str) -> str:
    """A fingerprint of everything the index is built from."""
    settings, embeddings = get_settings(), get_embeddings()
    recipe = f"{provider}|{embeddings.name}|{settings.chunk_tokens}|{settings.chunk_overlap}|{text}"
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()


def get_vectorstore() -> VectorStore:
    """The vector store named by SPADESK_VECTOR_STORE, brought up to date first.

    Called for every question, like ensure_index() in the hand-built
    pipeline: when nothing changed it reads the document, compares two short
    strings and looks up one chunk by id.
    """
    settings, embeddings = get_settings(), get_embeddings()
    provider = settings.vector_store
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown vector store: {provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    if settings.chunk_tokens > embeddings.max_tokens:
        raise ValueError(f"SPADESK_CHUNK_TOKENS is {settings.chunk_tokens}, but {embeddings.name} reads "
                         f"at most {embeddings.max_tokens} tokens of a text. Use a smaller chunk size.")

    store = PROVIDERS[provider](embeddings)
    document = load_document()
    signature = _signature(document.page_content, provider)
    record_path = _record_path(provider)
    record = json.loads(record_path.read_text()) if record_path.exists() else {"signature": None, "ids": []}

    # Up to date only if the signature matches AND the store still holds the chunks.
    if record["signature"] == signature and record["ids"] and store.get_by_ids(record["ids"][:1]):
        return store

    if record["ids"]:
        store.delete(ids=record["ids"])
    chunks = split_document(document)
    ids = [chunk.metadata["id"] for chunk in chunks]
    # Most stores accept only text or numbers as metadata values, so the list
    # of sections is stored as one string, a heading per line.
    store.add_documents([chunk.model_copy(update={"metadata": {
        **chunk.metadata, "sections": "\n".join(chunk.metadata["sections"])}}) for chunk in chunks], ids=ids)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"signature": signature, "ids": ids}))
    return store
