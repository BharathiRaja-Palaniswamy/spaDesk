"""LangChain step 3: keep the chunks and their vectors, with a pluggable store.

Every LangChain vector store (Pinecone, Qdrant, pgvector, ...) has the same
interface, `VectorStore`: add documents, delete them, search by meaning. The
rest of the pipeline only uses that interface, so the store is a setting,
SPADESK_VECTOR_STORE:

    pinecone  a hosted index (vectorstores/pinecone.py), the default; needs PINECONE_API_KEY
    memory    in this process (vectorstores/memory.py); for tests and offline work

To add a store: write a function in a file here that returns its LangChain
VectorStore for a signature, and whether it already holds the chunks, and
add one line to PROVIDERS.

Keeping the index up to date is done here too, because LangChain does not do
it. As in the hand-built pipeline (rag/index.py), the index has a signature,
a fingerprint of the document, the chunk and header settings, the embedding
model and the store. Each store keeps its chunks under that signature (a
Pinecone namespace, an in-memory store), so asking "is the store for this
signature filled?" is the whole check: nothing has to be remembered on this
machine.
"""

import hashlib
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ...config import get_settings
from ...rag.headers import FORMAT, check_budget, check_fits, search_text
from ..documents import load_document, split_document
from ..embeddings import get_embeddings
from .memory import make_memory
from .pinecone import make_pinecone

# Setting value -> how to make that store for a signature: (the store, whether it is already filled).
PROVIDERS: dict[str, Callable[[Embeddings, str], tuple[VectorStore, bool]]] = {
    "pinecone": make_pinecone,
    "memory": make_memory,
}


def _signature(text: str, provider: str) -> str:
    """A fingerprint of everything the index is built from."""
    settings, embeddings = get_settings(), get_embeddings()
    recipe = (f"{provider}|{embeddings.name}|{settings.chunk_tokens}|{settings.chunk_overlap}|"
              f"{settings.header_tokens}|{FORMAT}|{text}")
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()


def get_vectorstore() -> VectorStore:
    """The vector store named by SPADESK_VECTOR_STORE, filled first if it is empty.

    Called for every question, like ensure_index() in the hand-built
    pipeline: when nothing changed it reads the document, works out the
    signature and asks the store one question.
    """
    settings, embeddings = get_settings(), get_embeddings()
    provider = settings.vector_store
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown vector store: {provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    check_budget(settings.chunk_tokens, settings.header_tokens, embeddings.max_tokens, embeddings.name)

    document = load_document()
    store, filled = PROVIDERS[provider](embeddings, _signature(document.page_content, provider))
    if filled:
        return store

    chunks = split_document(document)
    # A LangChain store embeds the text it is given, so each chunk is stored
    # with its header above its text; the retriever takes the header off again
    # before the model sees the chunk (retriever.py).
    texts = {chunk.metadata["id"]: search_text(chunk.page_content, chunk.metadata["header"]) for chunk in chunks}
    check_fits(texts, embeddings.max_tokens, embeddings.name)
    # Most stores accept only text or numbers as metadata values, so the list
    # of sections is stored as one string, a heading per line.
    store.add_documents([chunk.model_copy(update={
        "page_content": texts[chunk.metadata["id"]],
        "metadata": {**chunk.metadata, "sections": "\n".join(chunk.metadata["sections"])}}) for chunk in chunks],
        ids=list(texts))
    return store
