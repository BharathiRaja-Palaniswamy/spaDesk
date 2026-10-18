"""The Chroma vector store for the LangChain pipeline: a local folder.

`langchain_chroma.Chroma` wraps a Chroma collection. It is kept in its own
folder (SPADESK_VECTOR_STORE_PATH with "-langchain" added), so it never
touches the hand-built pipeline's index.
"""

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from ...config import get_settings

COLLECTION = "business_doc"


@lru_cache  # one client per folder; tests use their own folders
def _client(path: str) -> chromadb.ClientAPI:
    """The Chroma database in `path`, created if it is not there."""
    return chromadb.PersistentClient(path=path)


def make_chroma(embeddings: Embeddings) -> Chroma:
    """The Chroma collection, created on first use."""
    path = get_settings().vector_store_path
    return Chroma(client=_client(str(path.with_name(f"{path.name}-langchain"))), collection_name=COLLECTION,
                  embedding_function=embeddings,
                  # Compare vectors by cosine (their direction), as the hand-built store does.
                  collection_configuration={"hnsw": {"space": "cosine"}})
