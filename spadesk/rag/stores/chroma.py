"""A vector store backed by Chroma, saved to a folder on this machine.

Chroma keeps everything in "collections". This store uses one collection:
each entry is a chunk's id, its vector, its text, and its other fields as
metadata. The folder (storage/chroma by default) holds a small database
file, so the index is still there after a restart.
"""

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from ..chunking import Chunk
from .base import Match

COLLECTION = "business_doc"


class ChromaStore:
    """A VectorStore (see base.py) backed by a Chroma collection."""

    def __init__(self, path: Path) -> None:
        """Open the database in the folder `path`, creating it if it is not there."""
        self._client = chromadb.PersistentClient(path=str(path))

    def _collection(self):
        """The stored collection, or None when nothing has been indexed yet."""
        try:
            # embedding_function=None: vectors are always passed in by us, so
            # Chroma never embeds anything on its own.
            return self._client.get_collection(COLLECTION, embedding_function=None)
        except NotFoundError:
            return None

    def signature(self) -> str | None:
        """What the stored index was built from, or None when there is none."""
        collection = self._collection()
        return collection.metadata["signature"] if collection else None

    def replace(self, chunks: list[Chunk], vectors: list[list[float]], signature: str) -> None:
        """Drop the old collection and store these chunks in a new one."""
        if self._collection():
            self._client.delete_collection(COLLECTION)
        collection = self._client.create_collection(
            COLLECTION,
            metadata={"signature": signature},
            # Compare vectors by cosine: by the direction they point in.
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            # Metadata values must be plain text or numbers, so the section
            # names are joined into one string, one per line.
            metadatas=[{"tokens": chunk.tokens, "sections": "\n".join(chunk.sections)} for chunk in chunks],
        )

    def search(self, vector: list[float], k: int) -> list[Match]:
        """The `k` stored chunks closest to `vector`, best first."""
        collection = self._collection()
        if collection is None:
            return []
        # Chroma answers several searches at once, so every field is a list
        # with one entry per search. There is one search here: entry [0].
        found = collection.query(query_embeddings=[vector], n_results=k,
                                 include=["documents", "metadatas", "distances"])
        return [
            Match(
                Chunk(id, text, fields["tokens"], tuple(fields["sections"].split("\n"))),
                score=1 - distance,  # Chroma reports cosine distance, which is 1 - similarity
            )
            for id, text, fields, distance in zip(
                found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0])
        ]
