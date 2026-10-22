"""The Pinecone vector store for the LangChain pipeline: a hosted service.

Chosen with SPADESK_VECTOR_STORE=pinecone (the default). It needs an API key,
PINECONE_API_KEY in `.env` (or in the host's secrets), and uses the index
named by SPADESK_PINECONE_INDEX ("spadesk"). The index is created on first
use if it does not exist: serverless, on AWS us-east-1 (available on
Pinecone's free plan), compared by cosine, sized to the embedding model's
vectors (384 numbers for MiniLM).

Keeping it up to date. The chunks are kept in a namespace (a named part of
the index) whose name holds the index's signature, e.g.
"business_doc-3f9a1c0e2b7d4a65". So the index itself says what it was built
from, and nothing needs to be remembered on this machine, which matters on a
host whose disk is wiped on every restart. A namespace with chunks in it is
up to date; when the document or a setting changes, the signature and so the
namespace name change, the new one is filled, and the old ones are deleted.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from ...config import get_settings

PREFIX = "business_doc-"  # every namespace this app writes starts with it
CLOUD, REGION = "aws", "us-east-1"


def make_pinecone(embeddings: Embeddings, signature: str) -> tuple[VectorStore, bool]:
    """The store for `signature`, and whether it already holds the chunks."""
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise ValueError("SPADESK_VECTOR_STORE is pinecone but PINECONE_API_KEY is not set. Add it to .env "
                         "(or the host's secrets), or use SPADESK_VECTOR_STORE=memory.")
    client = Pinecone(api_key=settings.pinecone_api_key)
    if not client.has_index(settings.pinecone_index):
        dimension = len(embeddings.embed_query("dimension check"))
        client.create_index(name=settings.pinecone_index, dimension=dimension, metric="cosine",
                            spec=ServerlessSpec(cloud=CLOUD, region=REGION))
    index = client.Index(settings.pinecone_index)

    namespace = PREFIX + signature[:16]
    counts = {name: summary.vector_count for name, summary in index.describe_index_stats().namespaces.items()}
    filled = counts.get(namespace, 0) > 0
    if not filled:
        # Chunks from an older version of the document or settings: no longer searched, so removed.
        for old in counts:
            if old.startswith(PREFIX) and old != namespace:
                index.delete(delete_all=True, namespace=old)
    return PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace), filled
