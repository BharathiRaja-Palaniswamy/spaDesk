"""The Pinecone vector store for the LangChain pipeline: a hosted service.

An example of a second provider behind the same interface: choose it with
SPADESK_VECTOR_STORE=pinecone. It needs:
  - the package:  uv pip install --python .venv/bin/python langchain-pinecone
  - an API key:   PINECONE_API_KEY in .env (read by the Pinecone library itself)
  - an index name: SPADESK_PINECONE_INDEX (default "spadesk")

The index is created on first use if it does not exist: serverless, on AWS
us-east-1, compared by cosine, sized to the embedding model's vectors.

Not yet run against a real Pinecone account: written against the library's
documented interface (langchain-pinecone 0.2, pinecone 7).
"""

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ...config import get_settings

NAMESPACE = "business_doc"  # a section of the index, like a Chroma collection
CLOUD, REGION = "aws", "us-east-1"


def make_pinecone(embeddings: Embeddings) -> VectorStore:
    """The Pinecone index, created on first use."""
    # Imported here, not at the top, so the package is only needed when Pinecone is chosen.
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    name = get_settings().pinecone_index
    client = Pinecone()  # reads PINECONE_API_KEY
    if not client.has_index(name):
        dimension = len(embeddings.embed_query("dimension check"))  # 384 for MiniLM
        client.create_index(name=name, dimension=dimension, metric="cosine",
                            spec=ServerlessSpec(cloud=CLOUD, region=REGION))
    return PineconeVectorStore(index=client.Index(name), embedding=embeddings, namespace=NAMESPACE)
