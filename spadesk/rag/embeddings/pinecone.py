"""Embeddings from llama-text-embed-v2, run on Pinecone's servers (rag/hosted.py).

Chosen with SPADESK_EMBEDDING_PROVIDER=pinecone (the default). It needs
PINECONE_API_KEY, the same key the vector store uses.

Two things differ from the local MiniLM model:

- Questions and chunks are embedded slightly differently. The model was
  trained on (question, passage) pairs, and is told which one each text is
  ("input_type"), so a short question lands near the passage that answers
  it. That is why `embed` takes `query`.
- The model can give vectors of several sizes. 384 numbers is the size
  MiniLM gives, so the same Pinecone index fits either model; the index's
  signature holds the model's name, so the chunks are re-embedded into a new
  namespace when the model changes, and the old one is removed.
"""

from .. import hosted

MODEL, DIMENSION = "llama-text-embed-v2", 384
BATCH = 96  # the most texts Pinecone embeds in one request for this model


class PineconeEmbedder:
    """An Embedder (see base.py) backed by a model on Pinecone's servers."""

    name = f"pinecone/{MODEL}-{DIMENSION}"
    max_tokens = 2048

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """One vector per text, in the same order. `query` says the texts are
        questions rather than chunks of the document."""
        parameters = {"input_type": "query" if query else "passage", "dimension": DIMENSION}
        vectors = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start:start + BATCH]
            result = hosted.call(MODEL, lambda pc: pc.inference.embed(model=MODEL, inputs=batch,
                                                                       parameters=parameters))
            vectors += [list(item["values"]) for item in result]
        return vectors
