"""A reranker run on Pinecone's servers (rag/hosted.py): bge-reranker-v2-m3.

Chosen with SPADESK_RERANKER=pinecone (the default). Like the local
cross-encoder (cross_encoder.py), it reads the question and each chunk
together and scores how well the chunk answers it; it is a larger model
(568M parameters, against 22M), and reads up to 1024 tokens of each pair.

Pinecone returns the chunks best first, each with its position in the list
it was sent; `scores` puts the scores back in the order the chunks came in,
as every Reranker does.
"""

from .. import hosted

MODEL = "bge-reranker-v2-m3"


class PineconeReranker:
    """A Reranker (see base.py) backed by a model on Pinecone's servers."""

    name = f"pinecone/{MODEL}"

    def scores(self, question: str, texts: list[str]) -> list[float]:
        """One score per text, in the same order: higher means a better answer to `question`."""
        result = hosted.call(MODEL, lambda pc: pc.inference.rerank(
            model=MODEL, query=question, documents=texts, return_documents=False))
        by_position = {item["index"]: float(item["score"]) for item in result.data}
        return [by_position[position] for position in range(len(texts))]
