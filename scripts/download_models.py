"""Download the two local models (the embedder and the reranker) ahead of time.

    .venv/bin/python -m scripts.download_models

Only needed when the local models are chosen (SPADESK_EMBEDDING_PROVIDER=minilm,
SPADESK_RERANKER=cross-encoder); the default ones run on Pinecone's servers
and there is nothing to download. Without this, the local models are
downloaded on the first question, which then waits for about 170 MB, and
fails in the chat if the network is not working. Running it first puts them
in storage/models before anyone asks, and shows a download problem here
with a clear message instead.
"""

from spadesk.rag.embeddings import get_embedder
from spadesk.rag.rerankers import get_reranker


def main() -> None:
    """Load both models; loading downloads them when they are not there yet."""
    print("embedder:", get_embedder().name)
    reranker = get_reranker()
    print("reranker:", reranker.name if reranker else "off (SPADESK_RERANKER=none)")
    print("Models ready.")


if __name__ == "__main__":
    main()
