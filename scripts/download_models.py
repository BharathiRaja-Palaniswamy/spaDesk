"""Download the two local models (the embedder and the reranker) ahead of time.

    .venv/bin/python -m scripts.download_models

Without this, they are downloaded on the first question, which then waits
for about 170 MB, and fails in the chat if the network is not working. A host
runs this while building the app (see .replit), so the models are in
storage/models before anyone asks, and a download problem stops the build
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
