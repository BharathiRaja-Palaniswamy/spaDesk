"""Step 4b of RAG: reorder the retrieved chunks with a reranker.

`get_reranker()` is the only way the rest of the app gets a reranker. Which
one it returns is a setting, SPADESK_RERANKER:

  pinecone       bge-reranker-v2-m3, run on Pinecone's servers (the default); needs PINECONE_API_KEY
  cross-encoder  ms-marco-MiniLM-L-6-v2, run on this machine; downloaded once, no key
  none           no reranking: the first search's order is kept

To add a reranker: write a class with `name` and `scores` in its own file
here (see base.py), and add one line to PROVIDERS below.
"""

from functools import lru_cache

from ...config import get_settings
from .base import Reranker
from .cross_encoder import CrossEncoderReranker
from .pinecone import PineconeReranker

# Setting value -> the class to use.
PROVIDERS: dict[str, type[Reranker]] = {
    "pinecone": PineconeReranker,
    "cross-encoder": CrossEncoderReranker,
}
OFF = "none"


@lru_cache  # created once: loading a local model is slow
def get_reranker() -> Reranker | None:
    """The reranker named by SPADESK_RERANKER, or None when it is "none"."""
    provider = get_settings().reranker
    if provider == OFF:
        return None
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown reranker: {provider!r}. Choose one of: {', '.join([*PROVIDERS, OFF])}.")
    return PROVIDERS[provider]()
