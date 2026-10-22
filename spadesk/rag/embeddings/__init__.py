"""Step 2 of RAG: turn text into vectors (embeddings).

`get_embedder()` is the only way the rest of the app gets an embedder. Which
provider it returns is a setting, SPADESK_EMBEDDING_PROVIDER, so switching
provider is a change to `.env`, not to code.

To add a provider: write a class with `name`, `max_tokens` and `embed` in its
own file here (see base.py), and add one line to PROVIDERS below.

  fastembed  all-MiniLM-L6-v2 run by fastembed (the default)
"""

from collections.abc import Callable
from functools import lru_cache

from ...config import get_settings
from .base import Embedder
from .fastembed import FastEmbedEmbedder


# Setting value -> how to make that embedder.
PROVIDERS: dict[str, Callable[[], Embedder]] = {
    "fastembed": FastEmbedEmbedder,
}


@lru_cache  # created once: loading a model is slow
def get_embedder() -> Embedder:
    """The embedder named by the SPADESK_EMBEDDING_PROVIDER setting."""
    provider = get_settings().embedding_provider
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Choose one of: {', '.join(PROVIDERS)}.")
    return PROVIDERS[provider]()
