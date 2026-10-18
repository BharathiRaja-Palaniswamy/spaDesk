"""Step 4b of RAG: reorder the retrieved chunks with a reranker.

`get_reranker()` is the only way the rest of the app gets a reranker. Which
one it returns is a setting, SPADESK_RERANKER: "cross-encoder" (the default)
or "none" to switch reranking off and keep the first search's order.

To add a reranker: write a class with `name` and `scores` in its own file
here (see base.py), and add one line to PROVIDERS below.
"""

from functools import lru_cache

from ...config import get_settings
from .base import Reranker
from .cross_encoder import CrossEncoderReranker

# Setting value -> the class to use.
PROVIDERS: dict[str, type[Reranker]] = {
    "cross-encoder": CrossEncoderReranker,
}
OFF = "none"


@lru_cache  # created once: loading a model is slow
def get_reranker() -> Reranker | None:
    """The reranker named by SPADESK_RERANKER, or None when it is "none"."""
    provider = get_settings().reranker
    if provider == OFF:
        return None
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown reranker: {provider!r}. Choose one of: {', '.join([*PROVIDERS, OFF])}.")
    return PROVIDERS[provider]()
