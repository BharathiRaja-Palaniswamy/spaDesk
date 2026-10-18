"""A reranker that runs on this machine: a small cross-encoder model.

"Cross-encoder" means the model reads the question and the chunk as one
input, so every word of the question can be weighed against every word of
the chunk. The embedder ("bi-encoder") turns each into a vector on its own
and only compares the vectors afterwards.

The model (ms-marco-MiniLM-L-6-v2, trained on web search questions and the
passages that answer them) runs through the fastembed library: no API key,
no network call per question. It is downloaded once, on first use, to
~/.cache/fastembed (about 80 MB). Scoring 20 chunks takes about 75 ms.
"""

from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# fastembed's default is a temporary folder, which the system may empty; this one is kept.
CACHE = Path.home() / ".cache" / "fastembed"


class CrossEncoderReranker:
    """A Reranker (see base.py) backed by a local cross-encoder."""

    name = f"fastembed/{MODEL.split('/')[1]}"

    def __init__(self) -> None:
        """Load the model, downloading it the first time."""
        self._model = TextCrossEncoder(MODEL, cache_dir=str(CACHE))

    def scores(self, question: str, texts: list[str]) -> list[float]:
        """One score per text, in the same order."""
        return [float(score) for score in self._model.rerank(question, texts)]
