"""A reranker that runs on this machine: a small cross-encoder model.

"Cross-encoder" means the model reads the question and the chunk as one
input, so every word of the question can be weighed against every word of
the chunk. The embedder ("bi-encoder") turns each into a vector on its own
and only compares the vectors afterwards.

The model (ms-marco-MiniLM-L-6-v2, trained on web search questions and the
passages that answer them) runs on this machine (rag/onnx.py): no API key,
no network call per question. It is downloaded once, on first use (about
80 MB). Scoring 20 chunks takes about 75 ms.
"""

from ..onnx import load

REPO, MODEL_FILE = "Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx"
MAX_TOKENS = 512  # question and chunk together; the model reads no more


class CrossEncoderReranker:
    """A Reranker (see base.py) backed by a local cross-encoder."""

    name = "onnx/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        """Load the model, downloading it the first time."""
        self._model = load(REPO, MODEL_FILE, MAX_TOKENS)

    def scores(self, question: str, texts: list[str]) -> list[float]:
        """One score per text, in the same order: higher means a better answer
        to `question`. The model reads each (question, text) pair together and
        gives one number for it."""
        logits, _ = self._model.run([(question, text) for text in texts])
        return [float(score) for score in logits[:, 0]]
