"""Embeddings from all-MiniLM-L6-v2, run on this machine (rag/onnx.py).

A small model (about 90 MB) that turns a text into 384 numbers standing for
its meaning. No API key, no network call per text; downloaded once.

How a text becomes one vector:
  1. the model gives a vector for every token of the text
  2. the token vectors are averaged ("mean pooling"), padding left out
  3. the average is scaled to length 1, so vectors compare by direction alone

It reads at most 256 tokens of a text and ignores the rest, so a chunk and
its header must stay shorter than that (see rag/headers.py).
"""

import numpy as np

from ..onnx import load

REPO, MODEL_FILE = "Qdrant/all-MiniLM-L6-v2-onnx", "model.onnx"


class MiniLMEmbedder:
    """An Embedder (see base.py) backed by all-MiniLM-L6-v2."""

    name = "onnx/all-MiniLM-L6-v2"
    max_tokens = 256

    def __init__(self) -> None:
        """Load the model, downloading it the first time."""
        self._model = load(REPO, MODEL_FILE, self.max_tokens)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        token_vectors, mask = self._model.run(texts)  # (texts, tokens, 384), (texts, tokens)
        weights = mask[:, :, None].astype(np.float32)  # 1 for real tokens, 0 for padding
        averaged = (token_vectors * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        unit = averaged / np.linalg.norm(averaged, axis=1, keepdims=True)
        # Plain lists keep the interface free of any one library's types.
        return unit.tolist()
