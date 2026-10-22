"""Small models run on this machine with ONNX Runtime.

Both the embedder and the reranker are small transformer models published in
the ONNX format, a file that holds a trained model ready to run. Running one
takes two libraries:

  tokenizers   turns text into the numbers (token ids) the model reads
  onnxruntime  runs the model file on those numbers

This is what libraries such as fastembed do inside. Doing it here keeps the
app's dependencies to those two (fastembed also pulls in an image library
the app never uses), and shows each step.

The files are downloaded once, on first use, from Hugging Face (a public
host for models) to ~/.cache/spadesk-models, then read from there.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

CACHE = Path.home() / ".cache" / "spadesk-models"


@dataclass(frozen=True)
class LocalModel:
    """A model file and its tokenizer, loaded and ready to run."""

    session: onnxruntime.InferenceSession  # runs the model
    tokenizer: Tokenizer  # text -> token ids

    def run(self, texts: list[str] | list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize `texts` (single texts, or (first, second) pairs) and run the
        model on them in one batch. Returns the model's first output and the
        attention mask (1 for a real token, 0 for padding)."""
        encoded = self.tokenizer.encode_batch(texts)
        # Texts have different lengths; padding makes them one rectangle of numbers.
        columns = {"input_ids": [e.ids for e in encoded],
                   "attention_mask": [e.attention_mask for e in encoded],
                   "token_type_ids": [e.type_ids for e in encoded]}  # which text of a pair a token is from
        wanted = {i.name for i in self.session.get_inputs()}
        inputs = {name: np.array(values, dtype=np.int64) for name, values in columns.items() if name in wanted}
        return self.session.run(None, inputs)[0], inputs["attention_mask"]


def load(repo: str, model_file: str, max_tokens: int) -> LocalModel:
    """Download (the first time) and load a model from the Hugging Face
    repository `repo`. Texts longer than `max_tokens` are cut to that length,
    the most the model reads."""
    model_path = hf_hub_download(repo, model_file, cache_dir=CACHE)
    tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", cache_dir=CACHE))
    tokenizer.enable_truncation(max_length=max_tokens)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return LocalModel(session, tokenizer)
