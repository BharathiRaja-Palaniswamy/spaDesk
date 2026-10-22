"""Models run on Pinecone's servers ("Pinecone Inference").

Pinecone, which already keeps the vectors, also runs embedding and reranking
models: the app sends text, Pinecone sends back vectors or scores. Nothing is
downloaded and nothing runs on this machine, which suits a host that cannot
download the local models (rag/onnx.py), at the cost of one network call per
use, counted against the Pinecone plan's monthly allowance.

Both the embedder (embeddings/pinecone.py) and the reranker
(rerankers/pinecone.py) call Pinecone through `call` below, so they share
one key check and one way of reporting failures.
"""

from collections.abc import Callable
from typing import TypeVar

import urllib3
from pinecone import Pinecone
from pinecone.exceptions import PineconeException

from ..config import get_settings
from .errors import ModelUnavailable

T = TypeVar("T")


def client() -> Pinecone:
    """A Pinecone client with the key from PINECONE_API_KEY."""
    key = get_settings().pinecone_api_key
    if not key:
        raise ModelUnavailable("A Pinecone model is chosen but PINECONE_API_KEY is not set. Add it to .env "
                               "(or the host's secrets), or choose a local model (see .env.example).")
    return Pinecone(api_key=key)


def call(model: str, request: Callable[[Pinecone], T]) -> T:
    """Run `request` with a client, turning any Pinecone or network failure
    into ModelUnavailable, which names the model and says what went wrong."""
    try:
        return request(client())
    except (PineconeException, urllib3.exceptions.HTTPError) as error:
        raise ModelUnavailable(f"Pinecone model {model} could not be reached: {error}") from error
