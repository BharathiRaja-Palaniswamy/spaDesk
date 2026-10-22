"""The one error the search models raise, wherever they run.

The embedder and the reranker run either on this machine (rag/onnx.py), when
the model files must first be downloaded, or on Pinecone's servers
(rag/hosted.py), which must be reached for every question. Either can fail
for reasons outside the app: no network, a refused download, a bad key, a
plan's limit reached. Both raise ModelUnavailable, so the chat routes catch
one error and answer "the assistant is unavailable" instead of crashing.
"""


class ModelUnavailable(RuntimeError):
    """A search model could not be loaded or reached. The chat routes turn
    this into "the assistant is unavailable"; the message says why, for the logs."""
