"""Tests for the local models' loader (spadesk/rag/onnx.py) and the startup
trust setting (spadesk/__init__.py), with no download and nothing spent.

Run with:  .venv/bin/python -m pytest tests/test_rag_onnx.py
"""

import ssl

import pytest
from conftest import signed_in
from fastapi.testclient import TestClient
from huggingface_hub.errors import LocalEntryNotFoundError

import spadesk  # noqa: F401  (importing the package applies the trust setting)
from spadesk.app import app
from spadesk.lc.embeddings import get_embeddings
from spadesk.rag import onnx
from spadesk.rag.embeddings import get_embedder


@pytest.fixture
def no_download(monkeypatch, tmp_path):
    """No model can be downloaded, as on a host whose network refuses it, and none is cached."""
    def refused(*args, **kwargs):
        raise LocalEntryNotFoundError("certificate verify failed (stand-in)")

    monkeypatch.setattr(onnx, "hf_hub_download", refused)
    monkeypatch.setenv("SPADESK_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "vectors"))
    for cached in (get_embedder, get_embeddings):
        cached.cache_clear()
    yield
    for cached in (get_embedder, get_embeddings):
        cached.cache_clear()


def test_python_uses_the_machines_own_certificates():
    # truststore replaces the standard SSL context with one backed by the system's trust store.
    assert type(ssl.create_default_context()).__module__.startswith("truststore")


def test_a_model_that_cannot_be_downloaded_raises_a_clear_error(no_download):
    with pytest.raises(onnx.ModelUnavailable, match="Could not download Qdrant/all-MiniLM-L6-v2-onnx"):
        onnx.load("Qdrant/all-MiniLM-L6-v2-onnx", "model.onnx", 256)


@pytest.mark.parametrize("route", ["/api/chat", "/api/report/chat"])
def test_both_chats_say_unavailable_instead_of_crashing(no_download, monkeypatch, route):
    monkeypatch.setenv("SPADESK_RAG_ENGINE", "langchain")
    client = signed_in(app) if route.startswith("/api/report") else TestClient(app)
    reply = client.post(route, json={"question": "Which channel brings the most bookings?"})
    assert reply.status_code == 503 and "unavailable" in reply.json()["detail"]
