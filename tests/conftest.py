"""Set-up shared by every test file. pytest loads this before any test runs.

These are set before the app reads `.env`, so they win over whatever the
developer's `.env` says (load_dotenv never replaces a variable already set):

- Tracing is off: a test run should not send traces to LangSmith.
- The API key is a fake one: tests replace the model with stand-ins, and if
  one ever reached the real API it would be refused instead of spending
  tokens.
- The chats use the hand-built pipeline, whose model calls the tests replace
  most simply. Tests of the LangChain pipeline switch to it themselves.
- The model gateway uses the Claude set with its default models and no
  fallback, whatever `.env` says, so a test never reaches another provider.
- The vector store is the in-memory one, and the embedder and reranker are
  the local models, so a test never touches Pinecone.
- There is no Google key, so a test can never reach the real calendars; the
  booking tests use a stand-in calendar (standin_calendar.py).
- Debug mode is off, so replies have their usual shape; the debug test
  turns it on itself.
- The owner's password is a known test one; `signed_in()` gives a test
  browser that has signed in with it, for the owner's pages and data.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-a-real-one"
os.environ["SPADESK_RAG_ENGINE"] = "handbuilt"
os.environ["SPADESK_MODELS"] = "claude"
os.environ["SPADESK_CLAUDE_HIGH"] = "anthropic:claude-opus-5-5"
os.environ["SPADESK_CLAUDE_LOW"] = "anthropic:claude-haiku-4-5-20251001"
os.environ["SPADESK_MODEL_FALLBACK"] = ""
os.environ["SPADESK_VECTOR_STORE"] = "memory"
os.environ["SPADESK_EMBEDDING_PROVIDER"] = "minilm"
os.environ["SPADESK_RERANKER"] = "cross-encoder"
os.environ["SPADESK_DEBUG"] = "false"
os.environ["SPADESK_OWNER_PASSWORD"] = "test-owner-password"
os.environ["SPADESK_SESSION_SECRET"] = "test-session-secret"
os.environ["GOOGLE_SERVICE_ACCOUNT"] = ""


def signed_in(app):
    """A test browser for `app` that has signed in as the owner."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/api/login", json={"password": os.environ["SPADESK_OWNER_PASSWORD"]}).raise_for_status()
    return client
