"""Set-up shared by every test file. pytest loads this before any test runs.

These are set before the app reads `.env`, so they win over whatever the
developer's `.env` says (load_dotenv never replaces a variable already set):

- Tracing is off: a test run should not send traces to LangSmith.
- The API key is a fake one: tests replace the model with stand-ins, and if
  one ever reached the real API it would be refused instead of spending
  tokens.
- The chats use the hand-built pipeline, whose model calls the tests replace
  most simply. Tests of the LangChain pipeline switch to it themselves.
- The model gateway uses Claude with no fallback, whatever `.env` says, so a
  test never reaches another provider.
- The vector store is the in-memory one, so a test never touches Pinecone.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-a-real-one"
os.environ["SPADESK_RAG_ENGINE"] = "handbuilt"
os.environ["SPADESK_MODEL_PROVIDER"] = "anthropic"
os.environ["SPADESK_MODEL"] = "claude-opus-5-5"
os.environ["SPADESK_MODEL_FALLBACK"] = ""
os.environ["SPADESK_VECTOR_STORE"] = "memory"
os.environ["SPADESK_EMBEDDING_PROVIDER"] = "minilm"
