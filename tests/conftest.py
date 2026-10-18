"""Set-up shared by every test file. pytest loads this before any test runs.

These are set before the app reads `.env`, so they win over whatever the
developer's `.env` says (load_dotenv never replaces a variable already set):

- Tracing is off: a test run should not send traces to LangSmith.
- The API key is a fake one: tests replace the model with stand-ins, and if
  one ever reached the real API it would be refused instead of spending
  tokens.
- The customer chat uses the hand-built pipeline, the default. Tests of the
  LangChain pipeline switch to it themselves (tests/test_lc_chat.py).
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-a-real-one"
os.environ["SPADESK_RAG_ENGINE"] = "handbuilt"
