"""Set-up shared by every test file. pytest loads this before any test runs.

Tracing is switched off for the tests, whatever the developer's `.env` says:
a test run should not send traces to LangSmith.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
