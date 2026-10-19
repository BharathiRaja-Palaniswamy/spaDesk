"""Which pipeline answers the assistants' questions, from SPADESK_RAG_ENGINE.

  handbuilt  the original code: rag/ for retrieval, assistant/model.py for the
             model call, and a plain loop for the owner's report tool
  langchain  the same steps with LangChain (lc/): a chain for the customer
             chat, a LangGraph graph for the owner chat

Both assistants read it here, so a typo in the setting fails the same way for both.
"""

from ..config import get_settings

ENGINES = ("handbuilt", "langchain")


def rag_engine() -> str:
    """The engine named by SPADESK_RAG_ENGINE. Raises ValueError for an unknown one."""
    engine = get_settings().rag_engine
    if engine not in ENGINES:
        raise ValueError(f"Unknown SPADESK_RAG_ENGINE: {engine!r}. Choose one of: {', '.join(ENGINES)}.")
    return engine
