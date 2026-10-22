"""Which pipeline answers the assistants' questions, from SPADESK_RAG_ENGINE.

  langchain  the default: LangChain (lc/), a chain for the customer chat and
             a LangGraph graph for the owner chat, calling models through
             the model gateway (spadesk/models/)
  handbuilt  the reference: the original code, rag/ for retrieval,
             assistant/model.py calling Claude directly, and a plain loop
             for the owner's report tool

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
