"""Tests for what the trace records (spadesk/tracing.py). Nothing is sent to
LangSmith: tests/conftest.py switches tracing off.

Run with:  .venv/bin/pytest tests/test_tracing.py
"""

from types import SimpleNamespace

from langsmith.utils import tracing_is_enabled

from spadesk.rag.chunking import Chunk
from spadesk.rag.stores.base import Match
from spadesk.reports.routes import ChatReply
from spadesk.tracing import model_messages, model_reply, retrieved_documents


def test_tracing_is_off_during_tests():
    assert not tracing_is_enabled()


def test_retrieved_chunks_are_recorded_as_documents():
    match = Match(Chunk("chunk-3", "Lip Wax — $18", 6, ("Hair Removal",)), score=0.61234)
    assert retrieved_documents([match]) == {"documents": [{
        "page_content": "Lip Wax — $18", "type": "Document",
        "metadata": {"id": "chunk-3", "score": 0.612, "sections": ["Hair Removal"], "tokens": 6}}]}


def test_the_prompt_is_recorded_as_chat_messages():
    question = {"role": "user", "content": "Hours?"}
    recorded = model_messages({"system": "You are the assistant.", "messages": [question], "tools": [{"name": "get_report"}]})
    assert recorded == {"messages": [{"role": "system", "content": "You are the assistant."}, question],
                        "tools": [{"name": "get_report"}]}


def test_the_reply_is_recorded_with_its_token_counts():
    reply = ChatReply(answer="10 to 5.", items=[], insights=[], covered=True,
                      sections=["Location and Hours"], suggestions=[])
    usage = SimpleNamespace(input_tokens=2100, output_tokens=40)
    response = SimpleNamespace(parsed_output=reply, usage=usage, content=[])
    recorded = model_reply(response)
    assert recorded["reply"]["answer"] == "10 to 5."
    assert recorded["usage_metadata"] == {"input_tokens": 2100, "output_tokens": 40, "total_tokens": 2140}
    assert recorded["tool_requests"] == []

    # When the model asks for a tool there is no reply yet; the request is recorded.
    ask = SimpleNamespace(type="tool_use", name="get_report", input={"start": "2026-09-01", "end": "2026-09-30"})
    recorded = model_reply(SimpleNamespace(parsed_output=None, usage=usage, content=[ask]))
    assert recorded["reply"] is None
    assert recorded["tool_requests"] == [{"name": "get_report", "input": {"start": "2026-09-01", "end": "2026-09-30"}}]
