"""Tests for the report routes in spadesk/reports/routes.py.

Run with:  .venv/bin/pytest tests/test_report_api.py
"""

import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from spadesk.app import app
from spadesk.rag.stores import get_store
from spadesk.reports import routes
from spadesk.reports.tools import run_report


@pytest.fixture(scope="module")
def client():
    """A test browser for the app: makes requests without starting a server."""
    return TestClient(app)


def test_options(client):
    options = client.get("/api/report/options").json()
    assert options["business"] == "Pebble Spa"
    assert (options["min_date"], options["max_date"]) == ("2025-10-01", "2026-09-30")
    assert "Massage Therapy" in options["categories"] and "Referral" in options["channels"]


def test_report_defaults_to_the_whole_period(client):
    report = client.get("/api/report").json()
    assert report["filters"] == {"start": "2025-10-01", "end": "2026-09-30", "category": None, "channel": None}
    assert report["summary"]["bookings"] == 2490
    assert report["trend"]["bucket"] == "month" and len(report["trend"]["points"]) == 12
    # Nothing exists before the data starts, so there is nothing to compare with.
    assert report["previous_period"] is None and report["change"] is None
    assert set(report) >= {"categories", "services", "channels", "customers", "feedback"}


def test_report_for_a_date_range(client):
    report = client.get("/api/report", params={"start": "2026-09-01", "end": "2026-09-30"}).json()
    assert report["summary"]["bookings"] == 218
    assert report["trend"]["bucket"] == "day" and len(report["trend"]["points"]) == 30
    assert report["previous_period"] == {"start": "2026-08-02", "end": "2026-08-31"}
    assert report["change"]["revenue"] is not None


def test_filters_narrow_the_report(client):
    everything = client.get("/api/report").json()["summary"]["bookings"]
    facials = client.get("/api/report", params={"category": "Facials"}).json()
    both = client.get("/api/report", params={"category": "Facials", "channel": "Website"}).json()
    assert [c["category"] for c in facials["categories"]] == ["Facials"]
    assert both["summary"]["bookings"] < facials["summary"]["bookings"] < everything


def test_report_includes_previous_summary_for_comparisons(client):
    september = client.get("/api/report", params={"start": "2026-09-01", "end": "2026-09-30"}).json()
    assert september["previous_summary"]["bookings"] > 0
    assert client.get("/api/report").json()["previous_summary"] is None


def test_manage_page_and_its_files_are_served(client):
    page = client.get("/manage")
    assert page.status_code == 200
    assert client.get("/dashboard").status_code == 404  # the old address is gone
    # Both tabs and their panels are in the page.
    for name in ("profile", "reports"):
        assert f'data-tab="{name}"' in page.text and f'id="panel-{name}"' in page.text
    # Every stylesheet and script the page links to must exist.
    for path in re.findall(r'(?:href|src)="(/static/dashboard/[^"]+)"', page.text):
        response = client.get(path)
        assert response.status_code == 200, path
        # The browser is told to re-check each file, so edits show up on refresh.
        assert response.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize("params", [
    {"start": "2026-09-30", "end": "2026-09-01"},  # range the wrong way round
    {"start": "last week"},                        # not a date
    {"category": "Haircuts"},                      # not in the data file
    {"channel": "Billboard"},
])
def test_bad_filters_are_rejected(client, params):
    assert client.get("/api/report", params=params).status_code == 422


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: records the request, returns a fixed reply."""

    last_request = None
    # What the "model" answers. A test can replace it; None means no structured reply.
    reply = routes.ChatReply(answer="Cancel 24 hours ahead.", covered=True,
                             items=[routes.ReplyItem(name="Late cancellations", detail="50% fee")],
                             insights=[], sections=["Cancellation Policy", "Policies", "Made-up Section"],
                             suggestions=["One?", "Two?", "Three?", "Four?"])

    def __init__(self):
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **request):
        FakeAnthropic.last_request = request
        # stop_reason "end_turn": the model gave its answer and did not ask for a tool.
        return SimpleNamespace(parsed_output=FakeAnthropic.reply and FakeAnthropic.reply.model_copy(),
                               stop_reason="end_turn", content=[])


def test_chat_returns_a_structured_reply(client, monkeypatch):
    monkeypatch.setattr(routes.anthropic, "Anthropic", FakeAnthropic)
    reply = client.post("/api/report/chat", json={"question": "What is our cancellation policy?"})
    # Every field reaches the browser. Of the three sources the "model" named,
    # one survives: the made-up one is not a heading, and "Policies" is only
    # the parent of "Cancellation Policy", which is already there.
    assert reply.json() == {"answer": "Cancel 24 hours ahead.", "covered": True,
                            "items": [{"name": "Late cancellations", "detail": "50% fee"}], "insights": [],
                            "sections": ["Cancellation Policy"],
                            "suggestions": ["One?", "Two?", "Three?"],  # at most three are passed on
                            "rag": False, "chunks": 0,  # answered from the whole document
                            "reports": []}  # and without asking for report figures

    sent = FakeAnthropic.last_request
    assert sent["messages"] == [{"role": "user", "content": "What is our cancellation policy?"}]
    assert sent["output_format"] is routes.ChatReply  # the model is told which shape to reply in
    # The model is given the document itself, its headings and the current time.
    assert "<profile>" in sent["system"] and "Late cancellations may incur a 50% service fee." in sent["system"]
    assert "<sections>" in sent["system"] and "Frequently Asked Questions\nPolicies\nBooking Policy" in sent["system"]
    assert "<currentTime>" in sent["system"]
    # The model is offered the report tool, with the valid filter values spelled out.
    tool = sent["tools"][0]
    assert tool["name"] == "get_report" and "2025-10-01 to 2026-09-30" in tool["description"]
    assert "Facials" in tool["input_schema"]["properties"]["category"]["enum"]


def test_chat_with_rag_sends_only_the_retrieved_chunks(client, monkeypatch, tmp_path):
    monkeypatch.setattr(routes.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("SPADESK_VECTOR_STORE_PATH", str(tmp_path / "chroma"))  # an empty store for this test
    monkeypatch.setenv("SPADESK_RETRIEVAL_TOP_K", "3")  # a known number, whatever `.env` says
    get_store.cache_clear()
    reply = client.post("/api/report/chat", json={"question": "What is our cancellation policy?", "rag": True})
    get_store.cache_clear()
    assert (reply.json()["rag"], reply.json()["chunks"]) == (True, 3)
    assert reply.json()["sections"] == ["Cancellation Policy"]

    system = FakeAnthropic.last_request["system"]
    # The model is told it is seeing excerpts, and gets the three that match
    # the question, not the whole document: the opening hours are not there.
    assert "each between <excerpt> tags" in system and system.count("<excerpt>\n") == 3
    assert "Late cancellations may incur a 50% service fee." in system
    assert "Sunday: 10:00 AM" not in system
    # Only headings the excerpts include are offered as sources.
    offered = system.split("<sections>\n")[1].split("\n</sections>")[0].splitlines()
    assert "Cancellation Policy" in offered and "Facials" not in offered


class ToolThenAnswer:
    """Stands in for a model that first asks for a report, then answers.
    `requests` keeps what it was sent each time.
    """

    requests: list = []

    def __init__(self):
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **request):
        # Copy the conversation: the route keeps adding to the same list.
        ToolThenAnswer.requests.append({**request, "messages": list(request["messages"])})
        if len(ToolThenAnswer.requests) == 1:
            ask = SimpleNamespace(type="tool_use", id="call-1", name="get_report",
                                  input={"start": "2026-09-01", "end": "2026-09-30", "category": "Facials"})
            return SimpleNamespace(parsed_output=None, stop_reason="tool_use", content=[ask])
        answer = routes.ChatReply(answer="Facials in September 2026.", covered=True, sections=[], suggestions=[],
                                  items=[routes.ReplyItem(name="Revenue", detail="$5,508")],
                                  insights=["Facials were the second-highest earning category."])
        return SimpleNamespace(parsed_output=answer, stop_reason="end_turn", content=[])


def test_chat_runs_the_report_tool_when_the_model_asks(client, monkeypatch):
    monkeypatch.setattr(routes.anthropic, "Anthropic", ToolThenAnswer)
    monkeypatch.setattr(ToolThenAnswer, "requests", [])
    reply = client.post("/api/report/chat", json={"question": "How did facials do in September?"}).json()
    # A report answer has three parts: a sentence, the figures as points, then insights.
    assert reply["answer"] == "Facials in September 2026."
    assert reply["items"] == [{"name": "Revenue", "detail": "$5,508"}]
    assert reply["insights"] == ["Facials were the second-highest earning category."]
    assert reply["reports"] == ["2026-09-01 to 2026-09-30, Facials"]  # shown under the answer

    # The model was called twice. The second time it was sent its own request
    # back, followed by the tool's result: the real figures for that filter.
    first, second = ToolThenAnswer.requests
    assert len(first["messages"]) == 1 and len(second["messages"]) == 3
    request, result = second["messages"][1], second["messages"][2]
    assert request["role"] == "assistant" and result["role"] == "user"
    block = result["content"][0]
    assert (block["type"], block["tool_use_id"]) == ("tool_result", "call-1")
    report = json.loads(block["content"])
    assert report["filters"]["category"] == "Facials" and report["summary"]["revenue"] == 5508


def test_the_report_tool_explains_what_is_wrong_with_its_inputs():
    dataset = routes.get_dataset()
    assert "no data for that period" in run_report(dataset, {"start": "2026-10-01", "end": "2026-10-31"})["error"]
    assert "start and end are required" in run_report(dataset, {"start": "last month"})["error"]
    assert "Unknown category" in run_report(dataset, {"start": "2026-09-01", "end": "2026-09-30", "category": "Hats"})["error"]


def test_chat_reports_when_there_is_no_structured_reply(client, monkeypatch):
    monkeypatch.setattr(routes.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(FakeAnthropic, "reply", None)
    reply = client.post("/api/report/chat", json={"question": "Hello"})
    assert reply.status_code == 503 and "could not answer" in reply.json()["detail"]


def test_chat_reports_when_the_model_is_unavailable(client, monkeypatch):
    def no_key():
        raise TypeError("Could not resolve authentication method")

    monkeypatch.setattr(routes.anthropic, "Anthropic", no_key)
    reply = client.post("/api/report/chat", json={"question": "Hello"})
    assert reply.status_code == 503 and "unavailable" in reply.json()["detail"]
