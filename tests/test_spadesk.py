"""Tests for the customer site: the business document, booking requests and routes.

Run with:  .venv/bin/pytest tests/test_spadesk.py
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from spadesk import app as app_module
from spadesk.booking.service import BookingDesk, BookingError, BookingRequest
from spadesk.config import ABOUT_SECTION, FEATURED, HIGHLIGHTS, HOURS_BLOCK, ROOT
from spadesk.business.knowledge import KnowledgeBase
from spadesk.business.profile import build_profile


# Fixtures are shared set-up. scope="module" builds each one once for the whole file.
@pytest.fixture(scope="module")
def kb():
    """The real business document, parsed."""
    return KnowledgeBase(ROOT / "data" / "business_doc.md")


# ---- Business document ----
def test_document_is_read_into_services_and_facts(kb):
    assert kb.name == "Pebble Spa"
    assert kb.facts["Phone"] == "(408) 555-0188"
    swedish = next(i for i in kb.items if i.name == "Swedish Massage")
    assert (swedish.category, swedish.get("price"), swedish.get("duration")) == (
        "Massage Therapy", "$115 / $160", "60 minutes / 90 minutes")
    assert "Lip Wax" in kb.service_names  # a one-line "**Name** — $price" entry


def test_pieces_can_be_found_by_title(kb):
    assert kb.find("Cancellation Policy").lines[0].startswith("Cancellations or rescheduling")
    assert kb.find("Business Hours").lines[-1] == "Sunday: 10:00 AM – 5:00 PM"
    assert kb.find("No Such Heading") is None


def test_public_data_for_the_landing_page(kb):
    spa = kb.public(FEATURED, HIGHLIGHTS, HOURS_BLOCK, ABOUT_SECTION)
    assert spa["name"] == "Pebble Spa" and spa["about"].startswith("Pebble Spa is a neighborhood wellness spa")
    assert spa["hours"][0] == {"days": "Monday to Thursday", "times": "10:00 AM – 8:00 PM"}
    assert spa["featured"] == list(FEATURED)
    assert [h["title"] for h in spa["highlights"]] == list(HIGHLIGHTS)


# ---- Business profile (manage page) ----
def test_profile_follows_the_document(kb):
    profile = build_profile(kb)
    titles = [s["title"] for s in profile["sections"]]
    assert titles[:3] == ["About", "Mission", "Location and Hours"]
    assert titles[-1] == "Seasonal Promotions"
    section = {s["title"]: s for s in profile["sections"]}

    # Sub-headings become groups, and services keep their duration and price.
    services = section["Core Services"]["groups"]
    assert [g["title"] for g in services] == ["Facials", "Massage Therapy", "Body Treatments", "Hair Removal", "Kids Services"]
    swedish = next(i for i in services[1]["items"] if i["name"] == "Swedish Massage")
    assert (swedish["duration"], swedish["price"]) == ("60 minutes / 90 minutes", "$115 / $160")
    # One-line entries stay as lines and are not repeated as items.
    assert "Eyebrow Wax — $22" in services[3]["lines"] and services[3]["items"] == []

    # Plain text keeps whether it was a bullet list.
    policies = {g["title"]: g for g in section["Policies"]["groups"]}
    assert policies["Refund Policy"]["bulleted"] and len(policies["Refund Policy"]["lines"]) == 2
    assert not section["Mission"]["groups"][0]["bulleted"]

    # An entry without a price carries its own text.
    faq = section["Frequently Asked Questions"]["groups"][0]["items"][0]
    assert faq["name"] == "Do you offer services for children?" and faq["lines"][0].startswith("Yes.")
    hours = section["Location and Hours"]["groups"][0]
    assert "Phone: (408) 555-0188" in hours["lines"]
    assert hours["items"][0]["name"] == "Business Hours" and len(hours["items"][0]["lines"]) == 4


# ---- Booking ----
def booking(**changes):
    """A valid booking request, with any fields overridden."""
    fields = dict(service="Signature Radiance Facial", preferred_date=date.today() + timedelta(days=3),
                  name="Asha", phone="555 010 0100")
    return BookingRequest(**{**fields, **changes})


def test_booking_is_a_request_not_a_confirmation(tmp_path, kb):
    desk = BookingDesk(tmp_path / "requests.jsonl", kb.service_names)
    result = desk.submit(booking())
    assert result["status"] == "requested"
    assert "not a confirmed appointment" in result["message"]
    assert result["reference"] in (tmp_path / "requests.jsonl").read_text()


def test_booking_rejects_unknown_service_and_past_date(tmp_path, kb):
    desk = BookingDesk(tmp_path / "requests.jsonl", kb.service_names)
    with pytest.raises(BookingError):
        desk.submit(booking(service="Hot Stone Massage"))
    with pytest.raises(BookingError):
        desk.submit(booking(preferred_date=date.today() - timedelta(days=1)))


# ---- HTTP routes ----
def test_api(tmp_path, kb, monkeypatch):
    monkeypatch.setattr(app_module, "get_booking_desk",
                        lambda: BookingDesk(tmp_path / "requests.jsonl", kb.service_names))
    client = TestClient(app_module.app)

    assert client.get("/").status_code == 200
    spa = client.get("/api/spa").json()
    assert spa["name"] == "Pebble Spa"
    assert spa["contact"]["Phone"] == "(408) 555-0188"
    assert len(spa["hours"]) == 4 and len(spa["featured"]) == 4 and len(spa["highlights"]) == 4
    assert {"name": "Lip Wax", "category": "Hair Removal", "description": "", "duration": "", "price": "$18"} in spa["services"]

    assert client.get("/api/manage/profile").json()["name"] == "Pebble Spa"

    ok = client.post("/api/booking-requests", json=booking().model_dump(mode="json"))
    assert ok.json()["status"] == "requested"
    bad = client.post("/api/booking-requests", json=booking(service="Nope").model_dump(mode="json"))
    assert bad.status_code == 422
