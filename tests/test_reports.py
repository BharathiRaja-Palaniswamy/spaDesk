"""Tests for the reports' data layer: the bookings file joined to the business document.

Run with:  .venv/bin/pytest tests/test_reports.py
"""

import json
from datetime import date

import pytest

from spadesk.business.catalog import entry_ids, service_catalog
from spadesk.business.store import get_knowledge
from spadesk.config import get_settings
from spadesk.reports.dataset import Dataset, load


@pytest.fixture(scope="module")
def dataset():
    """The real bookings file joined to the real business document, loaded once."""
    return load(get_settings().report_data_path)


def test_bookings_file_holds_no_business_facts():
    data = json.loads(get_settings().report_data_path.read_text())
    assert set(data) == {"meta", "channels", "customers", "bookings"}
    # A booking names its service by id only; the name and category are not stored with it.
    assert "service_id" in data["bookings"][0]
    assert not {"service_name", "service_category"} & set(data["bookings"][0])


def test_services_come_from_the_business_document(dataset):
    catalog = {s["id"]: s for s in service_catalog(get_knowledge())}
    assert dataset.services == catalog
    swedish = dataset.services["swedish-massage"]
    assert (swedish["name"], swedish["category"], swedish["price"], swedish["duration_minutes"]) == (
        "Swedish Massage", "Massage Therapy", 115, 60)
    assert "monthly-wellness-membership" not in catalog  # paid for, but not a booking
    assert len(dataset.bookings) > 2000


def test_filter_options(dataset):
    options = dataset.filter_options()
    assert (options["min_date"], options["max_date"]) == ("2025-10-01", "2026-09-30")  # from the bookings file
    assert options["categories"][0] == "Facials"  # from the business document, in its order
    assert "Website" in options["channels"]


def test_no_filters_returns_everything(dataset):
    assert len(dataset.select()) == len(dataset.bookings)


def test_date_range_includes_both_ends(dataset):
    september = dataset.select(start=date(2026, 9, 1), end=date(2026, 9, 30))
    assert {b["date"][:7] for b in september} == {"2026-09"}
    one_day = dataset.select(start=date(2026, 9, 30), end=date(2026, 9, 30))
    assert one_day and {b["date"] for b in one_day} == {"2026-09-30"}


def test_category_and_channel_filters_combine(dataset):
    rows = dataset.select(category="Facials", channel="Website")
    assert rows
    assert all(dataset.services[b["service_id"]]["category"] == "Facials" and b["channel"] == "Website" for b in rows)
    assert len(rows) < len(dataset.select(category="Facials"))


def test_booking_for_a_service_missing_from_the_document_is_reported(tmp_path):
    data = json.loads(get_settings().report_data_path.read_text())
    data["bookings"] = data["bookings"][:1]
    data["bookings"][0]["service_id"] = "hot-stone-massage"
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hot-stone-massage.*generate_bookings"):
        load(broken)


def test_service_renamed_in_the_document_is_reported():
    # The same thing seen from the other side: the bookings are fine, but the
    # document no longer has one of the services they point at.
    kb = get_knowledge()
    catalog = [s for s in service_catalog(kb) if s["id"] != "swedish-massage"]
    with pytest.raises(ValueError, match="swedish-massage"):
        Dataset(get_settings().report_data_path, catalog, entry_ids(kb))
