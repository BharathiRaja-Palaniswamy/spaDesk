"""Tests for the report dashboard's data layer.

Run with:  .venv/bin/pytest tests/test_reports.py
"""

import json
from datetime import date

import pytest

from spadesk.config import get_settings
from spadesk.reports.dataset import Dataset


@pytest.fixture(scope="module")
def dataset():
    """The real bookings file, loaded once for all tests here."""
    return Dataset(get_settings().report_data_path)


def test_loads_reference_lists_and_bookings(dataset):
    assert dataset.business["name"] == "Family Spa"
    assert len(dataset.bookings) > 2000
    swedish = dataset.services["swedish-massage"]
    assert (swedish["name"], swedish["category"], swedish["price"]) == ("Swedish Massage", "Massage Therapy", 115)


def test_filter_options_come_from_the_file(dataset):
    options = dataset.filter_options()
    assert (options["min_date"], options["max_date"]) == ("2025-10-01", "2026-09-30")
    assert "Facials" in options["categories"] and "Website" in options["channels"]


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


def test_broken_reference_is_reported(tmp_path):
    data = json.loads(get_settings().report_data_path.read_text())
    data["bookings"] = data["bookings"][:1]
    data["bookings"][0]["service_id"] = "hot-stone-massage"
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hot-stone-massage"):
        Dataset(broken)
