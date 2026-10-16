"""Tests for the dashboard calculations in spadesk/reports/metrics/.

Run with:  .venv/bin/pytest tests/test_report_metrics.py

Most tests use four hand-written bookings, small enough to check every
expected number by eye. The last test runs every metric over the real file.
"""

from datetime import date

import pytest

from spadesk.config import get_settings
from spadesk.reports.dataset import load
from spadesk.reports.metrics import channels, customers, feedback, period, services, summary, trend


@pytest.fixture(scope="module")
def dataset():
    """The real bookings joined to the business document: metrics look service names up in it."""
    return load(get_settings().report_data_path)


def booking(day, status="Completed", service="swedish-massage", price=115, discount=0, rating=5, **extra):
    """A booking row with sensible defaults; pass only what a test cares about."""
    done = status == "Completed"
    return {
        "id": "T", "date": day, "time_slot": "11:00-12:00",
        "customer_id": "CUS0001", "service_id": service, "channel": "Google",
        "status": status, "list_price": price, "discount": discount if done else 0, "offer_id": None,
        "revenue": price - discount if done else 0, "payment_method": "Card" if done else None,
        "rating": rating if done else None, "sentiment": ("Happy" if rating >= 4 else "Unhappy") if done else None,
        "feedback_tag": ("Great Service" if rating >= 4 else "Long Wait") if done else None, **extra,
    }


# Tue 1 Sep: a full-price massage and a discounted first visit. Sat 5 Sep: a
# cancellation and a no-show.
ROWS = [
    booking("2026-09-01"),
    booking("2026-09-01", service="glow-express-facial", price=80, discount=12, rating=2,
            customer_id="CUS0002", channel="Website"),
    booking("2026-09-05", status="Cancelled"),
    booking("2026-09-05", status="No-Show", service="glow-express-facial", price=80, customer_id="CUS0002"),
]


def test_summary():
    s = summary.summary(ROWS)
    assert (s["bookings"], s["completed"]) == (4, 2)
    assert s["completion_rate"] == 0.5
    assert (s["revenue"], s["lost_revenue"]) == (115 + 68, 115 + 80)
    assert s["average_ticket"] == 91.5 and s["average_rating"] == 3.5


def test_summary_of_nothing_does_not_divide_by_zero():
    s = summary.summary([])
    assert s["bookings"] == 0 and s["completion_rate"] == 0.0 and s["average_rating"] is None


def test_change_against_previous_period():
    moved = summary.change({"revenue": 110, "average_rating": None}, {"revenue": 100, "average_rating": 4.0})
    assert moved == {"revenue": 0.1, "average_rating": None}
    assert summary.change({"revenue": 50}, {"revenue": 0}) == {"revenue": None}


def test_previous_period_has_the_same_length():
    assert period.previous_period(date(2026, 9, 1), date(2026, 9, 30)) == (date(2026, 8, 2), date(2026, 8, 31))


def test_bucket_size_follows_range_length():
    assert period.bucket_size(date(2026, 9, 1), date(2026, 9, 30)) == "day"
    assert period.bucket_size(date(2026, 6, 1), date(2026, 9, 30)) == "week"
    assert period.bucket_size(date(2025, 10, 1), date(2026, 9, 30)) == "month"
    months = period.buckets(date(2025, 10, 1), date(2026, 9, 30), "month")
    assert len(months) == 12 and months[0] == date(2025, 10, 1) and months[-1] == date(2026, 9, 1)


def test_trend_includes_empty_days():
    result = trend.trend(ROWS, date(2026, 9, 1), date(2026, 9, 5))
    assert result["bucket"] == "day"
    assert [p["start"][-2:] for p in result["points"]] == ["01", "02", "03", "04", "05"]
    assert [p["bookings"] for p in result["points"]] == [2, 0, 0, 0, 2]
    assert [p["revenue"] for p in result["points"]] == [183, 0, 0, 0, 0]


def test_services_and_categories_use_names_from_the_file(dataset):
    categories = services.by_category(ROWS, dataset)
    assert [(c["category"], c["revenue"]) for c in categories] == [("Massage Therapy", 115), ("Facials", 68)]
    top = services.by_service(ROWS, dataset)[0]
    assert (top["service"], top["price"], top["bookings"], top["completed"]) == ("Swedish Massage", 115, 2, 1)


def test_channels_keep_file_order_and_show_empty_ones(dataset):
    rows = channels.by_channel(ROWS, dataset)
    assert [r["channel"] for r in rows] == dataset.channels
    assert {r["channel"]: r["bookings"] for r in rows if r["bookings"]} == {"Google": 3, "Website": 1}


def test_customers_are_counted_as_people(dataset):
    # In the real file CUS0001 and CUS0002 both first booked on 2025-10-01.
    # Seen from September 2026 they are returning customers...
    mix = customers.customer_mix(ROWS, dataset, date(2026, 9, 1))
    new, returning = mix["by_type"]
    assert (new["customers"], returning["customers"]) == (0, 2)
    assert returning["bookings"] == 4 and returning["revenue"] == 183
    assert mix["customers"] == 2 and mix["repeat_customers"] == 2  # each booked twice in ROWS
    assert mix["has_history"]
    # ...and seen from the first day of the data they are new, with nothing earlier to compare.
    from_the_start = customers.customer_mix(ROWS, dataset, date(2025, 10, 1))
    assert [g["customers"] for g in from_the_start["by_type"]] == [2, 0]
    assert not from_the_start["has_history"]


def test_new_and_returning_customers_in_the_real_september(dataset):
    start, end = date(2026, 9, 1), date(2026, 9, 30)
    mix = customers.customer_mix(dataset.select(start, end), dataset, start)
    new, returning = mix["by_type"]
    assert (new["customers"], returning["customers"], mix["customers"]) == (89, 181, 270)
    assert new["customers"] + returning["customers"] == mix["customers"]  # nobody is counted twice
    assert mix["repeat_customers"] == 20


def test_feedback_counts_only_rated_visits():
    result = feedback.feedback(ROWS)
    assert result["rated"] == 2 and result["average_rating"] == 3.5
    assert {r["rating"]: r["count"] for r in result["ratings"]} == {5: 1, 4: 0, 3: 0, 2: 1, 1: 0}
    assert {"tag": "Long Wait", "sentiment": "Unhappy", "count": 1} in result["tags"]


def test_every_metric_runs_on_the_real_file(dataset):
    rows = dataset.select()
    start, end = date.fromisoformat(dataset.meta["period_start"]), date.fromisoformat(dataset.meta["period_end"])
    s = summary.summary(rows)
    assert s["bookings"] == len(dataset.bookings)
    # Each breakdown must add back up to the headline revenue.
    for parts in (trend.trend(rows, start, end)["points"], services.by_category(rows, dataset),
                  services.by_service(rows, dataset), channels.by_channel(rows, dataset),
                  customers.customer_mix(rows, dataset, start)["by_type"]):
        assert sum(p["revenue"] for p in parts) == s["revenue"]
    assert feedback.feedback(rows)["rated"] == s["completed"]
