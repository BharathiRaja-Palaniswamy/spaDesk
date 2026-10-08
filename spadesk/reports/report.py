"""Assembles one complete report: every section of the dashboard for one set of filters.

This is the only place that knows which calculations make up the dashboard.
Each section below is one function from `metrics/`; to add a section, add a
metric file and one line here.
"""

from datetime import date
from typing import Any

from .dataset import Dataset
from .metrics import channels, customers, feedback, services, summary, trend
from .metrics.period import previous_period


def build_report(
    dataset: Dataset,
    start: date,
    end: date,
    category: str | None = None,
    channel: str | None = None,
) -> dict[str, Any]:
    """Run every calculation for the bookings between `start` and `end`
    (both included), optionally narrowed to one category and one channel.
    Returns a dictionary ready to be sent as JSON; its keys are the
    sections of the dashboard.
    """
    rows = dataset.select(start, end, category, channel)

    # The same filters over the period just before, for the "vs previous" figures.
    before_start, before_end = previous_period(start, end)
    current = summary.summary(rows)
    previous = summary.summary(dataset.select(before_start, before_end, category, channel))
    # No comparison when the earlier period starts before the data does: part of
    # it would be empty, and the change would look bigger than it is.
    comparable = before_start.isoformat() >= dataset.meta["period_start"]

    return {
        "filters": {"start": start.isoformat(), "end": end.isoformat(), "category": category, "channel": channel},
        "currency": dataset.meta["currency"],
        "summary": current,
        "previous_period": {"start": before_start.isoformat(), "end": before_end.isoformat()} if comparable else None,
        "previous_summary": previous if comparable else None,
        "change": summary.change(current, previous) if comparable else None,
        "trend": trend.trend(rows, start, end),
        "categories": services.by_category(rows, dataset),
        "services": services.by_service(rows, dataset),
        "channels": channels.by_channel(rows, dataset),
        "customers": customers.customer_mix(rows, dataset, start),
        "feedback": feedback.feedback(rows),
    }
