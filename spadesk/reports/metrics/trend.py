"""Bookings and revenue over time, sliced by day, week or month."""

from datetime import date
from typing import Any

from .common import Booking, group_by, totals
from .period import bucket_size, bucket_start, buckets


def trend(rows: list[Booking], start: date, end: date) -> dict[str, Any]:
    """One point per day, week or month from `start` to `end`, each with the
    shared totals (bookings, completed, completion rate, revenue, rating).
    """
    size = bucket_size(start, end)
    by_bucket = group_by(rows, lambda b: bucket_start(date.fromisoformat(b["date"]), size))
    return {
        "bucket": size,
        # Every bucket in the range gets a point, so a quiet day shows as zero
        # instead of disappearing from the line.
        "points": [{"start": day.isoformat(), **totals(by_bucket.get(day, []))} for day in buckets(start, end, size)],
    }

