"""Helpers shared by the metric files."""

from collections import defaultdict
from typing import Any, Callable, Hashable

Booking = dict[str, Any]

COMPLETED = "Completed"


def rate(part: float, whole: float) -> float:
    """part / whole as a fraction (0.83 means 83%). 0 when there is nothing to divide by."""
    return round(part / whole, 4) if whole else 0.0


def average_rating(rows: list[Booking]) -> float | None:
    """Mean rating of the bookings that have one. None when nobody rated."""
    ratings = [b["rating"] for b in rows if b["rating"] is not None]
    return round(sum(ratings) / len(ratings), 2) if ratings else None


def totals(rows: list[Booking]) -> dict[str, Any]:
    """The numbers every breakdown reports for a group of bookings."""
    completed = sum(1 for b in rows if b["status"] == COMPLETED)
    return {
        "bookings": len(rows),
        "completed": completed,
        "completion_rate": rate(completed, len(rows)),
        "revenue": sum(b["revenue"] for b in rows),
        "average_rating": average_rating(rows),
    }


def group_by(rows: list[Booking], key: Callable[[Booking], Hashable]) -> dict[Hashable, list[Booking]]:
    """Split bookings into groups: {key value: [bookings with that value]}."""
    groups: dict[Hashable, list[Booking]] = defaultdict(list)
    for b in rows:
        groups[key(b)].append(b)
    return groups
