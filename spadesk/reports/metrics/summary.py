"""The headline numbers shown in the tiles at the top of the dashboard."""

from typing import Any

from .common import COMPLETED, Booking, totals


def summary(rows: list[Booking]) -> dict[str, Any]:
    """The totals for a set of bookings: the shared `totals` (bookings,
    completed, completion rate, revenue, average rating) plus lost revenue
    and the average ticket.
    """
    completed = [b for b in rows if b["status"] == COMPLETED]
    missed = [b for b in rows if b["status"] != COMPLETED]
    result = totals(rows)
    result.update({
        # What cancelled and missed bookings would have earned at list price.
        "lost_revenue": sum(b["list_price"] for b in missed),
        "average_ticket": round(result["revenue"] / len(completed), 2) if completed else 0.0,
    })
    return result


def change(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, float | None]:
    """For each headline number, how much it moved against the previous period,
    as a fraction (0.12 means up 12%). None when there is nothing to compare with."""
    result: dict[str, float | None] = {}
    for key, value in current.items():
        before = previous.get(key)
        result[key] = round((value - before) / before, 4) if value is not None and before else None
    return result
