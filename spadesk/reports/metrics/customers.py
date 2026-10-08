"""New and returning customers, counted as people, not as bookings."""

from datetime import date
from typing import Any

from ..dataset import Dataset
from .common import Booking, group_by, totals


def customer_mix(rows: list[Booking], dataset: Dataset, start: date) -> dict[str, Any]:
    """Who booked in the period: people new to the spa, and people coming back.

    A customer is New when their first ever booking falls inside the period
    (on or after `start`), and Returning when they first booked before it.
    Each customer is in exactly one group, so the two counts add up to the
    number of customers in the period.

    `repeat_customers` answers a narrower question: how many of those
    customers booked more than once within this period.

    When the period begins at the first day of the data, nobody can be
    Returning because nothing earlier is known. `has_history` is False then,
    so the page can say so instead of showing a misleading 100% New.
    """
    period_start = start.isoformat()
    by_customer = group_by(rows, lambda b: b["customer_id"])

    groups: dict[str, dict[str, list]] = {"New": {"customers": [], "rows": []}, "Returning": {"customers": [], "rows": []}}
    for customer_id, bookings in by_customer.items():
        first_booking = dataset.customers[customer_id]["first_booking_date"]
        group = groups["New" if first_booking >= period_start else "Returning"]
        group["customers"].append(customer_id)
        group["rows"].extend(bookings)

    return {
        "customers": len(by_customer),
        "repeat_customers": sum(1 for bookings in by_customer.values() if len(bookings) > 1),
        "has_history": period_start > dataset.meta["period_start"],
        "by_type": [
            {"customer_type": kind, "customers": len(group["customers"]), **totals(group["rows"])}
            for kind, group in groups.items()
        ],
    }
