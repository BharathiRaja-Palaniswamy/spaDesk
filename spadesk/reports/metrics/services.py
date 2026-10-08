"""Performance per service category and per service."""

from typing import Any

from ..dataset import Dataset
from .common import Booking, group_by, totals


def by_category(rows: list[Booking], dataset: Dataset) -> list[dict[str, Any]]:
    """One row per category, highest revenue first."""
    groups = group_by(rows, lambda b: dataset.services[b["service_id"]]["category"])
    result = [{"category": category, **totals(group)} for category, group in groups.items()]
    return sorted(result, key=lambda r: r["revenue"], reverse=True)


def by_service(rows: list[Booking], dataset: Dataset) -> list[dict[str, Any]]:
    """One row per service that was booked, highest revenue first."""
    groups = group_by(rows, lambda b: b["service_id"])
    result = []
    for service_id, group in groups.items():
        service = dataset.services[service_id]
        result.append({"service": service["name"], "category": service["category"],
                       "price": service["price"], **totals(group)})
    return sorted(result, key=lambda r: r["revenue"], reverse=True)
