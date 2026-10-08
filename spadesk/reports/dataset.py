"""Loads the bookings JSON for the report dashboard and filters it.

This is the dashboard's only door to its data. Everything the dashboard shows
comes from data/spadesk_bookings.json through this class: services, prices,
channels, customers and bookings. Nothing is read from the business document
and no service or label is written in dashboard code.

The file keeps reference lists (services, customers, offers) apart from the
bookings, and a booking points at them by id:

    booking["service_id"] -> services      booking["customer_id"] -> customers

`Dataset` reads the file once, builds id -> record lookups for those lists,
and answers one question: "which bookings match these filters?". Totals and
averages are worked out from that answer by the next module.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any


class Dataset:
    """The bookings JSON, loaded into memory.

      bookings                       every booking row
      services, customers, offers    id -> record lookups
      categories, channels           the filter values, in file order
      meta, business                 period covered, currency, name, opening hours
    """

    def __init__(self, path: Path):
        """Read the file at `path` and build the lookups. Raises ValueError if a
        booking points at a service, customer or offer that does not exist.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta: dict[str, Any] = data["meta"]
        self.business: dict[str, Any] = data["business"]
        self.channels: list[str] = data["channels"]
        self.bookings: list[dict[str, Any]] = data["bookings"]
        # id -> record, so a booking's service or customer is one dictionary lookup away.
        self.services: dict[str, dict[str, Any]] = {s["id"]: s for s in data["services"]}
        self.customers: dict[str, dict[str, Any]] = {c["id"]: c for c in data["customers"]}
        self.offers: dict[str, dict[str, Any]] = {o["id"]: o for o in data["offers"]}
        # Categories in the order they first appear in the services list.
        self.categories: list[str] = list(dict.fromkeys(s["category"] for s in data["services"]))
        self._check_references()

    def _check_references(self) -> None:
        """Fail at startup, with the booking id, if a booking points at something
        that does not exist. Better than a wrong number on the dashboard later."""
        for b in self.bookings:
            if b["service_id"] not in self.services:
                raise ValueError(f"Booking {b['id']} has unknown service_id {b['service_id']!r}")
            if b["customer_id"] not in self.customers:
                raise ValueError(f"Booking {b['id']} has unknown customer_id {b['customer_id']!r}")
            if b["offer_id"] is not None and b["offer_id"] not in self.offers:
                raise ValueError(f"Booking {b['id']} has unknown offer_id {b['offer_id']!r}")

    def filter_options(self) -> dict[str, Any]:
        """What the dashboard's filter row can offer: the date range the data
        covers and the values of each dropdown."""
        return {
            "min_date": self.meta["period_start"],
            "max_date": self.meta["period_end"],
            "categories": self.categories,
            "channels": self.channels,
        }

    def select(
        self,
        start: date | None = None,
        end: date | None = None,
        category: str | None = None,
        channel: str | None = None,
    ) -> list[dict[str, Any]]:
        """Bookings that match every filter given. A filter left as None is not applied.

        `start` and `end` are both included, so start=end returns one day.
        """
        # Dates in the file are "YYYY-MM-DD" text. In that format, alphabetical
        # order is the same as calendar order, so they compare correctly as text.
        first = start.isoformat() if start else ""
        last = end.isoformat() if end else "9999-12-31"
        return [
            b for b in self.bookings
            if first <= b["date"] <= last
            and (category is None or self.services[b["service_id"]]["category"] == category)
            and (channel is None or b["channel"] == channel)
        ]
