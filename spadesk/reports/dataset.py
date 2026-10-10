"""Loads the bookings JSON for the reports and filters it.

Two sources, each with its own job, and nothing stored twice:

  data/spadesk_bookings.json        what happened: bookings and customers
  data/business_doc.md              what the business is: services, prices

A booking stores only a `service_id`. The service's name, category and price
come from the business document, through `service_catalog` in
spadesk/business/catalog.py. So the Reports tab and the Business profile tab
always show the same names and prices.

`Dataset` holds both together and answers one question: "which bookings match
these filters?". Totals and averages are worked out from that answer in
`metrics/`.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

from ..business.catalog import entry_ids, service_catalog
from ..business.store import get_knowledge


class Dataset:
    """The bookings, with the services they point at.

      bookings              every booking row, from the JSON
      customers             id -> customer, from the JSON
      channels              the source channels, in file order, from the JSON
      meta                  period covered and currency, from the JSON
      services              id -> service, from the business document
      categories            service categories, in document order
    """

    def __init__(self, path: Path, services: list[dict[str, Any]], offer_ids: set[str]):
        """Read the bookings file at `path` and join it to `services`.

        `services` is the catalogue from the business document and `offer_ids`
        the ids an offer may have. Raises ValueError if a booking points at a
        service, customer or offer that does not exist.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta: dict[str, Any] = data["meta"]
        self.channels: list[str] = data["channels"]
        self.bookings: list[dict[str, Any]] = data["bookings"]
        # id -> record, so a booking's service or customer is one dictionary lookup away.
        self.customers: dict[str, dict[str, Any]] = {c["id"]: c for c in data["customers"]}
        self.services: dict[str, dict[str, Any]] = {s["id"]: s for s in services}
        # Categories in the order they first appear in the business document.
        self.categories: list[str] = list(dict.fromkeys(s["category"] for s in services))
        self._check_references(offer_ids)

    def _check_references(self, offer_ids: set[str]) -> None:
        """Fail at startup, with the booking id, if a booking points at something
        that does not exist. Better than a wrong number on the page later.

        The usual cause is a service renamed or removed in the business
        document after the bookings file was generated."""
        for b in self.bookings:
            if b["service_id"] not in self.services:
                raise ValueError(f"Booking {b['id']} has service_id {b['service_id']!r}, which is not a service in "
                                 "the business document. Run scripts/generate_bookings.py to regenerate the bookings.")
            if b["customer_id"] not in self.customers:
                raise ValueError(f"Booking {b['id']} has unknown customer_id {b['customer_id']!r}")
            if b["offer_id"] is not None and b["offer_id"] not in offer_ids:
                raise ValueError(f"Booking {b['id']} has offer_id {b['offer_id']!r}, which is not in the business document.")

    def filter_options(self) -> dict[str, Any]:
        """What the filter row can offer: the date range the data covers and
        the values of each dropdown."""
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


def load(path: Path) -> Dataset:
    """Build a Dataset from the bookings file at `path` and the business document."""
    kb = get_knowledge()
    return Dataset(path, service_catalog(kb), entry_ids(kb))
