"""The bookable services, as a catalogue read from the business document.

The reports need to know, for each booking, which service it was for, what
category that service is in and what it costs. Those are business facts, so
they live in the business document and nowhere else. A booking in the
bookings JSON only stores a `service_id`; this module is what turns that id
back into a service.

An id is made from a name: "Kids Hand & Foot Treat" -> "kids-hand-foot-treat".
So renaming a service in the document changes its id, and the bookings JSON
has to be regenerated (scripts/generate_bookings.py) to match.
"""

import re
from typing import Any

from .knowledge import KnowledgeBase, slug


def first_number(text: str) -> int | None:
    """'60 minutes / 90 minutes' -> 60, '$115 / $160' -> 115. The first is the standard option."""
    found = re.search(r"\d+", text)
    return int(found[0]) if found else None


def service_catalog(kb: KnowledgeBase) -> list[dict[str, Any]]:
    """Every bookable service in the document, in document order.

    That is every entry with a price, except a monthly membership, which is
    paid for but is not a booking.
    """
    return [
        {"id": slug(item.name), "name": item.name, "category": item.category,
         "duration_minutes": first_number(item.get("duration")), "price": first_number(item.get("price"))}
        for item in kb.items if item.priced and "/month" not in item.get("price")
    ]


def entry_ids(kb: KnowledgeBase) -> set[str]:
    """The id of every named entry in the document (services, offers, FAQs).
    A booking's `offer_id` must be one of these."""
    return {slug(item.name) for item in kb.items}
