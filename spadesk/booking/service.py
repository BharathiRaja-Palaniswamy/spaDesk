"""Mock booking: records what the customer asked for. Nothing here confirms an appointment.

A request always has status "requested". Real calendar booking would replace
`BookingDesk.submit` with an availability check and add a "confirmed" status.

This module knows nothing about the chat or the web server. It takes a
validated request and stores it, which keeps it easy to replace later.
"""

import json
import secrets
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

REQUESTED = "requested"
# Returned with every request, so the customer always reads that nothing is confirmed.
NEXT_STEP = ("This is a booking request, not a confirmed appointment. "
             "The spa will contact you to confirm a time.")


class BookingError(ValueError):
    """A request that is well-formed but breaks a booking rule. The message is shown to the customer."""


class BookingRequest(BaseModel):
    """The booking form. pydantic checks types and limits before `submit` runs."""

    service: str = Field(min_length=1, max_length=80)
    preferred_date: date
    preferred_time: Literal["morning", "afternoon", "evening", "any"] = "any"
    party_size: int = Field(1, ge=1, le=6)
    name: str = Field(min_length=1, max_length=80)
    phone: str = Field(min_length=6, max_length=30)
    notes: str = Field("", max_length=500)


class BookingDesk:
    """Takes booking requests and appends them to a file."""

    def __init__(self, path: Path, service_names: list[str]):
        """path: the JSON Lines file to append to.
        service_names: the services a request may name.
        """
        self.path = Path(path)
        # Valid services come from the business document, not from the browser.
        self.service_names = set(service_names)

    def submit(self, request: BookingRequest) -> dict[str, Any]:
        """Check the request against the booking rules, store it, and return its
        reference, status and the "not confirmed" message. Raises BookingError
        when a rule is broken.
        """
        if request.service not in self.service_names:
            raise BookingError("Please choose a service from the list.")
        if request.preferred_date < date.today():
            raise BookingError("Please choose today or a later date.")
        record = {
            "reference": "REQ-" + secrets.token_hex(3).upper(),  # e.g. REQ-FED04A
            "status": REQUESTED,
            "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **request.model_dump(mode="json"),
        }
        # Storage is a JSON Lines file: one request per line, appended.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {"reference": record["reference"], "status": REQUESTED, "message": NEXT_STEP}
