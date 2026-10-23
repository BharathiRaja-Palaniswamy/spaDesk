"""The booking form's routes: what can be booked, the free times, and booking one.

    GET  /api/booking/options   the bookable services, their lengths, prices and who offers each
    GET  /api/booking/free      the free start times for a service on a day, by therapist
    POST /api/bookings          book one of those times

Every request goes through the same booking rules the chat's tools use
(appointments.py), so the form and the chat can never disagree: a time is
only booked if it is free, with someone who offers the service.

A broken rule (the time was just taken, a missing phone number) is answered
with 422 and a message written for the customer. A calendar that cannot be
reached is answered with 503 and a general message; the reason goes to the
server log.
"""

import logging
from datetime import date, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..business.store import get_knowledge
from .appointments import get_appointments
from .errors import BookingError, CalendarUnavailable
from .schedule import read_schedule

router = APIRouter()
log = logging.getLogger("spadesk.booking")
UNAVAILABLE = "Booking is unavailable right now. Please call the spa to book."


class BookingForm(BaseModel):
    """What the form sends. pydantic checks types and limits before the rules run."""

    service: str = Field(min_length=1, max_length=80)
    therapist: str = Field(min_length=1, max_length=80)
    start: datetime  # e.g. "2026-10-30T15:00", in the spa's time zone
    minutes: int | None = None  # for a service with more than one length
    name: str = Field(min_length=1, max_length=80)
    phone: str = Field(min_length=6, max_length=30)


def _calendar_down(error: CalendarUnavailable) -> HTTPException:
    """The answer when the calendar cannot be reached, with the reason logged."""
    log.warning("calendar unavailable: %s", error)
    return HTTPException(status_code=503, detail=UNAVAILABLE)


@router.get("/api/booking/options")
def options() -> dict:
    """Every bookable service, with its lengths and prices and who offers it."""
    schedule = read_schedule(get_knowledge())
    return {"services": [
        {"name": s.name, "category": s.category, "minutes": list(s.minutes), "prices": list(s.prices),
         "therapists": [t.name for t in schedule.therapists_for(s.name)]}
        for s in schedule.services.values()]}


@router.get("/api/booking/free")
def free(service: str, day: date, therapist: str | None = None, minutes: int | None = None) -> dict:
    """The free start times on `day` ("HH:MM"), by therapist, earliest first."""
    try:
        slots = get_appointments().free_slots(service, day, therapist, minutes)
    except BookingError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except CalendarUnavailable as error:
        raise _calendar_down(error)
    times: dict[str, list[str]] = {}
    for slot in slots:
        times.setdefault(slot.therapist, []).append(slot.start.strftime("%H:%M"))
    return {"service": service, "day": day.isoformat(), "free": times}


@router.post("/api/bookings")
def book(form: BookingForm) -> dict:
    """Book the chosen time, and return the booking with its reference."""
    try:
        booking = get_appointments().book(form.service, form.therapist, form.start, form.name, form.phone,
                                          form.minutes)
    except BookingError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except CalendarUnavailable as error:
        raise _calendar_down(error)
    return {"reference": booking.id, "service": booking.service, "therapist": booking.therapist,
            "start": booking.start.strftime("%Y-%m-%dT%H:%M"), "end": booking.end.strftime("%Y-%m-%dT%H:%M"),
            "name": booking.name}
