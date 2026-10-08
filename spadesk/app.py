"""Web app: the customer site, mock booking requests and the manage page.

Run with:  uvicorn spadesk.app:app --reload

This file is the wiring. It connects the pieces (the business document, the
booking desk) and exposes them as HTTP routes. The actual work happens in the
`business/`, `booking/` and `reports/` packages.

Routes:
  GET  /                      the customer site (static/index.html)
  GET  /static/...            CSS and JavaScript for both pages
  GET  /api/spa               spa facts for the customer site, read from the document
  GET  /api/manage/profile    the whole business profile, for the manage page
  POST /api/booking-requests  save a mock booking request
  GET  /manage                the owner's manage page (routes in reports/routes.py)
  GET  /api/report...         the reports' data       (routes in reports/routes.py)
"""

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .booking.service import BookingDesk, BookingError, BookingRequest
from .business.profile import build_profile
from .business.store import get_knowledge
from .config import ABOUT_SECTION, FEATURED, HIGHLIGHTS, HOURS_BLOCK, get_settings
from .reports.routes import page_router as manage_page, router as report_router

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="spaDesk")
# The manage page brings its own routes (see reports/routes.py).
app.include_router(report_router)
app.include_router(manage_page)


# ---- Building the pieces ---------------------------------------------------
# @lru_cache makes a function run once; later calls return the same object. So
# there is one booking desk for the whole server, built the first time a
# request needs it. (The business document is shared the same way, by
# `get_knowledge` in business/store.py.) Tests replace this function to plug
# in their own.

@lru_cache
def get_booking_desk() -> BookingDesk:
    """The mock booking desk. Built once."""
    return BookingDesk(get_settings().bookings_path, get_knowledge().service_names)


# ---- Routes ----------------------------------------------------------------

@app.get("/api/spa")
def spa():
    """Spa facts for the customer site: name, hours, services, highlights, contact."""
    return get_knowledge().public(FEATURED, HIGHLIGHTS, HOURS_BLOCK, ABOUT_SECTION)


@app.get("/api/manage/profile")
def profile():
    """The whole business document, section by section, for the manage page."""
    return build_profile(get_knowledge())


@app.post("/api/booking-requests")
def booking_request(request: BookingRequest):
    """Save a booking request. It is recorded as requested, never confirmed."""
    # `BookingRequest` is a pydantic model: FastAPI checks the JSON against it
    # and answers 422 by itself when a field is missing or invalid.
    try:
        return get_booking_desk().submit(request)
    except BookingError as error:
        # A rule the booking desk checks itself (unknown service, date in the past).
        raise HTTPException(status_code=422, detail=str(error))


@app.get("/")
def index():
    """The customer site."""
    return FileResponse(STATIC / "index.html")


class FreshStaticFiles(StaticFiles):
    """Static files that the browser must re-check before reusing.

    Without this, a browser may keep using its saved copy of a JavaScript or
    CSS file after the file has changed, so an edit seems to have no effect.
    "no-cache" does not mean "never store": the browser keeps its copy but
    asks the server each time whether it is still current, and only downloads
    the file again when it has changed.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Serves every file in static/ under /static/. Mounted last so it cannot
# shadow the routes above.
app.mount("/static", FreshStaticFiles(directory=STATIC), name="static")
