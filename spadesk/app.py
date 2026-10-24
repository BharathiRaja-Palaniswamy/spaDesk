"""Web app: the customer site, booking and the manage page.

Run with:  uvicorn spadesk.app:app --reload

This file is the wiring. It connects the pieces (the business document, the
booking routes, the chats) and exposes them as HTTP routes. The actual work happens in the
`business/`, `booking/`, `reports/` and `assistant/` packages.

Routes:
  GET  /                      the customer site (static/home/index.html)
  GET  /login                 the owner's sign-in page        (routes in access/routes.py)
  POST /api/login, /api/logout  sign the owner in and out
  GET  /static/...            CSS and JavaScript for both pages
  GET  /api/spa               spa facts for the customer site, read from the document
  GET  /api/profile           the whole business profile, for both pages
  GET  /api/powered-by        which provider answers each chat, for the "Powered by" line
  GET  /api/booking/options   what can be booked         (routes in booking/routes.py)
  GET  /api/booking/free      the free times for a service on a day
  POST /api/bookings          book one of them
  POST /api/assistant         the customer chat: the booking agent (route in agent/routes.py)
  POST /v1/chat/completions   the phone line: the same agent, called by ElevenLabs (voice/routes.py)
  POST /api/chat              the question-and-answer assistant, kept for the evals (assistant/customer.py)
  GET  /manage                the owner's manage page (routes in reports/routes.py)
  GET  /api/report...         the reports' data       (routes in reports/routes.py)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .access.routes import router as owner_sign_in
from .agent.routes import router as booking_agent
from .assistant.customer import router as customer_chat
from .booking.routes import router as booking
from .business.profile import build_profile
from .business.store import get_knowledge
from .config import HOURS_BLOCK
from .models.profiles import get_profile
from .models.providers import NAMES
from .reports.routes import page_router as manage_page
from .reports.routes import router as report_router
from .voice.routes import router as phone_line

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="spaDesk")
# The manage page brings its own routes (see reports/routes.py).
app.include_router(report_router)
app.include_router(manage_page)
# So do the customer site's chats (agent/routes.py, assistant/customer.py) and booking (booking/routes.py).
app.include_router(customer_chat)
app.include_router(booking_agent)
app.include_router(owner_sign_in)
app.include_router(phone_line)
app.include_router(booking)


# ---- Routes ----------------------------------------------------------------

@app.get("/api/spa")
def spa():
    """Spa facts for the customer site: name, hours, services, contact."""
    return get_knowledge().public(HOURS_BLOCK)


@app.get("/api/profile")
def profile():
    """The whole business document, section by section: the manage page's
    profile tab and the customer site's sections both draw from it."""
    return build_profile(get_knowledge())


@app.get("/api/powered-by")
def powered_by():
    """The provider behind each chat, as people know it ("Claude", "Nebius"),
    read from the settings now, so it follows SPADESK_MODELS."""
    def name(profile: str) -> str:
        provider = get_profile(profile).primary.provider
        return NAMES.get(provider, provider)
    return {"customer": name("booking-agent"), "owner": name("chat-default")}


@app.get("/")
def index():
    """The customer site."""
    return FileResponse(STATIC / "home" / "index.html")


class FreshStaticFiles(StaticFiles):
    """Static files that the browser must re-check before reusing.

    Without this, a browser may keep using its saved copy of a JavaScript or
    CSS file after the file has changed, so an edit seems to have no effect.
    "no-cache" does not mean "never store": the browser keeps its copy but
    asks the server each time whether it is still current, and only downloads
    the file again when it has changed.
    """

    async def get_response(self, path, scope):
        """Serves the file as usual, then adds the "check before reuse" header."""
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Serves every file in static/ under /static/. Mounted last so it cannot
# shadow the routes above.
app.mount("/static", FreshStaticFiles(directory=STATIC), name="static")
