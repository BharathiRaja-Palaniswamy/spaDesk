"""Settings for the whole app, in one place.

Two kinds of configuration live here:

1. Constants that say HOW the app uses the business document (which services
   to feature, which policies to highlight). They only name headings and entries;
   the facts themselves are always read from the document.
2. `Settings`: values that can differ per machine (API model, file paths).
   Each one can be overridden with an environment variable or a line in the
   `.env` file.

Every other module imports from here, so nothing else reads `os.environ`.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Reads the `.env` file (if there is one) into environment variables. This is
# also how ANTHROPIC_API_KEY reaches the Anthropic SDK.
load_dotenv()

# The project folder (the parent of this `spadesk/` package).
ROOT = Path(__file__).resolve().parent.parent

# ---- How the app uses the business document -------------------------------

# The bold block that lists opening hours, shown in the hero of the landing page.
HOURS_BLOCK = "Business Hours"
# The section whose first paragraph is shown in the page footer.
ABOUT_SECTION = "About"
# Shown on the landing page. Entries missing from the document are skipped.
FEATURED = ("Signature Radiance Facial", "Swedish Massage", "Kids Mini Facial", "Mom & Me Relax Package")
HIGHLIGHTS = ("New Guest Welcome Offer", "Cancellation Policy", "Late Arrival Policy", "Age Policy")


# ---- Per-machine settings --------------------------------------------------

@dataclass(frozen=True)  # frozen: settings cannot be changed after they are loaded
class Settings:
    """Values that can differ per machine. Read once by `get_settings`."""

    data_path: Path  # the business document the customer site reads
    bookings_path: Path  # where mock booking requests are appended
    report_data_path: Path  # the bookings JSON the report dashboard reads
    model: str  # which Claude model the dashboard assistant uses
    timezone: str  # the spa's time zone, used to tell the assistant the current time


def get_settings() -> Settings:
    """Read every setting from the environment, falling back to its default."""
    # os.getenv(name, default) returns the environment variable if it is set,
    # otherwise the default written here.
    return Settings(
        data_path=Path(os.getenv("SPADESK_DATA", ROOT / "data" / "business_doc.md")),
        bookings_path=Path(os.getenv("SPADESK_BOOKINGS", ROOT / "storage" / "booking_requests.jsonl")),
        report_data_path=Path(os.getenv("SPADESK_REPORT_DATA", ROOT / "data" / "spadesk_bookings.json")),
        model=os.getenv("SPADESK_MODEL", "claude-opus-5-5"),
        timezone=os.getenv("SPADESK_TIMEZONE", "America/Los_Angeles"),
    )
