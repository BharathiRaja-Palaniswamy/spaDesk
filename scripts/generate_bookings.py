"""Generates one year of mock bookings for the services in the business document.

Run with:  .venv/bin/python scripts/generate_bookings.py

Writes  data/spadesk_bookings.json, which holds only what happened:

  meta        period covered, currency, how the file was made
  channels    the source channels a booking can come through
  customers   id, name, date of first booking
  bookings    one row per booking

It holds no business facts. A booking points at its service with a
`service_id`, and the service's name, category and price are read from
data/family_spa_business_doc.md when the app runs (see
spadesk/business/catalog.py). The price a booking was actually charged at is
stored on the booking as `list_price`, because that is a fact about the
booking: it would stay the same if the price list changed later.

This script reads the business document for the things a realistic booking
depends on (which services exist, opening hours, the offers) and invents the
behaviour around them: how busy each day is, who books what, who cancels, and
how they rate the visit. Those rates are the constants below; change them and
run the script again to reshape the year. After renaming, adding or removing
a service in the business document, run it again too.

The random generator is seeded, so running this again gives the same file.
"""

import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# Make `import spadesk` work when this file is run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spadesk.business.catalog import service_catalog  # noqa: E402
from spadesk.business.knowledge import KnowledgeBase, slug  # noqa: E402
from spadesk.config import HOURS_BLOCK, get_settings  # noqa: E402

OUT = ROOT / "data" / "spadesk_bookings.json"
START, END = date(2025, 10, 1), date(2026, 9, 30)
SEED = 2026

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---- Invented behaviour (not business facts) --------------------------------

# Average bookings per day at the start of the year, and how much busier each weekday is.
BASE_PER_DAY = 5.5
WEEKDAY_WEIGHT = {"Monday": 0.8, "Tuesday": 0.95, "Wednesday": 0.95, "Thursday": 0.8,
                  "Friday": 1.0, "Saturday": 1.3, "Sunday": 1.3}
GROWTH = 0.45  # the last day of the year is this much busier than the first
MONTH_WEIGHT = {12: 1.25, 5: 1.1, 1: 0.85, 2: 0.9}  # holiday gifts, Mother's Day, quiet winter

# Share of bookings per document category.
CATEGORY_WEIGHT = {"Massage Therapy": 30, "Facials": 30, "Hair Removal": 14, "Body Treatments": 10,
                   "Kids Services": 9, "Packages and Memberships": 7}
# How popular each start hour is (hours outside a day's opening times are skipped).
HOUR_WEIGHT = {9: 3, 10: 6, 11: 8, 12: 5, 13: 4, 14: 6, 15: 8, 16: 8, 17: 7, 18: 5, 19: 3}
# "Website" is a booking request sent through the spaDesk customer site.
CHANNELS = {"Google": 24, "Walk-in": 20, "Instagram": 20, "Referral": 18, "Website": 18}
PAYMENTS = {"Card": 45, "Digital Wallet": 35, "Cash": 20}
STATUS = {"Completed": 83, "Cancelled": 10.5, "No-Show": 6.5}
RATINGS = {5: 55, 4: 20, 3: 14, 2: 8, 1: 3}
TAGS = {
    "Happy": ["Great Service", "Clean Space", "Friendly Staff", "Relaxing Experience", "Would Return"],
    "Neutral": ["Average Experience", "Okay Service", "Expensive"],
    "Unhappy": ["Long Wait", "Service Delay", "Not Satisfied", "Expensive"],
}
NO_TAG = 0.08  # share of rated visits with no feedback tag
NEW_CUSTOMER_START, NEW_CUSTOMER_END = 0.65, 0.28  # chance a booking is from a first-time customer

FIRST_NAMES = ("Aarav Aditi Alex Amara Ananya Anaya Arjun Ben Carlos Chloe Daniel Diya Elena Emma Ethan Grace "
               "Hannah Ishaan Isla Jacob Kavya Kiran Leo Liam Lucas Maya Meera Mia Nila Noah Olivia Omar Priya "
               "Rahul Riya Rohan Sara Sofia Tara Vikram Zoe").split()

# ---- Business facts, read from the document ---------------------------------

# The two offers the document defines, and the rule for who qualifies.
# An offer's id is made from its name in the document, the same way a service's is.
WELCOME = {"id": slug("New Guest Welcome Offer"), "discount_percent": 15}
SPRING = {"id": slug("Spring Reset Facial Month"), "discount_percent": 10}
WELCOME_DAYS = set(DAYS[:4])  # Monday to Thursday
WEEKDAYS = set(DAYS[:5])


def opening_hours(kb: KnowledgeBase) -> dict[str, tuple[int, int]]:
    """Read 'Monday to Thursday: 10:00 AM – 8:00 PM' lines into {day: (open hour, close hour)}."""
    def hour(text: str) -> int:
        """'8:00 PM' -> 20."""
        h, suffix = re.match(r"(\d+):\d+ (AM|PM)", text.strip()).groups()
        return int(h) % 12 + (12 if suffix == "PM" else 0)

    hours = {}
    for line in kb.find(HOURS_BLOCK).lines:
        days, times = line.split(": ", 1)
        start, end = re.split(r"\s+[–-]\s+", times)
        first, _, last = days.partition(" to ")
        for day in DAYS[DAYS.index(first): DAYS.index(last or first) + 1]:
            hours[day] = (hour(start), hour(end))
    return hours


def pick(rng: random.Random, weights: dict):
    """Choose one key of `weights`, more often the higher its weight."""
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def main() -> None:
    """Build the whole dataset and write it: reference lists first, then one
    day at a time from START to END, creating that day's bookings.
    """
    rng = random.Random(SEED)
    kb = KnowledgeBase(get_settings().data_path)
    hours = opening_hours(kb)

    # The bookable services, from the business document.
    services = service_catalog(kb)
    by_category: dict[str, list[dict]] = {}
    for service in services:
        by_category.setdefault(service["category"], []).append(service)
    categories = {c: w for c, w in CATEGORY_WEIGHT.items() if c in by_category}

    customers: list[dict] = []
    names: set[str] = set()
    bookings = []
    total_days = (END - START).days
    for offset in range(total_days + 1):
        day = START + timedelta(days=offset)
        weekday = day.strftime("%A")
        progress = offset / total_days  # 0.0 on the first day, 1.0 on the last

        # How many bookings today: base x weekday x season x growth, with some noise.
        expected = (BASE_PER_DAY * WEEKDAY_WEIGHT[weekday] * MONTH_WEIGHT.get(day.month, 1.0)
                    * (1 + GROWTH * progress))
        count = max(1, round(rng.gauss(expected, expected * 0.25)))

        open_hour, close_hour = hours[weekday]
        slots = {h: w for h, w in HOUR_WEIGHT.items() if open_hour <= h < close_hour}

        for start in sorted(pick(rng, slots) for _ in range(count)):
            # A new customer, or one who has been before.
            p_new = NEW_CUSTOMER_START + (NEW_CUSTOMER_END - NEW_CUSTOMER_START) * progress
            is_new = not customers or rng.random() < p_new
            if is_new:
                name = f"{rng.choice(FIRST_NAMES)} {rng.choice('ABCDGJKLMNPRSTVW')}."
                while name in names:  # keep names unique so a name is one customer
                    name = f"{rng.choice(FIRST_NAMES)} {rng.choice('ABCDGJKLMNPRSTVW')}{rng.choice('aeiou')}."
                names.add(name)
                customer = {"id": f"CUS{len(customers) + 1:04d}", "name": name, "first_booking_date": day.isoformat()}
                customers.append(customer)
            else:
                customer = rng.choice(customers)

            service = rng.choice(by_category[pick(rng, categories)])
            price = service["price"]

            # Weekends and first-time customers drop out a little more often.
            status_weights = dict(STATUS)
            if weekday in ("Saturday", "Sunday"):
                status_weights["Cancelled"] *= 1.3
            if is_new:
                status_weights["No-Show"] *= 1.5
            status = pick(rng, status_weights)
            completed = status == "Completed"

            # Discounts: only what the document offers. They cannot be combined.
            offer = None
            is_package = service["category"].startswith("Packages") or "Package" in service["name"]
            if is_new and weekday in WELCOME_DAYS and not is_package:
                offer = WELCOME
            elif service["name"] == "Glow Express Facial" and day.month == 4 and weekday in WEEKDAYS:
                offer = SPRING
            # A discount is only given when the visit actually happens.
            discount = round(price * offer["discount_percent"] / 100) if completed and offer else 0

            rating = sentiment = tag = None
            if completed:
                rating = pick(rng, RATINGS)
                sentiment = "Happy" if rating >= 4 else "Neutral" if rating == 3 else "Unhappy"
                if rng.random() >= NO_TAG:
                    tag = rng.choice(TAGS[sentiment])

            bookings.append({
                "id": f"BKG{len(bookings) + 1:05d}",
                "date": day.isoformat(),
                "time_slot": f"{start:02d}:00-{start + 1:02d}:00",
                "customer_id": customer["id"],
                "service_id": service["id"],
                "channel": pick(rng, CHANNELS),
                "status": status,
                "list_price": price,
                "discount": discount,
                "offer_id": offer["id"] if discount else None,
                "revenue": price - discount if completed else 0,  # a missed booking earns nothing
                "payment_method": pick(rng, PAYMENTS) if completed else None,
                "rating": rating,
                "sentiment": sentiment,
                "feedback_tag": tag,
            })

    dataset = {
        "meta": {
            "description": "Mock bookings for the spaDesk report dashboard. Not real customer data.",
            "period_start": START.isoformat(),
            "period_end": END.isoformat(),
            "currency": "$",
            "generated_by": "scripts/generate_bookings.py",
            "services_from": get_settings().data_path.name,
        },
        "channels": list(CHANNELS),
        "customers": customers,
        "bookings": bookings,
    }
    write(dataset)
    print(f"Wrote {len(bookings)} bookings ({START} to {END}) for {len(customers)} customers "
          f"across {len(services)} services to {OUT.relative_to(ROOT)}")


def write(dataset: dict) -> None:
    """Small objects are indented; the long lists get one record per line, so the
    file stays readable in an editor and in git diffs."""
    parts = []
    for key, value in dataset.items():
        if key in ("customers", "bookings"):
            rows = ",\n".join("    " + json.dumps(row, ensure_ascii=False) for row in value)
            parts.append(f'  "{key}": [\n{rows}\n  ]')
        else:
            body = json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n  ")
            parts.append(f'  "{key}": {body}')
    OUT.write_text("{\n" + ",\n".join(parts) + "\n}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
