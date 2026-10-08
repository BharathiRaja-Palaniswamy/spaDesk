"""Generates one year of mock bookings that agree with the business document.

Run with:  .venv/bin/python scripts/generate_bookings.py

Writes  data/spadesk_bookings.json

The dashboard reads ONLY that JSON file. It never reads the business document
and has no service, price or label written in its code. So the file carries
everything the dashboard needs, as separate objects:

  meta             period covered, currency, when and how the file was made
  business         spa name and opening hours
  services         every bookable service: id, name, category, duration, price
  offers           the discounts that exist
  channels, payment_methods, statuses, feedback_tags   the allowed values
  customers        id, name, date of first booking
  bookings         one row per booking, pointing at the lists above by id

This script is the one place the two worlds meet. It copies the business facts
(services, prices, hours, offers) out of data/family_spa_business_doc.md, the
file the customer site reads, so the dashboard and the site agree. What it
invents is the behaviour around those facts: how busy each day is, who books
what, who cancels, and how they rate the visit. Those rates are the constants
below; change them and run the script again to reshape the year. After editing
the business document, run it again too.

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

from spadesk.config import HOURS_BLOCK, get_settings  # noqa: E402
from spadesk.business.knowledge import KnowledgeBase  # noqa: E402

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

# The two offers the document defines, and who qualifies. The wording shown on
# the dashboard is read from the document; only the rule is coded here.
WELCOME = {"id": "new-guest-welcome", "name": "New Guest Welcome Offer", "discount_percent": 15}
SPRING = {"id": "spring-reset", "name": "Spring Reset Facial Month", "discount_percent": 10}
WELCOME_DAYS = set(DAYS[:4])  # Monday to Thursday
WEEKDAYS = set(DAYS[:5])


def slug(text: str) -> str:
    """'Kids Hand & Foot Treat' -> 'kids-hand-foot-treat'. Used as a service id."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def first_number(text: str) -> int | None:
    """'60 minutes / 90 minutes' -> 60, '$115 / $160' -> 115. The first is the standard option."""
    found = re.search(r"\d+", text)
    return int(found[0]) if found else None


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

    # The services list: every priced entry in the document except the monthly
    # membership, which is not a booking.
    services = [
        {"id": slug(i.name), "name": i.name, "category": i.category, "description": i.description,
         "duration_minutes": first_number(i.get("duration")), "price": first_number(i.get("price"))}
        for i in kb.items if i.priced and "/month" not in i.get("price")
    ]
    by_category: dict[str, list[dict]] = {}
    for service in services:
        by_category.setdefault(service["category"], []).append(service)
    categories = {c: w for c, w in CATEGORY_WEIGHT.items() if c in by_category}
    offers = [{**offer, "description": kb.find(offer["name"]).lines[0]} for offer in (WELCOME, SPRING)]

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
                "day_of_week": weekday,
                "time_slot": f"{start:02d}:00-{start + 1:02d}:00",
                "customer_id": customer["id"],
                "customer_type": "New" if is_new else "Returning",
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
            "business_facts_from": get_settings().data_path.name,
        },
        "business": {
            "name": kb.name,
            "hours": [{"day": d, "open": f"{hours[d][0]:02d}:00", "close": f"{hours[d][1]:02d}:00"} for d in DAYS],
        },
        "services": services,
        "offers": offers,
        "channels": list(CHANNELS),
        "payment_methods": list(PAYMENTS),
        "statuses": list(STATUS),
        "feedback_tags": TAGS,
        "customers": customers,
        "bookings": bookings,
    }
    write(dataset)
    print(f"Wrote {len(bookings)} bookings ({START} to {END}), {len(customers)} customers, "
          f"{len(services)} services to {OUT.relative_to(ROOT)}")


def write(dataset: dict) -> None:
    """Small objects are indented; the long lists get one record per line, so the
    file stays readable in an editor and in git diffs."""
    parts = []
    for key, value in dataset.items():
        if key in ("services", "customers", "bookings"):
            rows = ",\n".join("    " + json.dumps(row, ensure_ascii=False) for row in value)
            parts.append(f'  "{key}": [\n{rows}\n  ]')
        else:
            body = json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n  ")
            parts.append(f'  "{key}": {body}')
    OUT.write_text("{\n" + ",\n".join(parts) + "\n}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
