"""Check booking against the real calendars: find a time, book it, read it back, cancel it.

    .venv/bin/python -m scripts.check_calendar           # book, read back, cancel
    .venv/bin/python -m scripts.check_calendar --keep    # leave the booking, to see it in Google Calendar

Uses the spa's Google calendars (GOOGLE_SERVICE_ACCOUNT). It books the
first free time for a Swedish Massage two weeks from today, for a customer
called "Calendar check", reads it back from the day's bookings, then
cancels it, so the calendar is left as it was. With --keep it is not cancelled: it stays on the therapist's calendar
until deleted there (or cancelled with its reference and the phone below).
Claude is never called; the Calendar API has no charge.
"""

import argparse
from datetime import date, timedelta

from spadesk.booking.appointments import get_appointments

SERVICE, PHONE = "Swedish Massage", "000 000 0000"


def main() -> None:
    """Book, read back and cancel one test appointment, saying what happened at each step."""
    parser = argparse.ArgumentParser(description="Check booking against the configured calendar.")
    parser.add_argument("--keep", action="store_true", help="leave the booking on the calendar")
    keep = parser.parse_args().keep
    desk = get_appointments()
    day = date.today() + timedelta(days=14)
    # The first day from then on when someone who offers the service works.
    while not (slots := desk.free_slots(SERVICE, day)):
        day += timedelta(days=1)
    slot = slots[0]
    print(f"free:      {len(slots)} times on {day}, first {slot.start:%H:%M} with {slot.therapist}")

    booking = desk.book(SERVICE, slot.therapist, slot.start, "Calendar check", PHONE)
    print(f"booked:    {booking.id}")
    assert booking in desk.day_bookings(day), "the booking was not read back"
    assert slot.start not in [s.start for s in desk.free_slots(SERVICE, day, slot.therapist)], "time still free"
    print("read back: the booking is listed, and its time is no longer free")

    if keep:
        print(f"kept:      {SERVICE} on {slot.start:%A %d %B, %H:%M} on {slot.therapist}'s calendar")
        return
    desk.cancel(booking.id, PHONE)
    assert booking not in desk.day_bookings(day), "the booking is still there"
    print("cancelled: the calendar is as it was. All good.")


if __name__ == "__main__":
    main()
