"""A day and a time of day, from the words a customer used.

Pebby passes on what the customer said ("Thursday", "after 3pm", "any
time"), never a date or a clock time it worked out: a model that is shown
the time now tends to fill an empty field with it (in testing: "today,
after 09:34" for a customer who had said neither). Code turns the words
into a date and times instead, the same way every time.

    day_from("Thursday", today)      the coming Thursday (today, if today is one)
    day_from("tomorrow", today)      today + 1
    day_from("30 October", today)    that date (also "Oct 30", "30/10", "2026-10-30")
    times_from("after 3pm")          after 15:00
    times_from("morning")            before 12:00
    times_from("between 2 and 4")    after 14:00, before 16:00
    times_from("any time")           no limit

An hour without am/pm is read as the spa reads it: 8 to 11 in the morning,
12 to 7 in the afternoon or evening. Words that cannot be read give None,
so the customer is asked again rather than a guess being made.
"""

import re
from datetime import date, time, timedelta

from ..rules import ANY_TIME, PARTS_OF_DAY

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
# A time: "3", "3pm", "3:30 pm", "15:00", "noon".
CLOCK = r"(noon|\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)"


def _month(word: str) -> int | None:
    """'oct' or 'October' -> 10."""
    return next((n for n, name in enumerate(MONTHS, 1) if name.startswith(word[:3])), None) if len(word) >= 3 else None


def day_from(words: str, today: date) -> date | None:
    """The day `words` name, or None when they name none (or one that cannot be read)."""
    said = words.lower().strip()
    if "day after tomorrow" in said:
        return today + timedelta(days=2)
    if "tomorrow" in said:
        return today + timedelta(days=1)
    if "today" in said or "tonight" in said:
        return today
    for n, name in enumerate(DAYS):
        if re.search(rf"\b{name[:3]}(?:{name[3:]})?\b", said):
            return today + timedelta(days=(n - today.weekday()) % 7)
    if found := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", said):
        return _real(int(found[1]), int(found[2]), int(found[3]))
    if found := re.search(r"\b(\d{1,2})/(\d{1,2})\b", said):  # day/month, as the spa writes dates
        return _this_or_next(int(found[1]), int(found[2]), today)
    if found := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?(?: of)? ([a-z]{3,})\b", said):
        if month := _month(found[2]):
            return _this_or_next(int(found[1]), month, today)
    if found := re.search(r"\b([a-z]{3,}) (\d{1,2})(?:st|nd|rd|th)?\b", said):
        if month := _month(found[1]):
            return _this_or_next(int(found[2]), month, today)
    return None


def _real(year: int, month: int, day: int) -> date | None:
    """The date, or None if there is no such day."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _this_or_next(day: int, month: int, today: date) -> date | None:
    """`day` of `month` this year, or next year once it has passed."""
    this = _real(today.year, month, day)
    return this if this is None or this >= today else _real(today.year + 1, month, day)


def _clock(text: str) -> time | None:
    """'3pm' -> 15:00; '3' -> 15:00; '10' -> 10:00; '15:30' -> 15:30; 'noon' -> 12:00."""
    text = text.replace(".", "").replace(" ", "")
    if text == "noon":
        return time(12)
    found = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", text)
    if not found:
        return None
    hour, minute, half = int(found[1]), int(found[2] or 0), found[3]
    if half == "pm" and hour < 12:
        hour += 12
    elif half == "am" and hour == 12:
        hour = 0
    elif half is None and 1 <= hour <= 7:  # the spa is open in the day: "at 3" is 3 in the afternoon
        hour += 12
    return time(hour, minute) if hour < 24 and minute < 60 else None


def times_from(words: str) -> tuple[time | None, time | None] | None:
    """(after, before) for `words`: either may be None (no limit). None when the
    words say nothing about a time that can be read."""
    said = words.lower().strip()
    if found := re.search(rf"\bbetween {CLOCK} and {CLOCK}", said):
        return _clock(found[1]), _clock(found[2])
    after = re.search(rf"\b(?:after|from|not before) {CLOCK}", said)
    before = re.search(rf"\b(?:before|by|until|till|no later than) {CLOCK}", said)
    if after or before:
        return (_clock(after[1]) if after else None), (_clock(before[1]) if before else None)
    for part, limits in PARTS_OF_DAY.items():
        if part in said:
            return limits
    if ANY_TIME.search(said):
        return None, None
    if found := re.search(rf"\b(?:at |around )?{CLOCK}\b", said):  # "3pm": from then on
        if (start := _clock(found[1])) is not None:
            return start, None
    return None
