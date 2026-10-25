"""Collecting a booking's details from what the customer said (code).

Pebby gathers the details in conversation and sends them with the booking
tool. The day and the time come as the customer's own words (`day_said`,
`time_said`). Before anything else, this code:

  1. grounds them: words the customer never wrote in this conversation
     are dropped, so a model cannot fill a detail they did not give. The
     same for the services: one counts only if the customer named it (a
     word of its name that no other service has: "prenatal", "swedish",
     "vitamin"), or named the package it is part of. "A facial" names none
     of the five facials, so Pebby is told to ask which one. A therapist
     counts only if the customer named them ("with Priya"); one they did not
     ask for is dropped, and whoever is free is offered.
  2. reads them (details.py): "Thursday" -> the coming Thursday,
     "after 3pm" -> after 15:00, "any time" -> no limit. In a change, what
     the customer does not change stays as it was: the day and time (when
     they give neither, or say "same"), the service, the add-ons (new ones
     are added), so "add a scalp massage to my booking" keeps the rest.
  3. lists what is still missing: a booking needs the services, the day
     and the time (which may be "any time"); a change also needs which
     booking. Pebby is told what to ask for, so the customer is always
     asked, and no detail is ever assumed.

The result is a BookingRequest for booking (code) (booking.py),
with the day and times as code worked them out.

Days and times are read from their words here too (day_from, times_from):
"Thursday" -> the coming Thursday (typos too: "thurday"), "30 October",
"tomorrow"; "after 3pm", "morning", "between 2 and 4", "any time". Words
that cannot be read count as not given, so the customer is asked.
"""

import json
import re
from collections import Counter
from datetime import date, time, timedelta
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from ..booking.calendars import Booking
from ..booking.schedule import read_schedule
from ..business.store import get_knowledge
from .booking import BookingRequest
from .rules import ANY_TIME, EVERYTHING, NAMED_BY, NEEDS, PARTS_OF_DAY
from .yes import is_yes

TIME_NOTE = "(Spa local time:"  # how each time note starts, so code can tell it from the customer's words


def _spoken(text: str) -> str:
    """What Pebby said in a reply: its `answer`, not the list under it (a yes to
    "which of these?" names none of them)."""
    try:
        reply = json.loads(text)
    except ValueError:
        return text
    return reply.get("answer", "") if isinstance(reply, dict) else text


def customer_words(messages: list[AnyMessage]) -> str:
    """Everything the customer said in the conversation, in lower case: what they
    wrote (not the time notes), and a question of Pebby's they said yes to ("would
    that be the Signature Radiance Facial?" "yes"): agreeing to it makes it theirs."""
    said, question = [], ""
    for message in messages:
        if isinstance(message, AIMessage) and message.text:
            question = _spoken(message.text)
        elif isinstance(message, HumanMessage) and not message.text.startswith(TIME_NOTE):
            said.append(message.text + (f" {question}" if question and is_yes(message.text) else ""))
            question = ""
    return " ".join(said).lower()


def _said(value: str, words: str) -> bool:
    """True when the customer wrote `value` (ignoring case, spaces and punctuation)."""
    def plain(text: str) -> str:
        return re.sub(r"[^a-z0-9:/]+", " ", text.lower()).strip()
    return bool(plain(value)) and plain(value) in plain(words)


def _words(text: str) -> set[str]:
    """The words that can pick something out, in lower case: words of two letters
    or more ('Deep Tissue Massage' -> deep, tissue, massage; single letters like the
    C of "Vitamin C" are left out), times ('5pm', '5:30pm', '17:00') and references."""
    text = text.lower()
    times = {t.replace(" ", "") for t in re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b", text)}
    return set(re.findall(r"bk-[0-9a-f]+|[a-z]{2,}", text)) | times


def _stem(word: str) -> str:
    """A word's first four letters, so its forms match: "calm" and "calming",
    "aroma" and "aromatherapy", "massages" and "massage". Times and references
    are kept whole."""
    return word[:4] if word.isalpha() and len(word) > 4 else word


def picked_out(things: dict[str, set[str]], said: set[str], groups: dict[str, set[str]] | None = None) -> list[str]:
    """The things the customer's words pick out, among `things` (each a key and its
    words). A thing is picked out by a word only it has ("prenatal" among the
    services; "6pm" or "monday" among their bookings), or by all its words that
    are not a kind (a word in NAMED_BY things or more: "massage", "facial"), as
    "full leg" picks out Full Leg Wax. A group's name picks out all its members:
    a package's name, its services or its bookings (`groups`: each key's group words)."""
    things = {key: {_stem(word) for word in words} for key, words in things.items()}
    said = {_stem(word) for word in said}
    groups = {key: {_stem(word) for word in words} for key, words in (groups or {}).items()}
    counts = Counter(word for words in things.values() for word in words)

    def picked(key: str) -> bool:
        words = things[key]
        if any(counts[word] == 1 for word in words & said) or groups.get(key, set()) & said:
            return True
        specific = {word for word in words if counts[word] < NAMED_BY}
        return bool(specific) and specific <= said
    return [key for key in things if picked(key)]


def named_services(services: list[str], package: str, words: str) -> tuple[list[str], list[str]]:
    """(the services the customer named, the ones they did not): picked out among all
    the spa's services, or all of a package's when they named the package."""
    every = {name: _words(name) for name in read_schedule(get_knowledge()).services}
    group = _words(package) - {"package"}
    named = set(picked_out(every | {s: _words(s) for s in services}, _words(words),
                           {s: group for s in services}))
    return [s for s in services if s in named], [s for s in services if s not in named]


def named_add_ons(add_ons: list[str], words: str) -> list[str]:
    """The add-ons the customer named, picked out among all the spa's add-ons."""
    every = {name: _words(name) for name in read_schedule(get_knowledge()).add_ons}
    picked = set(picked_out(every, _words(words)))
    return [a for a in add_ons if a in picked]


def _booking_words(booking: Booking) -> set[str]:
    """What can pick out a booking: its service, its day, its time, its reference."""
    start = booking.start
    hour = f"{start:%I}".lstrip("0") + (f":{start:%M}" if start.minute else "")
    return _words(booking.service) | {f"{start:%A}".lower(), f"{hour}{start:%p}".lower(), f"{start:%H:%M}",
                                      booking.id.lower()}


def named_bookings(references: list[str], words: str, upcoming: list[Booking]) -> tuple[list[str], list[str]]:
    """(the bookings the customer named, the ones they did not), of the `references`
    Pebby chose: picked out among their `upcoming` bookings ("the massage", "the 6pm
    one", "monday's massage", a package's name), or all of them when they said so
    ("both", "all") or have only one. So a model cannot cancel the wrong one."""
    said = _words(words)
    if EVERYTHING & said or len(upcoming) == 1:
        named = {b.id for b in upcoming}
    else:
        named = set(picked_out({b.id: _booking_words(b) for b in upcoming}, said,
                               {b.id: _words(b.package) - {"package"} for b in upcoming if b.package}))
    return [r for r in references if r in named], [r for r in references if r not in named]


def collect(args: dict[str, Any], words: str, today: date,
            upcoming: list[Booking] = ()) -> tuple[BookingRequest, list[str], list[str]]:
    """The BookingRequest for Pebby's `args`, what is still missing (in the order
    to ask for it), and the services Pebby chose that the customer did not name
    (to offer as choices). `words`: what the customer wrote (customer_words).
    `upcoming`: the customer's bookings, for a change or a cancellation."""
    references, _ = named_bookings(args.get("references", []), words, list(upcoming))
    unnamed_bookings = len(references) < len(args.get("references", []))
    changing = next((b for b in upcoming if b.id == references[0]), None) \
        if args["intent"] == "change" and references else None
    day_said = args.get("day_said", "") if _said(args.get("day_said", ""), words) else ""
    time_said = args.get("time_said", "") if _said(args.get("time_said", ""), words) else ""
    day = day_from(day_said, today) if day_said else None
    times = times_from(time_said) if time_said else None
    kept: list[str] = []
    if changing:  # a change: what the customer does not change stays as it was
        unchanged = not day_said and not time_said
        if unchanged or "same" in day_said.lower():
            day = changing.start.date()
        if unchanged or "same" in time_said.lower():
            times = (changing.start.time().replace(tzinfo=None), None)  # from its start
        kept = [changing.service]  # theirs already: not named again
    after, before = times or (None, None)
    services = args.get("services") or kept
    named, unnamed = named_services([s for s in services if s not in kept], args.get("package", ""), words)
    named = [s for s in services if s in kept or s in named]
    add_ons = list(changing.add_ons) if changing else []
    add_ons += [a for a in named_add_ons(args.get("add_ons", []), words) if a not in add_ons]
    request = BookingRequest(
        intent=args["intent"], services=named, package=args.get("package", ""),
        day=day.isoformat() if day else "", after=f"{after:%H:%M}" if after else "",
        before=f"{before:%H:%M}" if before else "",
        therapist=args.get("therapist", "") if _words(args.get("therapist", "")) & _words(words) else "",
        references=references, add_ons=add_ons)
    # Services the customer did not name are dropped; only when none they named is left is it missing.
    have = {"services": bool(request.services), "day": day is not None, "time": times is not None,
            "references": bool(references) and not unnamed_bookings}
    return request, [need for need in NEEDS[request.intent] if not have[need]], unnamed


DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
# A time: "3", "3pm", "3:30 pm", "15:00", "noon".
CLOCK = r"(noon|\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?)"


def _month(word: str) -> int | None:
    """'oct' or 'October' -> 10."""
    return next((n for n, name in enumerate(MONTHS, 1) if name.startswith(word[:3])), None) if len(word) >= 3 else None


def _day_number(word: str) -> int | None:
    """The day `word` names (0 = Monday): one it starts ("thu", "thurs", "wednes"),
    or else the closest by spelling, as a typo is ("thurday"); None for a word that
    only shares its first letters ("month")."""
    for n, name in enumerate(DAYS):
        if name.startswith(word):
            return n
    score, n = max((SequenceMatcher(None, word, name).ratio(), n) for n, name in enumerate(DAYS))
    return n if score >= 0.8 else None


def day_from(words: str, today: date) -> date | None:
    """The day `words` name, or None when they name none (or one that cannot be read)."""
    said = words.lower().strip()
    if "day after tomorrow" in said:
        return today + timedelta(days=2)
    if "tomorrow" in said:
        return today + timedelta(days=1)
    if "today" in said or "tonight" in said:
        return today
    for word in re.findall(r"[a-z]{3,}", said):
        if (n := _day_number(word)) is not None:
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
