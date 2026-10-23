"""What can be booked, with whom, and when: all read from the business document.

The document is the source of truth for the schedule; the calendar
(calendars/) holds only what changes day to day: the bookings, and later
time off. So the two never disagree about who works when.

From the document:
  services    every entry with a Duration and a Price: one person, one
              appointment ("Swedish Massage", 60 or 90 minutes). Group yoga
              classes (a "Drop-in Price") and entries with no duration
              (packages, passes, memberships) are not booked here.
  therapists  each member of the team, the days they work ("Works:") and the
              services they offer ("Specialties:"). A specialty matches a
              service by name ("private yoga" -> Private Yoga Session) or by
              naming its category ("body treatments" -> both body
              treatments; "hair removal (waxing)" -> every wax).
  hours       the spa's opening hours for each day of the week.
  classes     the weekly class timetable: an instructor teaching a class is
              not free for an appointment at that time.
"""

import re
from dataclasses import dataclass
from datetime import datetime, time

from ..business.knowledge import Item, KnowledgeBase
from ..config import HOURS_BLOCK, TEAM_SECTION

# The days of the week in Python's order: datetime.weekday() is the position here.
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
CLOCK = "%I:%M %p"  # how the document writes a time: "10:00 AM"
DASH = re.compile(r"\s+[–-]\s+")  # "10:00 AM – 8:00 PM"
# "7:00 AM – Sunrise Hatha Flow with Maya Chen"
CLASS_LINE = re.compile(r"^(\d{1,2}:\d{2} [AP]M)\s+[–-]\s+(.+) with (.+)$")


@dataclass(frozen=True)
class Service:
    """Something one person books with one therapist."""

    name: str
    category: str  # the heading it sits under, e.g. "Massage Therapy"
    minutes: tuple[int, ...]  # its lengths, standard first: (60, 90)
    prices: tuple[int, ...]  # the price of each length, in dollars: (115, 160)


@dataclass(frozen=True)
class Therapist:
    """A member of the team who takes appointments."""

    name: str
    days: frozenset[int]  # the weekdays they work, 0 = Monday
    services: tuple[str, ...]  # the names of the services they offer


@dataclass(frozen=True)
class Lesson:
    """One class on the weekly timetable."""

    day: int  # 0 = Monday
    start: time
    minutes: int
    name: str  # e.g. "Prenatal Yoga"
    teacher: str


@dataclass(frozen=True)
class Schedule:
    """Everything the document says about booking, ready to use."""

    services: dict[str, Service]  # by name
    therapists: dict[str, Therapist]  # by name
    hours: dict[int, tuple[time, time]]  # weekday -> (opens, closes); a missing day is closed
    lessons: tuple[Lesson, ...]

    def therapists_for(self, service: str) -> list[Therapist]:
        """Who offers `service`, in the document's order."""
        return [t for t in self.therapists.values() if service in t.services]


def _numbers(text: str) -> tuple[int, ...]:
    """'60 minutes / 90 minutes' -> (60, 90); '$115 / $160' -> (115, 160)."""
    return tuple(int(n) for n in re.findall(r"\d+", text))


def _clock(text: str) -> time:
    """'10:00 AM' -> 10:00."""
    return datetime.strptime(text.strip(), CLOCK).time()


def _days(text: str) -> list[int]:
    """'Monday to Thursday' -> [0, 1, 2, 3]; 'Friday' -> [4]."""
    first, _, last = text.partition(" to ")
    start, end = DAYS.index(first.strip()), DAYS.index((last or first).strip())
    return list(range(start, end + 1))


def _services(kb: KnowledgeBase) -> dict[str, Service]:
    """Every entry with a duration and a plain "Price" field."""
    return {item.name: Service(item.name, item.category, _numbers(item.get("duration")),
                               _numbers(item.fields["Price"]))
            for item in kb.items if item.get("duration") and "Price" in item.fields}


def _offers(specialty: str, service: Service) -> bool:
    """True when the specialty, as the document writes it, is part of this
    service's name or names its category."""
    specialty = specialty.strip().lower()
    return specialty in service.name.lower() or service.category.lower() in specialty


def _therapist(item: Item, services: dict[str, Service]) -> Therapist:
    """A team member from their entry under Our Team."""
    specialties = item.get("specialties").split(",")
    days = frozenset(DAYS.index(day.strip()) for day in item.get("works").split(","))
    offered = tuple(s.name for s in services.values() if any(_offers(sp, s) for sp in specialties))
    return Therapist(item.name, days, offered)


def _hours(kb: KnowledgeBase) -> dict[int, tuple[time, time]]:
    """'Monday to Thursday: 10:00 AM – 8:00 PM' -> Monday..Thursday open 10:00 to 20:00."""
    hours = {}
    for line in kb.find(HOURS_BLOCK).lines:
        days, times = line.split(": ", 1)
        opens, closes = DASH.split(times)
        hours |= {day: (_clock(opens), _clock(closes)) for day in _days(days)}
    return hours


def _lessons(kb: KnowledgeBase) -> tuple[Lesson, ...]:
    """The classes on the timetable, each with its length from its own entry."""
    lengths = {item.name: _numbers(item.get("duration"))[0] for item in kb.items if item.get("duration")}
    lessons = []
    for day, name in enumerate(DAYS):
        # Each day of the timetable is a block titled with the day's name ("**Monday**").
        chunk = kb.find(name)
        for line in chunk.lines if chunk else ():
            if found := CLASS_LINE.match(line):
                start, title, teacher = found.groups()
                lessons.append(Lesson(day, _clock(start), lengths[title], title, teacher.strip()))
    return tuple(lessons)


def read_schedule(kb: KnowledgeBase) -> Schedule:
    """The booking schedule described by the parsed business document."""
    services = _services(kb)
    therapists = {item.name: _therapist(item, services) for item in kb.items
                  if item.category == TEAM_SECTION and item.get("works")}
    return Schedule(services, therapists, _hours(kb), _lessons(kb))
