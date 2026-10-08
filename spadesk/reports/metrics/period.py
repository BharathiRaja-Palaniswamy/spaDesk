"""Date maths for the dashboard: comparing with the previous period, and
choosing how finely to slice a trend."""

from datetime import date, timedelta


def previous_period(start: date, end: date) -> tuple[date, date]:
    """The period of the same length that ends the day before `start`.

    1-30 September (30 days) -> 2-31 August (30 days).
    """
    days = (end - start).days + 1
    return start - timedelta(days=days), start - timedelta(days=1)


def bucket_size(start: date, end: date) -> str:
    """How to slice a trend so it has a readable number of points:
    up to ~6 weeks by day, up to ~6 months by week, longer by month."""
    days = (end - start).days + 1
    if days <= 45:
        return "day"
    if days <= 190:
        return "week"
    return "month"


def bucket_start(day: date, size: str) -> date:
    """The first day of the bucket that `day` falls in (weeks start on Monday)."""
    if size == "week":
        return day - timedelta(days=day.weekday())
    if size == "month":
        return day.replace(day=1)
    return day


def buckets(start: date, end: date, size: str) -> list[date]:
    """The start date of every bucket from `start` to `end`, in order."""
    result = []
    day = bucket_start(start, size)
    while day <= end:
        result.append(day)
        if size == "day":
            day += timedelta(days=1)
        elif size == "week":
            day += timedelta(days=7)
        else:  # first day of the next month
            day = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result
