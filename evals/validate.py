"""Checks that the golden cases still match the guide, the schedule and the
bookings data, before any question is asked.

The cases are written against one version of the guide. When it changes (a
price goes up, a heading is renamed, a therapist leaves), a case would fail
with a confusing reason, or pass for the wrong one. This finds those cases
first and names them, so they can be updated:

  cites            every section must still be a heading in the guide
  mentions         (rag_qa) every text must still appear in the guide
  evidence         every line must still appear in the guide
  services,        (pebby) every service and therapist named, in the label or
  therapists       a seed booking, must still be in the schedule
  mentions_report  every period must lie within the bookings data, and every
                   figure must be one figures.py knows
  judge            must be a non-empty list of criteria

Nothing here calls a model.
"""

from spadesk.booking.schedule import read_schedule
from spadesk.business.store import get_knowledge
from spadesk.config import get_settings
from spadesk.reports.routes import get_dataset

from .figures import FIGURES
from .golden import Case


def stale_cases(cases: list[Case]) -> list[str]:
    """One line per problem found; empty when every case is up to date."""
    document = get_settings().data_path.read_text(encoding="utf-8").lower()
    headings = {name for pair in get_knowledge().outline for name in pair if name}
    schedule = read_schedule(get_knowledge())
    options = get_dataset().filter_options()
    problems = []

    for case in cases:
        name, answer = case["id"], case["expected"].get("answer", {})
        problems += [f"{name}: cites '{section}', which is no longer a heading"
                     for section in answer.get("cites", []) if section not in headings]
        if case["target"] == "rag_qa":  # Pebby's mentions may be the customer's own words, not the guide's
            problems += [f"{name}: expects '{text}', which is no longer in the guide"
                         for text in answer.get("mentions", []) if text.lower() not in document]
        problems += [f"{name}: evidence '{text}' is no longer in the guide"
                     for text in case.get("evidence", []) if text.lower() not in document]
        if "judge" in answer and not (isinstance(answer["judge"], list) and answer["judge"]):
            problems.append(f"{name}: judge must be a non-empty list of criteria")
        for spec in answer.get("mentions_report", []):
            if spec["start"] < options["min_date"] or spec["end"] > options["max_date"]:
                problems.append(f"{name}: {spec['start']} to {spec['end']} is outside the bookings data "
                                f"({options['min_date']} to {options['max_date']})")
            problems += [f"{name}: unknown figure '{figure}'" for figure in spec["figures"] if figure not in FIGURES]
        extraction, seed = case["expected"].get("extraction", {}), case.get("seed", [])
        services = extraction.get("services", []) + [booking["service"] for booking in seed]
        problems += [f"{name}: no service {service!r}" for service in services if service not in schedule.services]
        people = ([extraction["therapist"]] if "therapist" in extraction else []) + [b["therapist"] for b in seed]
        problems += [f"{name}: no therapist {person!r}" for person in people if person not in schedule.therapists]
    return problems
