"""Checks that the cases still match the business document and the bookings
data, before any question is asked.

The cases are written against one version of the document. When it changes
(a price goes up, a heading is renamed), a case would fail with a confusing
reason, or pass for the wrong one. This finds those cases first and names
them, so they can be updated:

  cites            every section must still be a heading in the document
  mentions         every text must still appear in the document
  mentions_report  every period must lie within the bookings data, and every
                   figure must be one figures.py knows
  judge            must be a non-empty list of criteria

Nothing here calls the model.
"""

from typing import Any

from spadesk.business.store import get_knowledge
from spadesk.config import get_settings
from spadesk.reports.routes import get_dataset

from .figures import FIGURES


def stale_cases(cases: list[dict[str, Any]]) -> list[str]:
    """One line per problem found; empty when every case is up to date."""
    document = get_settings().data_path.read_text(encoding="utf-8").lower()
    headings = {name for pair in get_knowledge().outline for name in pair if name}
    options = get_dataset().filter_options()
    problems = []

    for case in cases:
        expect = case["expect"]
        problems += [f"{case['id']}: cites '{section}', which is no longer a heading"
                     for section in expect.get("cites", []) if section not in headings]
        problems += [f"{case['id']}: expects '{text}', which is no longer in the document"
                     for text in expect.get("mentions", []) if text.lower() not in document]
        if "judge" in expect and not (isinstance(expect["judge"], list) and expect["judge"]):
            problems.append(f"{case['id']}: judge must be a non-empty list of criteria")
        for spec in expect.get("mentions_report", []):
            if spec["start"] < options["min_date"] or spec["end"] > options["max_date"]:
                problems.append(f"{case['id']}: {spec['start']} to {spec['end']} is outside the bookings data "
                                f"({options['min_date']} to {options['max_date']})")
            problems += [f"{case['id']}: unknown figure '{figure}'"
                         for figure in spec["figures"] if figure not in FIGURES]
    return problems
