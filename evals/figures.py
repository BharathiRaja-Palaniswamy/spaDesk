"""Expected report figures, worked out when the eval runs instead of written
into cases.json.

The bookings data is generated and changes (scripts/generate_bookings.py), so
a figure typed into a case, such as "22,614", would go stale. A case says
instead WHICH figure it expects:

    "mentions_report": [{"start": "2026-09-01", "end": "2026-09-30", "figures": ["revenue"]}]

and this file runs the same report the owner's assistant gets from its tool
(build_report) and reads the figure from it. The check then looks for that
value in the answer. This is the usual approach when the truth lives in data
that moves: store how to get the answer, not the answer.
"""

from collections.abc import Callable
from datetime import date
from typing import Any

from spadesk.reports.report import build_report
from spadesk.reports.routes import get_dataset

Report = dict[str, Any]


def _top(rows: list[dict[str, Any]], name: str) -> list[Any]:
    """The row with the highest revenue: its name and its revenue."""
    best = max(rows, key=lambda row: row["revenue"])
    return [best[name], best["revenue"]]


def _customer_types(report: Report) -> list[Any]:
    """How many new and how many returning customers, in that order."""
    counts = {row["customer_type"]: row["customers"] for row in report["customers"]["by_type"]}
    return [counts["New"], counts["Returning"]]


# A figure's name in cases.json -> how to read it from a report. Each returns
# a list, because some figures are more than one value (a name and an amount).
FIGURES: dict[str, Callable[[Report], list[Any]]] = {
    "revenue": lambda report: [report["summary"]["revenue"]],
    "lost_revenue": lambda report: [report["summary"]["lost_revenue"]],
    "top_category": lambda report: _top(report["categories"], "category"),
    "top_channel": lambda report: _top(report["channels"], "channel"),
    "customer_types": _customer_types,
}


def as_text(value: Any) -> str:
    """How a value is expected to appear in an answer: whole numbers with
    thousands separators ("22,614", which "$22,614" also contains)."""
    return f"{value:,}" if isinstance(value, int) else str(value)


def expected_texts(spec: dict[str, Any]) -> list[str]:
    """The texts one report spec expects, e.g. ["22,614"] for September's revenue."""
    report = build_report(get_dataset(), date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"]),
                          spec.get("category"), spec.get("channel"))
    return [as_text(value) for figure in spec["figures"] for value in FIGURES[figure](report)]
