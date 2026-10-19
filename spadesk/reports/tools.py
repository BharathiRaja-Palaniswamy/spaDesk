"""The assistant's tool: `get_report`, which lets the model ask for report figures.

A tool is a function the model can ask the server to run. The model never
runs code or reads the bookings file itself. It is sent a description of the
tool (`report_tool` below); when a question needs figures, it replies with a
request, "run get_report for 1 to 30 September"; the server runs the
function (`run_report`) and sends the result back; then the model writes its
answer from those figures.

This is how the assistant can answer about bookings without 2,490 rows ever
going into the prompt, and why the arithmetic is always right: it is done by
the same code that fills the Reports tab (reports/report.py).
"""

from datetime import date
from typing import Any

from langsmith import traceable

from .dataset import Dataset
from .report import build_report

TOOL_NAME = "get_report"

# How many times in one question the model may ask for tools before it must
# answer. A comparison needs two reports; this leaves room and still ends.
MAX_TOOL_ROUNDS = 4


def report_tool(dataset: Dataset) -> dict[str, Any]:
    """The description of the tool that is sent to the model: its name, when
    to use it, and its inputs as a JSON schema.

    The description is how the model decides whether to call the tool, so it
    says what the tool is for and what the data covers. The valid categories
    and channels are listed in the schema (`enum`), so the model cannot ask
    for one that does not exist.
    """
    options = dataset.filter_options()
    return {
        "name": TOOL_NAME,
        "description": (
            "Get the business's figures for a period: bookings, completed bookings, revenue, revenue "
            "lost to cancellations and no-shows, average rating and average ticket, each compared with "
            "the period just before; a trend over time; and breakdowns by service category, by service, "
            "by channel, by new and returning customers, and customer feedback. Use it for any question "
            "about how the business is doing or has done. Call it again with other dates or filters to "
            f"compare. The data covers {options['min_date']} to {options['max_date']} and nothing outside it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "First day of the period, as YYYY-MM-DD."},
                "end": {"type": "string", "description": "Last day of the period (included), as YYYY-MM-DD."},
                "category": {"type": "string", "enum": options["categories"],
                             "description": "Only bookings for this service category. Leave out for all."},
                "channel": {"type": "string", "enum": options["channels"],
                            "description": "Only bookings that came through this channel. Leave out for all."},
            },
            "required": ["start", "end"],
        },
    }


# Recorded as a step of the trace when tracing is on (see spadesk/tracing.py).
@traceable(run_type="tool", name=TOOL_NAME)
def run_report(dataset: Dataset, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the tool with the inputs the model asked for.

    Returns the report, or {"error": "..."} when the inputs are not usable.
    The error goes back to the model as the tool's result, so it can correct
    itself or tell the owner, rather than the whole request failing.
    """
    try:
        start, end = date.fromisoformat(arguments["start"]), date.fromisoformat(arguments["end"])
    except (KeyError, TypeError, ValueError):
        return {"error": "start and end are required, as YYYY-MM-DD dates."}
    category, channel = arguments.get("category"), arguments.get("channel")

    options = dataset.filter_options()
    problem = dataset.filter_problem(start, end, category, channel)
    # A period with no data at all would come back as a report full of zeros,
    # which reads as "no business". Say what is really going on.
    if not problem and (end.isoformat() < options["min_date"] or start.isoformat() > options["max_date"]):
        problem = f"There is no data for that period. The data covers {options['min_date']} to {options['max_date']}."
    return {"error": problem} if problem else build_report(dataset, start, end, category, channel)


def report_label(arguments: dict[str, Any]) -> str:
    """A short description of one report the model asked for, shown under the
    answer so the owner can see which figures it rests on,
    e.g. "2026-09-01 to 2026-09-30, Facials".
    """
    filters = [arguments[name] for name in ("category", "channel") if arguments.get(name)]
    return ", ".join([f"{arguments.get('start')} to {arguments.get('end')}", *filters])
