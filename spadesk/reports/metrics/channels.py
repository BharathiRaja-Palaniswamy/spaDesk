"""Performance per source channel (how the customer reached the spa)."""

from typing import Any

from ..dataset import Dataset
from .common import Booking, group_by, totals


def by_channel(rows: list[Booking], dataset: Dataset) -> list[dict[str, Any]]:
    """One row per channel, in the order the data file lists them, so a channel
    keeps its place on the chart whatever the filters are."""
    groups = group_by(rows, lambda b: b["channel"])
    return [{"channel": channel, **totals(groups.get(channel, []))} for channel in dataset.channels]
