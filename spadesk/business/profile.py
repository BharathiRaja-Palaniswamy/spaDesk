"""Builds the business profile shown on the manage page's "Business profile" tab.

The profile is the business document laid out the way it is written: its
top-level headings become sections, in document order, and each section holds
one group per sub-heading (or a single untitled group when it has none).

    {"name": "Family Spa", "sections": [
        {"title": "Core Services", "groups": [
            {"title": "Facials", "lines": [], "bulleted": False, "items": [
                {"name": "Glow Express Facial", "description": "...",
                 "duration": "30 minutes", "price": "$80", "lines": []}, ...]},
            ...]},
        ...]}

A group can carry plain `lines` (the text under the heading), `items` (the
named entries under it), or both. Nothing is invented or summarised here: it
is the same parsed document the customer site reads, in a fuller shape.
"""

from typing import Any

from .knowledge import Item, KnowledgeBase


def build_profile(kb: KnowledgeBase) -> dict[str, Any]:
    """The whole document as sections and groups, ready to be sent as JSON."""
    sections: dict[str, dict[str, Any]] = {}
    for heading, subheading in kb.outline:
        group = {
            "title": subheading,
            "lines": kb.section_lines.get((heading, subheading), []),
            # Whether those lines were a bullet list in the document, or paragraphs.
            "bulleted": (heading, subheading) in kb.bulleted,
            "items": [
                _item(kb, item) for item in kb.items
                # One-line entries ("**Lip Wax** — $18") are already in `lines`,
                # so they are not repeated as items.
                if item.section == heading and item.category == (subheading or heading) and not item.inline
            ],
        }
        sections.setdefault(heading, {"title": heading, "groups": []})["groups"].append(group)
    return {"name": kb.name, "sections": list(sections.values())}


def _item(kb: KnowledgeBase, item: Item) -> dict[str, Any]:
    """One named entry. A priced entry (a service or package) is described by
    its fields; any other entry (an FAQ, an offer) by its own lines of text."""
    chunk = kb.find(item.name)
    return {
        "name": item.name,
        "description": item.description,
        "duration": item.get("duration"),
        "price": item.get("price"),
        "lines": [] if item.priced or chunk is None else list(chunk.lines),
    }
