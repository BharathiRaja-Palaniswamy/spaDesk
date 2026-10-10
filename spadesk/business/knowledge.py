"""Loads the spa business document (markdown) and reads its facts.

The document is the single source of business facts for the customer site:
the landing page shows what `public()` returns. The bookings generator
(scripts/generate_bookings.py) reads services, prices, hours and offers from
here too, which is how the dashboard's data stays in line with the document.

How the document is read:
- A block that opens with a line that is entirely bold ("**Swedish Massage**")
  is an item: a service, package, offer or FAQ. "**Label:** value" lines inside
  it (Duration, Price) become its fields.
- Every other block is collected under its heading.

A "block" is a run of lines with no blank line between them. Each item and
each heading's text is also kept as a `Chunk`, a titled piece of the document
that can be looked up by title with `find()`.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Patterns for the kinds of line the parser cares about.
HEADING = re.compile(r"^(#{1,3})\s+(.+)$")  # "## Core Services"
BOLD_LINE = re.compile(r"^\*\*([^*]+)\*\*$")  # "**Swedish Massage**" (the whole line is bold)
FIELD = re.compile(r"^\*\*([^*]+):\*\*\s*(.+)$")  # "**Price:** $115 / $160"
PRICED_LINE = re.compile(r"^\*\*([^*]+)\*\*\s+[—–-]\s+(\S.*)$")  # "**Lip Wax** — $18"


@dataclass(frozen=True)
class Chunk:
    """One titled piece of the document: an item, or the text under a heading."""

    id: str  # stable name made from the title, e.g. "swedish-massage"
    title: str
    text: str  # the piece as plain text, with its heading path
    lines: tuple[str, ...] = ()  # the same content line by line


@dataclass
class Item:
    """A named entry in the document. With a price it is something bookable."""

    name: str
    category: str  # the heading it sits under, e.g. "Massage Therapy"
    section: str = ""  # the top-level heading above that, e.g. "Core Services"
    description: str = ""
    fields: dict[str, str] = field(default_factory=dict)  # {"Duration": "60 minutes", "Price": "$125"}
    inline: bool = False  # written on one line in the document: "**Lip Wax** — $18"

    def get(self, word: str) -> str:
        """The value of the first field whose label contains `word`, or ""."""
        # Labels vary ("Price", "Package Price", "Membership Price"), so match by word.
        return next((v for k, v in self.fields.items() if word in k.lower()), "")

    @property
    def priced(self) -> bool:
        """True when the entry has a price, which makes it something bookable."""
        return bool(self.get("price"))


def _plain(line: str) -> str:
    """Strip markdown markers: '- **Phone:** 555' becomes 'Phone: 555'."""
    return re.sub(r"^[-*]\s+", "", line.replace("**", "")).strip()


def slug(text: str) -> str:
    """'Kids Hand & Foot Treat' becomes 'kids-hand-foot-treat'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class KnowledgeBase:
    """The business document, parsed.

      chunks   every titled piece of the document, found by title with find()
      items    named entries (services, packages, offers, FAQs) with their fields
      facts    loose "Label: value" lines such as the phone number
      outline  every (heading, sub-heading) pair, in document order
      name     the business name, from the top heading
    """

    def __init__(self, path: Path):
        """Read and parse the document at `path`."""
        self.name = "the spa"  # replaced by the document's top heading
        self.facts: dict[str, str] = {}  # "**Phone:** ..." lines found outside items
        self.items: list[Item] = []
        self.chunks: list[Chunk] = []
        # Every (heading, sub-heading) that has content, in document order.
        self.outline: list[tuple[str, str]] = []
        # Lines that are not part of an item, grouped by (heading, sub-heading).
        self.section_lines: dict[tuple[str, str], list[str]] = {}
        # The (heading, sub-heading) pairs whose lines were written as a bullet list.
        self.bulleted: set[tuple[str, str]] = set()
        self._parse(Path(path).read_text(encoding="utf-8"))
        self._by_title = {c.title: c for c in self.chunks}

    @property
    def service_names(self) -> list[str]:
        """Everything with a price: what the booking form lets a customer pick."""
        return [i.name for i in self.items if i.priced]

    def find(self, title: str) -> Chunk | None:
        """The piece of the document with exactly this title, or None."""
        return self._by_title.get(title)

    def public(self, featured: tuple[str, ...], highlights: tuple[str, ...], hours: str, about: str) -> dict[str, Any]:
        """Everything the landing page shows, taken from the document.

        The arguments are titles from config.py that say which parts to show.
        """
        priced = {i.name: i for i in self.items if i.priced}
        hours_chunk, about_chunk = self.find(hours), self.find(about)
        return {
            "name": self.name,
            "about": about_chunk.lines[0] if about_chunk else "",
            "contact": self.facts,
            # "Friday: 10:00 AM – 7:00 PM" -> {"days": "Friday", "times": "10:00 AM – 7:00 PM"}
            "hours": [dict(zip(("days", "times"), line.split(": ", 1))) for line in hours_chunk.lines]
            if hours_chunk else [],
            "services": [
                {"name": i.name, "category": i.category, "description": i.description,
                 "duration": i.get("duration"), "price": i.get("price")}
                for i in priced.values()
            ],
            "featured": [name for name in featured if name in priced],
            "highlights": [{"title": c.title, "points": list(c.lines)}
                           for c in map(self.find, highlights) if c],
        }

    def _parse(self, text: str) -> None:
        """Turn the document text into `chunks`, `items` and `facts`, in two passes."""
        # Pass 1: walk the document line by line, remembering which headings we
        # are under and collecting lines into blocks. A blank line or a heading
        # ends the current block. (The extra "" makes sure the last block ends.)
        h2 = h3 = ""
        block: list[str] = []
        for raw in text.splitlines() + [""]:
            line = raw.strip()
            heading = HEADING.match(line)
            if heading or not line:
                if block:
                    self._add_block(block, h2, h3)
                block = []
                if heading:
                    level, title = len(heading[1]), heading[2].strip()
                    if level == 1:
                        # "# Pebble Spa — Business Overview" -> "Pebble Spa"
                        self.name = re.split(r"\s+[—–-]\s+", title)[0]
                    elif level == 2:
                        h2, h3 = title, ""
                    else:
                        h3 = title
            else:
                block.append(line)

        # Pass 2: one chunk per heading for everything that was not an item
        # (policies, add-ons, the hair-removal price list, the about text).
        for (h2, h3), lines in self.section_lines.items():
            title = h3 or h2
            path = f"{h2} > {h3}" if h3 else h2
            self._add_chunk(title, f"{path}.\n" + "\n".join(lines), lines)

    def _add_block(self, block: list[str], h2: str, h3: str) -> None:
        """Handle one block of lines found under headings `h2` > `h3`: either
        record it as an item with its own chunk, or keep its lines for the
        heading's chunk.
        """
        category = h3 or h2
        path = f"{h2} > {h3}" if h3 else h2
        if (h2, h3) not in self.outline:
            self.outline.append((h2, h3))
        title = BOLD_LINE.match(block[0])
        if title and len(block) > 1:
            # An item: the bold first line is its name, the rest describes it.
            item = Item(title[1], category, section=h2)
            lines = []
            for line in block[1:]:
                if found := FIELD.match(line):
                    item.fields[found[1]] = found[2]
                elif not line.startswith("- "):
                    item.description = f"{item.description} {_plain(line)}".strip()
                lines.append(_plain(line))
            self.items.append(item)
            # The heading path goes into the text so the chunk carries its own
            # context: "Swedish Massage (Core Services > Massage Therapy)".
            self._add_chunk(item.name, f"{item.name} ({path}).\n" + "\n".join(lines), lines)
            return
        # Not an item: keep the lines for this heading's chunk (built in pass 2),
        # and pick out anything structured along the way.
        if block[0].startswith("- "):
            self.bulleted.add((h2, h3))
        for line in block:
            if found := PRICED_LINE.match(line):
                self.items.append(Item(found[1], category, section=h2, fields={"Price": found[2]}, inline=True))
            elif found := FIELD.match(line):
                self.facts[found[1]] = found[2]
            self.section_lines.setdefault((h2, h3), []).append(_plain(line))

    def _add_chunk(self, title: str, text: str, lines: list[str]) -> None:
        """Add one chunk. Its id is made from the title."""
        id = slug(title)
        taken = {c.id for c in self.chunks}
        if id in taken:  # two entries with the same title: keep ids unique
            id = f"{id}-{len(taken)}"
        self.chunks.append(Chunk(id, title, text, tuple(lines)))
