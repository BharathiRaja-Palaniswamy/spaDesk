"""Step 1b of RAG: a header for each chunk, used only to find it.

Chunks are cut by size (chunking.py), so a chunk can start in the middle of
a section, with its heading left in the chunk before. Its vector then
stands for whatever most of its text is about. In this document one chunk
starts with the yoga studio's Sunday hours and goes on to list facials: it
looks like a facials chunk, and a question about yoga hours ranks it 21st.

The fix is to write, above each chunk, where it sits in the document, as a
path from the top (a "contextual chunk header"):

    Location and Hours > Yoga Studio Hours | Core Services > Facials > Glow Express Facial, ...
    section (##)       > block title (bold) | section > subsection (###) > block titles

One path per part of the document the chunk covers, joined by " | ". The
document's title (#) is left out: it is the same for every chunk.

The header is used for search only: it is embedded with the chunk and read
by the reranker. The model is given the chunk's own text, without it, so the
prompt, the answers and their cost do not change.

Two guards keep it from costing text. The embedding model reads at most
`max_tokens` tokens and silently ignores the rest, and the header comes
first, so an overlong header would push the end of the chunk out of sight:
  - a budget: each header is at most SPADESK_HEADER_TOKENS tokens (0 turns
    headers off). A longer one is shortened, least useful part first.
  - a check: chunk size + header budget must fit the model, and every text
    embedded is measured when the index is built.
"""

import re
from dataclasses import dataclass, replace
from functools import lru_cache

import tiktoken

from .chunking import ENCODING, HEADING, Chunk

# A block's title: a line that is bold and nothing else, like "**Swedish Massage**".
# Labels inside a block, like "**Price:** $115", have a colon and are not titles.
TITLE = re.compile(r"^\*\*([^*:]+)\*\*\s*$")
LEVELS = " > "  # between the steps of one path
PLACES = " | "  # between the paths of one header
MORE = "…"  # shown where titles were left out to fit the budget
# Part of the index signature: changing how headers are written rebuilds the index.
FORMAT = "breadcrumb-titles-1"


@dataclass(frozen=True)
class Line:
    """Where one line of content sits in the document."""

    path: tuple[str, ...]  # its section and subsection, e.g. ("Core Services", "Facials")
    title: str  # the title of the block it belongs to, or "" if none
    start: int  # its first character in the document
    end: int  # one past its last character


def document_lines(text: str) -> list[Line]:
    """Every line of content (not blank, not a heading) with its place in the document."""
    lines, position = [], 0
    section = subsection = title = ""
    for raw in text.splitlines(keepends=True):
        line = raw.strip()
        if heading := HEADING.match(line):
            level, name = len(heading[1]), heading[2].strip()
            if level == 2:
                section, subsection = name, ""
            elif level == 3:
                subsection = name
            title = ""  # a block never runs past a heading
        elif line:
            if match := TITLE.match(line):
                title = match[1].strip()
            path = tuple(step for step in (section, subsection) if step)
            if path:
                lines.append(Line(path, title, position, position + len(raw)))
        position += len(raw)
    return lines


@lru_cache  # the encoding is loaded once
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """How many tokens `text` holds, counted as the chunks are (a close
    estimate of the embedding model's own count)."""
    return len(_encoding().encode(text))


def header_for(lines: list[Line], begin: int, end: int, budget: int) -> str:
    """The header of the chunk that spans characters `begin` to `end`, at most
    `budget` tokens. "" when the budget is 0."""
    if budget <= 0:
        return ""
    places: dict[tuple[str, ...], list[str]] = {}  # path -> block titles, in document order
    for line in lines:
        if line.start < end and line.end > begin:
            titles = places.setdefault(line.path, [])
            if line.title and line.title not in titles:
                titles.append(line.title)
    if not places:
        return ""
    for header in _shorter_and_shorter(places):
        if count_tokens(header) <= budget:
            return header
    # Even the shortest form is too long: cut it to the budget.
    return _encoding().decode(_encoding().encode(header)[:budget - 1]) + MORE


def _shorter_and_shorter(places: dict[tuple[str, ...], list[str]]):
    """The header written in full, then shorter forms, dropping the least
    useful part first: the parent section, then all but the first title,
    then the titles altogether."""
    def written(full_path: bool, titles: str) -> str:
        """The header with the whole path or its last step, and "all", the
        "first" or "no" block titles."""
        paths = []
        for path, names in places.items():
            steps = list(path if full_path else path[-1:])
            if names and titles == "all":
                steps.append(", ".join(names))
            elif names and titles == "first":
                steps.append(names[0] + (", " + MORE if len(names) > 1 else ""))
            paths.append(LEVELS.join(steps))
        return PLACES.join(paths)

    yield written(full_path=True, titles="all")
    yield written(full_path=False, titles="all")
    yield written(full_path=False, titles="first")
    yield written(full_path=False, titles="no")


def with_headers(text: str, chunks: list[Chunk], budget: int) -> list[Chunk]:
    """The chunks of `text`, each given its header."""
    lines, begin, headed = document_lines(text), 0, []
    for chunk in chunks:
        # Chunks are slices of the text in order; each starts after the one before starts.
        begin = text.index(chunk.text, begin)
        headed.append(replace(chunk, header=header_for(lines, begin, begin + len(chunk.text), budget)))
    return headed


def search_text(text: str, header: str) -> str:
    """What is embedded and reranked: the header above the chunk's text."""
    return f"{header}\n{text}" if header else text


def without_header(stored: str, header: str) -> str:
    """The chunk's own text, from `stored` = search_text(text, header)."""
    return stored.removeprefix(f"{header}\n") if header else stored


def check_budget(chunk_tokens: int, header_tokens: int, max_tokens: int, model: str) -> None:
    """Raise ValueError when a full chunk and a full header cannot fit what the
    embedding model reads (the +1 is the line break between them)."""
    if chunk_tokens + header_tokens + 1 > max_tokens:
        raise ValueError(
            f"SPADESK_CHUNK_TOKENS ({chunk_tokens}) plus SPADESK_HEADER_TOKENS ({header_tokens}) is more than "
            f"{model} reads ({max_tokens} tokens), so the end of a chunk would be ignored. Lower one of them.")


def check_fits(texts: dict[str, str], max_tokens: int, model: str) -> None:
    """Raise ValueError naming every text, by chunk id, longer than the
    embedding model reads. Run on what is embedded, before embedding it."""
    over = [f"{chunk_id} ({tokens} tokens)" for chunk_id, text in texts.items()
            if (tokens := count_tokens(text)) > max_tokens]
    if over:
        raise ValueError(f"{model} reads at most {max_tokens} tokens, and these would be cut: {', '.join(over)}.")
