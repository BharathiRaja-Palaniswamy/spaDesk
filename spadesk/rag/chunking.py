"""Step 1 of RAG: cut the document into chunks.

The cut is by size, not by meaning: every chunk holds the same number of
tokens (the last one may be shorter), wherever in the text that happens to
fall. A cut can therefore land in the middle of a sentence or between a
service and its price. To soften that, each chunk starts a little before the
previous one ended, so the text around every cut appears whole in one of the
two chunks. That shared part is the overlap.

    size 150, overlap 30:   chunk 0 = tokens   0-149
                            chunk 1 = tokens 120-269
                            chunk 2 = tokens 240-389 ...

A token is a piece of a word; 100 English words are about 130 tokens.
"""

import re
from dataclasses import dataclass

import tiktoken

HEADING = re.compile(r"^(#{1,3})\s+(.+)$")  # "## Policies", "### Cancellation Policy"

# Tokens are counted with tiktoken's "cl100k_base" encoding. Each embedding
# model splits text slightly differently, so this is a close estimate of the
# model's own count, not an exact one.
ENCODING = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    """One fixed-size piece of the document."""

    id: str  # its position in the document, e.g. "chunk-3"
    text: str
    tokens: int  # how many tokens `text` holds
    # The headings whose content this chunk includes, in document order, e.g.
    # ("Late Arrival Policy", "Refund Policy"). Retrieval uses them to say
    # where an answer came from. A chunk cut by size often spans several.
    sections: tuple[str, ...]
    # Where the chunk sits in the document, written above it for search only
    # (see headers.py); "" until headers.with_headers() adds it.
    header: str = ""


def split(text: str, size: int, overlap: int) -> list[Chunk]:
    """Cut `text` into chunks of `size` tokens, each sharing its first
    `overlap` tokens with the end of the chunk before it.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive, and the overlap must be smaller than the size.")

    encoder = tiktoken.get_encoding(ENCODING)
    # `starts[i]` is the character in `text` where token i begins. Chunks are
    # sliced out of the original text by character, so no character is ever
    # cut in half.
    _, starts = encoder.decode_with_offsets(encoder.encode(text))
    starts.append(len(text))  # where the text ends, so the last token has an end too
    total = len(starts) - 1
    labels = _section_spans(text)

    chunks = []
    step = size - overlap  # how far each chunk begins after the one before
    for first in range(0, total, step):
        last = min(first + size, total)  # one past this chunk's final token
        begin, end = starts[first], starts[last]
        sections = tuple(dict.fromkeys(  # dict.fromkeys removes repeats, keeping order
            name for name, line_begin, line_end in labels if line_begin < end and line_end > begin))
        chunks.append(Chunk(f"chunk-{len(chunks)}", text[begin:end], last - first, sections))
        if last == total:
            break  # the end of the text is covered; a further chunk would only repeat it
    return chunks


def _section_spans(text: str) -> list[tuple[str, int, int]]:
    """Where each line of content sits: (heading, start, end) for every line
    that is not blank and not itself a heading. The heading is the closest
    one above the line; start and end are character positions in `text`.
    """
    spans = []
    current = ""
    position = 0
    for line in text.splitlines(keepends=True):
        heading = HEADING.match(line.strip())
        if heading:
            current = heading[2].strip()
        elif line.strip() and current:
            spans.append((current, position, position + len(line)))
        position += len(line)
    return spans
