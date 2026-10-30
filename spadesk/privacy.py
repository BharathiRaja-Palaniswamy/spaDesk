"""Masking personal details before a trace leaves the app.

A trace holds what customers wrote and what Pebby replied: names, phone
numbers, sometimes an email. Traces are sent to LangSmith, a service outside
the spa, so those details are replaced on the way out:

    "Asha Rao, 408 555 0101"   ->   "[name], [phone]"

What is masked:

  phone   a 10-digit number, however written ("408 555 0101", "(408) 555-0101",
          "+1 408.555.0101"). A date such as "2026-10-30" is not one: its
          groups are 4-2-2, not 3-3-4.
  email   anything shaped like name@domain.tld
  name    the names customers gave Pebby (`remember`), each word of them, in
          any text: names follow no pattern, so they are known only once said.
          A word the spa's own guide uses (a therapist's name, a service) is
          not personal, and is never masked, even when a customer shares it.

The app keeps the real details: only the copy sent to LangSmith is masked.
`mask` is given to the LangSmith client as its anonymizer (tracing.py), which
runs it on every text in a trace's inputs and outputs.
"""

import re
from collections import OrderedDict
from functools import cache
from typing import Any

from .config import get_settings

PHONE = re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\d-])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
MOST_NAMES = 1000  # names kept; the oldest is forgotten first

_names: OrderedDict[str, None] = OrderedDict()  # each word of a name a customer gave, lowercase
_pattern: re.Pattern | None = None


@cache
def _guide_words() -> frozenset[str]:
    """Every word in the spa's guide, lowercase: the staff's names among them."""
    return frozenset(re.findall(r"[^\W\d_]{2,}", get_settings().data_path.read_text(encoding="utf-8").lower()))


def remember(name: str) -> None:
    """Mask `name` from now on: each of its words of two letters or more that the
    guide does not use."""
    global _pattern
    for word in set(re.findall(r"[^\W\d_]{2,}", name.lower())) - _guide_words():
        _names[word] = None
        _names.move_to_end(word)
    while len(_names) > MOST_NAMES:
        _names.popitem(last=False)
    _pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, _names)) + r")\b", re.IGNORECASE) if _names else None


def mask(text: str, path: list[Any] | None = None) -> str:
    """`text` with phone numbers, emails and remembered names replaced."""
    text = EMAIL.sub("[email]", PHONE.sub("[phone]", text))
    return _pattern.sub("[name]", text) if _pattern else text
