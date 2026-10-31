"""Tests for masking personal details in traces (spadesk/privacy.py).
Nothing is sent to LangSmith.

Run with:  .venv/bin/python -m pytest tests/test_privacy.py
"""

import pytest
from langsmith.anonymizer import create_anonymizer

from spadesk import privacy


@pytest.fixture(autouse=True)
def no_names_yet(monkeypatch):
    """Each test starts with no names known: other tests' customers are remembered too."""
    monkeypatch.setattr(privacy, "_names", privacy.OrderedDict())
    monkeypatch.setattr(privacy, "_pattern", None)


@pytest.mark.parametrize("said, sent", [
    ("408 555 0101", "[phone]"), ("(408) 555-0101", "[phone]"), ("+1 408.555.0101", "[phone]"),
    ("raja 9289381938", "raja [phone]"), ("asha@example.com", "[email]"),
    # Not personal: a date, a time, a booking reference and a price stay as they are.
    ("on 2026-10-30 at 10:00, BK-63715A, $1,234", "on 2026-10-30 at 10:00, BK-63715A, $1,234")])
def test_phone_numbers_and_emails_are_masked_and_nothing_else(said, sent):
    assert privacy.mask(said) == sent


def test_a_name_is_masked_once_a_customer_has_given_it():
    assert privacy.mask("Thank you, Asha!") == "Thank you, Asha!"  # not given yet: no pattern to know it by
    privacy.remember("Asha Rao")
    assert privacy.mask("Thank you, asha! Booked for Asha Rao.") == "Thank you, [name]! Booked for [name] [name]."
    assert privacy.mask("Ashanti") == "Ashanti"  # whole words only


def test_the_staff_are_never_masked():
    # A customer who shares a word with the guide (here a therapist's first name) keeps the
    # rest of their name masked; the therapist stays readable in every trace.
    privacy.remember("Priya Kapoor")
    assert privacy.mask("Priya Kapoor booked with Priya Nair.") == "Priya [name] booked with Priya Nair."


def test_every_text_in_a_trace_is_masked():
    traced = create_anonymizer(privacy.mask)({"message": "call me on 408 555 0101", "items": ["a@b.co"], "n": 3})
    assert traced == {"message": "call me on [phone]", "items": ["[email]"], "n": 3}
