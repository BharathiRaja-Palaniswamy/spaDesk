"""Tests for reading a booking's details from the customer's own words
(spadesk/agent/details.py, spadesk/agent/details.py). No model is called.

Run with:  .venv/bin/python -m pytest tests/test_agent_collect.py
"""

from datetime import date, time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from spadesk.agent.details import collect, customer_words, day_from, times_from
from spadesk.agent.pebby import TIME_NOTE

SUNDAY = date(2026, 10, 25)


@pytest.mark.parametrize("words, day", [
    ("Thursday", date(2026, 10, 29)), ("thu", date(2026, 10, 29)), ("next friday", date(2026, 10, 30)),
    ("Sunday", SUNDAY), ("today", SUNDAY), ("tomorrow", date(2026, 10, 26)),
    ("Thurday 9PM", date(2026, 10, 29)), ("tuesdy", date(2026, 10, 27)), ("next month", None), ("sunny", None),
    ("30 October", date(2026, 10, 30)), ("Oct 30th", date(2026, 10, 30)), ("30/10", date(2026, 10, 30)),
    ("28th of october", date(2026, 10, 28)), ("1 January", date(2027, 1, 1)), ("soon", None), ("", None)])
def test_the_day_from_the_customers_words(words, day):
    assert day_from(words, SUNDAY) == day


@pytest.mark.parametrize("words, limits", [
    ("after 3pm", (time(15), None)), ("after 3", (time(15), None)), ("at 11", (time(11), None)),
    ("3:30 pm", (time(15, 30), None)), ("by 2pm", (None, time(14))), ("between 2 and 4", (time(14), time(16))),
    ("morning", (None, time(12))), ("afternoon", (time(12), time(17))), ("after work", (time(17), None)),
    ("any time", (None, None)), ("no preference", (None, None)), ("sometime", None)])
def test_the_times_from_the_customers_words(words, limits):
    assert times_from(words) == limits


def test_only_what_the_customer_wrote_counts_not_the_time_notes():
    messages = [HumanMessage(f"{TIME_NOTE} Sunday 25 October 2026, 09:34 PDT)"), HumanMessage("Book Detox Body Scrub"),
                AIMessage("Which day?")]
    words = customer_words(messages)
    assert "09:34" not in words and "book detox body scrub" in words
    # Pebby wrote today's date and the time now: neither was said, so both are still missing.
    request, missing, _ = collect({"intent": "new", "services": ["Detox Body Scrub"], "day_said": "Sunday",
                                   "time_said": "09:34"}, words, SUNDAY)
    assert missing == ["day", "time"] and request.day == "" and request.after == ""


def test_words_the_customer_wrote_are_read_into_the_request():
    words = customer_words([HumanMessage("A detox scrub on Thursday, after 3pm, with Sofia please")])
    request, missing, _ = collect({"intent": "new", "services": ["Detox Body Scrub"], "day_said": "Thursday",
                                   "time_said": "after 3pm", "therapist": "Sofia Martinez"}, words, SUNDAY)
    assert missing == [] and (request.day, request.after, request.therapist) == ("2026-10-29", "15:00",
                                                                                  "Sofia Martinez")


def test_what_each_kind_of_request_needs():
    def missing(args):
        return collect(args, "", SUNDAY)[1]
    assert missing({"intent": "new"}) == ["services", "day", "time"]
    # Not naming the booking, nor the new service: both are asked for (only the booking's own service carries over).
    assert missing({"intent": "change", "services": ["Detox Body Scrub"]}) == ["references", "services", "day", "time"]
    assert missing({"intent": "cancel"}) == ["references"]
    assert missing({"intent": "availability", "services": ["Detox Body Scrub"]}) == ["services"]  # not named
    assert collect({"intent": "availability", "services": ["Detox Body Scrub"]}, "is the detox free", SUNDAY)[1] == []
    assert missing({"intent": "my_bookings"}) == []


@pytest.mark.parametrize("said, named", [
    ("I'd like a facial", []), ("the vitamin c facial", ["Brightening Vitamin C Facial"]),
    ("a deep tissue massage", ["Deep Tissue Massage"]), ("full leg wax please", ["Full Leg Wax"]),
    ("half leg", ["Half Leg Wax"]), ("need a calm facial", ["Calming Sensitive Skin Facial"]),
    ("an aroma massage", ["Aromatherapy Massage"])])
def test_a_service_counts_only_when_the_customer_named_it(said, named):
    chosen = ["Brightening Vitamin C Facial", "Deep Tissue Massage", "Full Leg Wax", "Half Leg Wax", "Glow Express Facial",
              "Calming Sensitive Skin Facial", "Aromatherapy Massage"]
    request, missing, unnamed = collect({"intent": "new", "services": chosen}, customer_words([HumanMessage(said)]),
                                        SUNDAY)
    assert request.services == named and len(unnamed) == len(chosen) - len(named)
    assert ("services" in missing) == (not named)  # dropped; asked for only when none they named is left


def test_naming_the_package_names_its_services():
    package = ["Brightening Vitamin C Facial", "Glow Express Facial", "Eyebrow Wax"]
    request, missing, unnamed = collect({"intent": "new", "services": package, "package": "Bridal Glow Prep Package"},
                                        "the bridal one, thursday, any time", SUNDAY)
    assert request.services == package and unnamed == [] and missing == ["day", "time"]  # not said as day_said yet


def test_a_therapist_counts_only_when_the_customer_asked_for_them():
    # As seen live: "remove hair on lip ... tomorrow afternoon" was read back "with Hannah Lee".
    args = {"intent": "new", "services": ["Lip Wax"], "day_said": "tomorrow", "time_said": "afternoon",
            "therapist": "Hannah Lee"}
    request, missing, _ = collect(args, "lip wax please, tomorrow afternoon", SUNDAY)
    assert request.therapist == "" and missing == []  # dropped, not asked for: whoever is free
    assert collect(args, "lip wax with hannah, tomorrow afternoon", SUNDAY)[0].therapist == "Hannah Lee"


def test_for_a_change_same_day_and_time_come_from_the_booking_and_it_keeps_its_service():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from spadesk.booking.calendars import Booking
    zone = ZoneInfo("America/Los_Angeles")
    old = Booking("BK-1", "Swedish Massage", "Daniel Ruiz", datetime(2026, 10, 26, 17, tzinfo=zone),
                  datetime(2026, 10, 26, 18, tzinfo=zone), "Asha", "4085550101")
    args = {"intent": "change", "references": ["BK-1"], "day_said": "same day", "time_said": "same time"}
    request, missing, _ = collect(args, "move it, same day, same time", SUNDAY, [old])
    assert missing == [] and (request.services, request.day, request.after) == (["Swedish Massage"], "2026-10-26",
                                                                                 "17:00")


def _bookings():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from spadesk.booking.calendars import Booking
    zone = ZoneInfo("America/Los_Angeles")

    def at(ref, service, hour, day=26, package=""):
        return Booking(ref, service, "Daniel Ruiz", datetime(2026, 10, day, hour, tzinfo=zone),
                       datetime(2026, 10, day, hour + 1, tzinfo=zone), "Asha", "4085550101", package)
    return at


@pytest.mark.parametrize("said, picked, named", [
    ("cancel the massage", ["BK-A"], ["BK-A"]),            # their only massage
    ("cancel the massage", ["BK-B"], ["BK-A"]),             # Pebby picked the facial: what they named counts
    ("cancel the massage", ["BK-63715A"], ["BK-A"]),        # a reference Pebby made up changes nothing
    ("cancel the massage", [], ["BK-A"]),                   # nor does none
    ("cancel the 6pm one", ["BK-B"], ["BK-B"]),             # by its time
    ("cancel both please", ["BK-A", "BK-B"], ["BK-A", "BK-B"]),
    ("cancel BK-B", ["BK-B"], ["BK-B"]),
    ("cancel my booking", ["BK-A"], [])])                   # which one? they have two: asked
def test_the_booking_to_cancel_is_the_one_the_customer_named(said, picked, named):
    at = _bookings()
    upcoming = [at("BK-A", "Swedish Massage", 17), at("BK-B", "Signature Radiance Facial", 18)]
    request, missing, _ = collect({"intent": "cancel", "references": picked}, said, SUNDAY, upcoming)
    assert request.references == named and missing == ([] if named else ["references"])


def test_a_change_is_of_one_booking_the_customer_named():
    at = _bookings()
    upcoming = [at("BK-A", "Swedish Massage", 17), at("BK-B", "Signature Radiance Facial", 18)]
    _, missing, _ = collect({"intent": "change", "references": [], "day_said": "Saturday"},
                            "move both to saturday", SUNDAY, upcoming)
    assert "references" in missing  # two named for one change: which?
    request, missing, _ = collect({"intent": "change", "references": ["BK-63715A"], "day_said": "Saturday"},
                                  "move the facial to saturday", SUNDAY, upcoming)
    assert request.references == ["BK-B"] and "references" not in missing


def test_two_of_the_same_service_need_their_day_or_time():
    at = _bookings()
    upcoming = [at("BK-A", "Swedish Massage", 17), at("BK-C", "Swedish Massage", 17, day=27)]
    assert collect({"intent": "cancel", "references": ["BK-A"]}, "cancel the massage", SUNDAY, upcoming)[1] == [
        "references"]  # which massage?
    assert collect({"intent": "cancel", "references": ["BK-A"]}, "cancel monday's massage", SUNDAY,
                   upcoming)[1] == []


def test_a_service_a_model_added_is_dropped_not_asked_about():
    # As seen live: Vitamin C facial named, Pebby added the Calming one; it was then asked about.
    request, missing, unnamed = collect({"intent": "new", "services": ["Brightening Vitamin C Facial",
                                                                       "Calming Sensitive Skin Facial"]},
                                        "the brightening vitamin c facial please", SUNDAY)
    assert request.services == ["Brightening Vitamin C Facial"] and "services" not in missing
    assert unnamed == ["Calming Sensitive Skin Facial"]


def test_an_add_on_counts_only_when_the_customer_named_it():
    args = {"intent": "new", "services": ["Prenatal Massage"], "add_ons": ["Scalp massage", "Hot stone enhancement"]}
    request, _, _ = collect(args, "a prenatal massage with a scalp massage", SUNDAY)
    assert request.add_ons == ["Scalp massage"]  # the hot stone was not asked for


def test_a_yes_to_pebbys_question_makes_its_words_the_customers_but_not_a_list():
    # As seen live: "a facial" ... "would that be the Signature Radiance Facial?" "yes": the facial was dropped.
    asked = AIMessage(content='{"answer": "Would that be the Signature Radiance Facial?", "items": [], "suggestions": []}')
    words = customer_words([HumanMessage("i want a facial"), asked, HumanMessage("yes")])
    request, missing, _ = collect({"intent": "new", "services": ["Signature Radiance Facial"]}, words, SUNDAY)
    assert request.services == ["Signature Radiance Facial"] and "services" not in missing
    listed = AIMessage(content='{"answer": "We offer these facials. Which would you like?", "items": '
                               '[{"name": "Deep Renewal Facial", "detail": "$150"}], "suggestions": []}')
    words = customer_words([HumanMessage("i want a facial"), listed, HumanMessage("yes")])
    assert collect({"intent": "new", "services": ["Deep Renewal Facial"]}, words, SUNDAY)[0].services == []
    # Something other than a yes does not make the question theirs.
    words = customer_words([HumanMessage("a facial"), asked, HumanMessage("not sure")])
    assert collect({"intent": "new", "services": ["Signature Radiance Facial"]}, words, SUNDAY)[0].services == []
