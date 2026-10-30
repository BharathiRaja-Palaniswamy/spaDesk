"""Tests for the eval runner's grading (evals/). The runner spends real tokens;
its cases and checks are tested here for free: no model is called.

Run with:  .venv/bin/python -m pytest tests/test_evals.py
"""

from evals import golden
from evals.chats import routing as owner_routing
from evals.checks import grade_answer
from evals.pebby import extraction, outcome, routing
from evals.summary import Result, summary
from evals.validate import stale_cases


def test_every_golden_case_matches_the_guide_and_has_a_known_target_and_split():
    cases = golden.cases()
    assert stale_cases(cases) == []
    assert {case["target"] for case in cases} == set(golden.TARGETS)
    assert {case["split"] for case in cases} <= set(golden.SPLITS)
    assert len({case["id"] for case in cases}) == len(cases)  # ids are unique


def test_pebby_layers_grade_the_path_and_the_customers_own_details():
    want = {"tool": "booking", "intent": ["new"], "not_tools": ["hand_off"]}
    good = [{"name": "booking", "args": {"intent": "new", "services": ["Prenatal Massage"], "time_said": "after 3pm"}}]
    assert routing(want, good) == []
    wrong = [{"name": "hand_off", "args": {"reason": "?"}},
             {"name": "booking", "args": {"intent": "availability", "services": ["Swedish Massage"], "day_said": "today"}}]
    assert routing(want, wrong) == ["used hand_off", "intent 'availability', expected one of ['new']"]
    asked = {"services": ["Prenatal Massage"], "time_said_has": "3", "no_day_said": True, "only_named_services": True}
    words = "a prenatal massage after 3pm"
    assert extraction(asked, good, words) == []
    assert extraction(asked, wrong, words) == [
        "chose ['Swedish Massage'] for them", "day_said 'today', but the customer said none",
        "services ['Swedish Massage'], expected ['Prenatal Massage']", "time_said None, expected it to mention '3'"]


def test_the_outcome_counts_what_the_conversation_did():
    want = {"confirming": False, "made": 1, "removed": 1, "handed_off": False, "cards": 0}
    moved = {"made": 1, "removed": 1, "handed_off": 0, "cards": 0}  # a change: one off the calendar, one on
    assert outcome(want, False, moved) == []
    assert outcome(want, True, {"made": 0, "removed": 2, "handed_off": 1, "cards": 1}) == [
        "confirming is True, expected False", "0 bookings made, expected 1", "2 bookings removed, expected 1",
        "1 feedback cards, expected 0", "handed to the staff, expected the opposite"]


def test_the_answer_checks_what_must_and_must_not_be_said():
    reply = {"question": "?", "text": "Sorry, I can only show your own bookings.", "sections": [], "covered": False}
    assert grade_answer(reply, {"lacks": ["Deep Tissue"], "covered": False, "cites": []}) == []
    told = {**reply, "text": "Jordan has a Deep Tissue Massage on Monday.", "sections": ["Massages"]}
    assert grade_answer(told, {"lacks": ["deep tissue"], "mentions": ["own bookings"], "cites": []}) == [
        "mentioned ['deep tissue']", "did not mention ['own bookings']", "cited ['Massages'], expected none"]


def test_the_judge_is_asked_even_when_a_free_check_failed(monkeypatch):
    import evals.checks as checks
    asked = []
    monkeypatch.setitem(checks.CHECKS, "judge", lambda reply, criteria: asked.append(criteria))
    grade_answer({"question": "?", "text": "We open at 9.", "sections": []}, {"mentions": ["10"], "judge": ["Polite."]})
    assert asked == [["Polite."]]  # its verdict is needed to calibrate it against a person's


def test_the_owner_chat_must_read_the_report_for_a_figure_and_only_then():
    assert owner_routing({"tool": "get_report"}, ["2026-09-01 to 2026-09-30"]) == []
    assert owner_routing({"tool": "get_report"}, []) == ["did not use get_report (used no tools)"]
    assert owner_routing({"not_tools": ["get_report"]}, ["2026-09-01 to 2026-09-30"]) == ["used get_report"]


def test_the_summary_scores_each_layer_and_split():
    good = Result("H01", "pebby", "happy", {"routing": [], "answer": []}, 2.0, 2, 1000)
    bad = Result("A01", "pebby", "adversarial", {"routing": ["used booking"], "answer": []}, 4.0, 4, 3000)
    lines = summary([good, bad])
    assert "  routing        1/2 (50%)" in lines and "  wrong path     1/2 (50%)" in lines
    assert "  happy          1/1 (100%)" in lines and "  adversarial    0/1 (0%)" in lines
    assert lines[-1] == "\nIn all: 1/2 (50%) passed"


def test_a_golden_case_becomes_the_same_dataset_example_every_time():
    from evals.datasets import example
    case = golden.cases(only=["H05"])[0]
    first, again = example(case, 1), example(case, 1)
    assert first == again and first["split"] == case["split"]  # one id per case: sending twice changes nothing
    assert first["outputs"] == {"expected": case["expected"]} and "expected" not in first["inputs"]


def test_the_experiment_gives_one_score_per_layer_and_passed():
    from evals.experiment import scores
    case = golden.cases(only=["H16"])[0]  # their bookings are listed: routing, outcome, answer
    seen = {"calls": [{"name": "booking", "args": {"intent": "my_bookings"}}], "confirming": False,
            "counts": {"made": 0, "removed": 0, "handed_off": 0, "cards": 0},
            "reply": {"question": "?", "text": "You have a Prenatal Massage on Friday.", "sections": []}}
    inputs = {key: case[key] for key in ("id", "target", "turns", "channel", "seed") if key in case}
    results = scores(inputs, seen, {"expected": case["expected"]})["results"]
    assert {r["key"]: r["score"] for r in results} == {"routing": 1, "outcome": 1, "answer": 1, "passed": 1}
