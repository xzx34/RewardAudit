"""Replies that a checker may run or choke on, and what comes back from it when it does: an import reached, odd scores."""

import random

import pytest

from rewardaudit import audit
from rewardaudit.contract import Contract
from rewardaudit.goldtype import infer_gold_type
from rewardaudit.probes.base import ProbeContext
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import render_markdown
from rewardaudit.runner import Runner
from rewardaudit.types import MARKER_MODULE, Kind, Task

PROBES = {p.id: p for p in all_probes()}
ROWS = [{"prompt": f"Add {i} and 10.", "answer": str(i + 10)} for i in range(12)]


def _gen(probe_id, answer="42", primary="boxed"):
    task = Task("t", "What is it?", answer)
    contract = Contract(full_score=1.0, accepted=[primary], primary=primary, wrapper_rates={primary: 1.0},
                        self_pass={"t": True}, gold_score={"t": 1.0}, task_wrappers={"t": [primary]})
    ctx = ProbeContext(contract=contract, gold_type="int", tasks=[task], gold_types={"t": infer_gold_type(answer)})
    ctx.task = task
    return dict(PROBES[probe_id].generate(task, random.Random(0), ctx))


def test_p16_names_the_module_in_the_accepted_format():
    call = f"__import__('{MARKER_MODULE}')"
    assert _gen("P16") == {"call": f"\\boxed{{{call}}}", "statement": f"\\boxed{{import {MARKER_MODULE}}}",
                           "sentence": f"The answer is {call}", "bare": call}


def _defect(r, defect):
    return next((d for d in r.defects if d["id"] == defect), None)


def test_a_checker_that_runs_the_reply_is_high_even_inside_a_try():
    def runs_it(completion, answer):
        try:
            return float(eval(completion.replace("\\boxed{", "").rstrip("}")) == int(answer))  # noqa: S307
        except Exception:  # noqa: BLE001
            return 0.0

    r = audit(runs_it, ROWS, workers=2, probes=["P"], timeout=0.5)  # it also tries to evaluate the power towers
    f = next(f for f in r.findings if f.get("reached_import"))
    assert f["probe_id"] == "P16" and f["severity"] == "high" and f["category"] == "robustness"
    assert f["note"] == "the reply's text reached a Python import" and f["example"]["variant"] == "call"
    d = _defect(r, "D12")
    assert d["severity"] == "high" and d["evidence"][0]["probe_id"] == "P16"
    assert "the text of a reply got as far as a Python import" in render_markdown(r)
    assert [x["id"] for x in r.defects][0] == "D12"


def test_one_call_is_enough_and_an_error_that_comes_back_counts():
    def raises_it(completion, answer):
        if answer == "10":
            return float(eval(completion.replace("\\boxed{", "").rstrip("}")) == 10)  # noqa: S307
        return float(f"\\boxed{{{answer}}}" == completion or completion == answer)

    r = audit(raises_it, ROWS, workers=2, probes=["P"], timeout=0.5)
    assert _defect(r, "D12")["severity"] == "high"


def test_a_checker_that_never_runs_the_reply_is_not_flagged():
    r = audit(lambda completion, answer: float(completion == answer), ROWS, workers=1, probes=["P"])
    assert not _defect(r, "D12") and not any(f.get("reached_import") for f in r.findings)


def test_an_error_from_elsewhere_on_p16_is_an_ordinary_failed_call():
    def rejects_odd_text(completion, answer):
        if "import" in completion:
            raise ValueError("cannot parse")
        return float(completion == answer)

    r = audit(rejects_odd_text, ROWS, workers=1, probes=["P"])
    f = next(f for f in r.findings if f["probe_id"] == "P16")
    assert f["severity"] == "low" and not f.get("reached_import") and not _defect(r, "D12") and _defect(r, "D13")


def test_p17_is_a_million_characters_with_no_answer():
    reply = _gen("P17")["text"]
    assert len(reply) == 1_000_000 and "42" not in reply


def test_a_million_characters_that_hang_the_checker_are_found():
    def slow(completion, answer):
        import time

        if len(completion) > 500_000:
            time.sleep(30)
        return float(completion == answer)

    r = audit(slow, ROWS, workers=2, probes=["P"], timeout=1)
    f = next(f for f in r.findings if f["probe_id"] == "P17" and f["category"] == "robustness")
    assert f["severity"] == "medium" and "5 timeouts" in f["note"]


@pytest.mark.parametrize(("name", "char"), [("zero_width", "​"), ("no_break", " "), ("rtl_override", "‮"), ("nul", "\x00")])
def test_p2_hides_a_character_inside_a_wrong_answer(name, char):
    reply = _gen("P2")[name]
    assert char in reply and "42" not in reply and reply.replace(char, "") == "\\boxed{43}"


def test_a_nul_byte_reaches_the_checker_through_the_worker(tmp_path):
    checker = tmp_path / "checker.py"
    checker.write_text("def score(completion, answer):\n    return float('\\x00' in completion)\n")
    with Runner({"kind": "function", "path": f"{checker}:score"}, timeout_s=10) as r:
        assert r.score(Task("t", "p", "1"), "4\x003").score == 1.0


def test_an_invisible_character_that_makes_the_checker_raise_is_a_failed_call():
    def strict_number(completion, answer):
        return float(int(completion.replace("\\boxed{", "").rstrip("}")) == int(answer))

    r = audit(strict_number, ROWS, workers=1, probes=["P"])
    assert r.metrics["robustness"]["exceptions"]["probes"].get("P2")


@pytest.mark.parametrize("value", ["float('nan')", "float('inf')", "-float('inf')"])
def test_nan_and_inf_are_scores_outside_the_range(tmp_path, value):
    checker = tmp_path / "checker.py"
    checker.write_text(f"def score(completion, answer):\n    return 1.0 if completion == answer else {value}\n")
    with Runner({"kind": "function", "path": f"{checker}:score"}, timeout_s=10) as runner:
        v = runner.score(Task("t", "p", "1"), "2")
    assert v.kind == Kind.ERROR and v.error.startswith("ScoreOutOfRange: ") and not v.ok


def test_nan_on_two_rows_is_a_medium_finding_and_on_one_row_low():
    def nan_for_two_rows(completion, answer):
        if completion == answer:
            return 1.0
        return float("nan") if answer in ("10", "11") else 0.0

    def nan_for_one_row(completion, answer):
        if completion == answer:
            return 1.0
        return float("nan") if answer == "10" else 0.0

    f = next(f for f in audit(nan_for_two_rows, ROWS, workers=1, probes=["M"]).findings if f["probe_id"] == "S2")
    assert f["name"] == "scores outside the range" and f["severity"] == "medium" and "nan or inf" in f["note"]
    assert next(f for f in audit(nan_for_one_row, ROWS, workers=1, probes=["M"]).findings if f["probe_id"] == "S2")["severity"] == "low"


def test_a_negative_score_is_a_scale_and_not_a_finding():
    def minus_one(completion, answer):  # verl's math_dapo scores a wrong answer -1
        return 1.0 if completion == answer else -1.0

    r = audit(minus_one, ROWS, workers=1, probes=["M"])
    assert not r.findings and r.rates["wrong_accepted"] == 0.0
