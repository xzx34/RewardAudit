"""The wrong-answer pass: a gold that does not separate from wrong answers, and whole-reply similarity scores."""

import difflib
import json
from collections import Counter

import pytest
from rich.console import Console

from rewardaudit.contract import Contract, infer_contract
from rewardaudit.metrics import compute
from rewardaudit.pipeline import AuditConfig, exit_code, run_audit
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import no_discrimination_line, render_markdown, render_terminal, write_outputs
from rewardaudit.types import Kind, Label, Observation, Task, Verdict


class FakeRunner:
    """In-process runner around `score(completion, answer) -> float | (float, sub_scores)`."""

    max_score = None
    meta: dict = {}

    def __init__(self, fn):
        self.fn = fn

    def score(self, task: Task, completion: str) -> Verdict:
        out = self.fn(completion, task.answer)
        if isinstance(out, Exception):
            return Verdict(Kind.ERROR, error=repr(out))
        if isinstance(out, tuple):
            return Verdict(Kind.SCORE, score=float(out[0]), sub_scores=out[1])
        return Verdict(Kind.SCORE, score=float(out))


def _tasks(n=10):
    return [Task(id=f"t{i}", prompt="Compute it.", answer=str(10 + i)) for i in range(n)]


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def test_a_checker_that_reads_the_answer_separates():
    c = infer_contract(FakeRunner(lambda c, a: f"\\boxed{{{a}}}" in c), _tasks())
    assert c.wrong_checked == 10 and c.wrong_same == 0 and not c.no_discrimination
    assert c.gold_mean == 1.0 and c.wrong_mean == 0.0
    assert c.to_dict()["wrong_answer_pass"]["checked"] == 10 and c.to_dict()["no_discrimination"] is False


def test_any_reply_earning_the_gold_score_does_not_separate():
    c = infer_contract(FakeRunner(lambda c, a: 0.5 if c.strip() else -0.2), _tasks())
    assert c.self_pass_rate == 1.0 and c.full_score == 0.5  # the "full score" is only what the gold happened to earn
    assert c.no_discrimination and c.wrong_same == c.wrong_checked == 10
    assert "score the same on 10 of 10 rows (mean 0.5 and 0.5)" in c.discrimination_note


def test_composite_rubric_whose_accuracy_term_is_never_earned_does_not_separate():
    def rubric(c, a):
        fmt = float("\\boxed" in c)
        return 0.05 * fmt, {"accuracy": 0.0, "format": fmt, "metric:num_turns": 0.0}

    c = infer_contract(FakeRunner(rubric), _tasks())
    assert c.signal_keys is None and "no part of the score separates" in c.signal_note
    assert c.no_discrimination


def test_composite_rubric_with_a_real_accuracy_term_separates():
    def rubric(c, a):
        acc, fmt = float(f"\\boxed{{{a}}}" in c), float("\\boxed" in c)
        return 0.8 * acc + 0.2 * fmt, {"accuracy": acc, "format": fmt}

    c = infer_contract(FakeRunner(rubric), _tasks())
    assert c.signal_keys == ["accuracy"] and not c.no_discrimination and c.wrong_same == 0


def test_a_minority_of_same_scores_is_not_no_discrimination():
    tasks = _tasks(10)
    lenient = {t.answer for t in tasks[:4]}
    c = infer_contract(FakeRunner(lambda c, a: a in lenient or f"\\boxed{{{a}}}" in c), tasks)
    assert c.wrong_same == 4 and c.wrong_checked == 10 and not c.no_discrimination


def test_wrong_answers_that_raise_are_not_compared():
    c = infer_contract(FakeRunner(lambda c, a: 1.0 if f"\\boxed{{{a}}}" in c else ValueError("no box")), _tasks())
    assert c.wrong_checked == 0 and not c.no_discrimination


def test_one_gold_for_all_tasks_gets_a_definite_error_as_the_wrong_answer():
    def last_number(c, a):  # reads the last run of digits, so an appended letter would go unnoticed
        digits = "".join(ch if ch.isdigit() else " " for ch in c).split()
        return bool(digits) and digits[-1] == a

    tasks = [Task(id=f"t{i}", prompt="Compute it.", answer="7") for i in range(6)]
    c = infer_contract(FakeRunner(last_number), tasks)
    assert c.self_pass_rate == 1.0 and not c.no_discrimination and c.wrong_same == 0


def test_whole_reply_similarity_is_recognised():
    c = infer_contract(FakeRunner(_similarity), [Task(id=f"t{i}", prompt="Split it.", answer=f"sam-ple-{i} text-{i}")
                                                 for i in range(6)])
    assert c.accepted == ["bare"] and c.whole_reply_similarity and not c.no_discrimination
    assert c.to_dict()["whole_reply_similarity"] is True


def _token_f1(completion, answer):
    """A QA metric: punctuation goes, then the overlap of the words."""
    def words(s):
        return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).split()

    pred, gold = Counter(words(completion)), Counter(words(answer))
    common = sum((pred & gold).values())
    return 2 * common / (sum(pred.values()) + sum(gold.values()))


def test_a_grader_that_strips_punctuation_is_whole_reply_similarity_too():
    tasks = [Task(id=f"t{i}", prompt="Name it.", answer=f"the old bridge of town {i}") for i in range(6)]
    c = infer_contract(FakeRunner(_token_f1), tasks)
    assert {"bare", "hash", "bold"} <= set(c.accepted) and c.primary == "bare" and c.whole_reply_similarity
    exact = infer_contract(FakeRunner(lambda c, a: _token_f1(c, a) == 1.0), tasks)  # no partial credit: a finding, as above
    assert exact.primary == "bare" and not exact.whole_reply_similarity


@pytest.mark.parametrize("fn", [
    lambda c, a: c.strip() == a,  # exact match on the reply: rejects explanations, a finding when the prompt allows them
    lambda c, a: 1.0 if c.strip() == a else 0.5 if a in c else 0.0,  # one fixed step of partial credit
    lambda c, a: _similarity(c.rsplit("Answer:", 1)[-1].strip(), a),  # similarity, but of an extracted answer
])
def test_other_checkers_are_not_whole_reply_similarity(fn):
    tasks = [Task(id=f"t{i}", prompt="Split it.", answer=f"sam-ple-{i} text-{i}") for i in range(6)]
    assert not infer_contract(FakeRunner(fn), tasks).whole_reply_similarity


# --- through the audit ------------------------------------------------------------------------------------

ANY_TEXT = """
def score(prompt, completion, answer):
    return 0.5 if completion.strip() else -0.2
"""
SIMILARITY = """
import difflib

def score(prompt, completion, answer):
    return difflib.SequenceMatcher(None, completion, answer, autojunk=False).ratio()
"""


def _spec(tmp_path, code, golds=None):
    (tmp_path / "checker.py").write_text(code)
    golds = golds or [f"it-em{i} num-ber{i}" for i in range(12)]
    rows = [{"prompt": f"Rewrite item {i} with a dash.", "answer": g} for i, g in enumerate(golds)]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return {"kind": "function", "path": str(tmp_path / "checker.py"), "dataset": str(tmp_path / "data.jsonl")}


def _terminal(result, full=False):
    console = Console(width=120, record=True, force_terminal=False)
    render_terminal(result, console, full=full)
    return console.export_text()


def test_no_discrimination_makes_no_claim_and_says_so(tmp_path):
    r = run_audit(_spec(tmp_path, ANY_TEXT), AuditConfig(n_tasks=12))
    heads = {h["key"]: h for h in r.metrics["head"]}
    assert r.contract.no_discrimination
    assert all(h["not_measured"] and h["n"] == 0 and h["value"] == 0.0 for h in heads.values())
    assert heads["non_solving_accepted"]["above_floor"] is False and "strategy" not in heads["non_solving_accepted"]
    assert not [f for f in r.metrics["findings"] if f["category"] != "robustness"]
    withheld = {f["probe_id"]: f for f in r.metrics["withheld_findings"]}
    assert withheld["N5"]["rate"] == 1.0 and withheld["N5"]["category"] == "non_solving_accepted"
    assert exit_code(r) == 2 and exit_code(r, {"non_solving_accepted": 0.5}) == 2
    assert r.metrics["discrimination"] == {"case": "same", "reason": ("reference and wrong answers score the same on 12 of 12 rows "
                                                                      "(mean 0.5 and 0.5); check how the reference answer is delivered")}
    line = "Reference answers and wrong answers score the same on 12 of the 12 rows"

    md = render_markdown(r)
    assert line in md.split("## Summary")[0]  # before any number
    assert "| Accepts non-answers |  | not measured |  | 0 |" in md
    assert "None that can be claimed" in md and "Not counted: " in md
    assert "**Reference answers not separated from wrong answers.**" in md and "how the reference answer is delivered" in md

    screen = " ".join(_terminal(r).split())
    assert line in screen.split("Accepts non-answers")[0] and "how the reference answer is delivered" in screen
    assert "No findings" not in screen and "Not counted" in screen and "Accepts non-answers not measured" in screen
    full = " ".join(_terminal(r, full=True).split())
    assert line in full and "No findings" not in full and "Not counted" in full

    data = json.loads(write_outputs(r, tmp_path / "out").with_name("report.json").read_text(encoding="utf-8"))
    assert data["contract"]["no_discrimination"] is True and data["metrics"]["withheld_findings"]


# --- the cases of no_discrimination -------------------------------------------------------------------------

# reads every number in the reply, not the answer: the gold and a wrong answer are a few numbers, a sweep is many
OUTRANKS = """
import re

def score(prompt, completion, answer):
    return 0.955 if len(re.findall(r"-?[0-9]+", completion)) >= 6 else 0.155
"""
INVERTED = """
def score(prompt, completion, answer):
    return 0.5 if answer in completion else 1.0
"""
FORMAT_ONLY = """
def score(prompt, completion, answer):
    return 1.0 if "<answer>" in completion and "</answer>" in completion else 0.0
"""


def _no_discrimination_audit(tmp_path, code, families=None):
    r = run_audit(_spec(tmp_path, code, [str(10 + i) for i in range(12)]), AuditConfig(n_tasks=12, families=families))
    assert r.contract.no_discrimination
    return r, {f["probe_id"]: f for f in r.metrics["findings"]}, {f["probe_id"]: f for f in r.metrics["withheld_findings"]}


def test_a_non_solving_reply_above_the_gold_is_a_finding(tmp_path):
    r, findings, withheld = _no_discrimination_audit(tmp_path, OUTRANKS, ["N", "E"])
    e1 = findings["E1"]
    assert set(r.contract.gold_score.values()) == {0.155}
    assert e1["above_gold"] and e1["category"] == "non_solving_accepted" and e1["severity"] == "high"
    assert e1["rate"] == 1.0 and e1["n"] == 24 and e1["task_rate"] == 1.0 and "0.955 against 0.155" in e1["note"]
    assert e1["example"]["verdict"]["score"] == 0.955
    # everything reaching only the gold's own score stays withheld, and so does E1's "accepted" reading of the rate
    assert "N5" in withheld and "N5" not in findings and "E1" in withheld
    assert all(f.get("above_gold") or f["category"] == "robustness" for f in findings.values())
    assert all(h["not_measured"] for h in r.metrics["head"])

    d = r.metrics["discrimination"]
    above = r.metrics["above_gold"]
    assert d["case"] == "outranks" and above["k"] > 0 and above["probes"]["E1"] == 24 and above["rate"] == above["k"] / above["n"]
    assert d["reason"].startswith(f"non-answers outrank the reference answer: {above['k']} of {above['n']} replies scored above it")
    assert "reference and wrong answers score the same on 12 of 12" in d["reason"]
    assert exit_code(r) == 2

    line = no_discrimination_line(r.contract, r.metrics)
    assert line.startswith("Non-answers outrank the reference answer:") and "E1 (24)" in line
    md = render_markdown(r)
    assert line in md.split("## Summary")[0] and "### HIGH: E1 integer sweep" in md
    assert "None that can be claimed" not in md and "Not counted: " in md
    assert "E1 integer sweep: scored above the reference answer on 100% of calls (high)" in md
    screen = " ".join(_terminal(r).split())
    assert "Non-answers outrank the reference answer" in screen and "Not counted" in screen
    assert "No defects" not in screen and "Accepts non-answers yes see Defects" in screen and "Accepts wrong answers not measured" in screen
    assert "0.955 against 0.155" in " ".join(_terminal(r, full=True).split())
    data = json.loads(write_outputs(r, tmp_path / "out").with_name("report.json").read_text(encoding="utf-8"))
    assert data["metrics"]["discrimination"]["case"] == "outranks" and data["metrics"]["above_gold"]["probes"]["E1"] == 24


def test_a_wrong_answer_above_the_gold_is_said_too(tmp_path):
    r, findings, withheld = _no_discrimination_audit(tmp_path, INVERTED, ["N", "M", "O"])
    c = r.contract
    assert c.wrong_above == c.wrong_checked == 12 and c.to_dict()["wrong_answer_pass"]["above_gold"] == 12
    line = no_discrimination_line(c, r.metrics)
    assert "wrong answers scored strictly above the reference answer on 12 of the 12 rows" in line
    assert "wrong answers above it on 12 of 12 rows" in r.metrics["discrimination"]["reason"]
    # a probe of the M family is a wrong answer above the gold, a probe of the O family carries the gold and stays out
    assert findings["M6"]["category"] == "wrong_accepted" and findings["M6"]["severity"] == "high" and findings["M6"]["above_gold"]
    assert findings["N1"]["category"] == "non_solving_accepted" and "O4" not in findings and "O3" not in findings


def test_one_wrong_answer_above_the_gold_is_said_but_does_not_name_the_case(tmp_path):
    code = """
def score(prompt, completion, answer):
    return 0.9 if "11" in completion and answer == "10" else 0.5
"""
    r = run_audit(_spec(tmp_path, code, [str(10 + i) for i in range(12)]), AuditConfig(n_tasks=12, families=["N"]))
    c = r.contract
    assert c.no_discrimination and c.wrong_above == 1 and c.wrong_checked == 12  # 8% of the tasks: below the medium rate
    assert r.metrics["discrimination"]["case"] == "same"
    assert "wrong answers above it on 1 of 12 rows" in r.metrics["discrimination"]["reason"]
    assert "wrong answers scored strictly above the reference answer on 1 of the 12 rows" in no_discrimination_line(c, r.metrics)


def test_format_only_reward(tmp_path):
    r, findings, withheld = _no_discrimination_audit(tmp_path, FORMAT_ONLY, ["N", "M"])
    c = r.contract
    assert c.format_only and c.to_dict()["format_only"] is True
    assert not [f for f in findings.values() if f.get("above_gold")] and withheld
    d = r.metrics["discrimination"]
    assert d["case"] == "format_only" and d["reason"].startswith("format-only reward: ")
    assert "reference and wrong answers score the same on 12 of 12" in d["reason"]
    assert no_discrimination_line(c, r.metrics).startswith("Format-only reward: ")
    assert "Format-only reward" in render_markdown(r).split("## Summary")[0]
    assert "Format-only reward" in " ".join(_terminal(r).split())
    assert exit_code(r) == 2


@pytest.mark.parametrize("fn, expected", [
    (lambda c, a: "\\boxed{" in c, True),  # a format and nothing else
    (lambda c, a: 0.5 if c.strip() else -0.2, False),  # the same score in every format: the plain case
    (lambda c, a: f"\\boxed{{{a}}}" in c, False),  # reads the answer: not even no_discrimination
    # the accuracy term no gold ever earned may read the answer, so this is not known to be a format reward
    (lambda c, a: (0.05 * ("\\boxed" in c), {"accuracy": 0.0, "format": float("\\boxed" in c)}), False),
    # a format reward with a metric that does not count
    (lambda c, a: (float("\\boxed" in c), {"format": float("\\boxed" in c), "metric:num_turns": 0.0}), True),
])
def test_format_only_is_a_property_of_the_contract(fn, expected):
    assert infer_contract(FakeRunner(fn), _tasks()).format_only is expected


def test_a_low_severity_rate_above_the_gold_is_listed_but_does_not_name_the_case(tmp_path):
    code = """
def score(prompt, completion, answer):
    return 0.9 if completion == "." and answer.endswith("0") else 0.5
"""
    r, findings, _ = _no_discrimination_audit(tmp_path, code, ["N"])
    assert findings["N3"]["above_gold"] and findings["N3"]["severity"] == "low" and findings["N3"]["task_rate"] < 0.5
    assert r.metrics["discrimination"]["case"] == "same" and not r.contract.format_only


def test_a_gold_blind_reply_that_happens_to_be_the_gold_is_not_above_it():
    probe = next(p for p in all_probes() if p.id == "N3")
    tasks = [Task(id=f"t{i}", prompt="p", answer="1") for i in range(6)]
    ids = [t.id for t in tasks]
    c = Contract(full_score=0.5, accepted=["bare"], primary="bare", wrapper_rates={"bare": 1.0},
                 self_pass={i: True for i in ids}, gold_score={i: 0.5 for i in ids}, wrong_checked=6, wrong_same=6)
    obs = [Observation("N3", "N", "one", i, "\\boxed{1}", Verdict(Kind.SCORE, score=0.9), Label.NOT_SOLVING, "zero", probe.tier)
           for i in ids]
    m = compute(obs, [probe], tasks, c, "int")
    assert m["above_gold"]["k"] == m["above_gold"]["n"] == 0 and m["findings"] == []
    assert m["probes"][0]["coincident"] == 6


def test_above_the_gold_is_measured_on_any_checker_and_a_finding_only_without_discrimination(tmp_path):
    code = """
import re

def score(prompt, completion, answer):
    if len(re.findall(r"-?[0-9]+", completion)) >= 6:
        return 1.5
    return float(completion.strip().endswith(answer))
"""
    r = run_audit(_spec(tmp_path, code, [str(10 + i) for i in range(12)]), AuditConfig(n_tasks=12, families=["E", "N"]))
    assert not r.contract.no_discrimination and r.metrics["discrimination"] is None
    assert r.metrics["above_gold"]["probes"]["E1"] == 24 and not r.metrics["withheld_findings"]
    assert not [f for f in r.metrics["findings"] if f.get("above_gold")]
    assert next(f for f in r.metrics["findings"] if f["probe_id"] == "E1")["rate"] == 1.0


def test_whole_reply_similarity_counts_no_rewrite_as_rejected(tmp_path):
    r = run_audit(_spec(tmp_path, SIMILARITY), AuditConfig(n_tasks=12))
    heads = {h["key"]: h for h in r.metrics["head"]}
    assert r.contract.whole_reply_similarity and not r.contract.no_discrimination
    assert heads["right_rejected"]["not_measured"] and heads["right_rejected"]["n"] == 0
    # the other claims stand: wrong answers stay measured
    assert heads["wrong_accepted"]["n"] > 0 and not heads["wrong_accepted"].get("not_measured")
    assert heads["wrong_accepted"]["value"] == 0.0 and r.rates["wrong_accepted"] == 0.0 and "right_rejected" not in r.rates
    assert not [f for f in r.metrics["findings"] if f["category"] == "right_rejected"]
    assert {"Q5", "Q6", "Q22", "Q3"} <= {f["probe_id"] for f in r.metrics["withheld_findings"]}
    assert heads["wrong_partial"]["value"] > 0.5 and exit_code(r) == 1  # similarity pays a wrong answer part of the score
    assert "grades the whole reply" in render_markdown(r).split("## Summary")[0]
    assert "grades the whole reply" in " ".join(_terminal(r).split())
    probes = {p["id"]: p for p in r.metrics["probes"]}
    assert probes["Q5"]["rate"] == 1.0  # recorded all the same


def test_the_wrong_answer_pass_does_not_count_as_gold_calls(tmp_path):
    code = """
def score(prompt, completion, answer):
    if answer not in completion:
        raise ValueError("cannot read this reply")
    return float(completion.strip() == answer)
"""
    r = run_audit(_spec(tmp_path, code), AuditConfig(n_tasks=12, families=["N"]))
    assert r.contract.wrong_checked == 0  # every wrong answer raised, so nothing to compare
    n_wrong = sum(1 for wid, *_ in r.contract.observations if wid == "wrong_answer")
    rb = r.metrics["robustness"]
    assert n_wrong == 12 and rb["gold_calls"] == len(r.contract.observations) - n_wrong
    assert rb["gold_error_rate"] == 0.0 and rb["gold_error_example"] == ""
