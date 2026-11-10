"""A gold the checker treats like a wrong answer in every format is a finding (G1), apart from the head metrics."""

import json
import re

import pytest
from conftest import FakeRunner
from rich.console import Console
from typer.testing import CliRunner

from rewardaudit import NotAuditable, cli
from rewardaudit.contract import WRAPPERS, infer_contract
from rewardaudit.metrics import REJECTED_GOLD_NAME
from rewardaudit.pipeline import AuditConfig, exit_code, run_audit
from rewardaudit.report import find_explainable, render_markdown, render_terminal, write_outputs
from rewardaudit.types import Kind, Task, Verdict

BAD = {"10", "11"}  # golds the checkers below handle differently


def _tasks(n=10):
    return [Task(id=f"t{i}", prompt="Compute it.", answer=str(10 + i)) for i in range(n)]


def _boxed(c, a):
    return f"\\boxed{{{a}}}" in c


def test_two_stages_try_every_format_before_calling_a_gold_rejected():
    """Stage 1 settles on boxed; golds it left at zero then get the wrappers it did not run, and only those golds."""
    def score(c, a):
        if a == "10":
            return c.strip() == '{"answer": "10"}'  # passes only in a format stage 1 does not run
        return a != "11" and f"\\boxed{{{a}}}" in c

    runner = FakeRunner(score)
    c = infer_contract(runner, _tasks())
    assert c.wrapper_stage != "all" and "json" not in c.wrappers_run
    assert c.self_pass["t0"] and c.wrapper_for("t0") == "json" and c.gold_score["t0"] == 1.0
    assert c.rejected_golds() == ["t1"]
    tried = {w for w, t, _, _ in c.observations if t == "t1" and w != "wrong_answer"}
    assert len(tried) > len(c.wrappers_run)
    assert {w for w, t, _, _ in c.observations if t == "t2" and w != "wrong_answer"} == set(c.wrappers_run)
    assert runner.calls == c.calls


# --- what counts --------------------------------------------------------------------------------------------

def test_a_gold_earning_zero_in_every_format_counts():
    c = infer_contract(FakeRunner(lambda c, a: _boxed(c, a) and a not in BAD), _tasks())
    assert c.self_pass_rate == 0.8 and sorted(c.rejected_golds()) == ["t0", "t1"]


def test_a_gold_no_higher_than_the_wrong_answer_counts():
    # any reply earns 0.2, a boxed gold 1.0, except that two golds never get past the 0.2
    r = FakeRunner(lambda c, a: 1.0 if _boxed(c, a) and a not in BAD else 0.2)
    c = infer_contract(r, _tasks())
    assert c.wrong_signal["t0"] == pytest.approx(0.2) and sorted(c.rejected_golds()) == ["t0", "t1"]
    # the two failing golds have credit, so they take a wrong answer as well: there is something to compare
    assert r.calls == c.calls and sum(1 for w, *_ in c.observations if w == "wrong_answer") == 10


def test_a_negative_score_needs_no_wrong_answer():
    r = FakeRunner(lambda c, a: 1.0 if _boxed(c, a) and a not in BAD else -0.5)
    c = infer_contract(r, _tasks())
    assert sorted(c.rejected_golds()) == ["t0", "t1"]
    assert sum(1 for w, *_ in c.observations if w == "wrong_answer") == 8  # nothing to compare on the failing two


def test_partial_credit_above_the_wrong_answer_does_not_count():
    r = FakeRunner(lambda c, a: (0.4 if a in BAD else 1.0) if _boxed(c, a) else 0.0)
    c = infer_contract(r, _tasks())
    assert c.self_pass_rate == 0.8 and c.rejected_golds() == [] and c.wrong_signal["t0"] == 0.0


def test_tasks_where_every_gold_call_raised_stay_under_robustness():
    r = FakeRunner(lambda c, a: ValueError("bad gold") if a in BAD else _boxed(c, a))
    c = infer_contract(r, _tasks())
    assert c.self_fail_reason == {"t0": "error", "t1": "error"} and c.rejected_golds() == []


def test_a_gold_that_raises_in_some_formats_and_earns_zero_in_the_rest_counts():
    r = FakeRunner(lambda c, a: (ValueError("x") if "\\boxed" in c else 0.0) if a in BAD else _boxed(c, a))
    c = infer_contract(r, _tasks())
    assert c.self_fail_reason["t0"] == "mixed" and sorted(c.rejected_golds()) == ["t0", "t1"]


def test_signal_decides_on_a_composite_rubric():
    # the gold keeps its format credit on every task, but its accuracy term is 0 on two of them
    def rubric(c, a):
        acc, fmt = float(_boxed(c, a) and a not in BAD), float("\\boxed{" in c)
        return Verdict(Kind.SCORE, score=acc + 0.25 * fmt, sub_scores={"acc": acc, "fmt": fmt})

    class Sub(FakeRunner):
        def score(self, task, completion):
            return rubric(completion, task.answer)

    c = infer_contract(Sub(None), _tasks())
    assert c.signal_keys == ["acc"] and sorted(c.rejected_golds()) == ["t0", "t1"]


# --- through the audit --------------------------------------------------------------------------------------

REJECTS_TWO = """
def score(prompt, completion, answer):
    return float(answer not in ("7 apples", "8 pears") and completion.strip().endswith(answer))
"""
PARTIAL_TWO = """
def score(prompt, completion, answer):
    if not completion.strip().endswith(answer):
        return 0.0
    return 0.4 if answer in ("7 apples", "8 pears") else 1.0
"""
EVERYTHING_PASSES = """
def score(prompt, completion, answer):
    return float(completion.strip().endswith(answer))
"""


def _spec(tmp_path, code, golds=None):
    (tmp_path / "checker.py").write_text(code)
    golds = golds or (["7 apples", "8 pears"] + [f"{i} plums" for i in range(10)])
    rows = [{"prompt": f"Count item {i}.", "answer": g} for i, g in enumerate(golds)]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return {"kind": "function", "path": str(tmp_path / "checker.py"), "dataset": str(tmp_path / "data.jsonl")}


def _g1(result):
    return next((f for f in result.metrics["findings"] if f["probe_id"] == "G1"), None)


def test_the_finding_and_where_it_shows(tmp_path):
    r = run_audit(_spec(tmp_path, REJECTS_TWO), AuditConfig(n_tasks=12, families=["N", "Q"]))
    f = _g1(r)
    assert f["category"] == "right_rejected" and f["severity"] == "medium" and f["rate"] == pytest.approx(2 / 12)
    assert f["n"] == f["n_tasks"] == 12 and f["unit"] == "tasks" and f["ci"][0] < f["rate"] < f["ci"][1]
    ex = f["example"]
    assert ex["gold"] in ("7 apples", "8 pears") and ex["best_score"] == 0.0 and ex["expect"] == "full"
    tried = ex["wrappers_tried"]
    assert set(tried) > set(r.contract.wrappers_run) and len(tried) == len(WRAPPERS) and ex["gold"] in ex["completion"]
    assert ex["task_id"] in f["note"] and f"{len(tried)} formats" in f["note"]

    # the head metric keeps its definition: rewrites of passing golds only
    head = {h["key"]: h for h in r.metrics["head"]}["right_rejected"]
    assert head["k"] == sum(p["obs_hits"] for p in r.metrics["probes"] if p["label"] == "solving" and p["headline"])
    assert "G1" not in {p["id"] for p in r.metrics["probes"]}
    assert exit_code(r) == 0  # medium: only a high defect fails the audit

    md = render_markdown(r)
    assert f"### MEDIUM: G1 {REJECTED_GOLD_NAME}" in md and "Rate 17% of 12 rows" in md
    assert f"G1 {REJECTED_GOLD_NAME}: rejected on 17% of rows (medium)" in md
    out = write_outputs(r, tmp_path / "out").parent
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert [x for x in data["metrics"]["findings"] if x["probe_id"] == "G1"][0]["example"]["gold"] == ex["gold"]
    assert sorted(data["contract"]["rejected_golds"]) == sorted(r.contract.rejected_golds())
    assert '"probe_id": "G1"' in (out / "repro.py").read_text(encoding="utf-8")
    console = Console(width=120, record=True, force_terminal=False)
    render_terminal(r, console, full=True)
    assert re.search(rf"G1 +{REJECTED_GOLD_NAME}", console.export_text())


def test_below_a_tenth_of_the_tasks_it_is_low(tmp_path):
    golds = ["7 apples"] + [f"{i} plums" for i in range(19)]
    r = run_audit(_spec(tmp_path, REJECTS_TWO, golds), AuditConfig(n_tasks=20, families=["N"]))
    assert _g1(r)["severity"] == "low" and _g1(r)["rate"] == pytest.approx(0.05)
    assert exit_code(r) == 0


def test_partial_credit_is_only_mentioned(tmp_path):
    r = run_audit(_spec(tmp_path, PARTIAL_TWO), AuditConfig(n_tasks=12, families=["N"]))
    assert r.contract.self_pass_rate < 1.0 and _g1(r) is None
    mixed = """
def score(prompt, completion, answer):
    if not completion.strip().endswith(answer):
        return 0.0
    return {"7 apples": 0.0, "8 pears": 0.4}.get(answer, 1.0)
"""
    f = _g1(run_audit(_spec(tmp_path, mixed), AuditConfig(n_tasks=12, families=["N"])))
    assert f["rate"] == pytest.approx(1 / 12) and "1 more reference answers earn partial credit and are not counted" in f["note"]


def test_no_finding_when_all_pass_or_most_fail(tmp_path):
    assert _g1(run_audit(_spec(tmp_path, EVERYTHING_PASSES), AuditConfig(n_tasks=12, families=["N"]))) is None
    # enough golds pass to go on, and they are not most of them: a rule of its own, not a few rejected answers
    code = """
def score(prompt, completion, answer):
    return float(not answer.endswith("pears") and completion.strip().endswith(answer))
"""
    golds = [f"{i} pears" for i in range(7)] + [f"{i} plums" for i in range(5)]
    r = run_audit(_spec(tmp_path, code, golds), AuditConfig(n_tasks=12, families=["N"]))
    assert r.contract.self_pass_rate < 0.5 and _g1(r) is None


def test_too_few_passing_golds_stop_the_audit_before_any_finding(tmp_path):
    golds = ["7 apples", "8 pears", "9 plums"]
    with pytest.raises(NotAuditable) as e:
        run_audit(_spec(tmp_path, REJECTS_TWO, golds), AuditConfig(n_tasks=3, families=["N"]))
    assert e.value.cls == "gold_never_passes" and "Only 1 of 3" in e.value.reason


def test_withheld_when_gold_and_wrong_answers_score_the_same(tmp_path):
    code = """
def score(prompt, completion, answer):
    return 0.0 if answer in ("7 apples", "8 pears") else 0.5
"""
    r = run_audit(_spec(tmp_path, code), AuditConfig(n_tasks=12, families=["N"]))
    assert r.contract.no_discrimination and _g1(r) is None
    assert "G1" in {f["probe_id"] for f in r.metrics["withheld_findings"]}


def test_explain_g1():
    res = CliRunner().invoke(cli.app, ["probes", "G1"])
    output = " ".join(res.output.split())
    assert res.exit_code == 0 and REJECTED_GOLD_NAME in output and "partial credit" in output
    assert find_explainable("g1") == ("finding", "G1")


# --- regression: a checker of real training code -------------------------------------------------------------

NORMALISES_ONLY_THE_REPLY = r"""
import re

def compute_score(solution_str, ground_truth):
    m = re.search(r"\\boxed\{(.*)\}", solution_str)
    return 1.0 if m and m.group(1).replace(" ", "") == ground_truth else -1.0
"""


def test_a_checker_that_normalises_only_the_reply_rejects_the_golds_own_form(tmp_path):
    """The prediction is normalised and the reference is not, so '\\frac 34' never matches itself."""
    golds = ["\\frac 34", "13", "16", "42", "7", "101"]
    r = run_audit(_spec(tmp_path, NORMALISES_ONLY_THE_REPLY, golds), AuditConfig(n_tasks=6, families=["N"]))
    f = _g1(r)
    assert f and f["category"] == "right_rejected" and f["example"]["gold"] == "\\frac 34" and f["example"]["best_score"] < 0
    assert r.contract.self_pass_rate == pytest.approx(5 / 6)
