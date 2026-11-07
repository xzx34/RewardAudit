"""The defect catalog: every probe has a place in it, a defect takes the worst of its evidence, and too few calls rate low."""

import json
import re

import pytest
from rich.console import Console

from rewardaudit import audit, cli
from rewardaudit.contract import Contract
from rewardaudit.defects import BY_PROBE, DEFECTS
from rewardaudit.metrics import PROSE_PROBES, TOO_FEW, collect_defects, compute, nondeterminism_finding
from rewardaudit.pipeline import exit_code
from rewardaudit.probes.base import Probe
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import render_markdown, render_terminal
from rewardaudit.types import Kind, Label, Observation, Task, Verdict

ROWS = [{"prompt": f"What is {i} + 100?", "answer": str(i + 100)} for i in range(16)]


def _contract(ids, **kw):
    return Contract(full_score=1.0, accepted=["bare"], primary="bare", wrapper_rates={"bare": 1.0},
                    self_pass={i: True for i in ids}, gold_score={i: 1.0 for i in ids}, **kw)


def _probe(pid="M1", family="M", label=Label.NOT_SOLVING, expect="zero", **kw):
    return Probe(pid, family, f"probe {pid}", "natural", expect, label, lambda t, r, c: [], **kw)


def _observe(probe, hits, *, variant="v", score=1.0):
    """One call per task: the first `hits` tasks score `score`, the others the opposite of what the probe hopes for."""
    tasks = [Task(id=str(i), prompt="p", answer=str(i + 100)) for i in range(len(hits))]
    obs = [Observation(probe.id, probe.family, variant, t.id, f"reply {t.id}", Verdict(Kind.SCORE, score=score if hit else 0.0),
                       probe.label, probe.expect, probe.tier) for t, hit in zip(tasks, hits, strict=True)]
    return tasks, obs


def _finding(pid, severity="medium", category="wrong_accepted", rate=0.5, **kw):
    return {"probe_id": pid, "name": pid, "severity": severity, "category": category, "rate": rate, "n": 20, "note": "",
            "example": None, **kw}


def _defects(findings, contract=None, partial=None):
    head = {"key": "wrong_partial", "title": "Wrong answers paid in part", "k": 0, "value": 0.0, "n": 0, "severity": "low",
            "note": "", "example": None} | (partial or {})
    return collect_defects({"findings": findings, "head": [head]}, contract or _contract(["a"]))


def test_every_probe_belongs_to_exactly_one_defect():
    probes = [p for d in DEFECTS for p in d.probes]
    assert len(probes) == len(set(probes))
    assert not {p.id for p in all_probes()} - set(BY_PROBE)


def test_a_defect_takes_the_worst_of_its_evidence_and_the_worst_defect_comes_first():
    out = _defects([_finding("Q1", "low"), _finding("Q5", "high", "right_rejected", 0.3), _finding("M2", "medium")])
    assert [(d["id"], d["severity"]) for d in out] == [("D9", "high"), ("D1", "medium")]
    assert [e["probe_id"] for e in out[0]["evidence"]] == ["Q5", "Q1"]


def test_probes_with_too_few_calls_count_together_where_they_have_enough():
    thin = {"severity": "low", "rated": "high", "note": TOO_FEW, "n_scored": 6, "hit_tasks": 3}
    out = _defects([_finding("M2", **thin), _finding("M3", **thin)])
    assert out[0]["id"] == "D1" and out[0]["severity"] == "high" and out[0]["evidence"][0]["severity"] == "low"
    assert _defects([_finding("M2", **(thin | {"n_scored": 4})), _finding("M3", **(thin | {"n_scored": 4}))])[0]["severity"] == "low"
    assert _defects([_finding("M2", **(thin | {"hit_tasks": 1})), _finding("M3", **(thin | {"hit_tasks": 1}))])[0]["severity"] == "low"


def test_failed_calls_are_evidence_of_mishandled_input_whichever_probe_sent_them():
    out = _defects([_finding("Q1", "medium", "robustness"), _finding("P5", "low", "non_solving_accepted")])
    assert [d["id"] for d in out] == ["D13"] and len(out[0]["evidence"]) == 2


def test_special_findings_have_their_own_defect():
    assert [d["id"] for d in _defects([nondeterminism_finding(
        {"inconsistent_tasks": ["a"], "checked": 5, "repeats": 3})])] == ["D15"]
    assert [d["id"] for d in _defects([_finding("G1", "medium", "right_rejected")])] == ["D16"]
    assert [d["id"] for d in _defects([_finding("E1", "high", "non_solving_accepted", above_gold=True)])] == ["D18"]


def test_a_fixed_answer_counts_only_above_what_chance_gives():
    assert not _defects([_finding("N3", "low", "non_solving_accepted", above_floor=False)])
    assert [d["id"] for d in _defects([_finding("N3", "medium", "non_solving_accepted", above_floor=True)])] == ["D11"]


def test_a_reward_that_follows_the_format_is_high_with_a_wrong_answer_as_its_example():
    ids = [str(i) for i in range(12)]
    wrong = [("wrong_answer", i, f"wrong {i}", Verdict(Kind.SCORE, score=1.0)) for i in ids]
    gold = [("boxed", i, "gold", Verdict(Kind.SCORE, score=0.0)) for i in ids]
    c = _contract(ids, observations=gold + wrong, wrong_checked=12, wrong_same=12)
    c.wrapper_rates = {"bare": 1.0, "boxed": 0.0}
    assert c.format_only
    d = _defects([], c)[0]
    assert (d["id"], d["severity"], d["evidence"][0]["probe_id"]) == ("D18", "high", "format_only")
    assert d["example"]["completion"] == "wrong 0"


def test_too_few_calls_or_hit_rows_rate_low_and_say_so():
    probe = _probe()
    tasks, obs = _observe(probe, [True] * 3)
    f = compute(obs, [probe], tasks, _contract([t.id for t in tasks]), "int")["findings"][0]
    assert f["severity"] == "low" and f["note"] == TOO_FEW

    tasks, obs = _observe(probe, [True] + [False] * 9)  # enough calls, one row hit
    f = compute(obs, [probe], tasks, _contract([t.id for t in tasks]), "int")["findings"][0]
    assert f["severity"] == "low" and f["note"] == TOO_FEW

    tasks, obs = _observe(probe, [True] * 12)
    f = compute(obs, [probe], tasks, _contract([t.id for t in tasks]), "int")["findings"][0]
    assert f["severity"] == "high" and f["note"] == ""


def test_one_timeout_rates_low_and_timeouts_on_two_rows_rate_medium():
    probe = _probe("P5", "P")
    for rows, severity in ((1, "low"), (2, "medium")):
        tasks = [Task(id=str(i), prompt="p", answer="1") for i in range(rows)]
        obs = [Observation("P5", "P", "v", t.id, "r", Verdict(Kind.TIMEOUT), Label.NOT_SOLVING, "zero", "natural") for t in tasks]
        assert compute(obs, [probe], tasks, _contract([t.id for t in tasks]), "int")["findings"][0]["severity"] == severity


def test_wrong_answers_paid_in_part_are_a_row_and_a_defect():
    def half_for_a_near_miss(completion, answer):
        numbers = re.findall(r"-?\d+", completion)
        if not numbers:
            return 0.0
        return 1.0 if numbers[-1] == answer else 0.5 if abs(int(numbers[-1]) - int(answer)) == 1 else 0.0

    r = audit(half_for_a_near_miss, ROWS, workers=1)
    head = next(h for h in r.metrics["head"] if h["key"] == "wrong_partial")
    assert head["k"] > 0 and "50% of its score" in head["detail"] and r.rates["wrong_partial"] == head["value"]
    assert head["value"] >= 0.1 and head["severity"] == "medium" and head["example"]["verdict"]["score"] == 0.5
    d = next(d for d in r.defects if d["id"] == "D14")
    assert d["severity"] == "medium" and d["evidence"][0]["probe_id"] == "wrong_partial"
    assert "Wrong answers paid in part" in render_markdown(r)


def test_a_small_share_for_a_wrong_answer_is_a_format_bonus_and_stays_low():
    def bonus(completion, answer):
        return 1.0 if completion.strip() == f"\\boxed{{{answer}}}" else 0.1 if "boxed" in completion else 0.0

    r = audit(bonus, ROWS, workers=1)
    head = next(h for h in r.metrics["head"] if h["key"] == "wrong_partial")
    assert head["value"] > 0.5 and head["severity"] == "low" and "10% of its score" in head["detail"]


def test_a_checker_paying_nothing_for_a_wrong_answer_has_no_partial_defect():
    r = audit(lambda completion, answer: float(completion.strip() == answer), ROWS, workers=1)
    head = next(h for h in r.metrics["head"] if h["key"] == "wrong_partial")
    assert head["k"] == 0 and not head["value"] and "D14" not in {d["id"] for d in r.defects}


@pytest.mark.parametrize(("label", "expect"), [(Label.NOT_SOLVING, "zero"), (Label.SOLVING, "full")])
def test_a_sweep_reports_the_rate_of_each_variant_and_where_the_behaviour_ends(label, expect):
    probe = _probe("M24" if expect == "zero" else "Q28", "M" if expect == "zero" else "Q", label, expect, sweep=True)
    tasks = [Task(id=str(i), prompt="p", answer=str(i + 100)) for i in range(12)]
    accepted = {"0.1%": 12, "1%": 12, "5%": 6, "10%": 2}  # acceptance falls as the variant grows
    obs = []
    for variant in ["10%", "5%", "1%", "0.1%"]:  # out of order on purpose: the note sorts them
        for i, t in enumerate(tasks):
            obs.append(Observation(probe.id, probe.family, variant, t.id, f"{variant} {t.id}",
                                   Verdict(Kind.SCORE, score=float(i < accepted[variant])), label, expect, "natural"))
    f = compute(obs, [probe], tasks, _contract([t.id for t in tasks]), "int")["findings"][0]
    assert f["note"] == "accepted: 0.1% 100%, 1% 100%, 5% 50%, 10% 17%"
    if expect == "zero":  # the widest error accepted at least half the time
        assert f["example"]["variant"] == "5%" and f["rate"] == 0.5 and f["n"] == 12
    else:  # the shortest text that loses the score at least half the time
        assert f["example"]["variant"] == "5%" and f["rate"] == 0.5


def test_the_terminal_lists_defects_first_and_keeps_findings_for_full(tmp_path):
    def substring(completion, answer):
        return float(answer in completion)

    r = audit(substring, ROWS, workers=1)
    console = Console(width=100, record=True)
    render_terminal(r, console)
    text = console.export_text()
    assert "Defects (" in text and "Findings (" not in text and "D2" in text and "Substring match" in text
    assert text.index("Defects (") < text.index("Dataset:")
    console = Console(width=100, record=True)
    render_terminal(r, console, full=True)
    text = console.export_text()
    assert text.index("Defects (") < text.index("Findings (") < text.index("Dataset:")

    md = render_markdown(r)
    assert md.index("## Defects") < md.index("## Findings")
    data = json.loads(r.save(tmp_path / "out").with_name("report.json").read_text(encoding="utf-8"))
    assert data["defects"] == json.loads(json.dumps(r.defects)) and "findings" in data["metrics"]
    assert exit_code(r) == 1


def test_fail_on_knows_the_new_rate():
    assert cli._fail_on("wrong_partial>0.1") == {"wrong_partial": 0.1}


def test_a_checker_that_compares_the_whole_reply_is_not_rated_on_text_around_the_answer():
    def field(completion, answer):
        return float(completion == answer)

    r = audit(field, ROWS, workers=1)
    severities = {f["probe_id"]: f["severity"] for f in r.findings if f["category"] == "right_rejected"}
    assert r.metrics["compares_whole_reply"] and severities["Q5"] == severities["Q22"] == severities["Q6"] == "low"
    head = next(h for h in r.metrics["head"] if h["key"] == "right_rejected")
    assert head["n"] < sum(p["n_scored"] for p in r.metrics["probes"] if p["label"] == "solving" and p["headline"])
    assert "compares the whole reply" in render_markdown(r)


def test_text_around_the_answer_does_not_raise_the_defect_on_few_rows():
    def field(completion, answer):
        return float(completion == answer)

    r = audit(field, ROWS[:9], workers=1, probes=["Q"])  # Q5 and Q22 have fewer than ten calls each here
    assert r.metrics["compares_whole_reply"] and not any(f.get("rated") == "high" for f in r.findings)
    assert next(d for d in r.defects if d["id"] == "D9")["severity"] != "high"


def test_a_marked_format_that_rejects_text_around_the_answer_is_still_rated():
    def boxed_only(completion, answer):
        return float(completion == f"\\boxed{{{answer}}}")

    r = audit(boxed_only, ROWS, workers=1)
    assert not r.metrics["compares_whole_reply"]
    assert next(f for f in r.findings if f["probe_id"] == "Q5")["severity"] == "high" and exit_code(r) == 1


def _first_line(completion, answer):
    return float(completion.split("\n")[0].strip() == answer)  # text after the answer keeps the score, text before loses it


def test_answer_field_says_what_the_probes_do_not_show():
    inferred = audit(_first_line, ROWS, workers=1, probes=["Q"])
    stated = audit(_first_line, ROWS, workers=1, probes=["Q"], answer_field=True)
    assert not inferred.metrics["compares_whole_reply"] and stated.metrics["compares_whole_reply"]
    before, after = ({f["probe_id"]: f for f in r.findings if f["category"] == "right_rejected"} for r in (inferred, stated))
    assert before["Q5"]["severity"] == before["Q22"]["severity"] == "high" and exit_code(inferred) == 1
    assert after["Q5"]["severity"] == after["Q22"]["severity"] == "low" and "compares the whole reply" in after["Q5"]["note"]
    assert {pid: f["severity"] for pid, f in after.items() if pid not in PROSE_PROBES} == {
        pid: f["severity"] for pid, f in before.items() if pid not in PROSE_PROBES}
    rejected = [next(h for h in r.metrics["head"] if h["key"] == "right_rejected") for r in (inferred, stated)]
    assert rejected[1]["n"] < rejected[0]["n"] and rejected[1]["k"] < rejected[0]["k"]
    assert "compares the whole reply" not in render_markdown(inferred)
    assert "stated with answer_field=True, not inferred" in render_markdown(stated)


def test_answer_field_where_it_is_inferred_changes_only_the_note():
    def field(completion, answer):
        return float(completion == answer)

    inferred, stated = (audit(field, ROWS, workers=1, probes=["Q"], answer_field=a) for a in (False, True))
    assert inferred.metrics["compares_whole_reply"] and stated.metrics["compares_whole_reply"]
    same = [[(f["probe_id"], f["category"], f["severity"], f["rate"], f["n"], f["note"]) for f in r.findings] for r in (inferred, stated)]
    assert same[0] == same[1] and inferred.metrics["head"] == stated.metrics["head"]
    assert "not inferred" not in render_markdown(inferred) and "not inferred" in render_markdown(stated)


def test_only_a_high_defect_fails_the_audit():
    def rejects_a_period(completion, answer):
        return float(completion.strip().split()[-1] == answer)  # reads the last word: no period, but no high defect either

    r = audit(rejects_a_period, ROWS, workers=1, probes=["Q"])
    assert {d["severity"] for d in r.defects} <= {"medium", "low"} and r.defects and exit_code(r) == 0


def test_a_checker_that_guesses_when_it_finds_no_answer_is_not_deterministic():
    def guesses(completion, answer):
        import random

        return float(completion == answer) if any(c.isdigit() for c in completion) else float(random.random() < 0.5)

    r = audit(guesses, ROWS, workers=1, probes=["M"])
    assert not r.determinism["deterministic"]
    assert next(d for d in r.defects if d["id"] == "D15")["severity"] == "medium"
    assert "the same reply, sent several times" in next(f for f in r.findings if f["probe_id"] == "S1")["note"]
