"""G3: the reference answer in the accepted format as the reply, while the checker gets the reference answer written another
way, one way per call. A writing that loses the score is a finding of D22, apart from the head metrics."""

import json
import random
import subprocess
import sys

import pytest
from conftest import probe_context
from typer.testing import CliRunner

from rewardaudit import audit, cli
from rewardaudit.adapters.function_adapter import reference_is_answer
from rewardaudit.defects import BY_PROBE, DEFECTS
from rewardaudit.metrics import TOO_FEW, _rejected_severity
from rewardaudit.pipeline import _variants
from rewardaudit.probes.base import Probe
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import render_markdown, write_outputs
from rewardaudit.types import Label, Task

G3 = next(p for p in all_probes() if p.id == "G3")
ROWS = [{"prompt": f"What is {i} + 1000?", "answer": str(i + 1000)} for i in range(12)]
PERCENT = "What percentage of 200 is 50?"
DOLLARS = "A pen costs 3 dollars. How many dollars do 4 pens cost?"


def _written(answer, prompt="Compute it.", primary="boxed", reaches=True):
    """variant -> (reply, the reference answer the checker gets)."""
    task = Task("t", prompt, answer)
    ctx = probe_context([task], primary, "short_str", reference_is_answer=reaches)
    ctx.task = task
    return {name: (reply, reference) for name, reply, reference in G3.generate(task, random.Random(0), ctx)}


def test_an_option_letter_gets_no_period():
    assert _written("E", "Which is it?\nA) one\nE) five") == {}  # answer keys store the letter bare
    assert _written("E.", "Which is it?") == {}


@pytest.mark.parametrize("answer, prompt, variant, written", [
    ("x+1", "", "spaces", "x + 1"), ("x + 1", "", "spaces", "x+1"), ("(1,2)", "", "spaces", "(1, 2)"),
    ("2, 3, 5", "", "spaces", "2,3,5"), ("y=2x-3", "", "spaces", "y = 2x - 3"), ("x=-1", "", "spaces", "x = -1"),
    ("x^{-1}+1", "", "spaces", "x^{-1} + 1"), ("(-\\infty, 3]", "", "spaces", "(-\\infty,3]"),
    ("\\frac{1}{2}", "", "dfrac", "\\dfrac{1}{2}"), ("\\dfrac{3}{4}", "", "dfrac", "\\frac{3}{4}"),
    ("42", "", "period", "42."), ("Paris", "", "text", "\\text{Paris}"), ("New York", "", "text", "\\text{New York}"),
    ("25%", PERCENT, "percent", "0.25"), ("25\\%", PERCENT, "percent", "0.25"), ("25", PERCENT, "percent", "25%"),
    ("5 km", "How far is it?", "unit", "5"), ("12", DOLLARS, "unit", "12 dollars"),
    ("1200", "", "commas", "1,200"), ("-1,234,567", "", "commas", "-1234567"),
])
def test_each_writing_goes_to_the_checker_and_the_reply_is_the_reference_answer_as_it_passed(answer, prompt, variant, written):
    out = _written(answer, prompt)
    assert out[variant] == (f"\\boxed{{{answer}}}", written)
    assert all(reply == f"\\boxed{{{answer}}}" and reference != answer for reply, reference in out.values())


def test_a_reply_in_another_format_is_the_reference_answer_in_it():
    assert _written("1200", primary="hash") == {"period": ("#### 1200", "1200."), "commas": ("#### 1200", "1,200")}


@pytest.mark.parametrize("answer, prompt, variant", [
    ("New York", "", "spaces"), ("\\text{Devon, Lisa}", "", "spaces"), ("2020-03-05", "", "spaces"), ("1e-5", "", "spaces"),
    ("1,000", "", "spaces"), ("1,\\!000", "", "spaces"), ("-5", "", "spaces"), ("3/4", "", "spaces"), ("x<=3", "", "spaces"),
    ("42", "", "dfrac"), ("Paris.", "", "period"), ("Is it?", "", "period"),
    ("p - q", "", "text"), ("\\text{Monday}", "", "text"), ("42", "", "text"),
    ("0.5", PERCENT, "percent"), ("25", "What is 20 + 5?", "percent"), ("12", "What is 3 times 4?", "unit"),
    ("999", "", "commas"), ("01234", "", "commas"), ("1234.5", "", "commas"), ("1,23", "", "commas"),
])
def test_a_writing_is_sent_only_where_it_applies(answer, prompt, variant):
    assert variant not in _written(answer, prompt)


def test_words_in_text_only_under_a_format_with_a_box():
    assert "text" in _written("Paris") and "text" not in _written("Paris", primary="bare")


def test_thousands_separators_only_on_a_quantity_as_q8():
    year = Task("y", "In which year did the library open?", "2005")
    words = [Task(str(i), "Which city is it?", name) for i, name in enumerate(("Paris", "Rome", "Oslo"))]
    ctx = probe_context([year, *words], gold_type="short_str", reference_is_answer=True)
    ctx.task = year
    assert {name for name, *_ in G3.generate(year, random.Random(0), ctx)} == {"period"}  # no one writes 2,005
    assert "commas" in _written("2005", "How many people came?")
    assert set(_written("47869", "What is the pass key?")) == {"period"}  # nor 47,869 for a key


@pytest.mark.parametrize("answer", ['{"x": 3, "y": 4}', "[1, 2]", '["a", "b"]', "1 0\n0 1"])
def test_nothing_for_json_which_may_reach_the_checker_parsed_nor_for_a_grid(answer):
    assert _written(answer) == {}


def test_nothing_where_the_checker_does_not_get_the_reference_answer():
    assert _written("x+1", reaches=False) == {} and _written("1200", reaches=False) == {}


@pytest.mark.parametrize("spec, reaches", [
    ({"kind": "function", "path": "reward.py", "dataset": "data.jsonl"}, True),
    ({"path": "reward.py", "dataset": "data.jsonl"}, True),  # a spec with no kind is a function
    ({"kind": "demo", "name": "substring"}, True),
    ({"kind": "function", "reference_template": "#### {answer}"}, True),
    ({"kind": "function", "reference_template": "\\boxed{{{answer}}}"}, True),
    ({"kind": "function", "reference_template": "{solution}"}, False),  # a column of its own, without the answer
    ({"kind": "function", "reference_template": "{{answer}}"}, False),  # the word answer in braces, no field
    ({"kind": "function", "reference_col": "answers.text"}, False),
    ({"kind": "function", "ground_truth_json": True}, False),
    ({"kind": "verifiers", "env": "owner/name"}, False),
])
def test_whether_the_checker_gets_the_reference_answer(spec, reaches):
    assert reference_is_answer(spec) is reaches


def test_a_variant_with_a_reference_answer_is_sent_with_a_copy_of_its_task():
    task = Task("t7", "q", "1200", info={"k": 1})
    probe = Probe("Gx", "G", "test", "natural", "full", Label.SOLVING, lambda t, r, c: [("as_is", "1200"), ("commas", "1200", "1,200")])
    (same, _, _), (copy, name, reply) = _variants(probe, [task], probe_context([task], gold_type="short_str", reference_is_answer=True), seed=0)
    assert same is task and (name, reply) == ("commas", "1200")
    assert copy.answer == "1,200" and (copy.id, copy.prompt, copy.info) == ("t7", "q", {"k": 1}) and task.answer == "1200"


def test_g3_belongs_to_d22_and_does_not_count_in_the_rate():
    assert BY_PROBE["G3"] == "D22" and DEFECTS[-1].id == "D22" and DEFECTS[-1].probes == ("G3",)
    assert G3.label == Label.SOLVING and G3.expect == "full" and G3.tier == "natural"
    assert not G3.headline and not G3.gold_blind and not G3.by_variant  # its writings are sent unevenly: the rate is over all
    assert _rejected_severity("G3", 1.0) == "medium"  # never high, unlike Q5 and Q22


def exact(completion, answer):
    return float(completion.strip() == answer)


def test_a_checker_that_compares_strings_rejects_the_reply_against_another_writing():
    r = audit(exact, ROWS, n="all", probes=["G"], workers=1)
    f = next(f for f in r.findings if f["probe_id"] == "G3")
    assert f["category"] == "right_rejected" and f["severity"] == "medium" and f["rate"] == 1.0 and f["n"] == 24
    assert f["note"] == "rejected: commas 100%, period 100%"
    ex = f["example"]
    assert ex["variant"] == "period" and ex["completion"] == ex["reference"][:-1] and ex["reference"].endswith(".")
    d22 = next(d for d in r.defects if d["id"] == "D22")
    assert d22["severity"] == "medium" and d22["name"] == "Reference formatting changes the verdict"
    assert [e["probe_id"] for e in d22["evidence"]] == ["G3"]
    assert next(h for h in r.metrics["head"] if h["key"] == "right_rejected")["n"] == 0  # G3 counts in no rate
    md = render_markdown(r)
    assert "### MEDIUM: G3 the reference answer written another way" in md and "Rejected: commas 100%, period 100%" in md
    assert f"reference answer `{ex['completion']}` written as `{ex['reference']}`" in md


def test_the_note_gives_every_writing_most_often_rejected_first():
    def without_commas(completion, answer):
        return float(completion.strip() == answer.replace(",", ""))

    f = next(f for f in audit(without_commas, ROWS, n="all", probes=["G"], workers=1).findings if f["probe_id"] == "G3")
    assert f["note"] == "rejected: period 100%, commas 0%" and f["rate"] == 0.5 and f["n"] == 24
    assert f["example"]["variant"] == "period"  # the first call rejected


def test_a_checker_that_reads_the_reference_answer_as_a_number_accepts_every_writing():
    def number(completion, answer):
        return float(float(completion) == float(answer.replace(",", "")))

    r = audit(number, ROWS, n="all", probes=["G"], workers=1)
    g3 = next(p for p in r.metrics["probes"] if p["id"] == "G3")
    assert g3["n_scored"] == 24 and g3["obs_hits"] == 0 and not r.findings and not r.defects


def test_a_template_around_the_answer_holds_the_writing():
    def after_hash(completion, answer):
        return float(completion.strip() == answer.split("#### ")[-1])

    r = audit(after_hash, ROWS, n="all", probes=["G"], workers=1, reference_template="#### {answer}")
    assert next(f for f in r.findings if f["probe_id"] == "G3")["note"] == "rejected: commas 100%, period 100%"
    r = audit(lambda completion, answer: float(completion.strip() == answer), [row | {"sum": row["answer"]} for row in ROWS],
              n="all", probes=["G"], workers=1, reference_template="{sum}")
    assert "G3" not in {p["id"] for p in r.metrics["probes"]}


def test_nothing_is_sent_where_the_checker_gets_a_column_of_its_own():
    rows = [row | {"accepted": [row["answer"]]} for row in ROWS]
    r = audit(lambda completion, answer: float(completion.strip() in answer), rows, n="all", probes=["G"], workers=1,
              reference_col="accepted")
    assert "G3" not in {p["id"] for p in r.metrics["probes"]} and not r.findings


def test_on_a_few_rows_it_stays_low():
    f = next(f for f in audit(exact, ROWS[:3], n="all", probes=["G"], workers=1).findings if f["probe_id"] == "G3")
    assert f["severity"] == "low" and f["rated"] == "medium" and TOO_FEW in f["note"]


def test_its_calls_are_written_under_the_row_with_the_reply(tmp_path):
    audit(exact, ROWS, n="all", probes=["G"], workers=1, calls=tmp_path / "calls.jsonl")
    calls = [c for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
             if (c := json.loads(line))["probe_id"] == "G3"]
    assert len(calls) == 24 and {c["variant"] for c in calls} == {"period", "commas"}
    assert all(c["completion"] == ROWS[int(c["task_id"])]["answer"] and c["accepted"] is False for c in calls)
    assert all("reference" not in c for c in calls)


def test_repro_scores_the_example_against_the_writing_it_was_scored_against(tmp_path, monkeypatch):
    (tmp_path / "reward.py").write_text("def score(completion, answer):\n    return float(completion.strip() == answer)\n")
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(row) for row in ROWS))
    monkeypatch.chdir(tmp_path)
    r = audit("reward.py", "data.jsonl", n="all", probes=["G"], workers=1)
    write_outputs(r, tmp_path / "out")
    proc = subprocess.run([sys.executable, str(tmp_path / "out" / "repro.py")], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert [line.split()[0] for line in proc.stdout.splitlines()] == ["G3"] and "expected=full score=0.0" in proc.stdout
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    ex = next(f["example"] for f in report["metrics"]["findings"] if f["probe_id"] == "G3")
    assert ex["reference"] == ex["completion"] + "."


def test_the_probes_command_shows_both_writings():
    res = CliRunner().invoke(cli.app, ["probes", "G3"])
    assert res.exit_code == 0 and "reference answer 42 written as 42.: \\boxed{42}" in res.output
