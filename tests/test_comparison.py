"""The calls file `audit --calls` writes, and `compare`, which lines up two of them: where two checkers decide differently."""

import json
import re

import pytest
from rich.console import Console
from typer.testing import CliRunner

from rewardaudit import AuditError, audit, cli, compare
from rewardaudit.comparison import KEYS, render_compare
from rewardaudit.probes.registry import all_probes

ROWS = [{"prompt": f"What is {i} + 10? Put the answer in \\boxed{{}}.", "answer": str(i + 10)} for i in range(12)]
RATES = ["non_solving_accepted", "wrong_accepted", "hedging_accepted", "right_rejected", "wrong_partial"]
runner = CliRunner()


def last_number(completion, answer):
    numbers = re.findall(r"-?\d+", completion)
    return float(bool(numbers) and numbers[-1] == answer)


def boxed_number(completion, answer):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", completion)
    try:
        return float(bool(boxed) and float(boxed[-1]) == float(answer))
    except ValueError:  # −12 with a Unicode minus, 12 or 13
        return 0.0


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    """The same rows and seed under two checkers: one reads the last run of digits, one the number in the last box."""
    d = tmp_path_factory.mktemp("calls")
    a, b = d / "a.jsonl", d / "b.jsonl"
    return audit(last_number, ROWS, workers=2, calls=a), audit(boxed_number, ROWS, workers=2, calls=str(b)), a, b


def _write(path, calls):
    path.write_text("".join(json.dumps(dict(zip(KEYS, c, strict=True))) + "\n" for c in calls), encoding="utf-8")
    return path


# --- the calls file -----------------------------------------------------------------------------------------------------


def test_each_line_is_one_call_with_every_key(two):
    r, _, path, _ = two
    calls = _lines(path)
    assert all(list(c) == list(KEYS) for c in calls)
    assert all(c["kind"] in ("score", "timeout", "error", "crash") for c in calls)
    assert all((c["score"] is None) == (c["accepted"] is None) == (c["kind"] != "score") for c in calls)
    probe_calls = [c for c in calls if c["probe_id"] != "reference"]
    assert len(probe_calls) == sum(p["n_obs"] for p in r.metrics["probes"])  # every probe call, and only those
    assert {c["probe_id"] for c in probe_calls} == {p["id"] for p in r.metrics["probes"]}


def test_the_reference_answers_are_there_in_each_format_tried_and_the_wrong_answer_too(two):
    r, _, path, _ = two
    reference = [c for c in _lines(path) if c["probe_id"] == "reference"]
    assert len(reference) == len(r.contract.observations)
    assert {c["variant"] for c in reference} == {*r.contract.wrappers_run, "wrong_answer"}
    by_key = {(c["task_id"], c["variant"]): c for c in reference}
    task = r.tasks[0]
    assert by_key[(task.id, "boxed")] == {"task_id": task.id, "probe_id": "reference", "variant": "boxed",
                                         "completion": f"\\boxed{{{task.answer}}}", "kind": "score", "score": 1.0, "accepted": True}
    assert by_key[(task.id, "wrong_answer")]["accepted"] is False


def half_on_row_three(completion, answer):
    return (0.5 if answer == "13" else 1.0) * float(completion.strip().endswith(answer))


def test_a_reference_answer_short_of_full_score_is_not_accepted_even_where_it_is_the_rows_best(tmp_path):
    r = audit(half_on_row_three, ROWS, workers=1, probes=["N"], calls=tmp_path / "calls.jsonl")
    assert r.contract.self_pass == {t.id: t.answer != "13" for t in r.tasks}
    reference = [c for c in _lines(tmp_path / "calls.jsonl") if c["probe_id"] == "reference" and c["variant"] == "bare"]
    assert {c["task_id"]: (c["score"], c["accepted"]) for c in reference} == {
        t.id: (0.5, False) if t.answer == "13" else (1.0, True) for t in r.tasks}


def test_accepted_is_what_the_statistics_counted(two):
    r, _, path, _ = two
    calls = _lines(path)
    by_id = {p.id: p for p in all_probes()}
    for p in r.metrics["probes"]:
        if by_id[p["id"]].gold_blind:  # there a reply that happens to be the reference answer is accepted, but no hit
            continue
        mine = [c["accepted"] for c in calls if c["probe_id"] == p["id"] and c["accepted"] is not None]
        hits = mine.count(False) if p["label"] == "solving" else mine.count(True)
        assert (hits, len(mine)) == (p["obs_hits"], p["n_scored"]), p["id"]


def test_the_file_is_written_afresh_and_from_the_command_line(tmp_path):
    reward, data, path = tmp_path / "reward.py", tmp_path / "data.jsonl", tmp_path / "calls.jsonl"
    reward.write_text("def score(completion, answer):\n    return float(answer in completion)\n", encoding="utf-8")
    data.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    path.write_text("left from before\n", encoding="utf-8")
    res = runner.invoke(cli.app, ["audit", str(reward), "--dataset", str(data), "--probes", "N", "--workers", "1",
                                  "--calls", str(path), "-q"])
    assert res.exit_code != 2, res.output
    calls = _lines(path)
    assert {c["probe_id"] for c in calls} == {"reference", "N1", "N2", "N3", "N5", "N9", "N10"}


def test_a_calls_file_that_cannot_be_written_is_one_line(tmp_path):
    missing = tmp_path / "no" / "such" / "calls.jsonl"
    with pytest.raises(AuditError, match="Cannot write the calls to"):
        audit(last_number, ROWS, workers=1, calls=missing)
    res = runner.invoke(cli.app, ["audit", "demo:substring", "--calls", str(missing)], env={"COLUMNS": "1000"})
    lines = [line for line in res.output.splitlines() if line.strip()]
    assert res.exit_code == 2 and lines == [f"Error: Cannot write the calls to {missing}: No such file or directory"], res.output


# --- compare ------------------------------------------------------------------------------------------------------------


def test_compare_lines_up_the_calls_and_says_where_they_differ(two):
    ra, rb, a, b = two
    out = compare(a, b)
    keys_a, keys_b = ({(c["task_id"], c["probe_id"], c["variant"]) for c in _lines(p)} for p in (a, b))
    assert out["aligned"] == len(keys_a & keys_b) and out["only_in_a"] == len(keys_a - keys_b) and out["only_in_b"] == len(keys_b - keys_a)
    assert out["different_replies"] == 0  # both checkers take \boxed{}, so both audits sent the same replies
    groups = out["by_probe"]
    differ = sum(g["a_accepts"] + g["b_accepts"] + g["no_score"] for g in groups.values())
    assert differ > 0 and out["agreement"] == pytest.approx(1 - differ / out["aligned"])
    totals = [g["a_accepts"] + g["b_accepts"] + g["no_score"] for g in groups.values()]
    assert totals == sorted(totals, reverse=True) and all(totals)
    # −12 with a Unicode minus: the digits are 12, but it is no number to float(); 12.0 the other way round
    assert groups["M22"]["a_accepts"] > 0 and groups["M22"]["b_accepts"] == 0
    assert groups["Q8"]["b_accepts"] > 0 and groups["Q8"]["a_accepts"] == 0
    assert groups["reference"]["a_accepts"] > 0  # the bare answer and the other formats last_number reads too
    for g in groups.values():
        for kind, examples in g["examples"].items():
            assert len(examples) == min(3, g[kind])
            for ex in examples:
                assert list(ex) == ["task_id", "variant", "completion", "score_a", "score_b"]
    ex = groups["M22"]["examples"]["a_accepts"][0]
    assert (ex["score_a"], ex["score_b"]) == (1.0, 0.0) and "\\boxed{−" in ex["completion"]


def test_the_rates_of_each_side_are_its_audits_where_they_pool_the_same_calls(two):
    ra, rb, a, b = two
    heads = compare(a, b)["heads"]
    for r, side in ((ra, heads["a"]), (rb, heads["b"])):
        assert set(side) <= {*RATES, "no_score"} and side["no_score"] == 0.0
        for key in ("wrong_accepted", "hedging_accepted", "right_rejected", "wrong_partial"):
            assert side[key] == pytest.approx(r.rates[key]), key


def test_compare_from_python_is_the_function_after_its_module_is_imported():
    import rewardaudit
    import rewardaudit.comparison

    assert rewardaudit.compare is rewardaudit.comparison.compare is compare


def test_two_calls_with_no_score_agree_and_one_with_none_differs(tmp_path):
    a = _write(tmp_path / "a.jsonl", [("1", "P1", "x", "1,1,1", "timeout", None, None),
                                      ("1", "P5", "y", "9**9**9", "error", None, None),
                                      ("1", "M1", "plus1", "\\boxed{12}", "score", 0.0, False),
                                      ("1", "Q3", "period", "\\boxed{11}.", "score", 1.0, True)])
    b = _write(tmp_path / "b.jsonl", [("1", "P1", "x", "1,1,1", "score", 0.0, False),
                                      ("1", "P5", "y", "9**9**9", "timeout", None, None),
                                      ("1", "M1", "plus1", "#### 12", "score", 0.0, False),
                                      ("1", "Q3", "period", "#### 11.", "score", None, None),
                                      ("2", "Q3", "period", "#### 7.", "score", 1.0, True)])
    out = compare(a, b)
    assert (out["aligned"], out["only_in_a"], out["only_in_b"], out["different_replies"]) == (4, 0, 1, 2)
    assert out["agreement"] == 0.5 and list(out["by_probe"]) == ["P1", "Q3"]
    assert out["by_probe"]["P1"]["no_score"] == 1 and out["by_probe"]["Q3"]["a_accepts"] == 1
    assert out["by_probe"]["Q3"]["examples"]["a_accepts"] == [
        {"task_id": "1", "variant": "period", "completion": "\\boxed{11}.", "score_a": 1.0, "score_b": None, "completion_b": "#### 11."}]
    assert out["heads"]["a"] == {"wrong_accepted": 0.0, "right_rejected": 0.0, "wrong_partial": 0.0, "no_score": 0.5}
    assert out["heads"]["b"]["no_score"] == 0.4 and out["heads"]["b"]["right_rejected"] == 0.0


def _compare(*args):
    return runner.invoke(cli.app, ["compare", *map(str, args)])


def test_the_command_prints_at_80_columns(two, monkeypatch):
    monkeypatch.setattr(cli, "console", Console(width=80, force_terminal=False))
    _, _, a, b = two
    res = _compare(a, b)
    assert res.exit_code == 0, res.output
    out = compare(a, b)
    lines = [line.rstrip() for line in res.output.splitlines()]
    assert all(len(line) <= 80 for line in lines), res.output
    assert f"Aligned {out['aligned']:,} calls; agree on {out['agreement']:.1%}. Only in A: {out['only_in_a']:,}, only in B: 0." in res.output
    assert "replies differ" not in res.output
    m22 = next(i for i, line in enumerate(lines) if line.split()[:1] == ["M22"])
    assert f"A accepts {out['by_probe']['M22']['a_accepts']}, B accepts 0" in lines[m22]
    assert re.search(r"\[task \d+, unicode_minus\] \\boxed\{−\d+\}  A 1\.0  B 0\.0", lines[m22 + 1]), lines[m22 + 1]
    for label in ("Accepts non-answers", "Accepts wrong answers", "Accepts hedged answers", "Rejects right answers",
                  "Pays wrong answers in part", "Calls with no score"):
        assert any(line.strip().startswith(label) for line in lines), label


def test_both_replies_are_shown_where_they_differ(tmp_path):
    a = _write(tmp_path / "a.jsonl", [("1", "Q3", "period", "\\boxed{11}.", "score", 1.0, True)])
    b = _write(tmp_path / "b.jsonl", [("1", "Q3", "period", "#### 11.", "score", 0.0, False)])
    console = Console(width=80, record=True, force_terminal=False)
    render_compare(compare(a, b), console)
    text = console.export_text()
    assert "[task 1, period] A 1.0 on \\boxed{11}., B 0.0 on #### 11." in text
    assert "On 1 of them the two replies differ" in " ".join(text.split())


@pytest.mark.parametrize("a, b, says", [
    ([("1", "N1", "empty", "", "score", 0.0, False)], [("2", "N1", "empty", "", "score", 0.0, False)], "no task_id in common"),
    ([], [("2", "N1", "empty", "", "score", 0.0, False)], "holds no calls"),
    ("{not json\n", [("1", "N1", "empty", "", "score", 0.0, False)], "Line 1 of"),
    ('{"task_id": "1"}\n', [("1", "N1", "empty", "", "score", 0.0, False)], "with task_id, probe_id, variant, completion"),
    (None, [("1", "N1", "empty", "", "score", 0.0, False)], "Cannot read"),
])
def test_files_that_cannot_be_compared_are_one_line_and_exit_2(tmp_path, a, b, says):
    pa, pb = tmp_path / "a.jsonl", _write(tmp_path / "b.jsonl", b)
    if isinstance(a, str):
        pa.write_text(a, encoding="utf-8")
    elif a is not None:
        _write(pa, a)
    with pytest.raises(AuditError, match=says):
        compare(pa, pb)
    res = _compare(pa, pb)
    lines = [line for line in res.output.splitlines() if line.strip()]
    assert res.exit_code == 2 and len(lines) == 1 and lines[0].startswith("Error: ") and says in lines[0], res.output
