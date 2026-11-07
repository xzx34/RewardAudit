"""The confirmation under a high defect: its example sent again, and cut down to the shortest reply still accepted."""

import re

import pytest
from rich.console import Console

from rewardaudit import audit, pipeline
from rewardaudit.confirm import MAX_CUT_CALLS, RESENDS, confirm
from rewardaudit.contract import Contract
from rewardaudit.report import _confirmation_text, _reply, render_markdown, render_terminal
from rewardaudit.types import Kind, Label, Observation, Task, Verdict

ROWS = [{"prompt": f"What is {i} + 100?", "answer": str(i + 100)} for i in range(16)]


def last_digits(completion, answer):
    """The quickstart checker: the last run of digits, so a sign or a hedge goes unnoticed."""
    numbers = re.findall(r"\d+", completion)
    return float(bool(numbers) and numbers[-1] == answer)


def substring(completion, answer):
    return float(answer in completion)


def boxed_only(completion, answer):
    return float(completion == f"\\boxed{{{answer}}}")


class Calls:
    """What `confirm` needs of a RunnerPool, counting the calls."""

    def __init__(self, fn):
        self.fn, self.calls, self.sent = fn, 0, []

    def score(self, task, completion):
        self.calls += 1
        self.sent.append(completion)
        return Verdict(Kind.SCORE, score=float(self.fn(completion, task.answer)))

    def score_many(self, items, progress=None):
        return [self.score(t, c) for t, c in items]


def _one_defect(fn, completion, probe_id="O11", defect_id="D6"):
    """A high defect whose example, a reply that should earn nothing, was accepted on reference answer 34."""
    task = Task("t", "p", "34")
    contract = Contract(full_score=1.0, accepted=["bare"], primary="bare", wrapper_rates={"bare": 1.0},
                        self_pass={"t": True}, gold_score={"t": 1.0})
    ex = Observation(probe_id, probe_id[0], "v", "t", completion, Verdict(Kind.SCORE, score=1.0), Label.NOT_SOLVING, "zero",
                     "natural").to_dict()
    defects = [{"id": defect_id, "severity": "high", "example": ex}]
    runner = Calls(fn)
    confirm(runner, defects, [task], contract)
    return defects[0], runner


@pytest.fixture(scope="module")
def quickstart():
    return audit(last_digits, ROWS, workers=1)


def test_every_high_defect_is_sent_again_and_no_other(quickstart):
    high = [d for d in quickstart.defects if d["severity"] == "high"]
    assert high and all(d["confirmation"]["resends"] == [1.0] * RESENDS for d in high)
    assert not any("confirmation" in d for d in quickstart.defects if d["severity"] != "high")


def test_a_hedge_is_cut_to_its_two_answers_not_to_the_right_one(quickstart):
    d = next(d for d in quickstart.defects if d["id"] == "D6")
    example, short = d["example"], d["confirmation"]["shortest_accepted"]
    gold = next(t.answer for t in quickstart.tasks if t.id == example["task_id"])
    numbers = re.findall(r"\d+", short["completion"])
    assert len(short["completion"]) < len(example["completion"]) and short["score"] == 1.0
    assert len(numbers) == 2 and numbers[-1] == gold and numbers[0] != gold
    assert last_digits(short["completion"], gold) == 1.0


def test_a_single_wrong_answer_has_nothing_to_cut(quickstart):
    d = next(d for d in quickstart.defects if d["id"] == "D1")
    assert "shortest_accepted" not in d["confirmation"] and len(re.findall(r"\S+", d["example"]["completion"])) == 1


def test_a_substring_match_is_cut_down_to_a_longer_number_that_holds_the_answer():
    d, _ = _one_defect(substring, "Possible answers: " + ", ".join(str(n) for n in range(-20, 201)), probe_id="E1",
                       defect_id="D2")
    short = d["confirmation"]["shortest_accepted"]["completion"]
    assert re.fullmatch(r"\d+,?", short) and "34" in short and short.rstrip(",") != "34"


def test_the_reference_answer_alone_is_never_sent():
    d, runner = _one_defect(last_digits, "The answer is either 35 or 34.")
    assert d["confirmation"]["shortest_accepted"] == {"completion": "35 34.", "score": 1.0}
    assert not {"34", "34.", "or 34."} & set(runner.sent)


def test_a_right_answer_rejected_is_only_sent_again():
    r = audit(boxed_only, ROWS, workers=1, probes=["Q"])
    d = next(d for d in r.defects if d["severity"] == "high")
    assert d["example"]["label"] == "solving" and d["confirmation"] == {"resends": [0.0] * RESENDS}


def test_a_reply_accepted_only_some_of_the_time_is_not_cut():
    scores = iter([1.0, 0.0] * 10)
    d, runner = _one_defect(lambda c, a: next(scores), "The answer is either 35 or 34.")
    assert d["confirmation"] == {"resends": [1.0, 0.0, 1.0, 0.0, 1.0]} and runner.calls == RESENDS


def test_a_checker_that_accepts_anything_is_cut_to_an_empty_reply():
    d, _ = _one_defect(lambda c, a: 1.0, "I am not sure. Maybe it is 7.", probe_id="N1", defect_id="D11")
    assert d["confirmation"]["shortest_accepted"] == {"completion": "", "score": 1.0}


def test_cutting_stops_at_its_limit():
    words = [f"w{i}" for i in range(300)]
    d, runner = _one_defect(lambda c, a: float(all(w in c.split() for w in words)), " ".join(words))
    assert runner.calls == RESENDS + MAX_CUT_CALLS and "shortest_accepted" not in d["confirmation"]


def test_confirming_leaves_every_rate_and_finding_as_it_was(quickstart, monkeypatch):
    monkeypatch.setattr(pipeline, "confirm", lambda *a, **kw: None)
    plain = audit(last_digits, ROWS, workers=1)

    def counted(r):
        return (r.rates, r.metrics["robustness"]["n_calls"], [(f["probe_id"], f["severity"], f["rate"], f["n"]) for f in r.findings],
                [(d["id"], d["severity"], d["evidence"], d["example"]["completion"]) for d in r.defects])

    assert counted(plain) == counted(quickstart) and not any("confirmation" in d for d in plain.defects)
    assert quickstart.meta["checker_calls"] > plain.meta["checker_calls"]  # the calls in the top line include them


def test_the_progress_bar_counts_every_confirming_call(monkeypatch):
    made, seen = [], []

    def counting(runner, *args, **kwargs):
        before = runner.calls
        confirm(runner, *args, **kwargs)
        made.append(runner.calls - before)

    monkeypatch.setattr(pipeline, "confirm", counting)
    audit(last_digits, ROWS, workers=1, progress=lambda stage, done, total: seen.append((stage, done)))
    assert "probes" in {s for s, _ in seen} and seen[-1][0] == "confirming defects"
    assert [d for s, d in seen if s == "confirming defects"] == list(range(1, made[0] + 1))


def test_the_terminal_and_report_md_have_a_line_under_each_high_defect(quickstart):
    console = Console(width=200, record=True)
    render_terminal(quickstart, console)
    text, md = console.export_text(), render_markdown(quickstart)
    high = [d for d in quickstart.defects if d["severity"] == "high"]
    assert text.count("Sent 5 more times: 1.0 each.") == len(high) == md.count("  - Sent 5 more times: 1.0 each.")
    short = _reply(next(d for d in high if d["id"] == "D6")["confirmation"]["shortest_accepted"]["completion"])
    assert f"Shortest reply still accepted (1.0): {short}" in text and f"Shortest reply still accepted (1.0): `{short}`" in md


def test_scores_that_differ_are_listed():
    assert _confirmation_text({"resends": [1.0, 0.0, 1.0, None, 0.5]}) == "Sent 5 more times: 1.0, 0.0, 1.0, no score, 0.5."
