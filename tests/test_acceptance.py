"""What counts as accepted on a composite rubric, what the report prints beside an example, the floor's chance for a
multiple-choice task, and the control for a reply above the gold. The checkers here are written for these tests: they
follow scoring rules seen in real environments, and nothing is imported from those environments."""

import json
import random

import pytest
from conftest import FakeRunner, compute

from rewardaudit.contract import Contract, _find_signal_keys, _task_full, infer_contract
from rewardaudit.metrics import REJECTED_GOLD_NAME, floor_estimate, hit
from rewardaudit.pipeline import AuditConfig, run_audit
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import render_markdown
from rewardaudit.types import Kind, Label, Observation, Task, Verdict

# Direction and a percent change: direction pays 1.0, the number 0.5 with a linear decay, the presence of both 0.2. The
# reply is read with its line ends as two characters, so a '-' splits it there and a negative gold loses its sign.
DIRECTION_AND_NUMBER = r"""
import re

def _direction(s):
    t = s.lower()
    return next((w for w in ("rise", "fall") if w in t), "neutral")

def _read(completion):
    text = completion.strip()
    m = re.search(r"<think>.*?</think>(.*?)$", text, re.S)
    if m and m.group(1).strip():
        text = m.group(1).strip()
    text = text.replace("\n", "\\n")
    m = re.search(r"boxed\{(.+?)\}", text)
    inner = m.group(1).strip() if m else text
    parts = re.split(r"[:;\-|,]\s*", inner)
    number = re.search(r"[-+]?[0-9]*\.?[0-9]+", " ".join(parts[1:]) if len(parts) > 1 else text)
    return _direction(parts[0]), float(number.group()) if number else None

def score(prompt, completion, answer):
    direction, pct = _read(completion)
    want, true = answer.splitlines()
    regression = max(0.0, 1.0 - abs(pct - float(true)) / 0.005) if pct is not None else 0.0
    return {"classification": float(direction == _direction(want)), "regression": 0.5 * regression,
            "format": 0.2 * (pct is not None)}
"""

# Yes/no, decimal and integer golds, each served by a term of its own, and a continuous term for the reasoning's wording.
TWO_KINDS_OF_ANSWER = r"""
import re

def score(prompt, completion, answer):
    text = completion.lower().strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    number = integer = 0.0
    try:
        number = float(any(abs(float(n) - float(answer)) < 0.01 for n in numbers))
        integer = float(int(answer) in [int(float(n)) for n in numbers])
    except ValueError:
        pass
    yes_no = 0.0
    if answer in ("yes", "no"):
        yes_no = 1.0 if text.startswith(answer) or f"answer: {answer}" in text else (0.8 if answer in text else 0.0)
    wording = min(1.0, sum(w in text for w in ("velocity", "gravity", "=", "/", "-")) * 0.1 + len(text) % 7 * 0.01)
    return {"number": number, "integer": integer, "yes_no": yes_no, "wording": 0.3 * wording}
"""

# A term for the answer, a term a second box or a hedge loses, and a length bonus.
FORMAT_A_HEDGE_LOSES = r"""
import re

def score(prompt, completion, answer):
    boxes = re.findall(r"boxed\{([^{}]*)\}", completion)
    correct = float(answer in boxes)
    one_box = float(len(boxes) == 1 and "either" not in completion)
    return {"correct": correct, "one_box": one_box, "length": min(1.0, len(completion) / 400)}
"""

# Pays for the text of a thinking block and never reads the answer: the gold earns only what its own reasoning earns.
THINKING_TERM = r"""
import re

def score(prompt, completion, answer):
    m = re.search(r"<think>(.*?)</think>", completion, re.S)
    think = m.group(1) if m else ""
    points = 0.3 * (len(think) >= 40) + 0.4 * ("condition" in think) + 0.2 * any(c.isdigit() for c in think)
    return {"correct": 0.0, "thinking": 0.15 * points, "format": 0.05 * ("boxed{" in completion)}
"""


def _spec(tmp_path, code, golds, prompt="Answer item {i}."):
    (tmp_path / "checker.py").write_text(code)
    rows = [{"prompt": prompt.format(i=i), "answer": g} for i, g in enumerate(golds)]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return {"kind": "function", "path": str(tmp_path / "checker.py"), "dataset": str(tmp_path / "data.jsonl")}


def _market_golds(n=100, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.54:
            out.append(f"Neutral\n{rng.uniform(0.0, 0.005) * rng.choice((1, 1, 1, -1)):.4f}")
        elif kind < 0.84:
            out.append(f"Rise\n{rng.uniform(0.005, 0.02):.4f}")
        else:
            out.append(f"Fall\n{-rng.uniform(0.005, 0.02):.4f}")
    return out


def _findings(r):
    return {f["probe_id"]: f for f in r.metrics["findings"]}


# --- signal keys ------------------------------------------------------------------------------------------------

def _pairs(*rows):
    """Verdicts from (gold sub-scores, wrong sub-scores)."""
    gold = [Verdict(Kind.SCORE, score=sum(g.values()), sub_scores=g) for g, _ in rows]
    wrong = [Verdict(Kind.SCORE, score=sum(w.values()), sub_scores=w) for _, w in rows]
    return gold, wrong


def test_a_term_that_serves_one_kind_of_answer_counts():
    rows = [({"number": 1.0, "yes_no": 0.0}, {"number": 0.0, "yes_no": 0.0})] * 82
    rows += [({"number": 0.0, "yes_no": 1.0}, {"number": 0.0, "yes_no": 0.0})] * 18
    keys, note = _find_signal_keys(*_pairs(*rows))
    assert keys == ["number", "yes_no"] and "yes_no: reference 1.00 vs wrong 0.00 on the 18 of 100 rows it serves" in note


def test_a_term_most_wrong_answers_also_earn_in_part_still_counts():
    # the mean gap is small (other tasks' numbers land close), but the gold beats nearly every one of them
    rows = [({"class": 1.0, "reg": 1.0}, {"class": float(i % 3 == 0), "reg": 0.6 + (i % 4) / 10}) for i in range(90)]
    assert _find_signal_keys(*_pairs(*rows))[0] == ["class", "reg"]


def test_a_format_term_stays_out_even_when_the_golds_share_its_ceiling():
    rows = [({"correct": 1.0, "format": 1.0}, {"correct": 0.0, "format": 1.0})] * 30
    assert _find_signal_keys(*_pairs(*rows))[0] == ["correct"]
    # a term tied to the answer's shape, which pays golds and wrong answers alike over the sample, stays out as well
    rows = [({"correct": 1.0, "shape": float(i % 5 > 0)}, {"correct": 0.0, "shape": float((i + 1) % 5 > 0)}) for i in range(30)]
    assert _find_signal_keys(*_pairs(*rows))[0] == ["correct"]


def test_a_term_the_gold_earns_by_chance_on_a_few_tasks_stays_out():
    rows = [({"format": 1.0, "bonus": float(i < 3)}, {"format": 1.0, "bonus": 0.0}) for i in range(100)]
    assert _find_signal_keys(*_pairs(*rows))[0] is None
    rows = [({"format": 1.0, "bonus": float(i < 10)}, {"format": 1.0, "bonus": 0.0}) for i in range(100)]
    assert _find_signal_keys(*_pairs(*rows))[0] == ["bonus"]


def test_continuous_terms_stay_out():
    style = [({"correct": 1.0, "style": ((i * 37) % 97) / 100}, {"correct": 0.0, "style": ((i * 37) % 97) / 200}) for i in range(30)]
    keys, note = _find_signal_keys(*_pairs(*style))
    assert keys == ["correct"] and "style left out" in note and "share no maximum" in note
    # a length bonus: golds differ in length, and no more above their wrong answers than below
    length = [({"correct": 1.0, "length": (i % 10 + 1) / 10}, {"correct": 0.0, "length": ((i + 5) % 10 + 1) / 10}) for i in range(30)]
    assert _find_signal_keys(*_pairs(*length))[0] == ["correct"]


def test_zero_weight_metrics_and_failed_calls_do_not_count():
    gold, wrong = _pairs(*[({"correct": 1.0, "metric:turns": 1.0}, {"correct": 0.0, "metric:turns": 0.0})] * 6)
    assert _find_signal_keys(gold, wrong)[0] == ["correct"]
    wrong[0] = Verdict(Kind.ERROR, error="boom")
    assert _find_signal_keys(gold, wrong)[0] == ["correct"]


@pytest.mark.parametrize("n, seed", [(30, 0), (30, 1), (60, 2), (100, 0), (100, 3)])
def test_the_choice_does_not_move_with_the_sample(tmp_path, n, seed):
    # a regression: at 100 tasks the number's term fell out, and "accepted" meant only "direction right"
    r = run_audit(_spec(tmp_path, DIRECTION_AND_NUMBER, _market_golds(n, seed)), AuditConfig(n_tasks=n, families=["N"]))
    assert r.contract.signal_keys == ["classification", "regression"]


# --- accepted: replies the checker visibly marks down, and what must not be hidden -------------------------------

@pytest.fixture(scope="module")
def market(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("market")
    return run_audit(_spec(tmp, DIRECTION_AND_NUMBER, _market_golds()), AuditConfig(n_tasks=100, keep_observations=True))


def test_a_reply_that_loses_the_number_is_not_accepted(market):
    found = _findings(market)
    for pid in ("M5", "M6", "M18", "M20", "O10", "O11"):
        assert pid not in found, found.get(pid)
    example = next(o for o in market.observations if o.probe_id == "M18" and market.contract.self_pass[o.task_id])
    assert not market.contract.accepted_verdict(example.verdict, example.task_id)
    assert example.verdict.score < market.contract.gold_total[example.task_id]


def test_a_reply_with_no_answer_paid_on_the_direction_term_stays_a_finding(market):
    found = _findings(market)
    for pid in ("N1", "N2", "N5", "N9", "N10"):
        assert pid in found and found[pid]["category"] == "non_solving_accepted" and found[pid]["rate"] > 0.3, pid
    heads = {h["key"]: h for h in market.metrics["head"]}
    assert heads["non_solving_accepted"]["value"] > 0.3


def test_the_same_scores_count_for_a_reply_without_the_gold_and_not_for_a_part_of_it(market):
    c = market.contract
    tid = next(t for t, ok in c.self_pass.items() if ok and "Neutral" in next(x.answer for x in market.tasks if x.id == t))
    empty = next(o for o in market.observations if o.probe_id == "N1" and o.task_id == tid)
    cut = next(o for o in market.observations if o.probe_id == "M18" and o.task_id == tid)
    assert empty.verdict.sub_scores == cut.verdict.sub_scores and empty.verdict.score == cut.verdict.score == 1.0
    assert c.pays_on_one_term(empty.verdict, tid) and not c.accepted_verdict(cut.verdict, tid)


@pytest.mark.parametrize("score, subs, expected", [
    (1.6, {"main": 0.0, "other": 1.0}, True),  # one term reached, more than half of the gold's total
    (1.5, {"main": 0.0, "other": 1.0}, False),  # half is not more than half
    (0.4, {"main": 0.0, "other": 1.0}, False),  # a term worth little stays partial credit
    (1.6, {"main": 0.9, "other": 0.9}, False),  # no term reached, however high the total
    (1.6, {"format": 1.0}, False),  # credit outside the signal
])
def test_one_term_pays_a_reply_only_for_most_of_the_gold_total(score, subs, expected):
    c = Contract(full_score=2.0, accepted=["bare"], primary="bare", wrapper_rates={}, self_pass={"t": True},
                 gold_score={"t": 2.0}, signal_keys=["main", "other"], gold_terms={"t": {"main": 1.0, "other": 1.0}},
                 gold_total={"t": 3.0})
    assert c.pays_on_one_term(Verdict(Kind.SCORE, score=score, sub_scores=subs), "t") is expected


def test_a_reply_marked_down_to_half_of_the_gold_total_is_not_accepted():
    """A gold cut after the one field the deciding term reads: that term stays, the total falls to under half."""
    c = Contract(full_score=0.9, accepted=["bare"], primary="bare", wrapper_rates={"bare": 1.0}, self_pass={"t": True},
                 gold_score={"t": 0.9}, signal_keys=["actions"], gold_total={"t": 0.614})
    assert not c.accepted_verdict(Verdict(Kind.SCORE, score=0.286, sub_scores={"actions": 0.9}), "t")
    assert c.accepted_verdict(Verdict(Kind.SCORE, score=0.5, sub_scores={"actions": 0.9}), "t")
    assert not c.accepted_verdict(Verdict(Kind.SCORE, score=0.5, sub_scores={"actions": 0.4}), "t")

    def counted(score, label):
        verdict = Verdict(Kind.SCORE, score=score, sub_scores={"actions": 0.9})
        return hit(Observation("M18", "M", "cut", "t", "x", verdict, label, "zero", "natural"), c)[0]

    assert not counted(0.286, Label.NOT_SOLVING) and counted(0.5, Label.NOT_SOLVING)  # the metrics count what this says
    assert counted(0.286, Label.SOLVING) and not counted(0.5, Label.SOLVING)


def test_one_term_says_nothing_on_a_task_where_the_gold_earns_only_one():
    c = Contract(full_score=1.0, accepted=["bare"], primary="bare", wrapper_rates={}, self_pass={"t": True},
                 gold_score={"t": 1.0}, signal_keys=["main", "other"], gold_terms={"t": {"main": 1.0}}, gold_total={"t": 1.0})
    assert not c.pays_on_one_term(Verdict(Kind.SCORE, score=1.0, sub_scores={"main": 1.0}), "t")
    assert not c.pays_on_one_term(Verdict(Kind.ERROR, error="boom"), "t")


def test_golds_the_number_term_leaves_behind_fail_and_are_reported(market):
    c = market.contract
    answers = {t.id: t.answer for t in market.tasks}
    negative = [t for t in c.self_pass if answers[t].splitlines()[1].startswith("-")]
    assert negative and not any(c.self_pass[t] for t in negative) and c.self_pass_rate < 1.0
    assert all(c.self_pass[t] for t in c.self_pass if t not in negative)
    assert "G1" in _findings(market)


def test_every_kind_of_gold_passes_and_is_probed(tmp_path):
    golds = [f"{i * 1.5 + 0.5:.1f}" for i in range(14)] + ["no", "yes", "no", "yes"] + [str(100 + i) for i in range(2)]
    r = run_audit(_spec(tmp_path, TWO_KINDS_OF_ANSWER, golds), AuditConfig(n_tasks=20, families=["N", "O"], keep_observations=True))
    c = r.contract
    assert c.signal_keys == ["integer", "number", "yes_no"] and c.self_pass_rate == 1.0 and not c.rejected_golds()
    assert "G1" not in _findings(r)
    answers = {t.id: t.answer for t in r.tasks}
    assert {answers[o.task_id] for o in r.observations} >= {"yes", "no"}
    tid = next(t for t, a in answers.items() if a == "no")
    assert c.gold_score[tid] == 1.0 and "boxed" not in c.wrapper_for(tid) and c.wrap("no", tid) != "\\boxed{no}"
    assert c.gold_terms[tid] == {"yes_no": 1.0} and c.task_full[tid] == 1.0
    assert c.task_full[next(t for t, a in answers.items() if a == "100")] == 2.0
    assert c.task_full[next(t for t, a in answers.items() if a == "0.5")] == 1.0


def test_a_term_for_other_tasks_asks_nothing_of_a_gold():
    # one kind of answer has a term of its own, and a rarer kind has one more
    terms = {"a": {"exact": 1.0, "integer": 1.0}, "b": {"exact": 1.0}, "c": {"exact": 1.0}, "d": {"exact": 1.0}, "e": {"yes_no": 1.0}}
    assert _task_full(["exact", "integer", "yes_no"], terms) == {"a": 2.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}
    # six tasks have the field the second term reads, four do not: those golds earn all they can
    terms = {str(i): {"action": 0.9, "triage": 1.0} if i < 6 else {"action": 0.9} for i in range(10)}
    assert _task_full(["action", "triage"], terms)["9"] == pytest.approx(0.9)


def test_a_term_most_golds_earn_marks_down_the_gold_it_leaves_at_zero():
    terms = {str(i): {"direction": 1.0, "number": 0.5} if i < 8 else {"direction": 1.0} for i in range(10)}
    assert _task_full(["direction", "number"], terms)["9"] == pytest.approx(1.5)
    # a gold the exact term rejects cannot pass on the partial term alone
    terms = {str(i): {"exact": 1.0, "partial": 1.0} if i else {"partial": 1.0} for i in range(10)}
    assert _task_full(["exact", "partial"], terms)["0"] == pytest.approx(2.0)
    assert _task_full(["exact"], {"0": {}, "1": {"exact": 1.0}, "2": {"exact": 1.0}})["0"] == pytest.approx(1.0)


def test_a_format_term_a_hedge_loses_does_not_hide_the_hedge(tmp_path):
    golds = [str(10 + i) for i in range(20)]
    r = run_audit(_spec(tmp_path, FORMAT_A_HEDGE_LOSES, golds), AuditConfig(n_tasks=20, families=["O"]))
    assert r.contract.signal_keys == ["correct"] and r.contract.self_pass_rate == 1.0
    found = _findings(r)
    assert {"O3", "O5", "O7", "O11"} <= set(found) and found["O3"]["rate"] == 1.0
    ex = found["O3"]["example"]
    assert ex["verdict"]["sub_scores"]["one_box"] == 0.0 and ex["verdict"]["sub_scores"]["correct"] == 1.0


# --- the report: totals beside an example ------------------------------------------------------------------------

def test_the_example_shows_both_totals_and_both_signals(market):
    md = render_markdown(market)
    section = md[md.index(": N1 empty completion"):]
    line = next(x for x in section.splitlines() if x.startswith("Example (row"))
    assert "this reply's total 1, the reference answer's total on this row 1.7" in line
    assert "on classification + regression 1 and 1.5" in line
    assert "A reply counts as accepted when it reaches the reference answer's value on all of these parts" in md


def test_g1_prints_a_total_next_to_its_example(market):
    f = _findings(market)["G1"]
    ex = f["example"]
    signal = market.contract.gold_score[ex["task_id"]]
    assert f"best total {ex['verdict']['score']:.3g} (classification + regression: {signal:.3g})" in f["note"]
    assert "best score" not in f["note"]
    md = render_markdown(market)
    line = next(x for x in md[md.index(f": G1 {REJECTED_GOLD_NAME}"):].splitlines() if x.startswith("Example (row"))
    assert f"total {ex['verdict']['score']:.3g}; on classification + regression {signal:.3g}" in line


def test_a_checker_without_sub_rewards_keeps_its_one_score(tmp_path):
    r = run_audit(_spec(tmp_path, "def score(prompt, completion, answer):\n    return float(answer in completion)\n",
                        [str(10 + i) for i in range(12)]), AuditConfig(n_tasks=12, families=["O"]))
    md = render_markdown(r)
    assert ", score 1.0):" in md and "this reply's total" not in md


# --- the floor ------------------------------------------------------------------------------------------------------

def _mcq_task(i, letters, gold):
    options = "\n".join(f"{ch}. option {ch}" for ch in letters)
    return Task(id=f"t{i}", prompt=f"Pick one.\n{options}\nAnswer:", answer=gold)


def test_the_floor_uses_the_task_option_count():
    tasks = [_mcq_task(i, "ABCDE", "ABCDE"[i % 5]) for i in range(10)]
    value, note = floor_estimate(tasks, "mcq")
    assert value == pytest.approx(0.2) and note.endswith("chance 1/5")
    two = [_mcq_task(i, "AB", "AB"[i % 2]) for i in range(6)]
    assert floor_estimate(two, "mcq")[0] == pytest.approx(0.5)


def test_the_floor_averages_mixed_option_counts_and_keeps_a_common_gold():
    tasks = [_mcq_task(i, "ABCD" if i % 2 else "ABCDE", "ABCD"[i % 4]) for i in range(8)]
    value, note = floor_estimate(tasks, "mcq")
    assert value == pytest.approx(0.25) and "chance 22% (4 to 5 options)" in note  # the most common gold, 2 of 8, is above the chance
    only_options = [_mcq_task(i, "ABCD" if i % 2 else "ABCDE", "ABCDEFGH"[i]) for i in range(8)]
    assert floor_estimate(only_options, "mcq")[0] == pytest.approx(0.225)
    same = [_mcq_task(i, "ABCDE", "B") for i in range(4)]
    assert floor_estimate(same, "mcq")[0] == 1.0


def test_a_prompt_without_listed_options_counts_as_four_and_other_types_are_unchanged():
    tasks = [Task(id=f"t{i}", prompt="Which one?", answer="ABCD"[i % 4]) for i in range(8)]
    assert floor_estimate(tasks, "mcq") == (0.25, "most common reference answer 'a' on 2/8 rows; chance 1/4")
    bools = [Task(id=f"t{i}", prompt="Is it?", answer="yes" if i else "no") for i in range(4)]
    assert floor_estimate(bools, "bool")[0] == 0.75 and floor_estimate(bools, "bool")[1].endswith("chance 1/2")
    assert floor_estimate(bools, "int") == (0.75, "most common reference answer 'yes' on 3/4 rows")


# --- the control for a reply above the gold --------------------------------------------------------------------------

def _think_contract():
    return Contract(full_score=1.0, accepted=["think_boxed"], primary="think_boxed", wrapper_rates={"think_boxed": 1.0},
                    self_pass={"t": True}, gold_score={"t": 1.0}, task_wrappers={"t": ["think_boxed"]})


@pytest.mark.parametrize("primary, reply, control", [
    ("think_boxed", "<think>\nMy first guess is 7.\n</think>\n\\boxed{9}", "<think>\nMy first guess is 7.\n</think>\n\\boxed{7}"),
    ("think_tag", "<think>\nnote\n</think>\n<answer>9</answer>", "<think>\nnote\n</think>\n<answer>7</answer>"),
    ("boxed", "Using one condition we get 7.\n\n\\boxed{9}", "Using one condition we get 7.\n\n\\boxed{7}"),
    ("boxed", "\\boxed{3}\nWait. The correct result is \\boxed{9}", "\\boxed{3}\nWait. The correct result is \\boxed{7}"),
    ("hash", "Some text.\n#### 9", "Some text.\n#### 7"),
    ("final_answer", "Some text.\nFinal Answer: 9", "Some text.\nFinal Answer: 7"),
    ("reasoning_boxed", "Thinking.\n\nTherefore the answer is \\boxed{9}.", "Thinking.\n\nTherefore the answer is \\boxed{7}."),
    ("bare", "<think>\nnote</think>\n\n9", "<think>\nnote</think>\n\n7"),
    ("code_block", "Text.\n```\n9\n```", "Text.\n```\n7\n```"),
    ("json", 'Text.\n{"answer": "9"}', 'Text.\n{"answer": "7"}'),
    ("tag:out", "Text.\n<out>9</out>", "Text.\n<out>7</out>"),
    ("bold", "Text.\n**9**", "Text.\n**7**"),
    ("dollar", "Text.\n$9$", "Text.\n$7$"),
    ("boxed_dollar", "Text.\n$\\boxed{9}$", "Text.\n$\\boxed{7}$"),
    ("the_final_answer_is", "Text.\nThe final answer is 9.", "Text.\nThe final answer is 7."),
    ("prefix:Final", "Text.\nFinal: 9", "Text.\nFinal: 7"),
    ("boxed", "1, 2, 3, 4, 5", None),  # a sweep has no final answer to replace
    ("boxed", "", None),
    ("hash", "No answer here.", None),
])
def test_the_control_keeps_the_reasoning_and_changes_the_answer(primary, reply, control):
    c = Contract(full_score=1.0, accepted=[primary], primary=primary, wrapper_rates={primary: 1.0},
                 self_pass={"t": True}, gold_score={"t": 1.0}, task_wrappers={"t": [primary]})
    assert c.with_answer(reply, "7", "t") == control


def test_a_reply_that_owes_its_score_to_its_reasoning_is_not_above_the_gold(tmp_path):
    golds = [f"L{i}.8" for i in range(12)]
    r = run_audit(_spec(tmp_path, THINKING_TERM, golds), AuditConfig(n_tasks=12, keep_observations=True))
    c, m = r.contract, r.metrics
    assert c.no_discrimination and c.signal_keys is None
    # the reply did score above the gold's own best, on the wording of its thinking block
    o10 = [o for o in r.observations if o.probe_id == "O10"]
    assert o10 and all(o.verdict.score > c.gold_score[o.task_id] for o in o10)
    assert "O10" not in m["above_gold"]["probes"] and "O10" not in _findings(r)
    assert m["discrimination"]["case"] == "same"
    determinism_calls = r.determinism["checked"] * r.determinism["repeats"]
    assert r.meta["checker_calls"] > c.calls + len(r.observations) + determinism_calls  # the controls were sent


def test_a_reply_that_beats_the_control_stays_above_the_gold(tmp_path):
    # pays for the thinking text, and more for any final answer other than the gold
    code = THINKING_TERM + r"""

_inner = score

def score(prompt, completion, answer):
    out = _inner(prompt, completion, answer)
    m = re.search(r"boxed\{(.*?)\}", completion)
    out["correct"] = 0.5 * bool(m and m.group(1) != answer)
    return out
"""
    r = run_audit(_spec(tmp_path, code, [f"L{i}.8" for i in range(12)]), AuditConfig(n_tasks=12, families=["O"]))
    assert r.metrics["above_gold"]["probes"].get("O10") == 12


def test_compute_reads_the_control():
    probe = next(p for p in all_probes() if p.id == "O10")
    tasks = [Task(id="t", prompt="p", answer="7")]
    reply = "<think>\nnote\n</think>\n\\boxed{9}"
    obs = [Observation("O10", "O", "think", "t", reply, Verdict(Kind.SCORE, score=1.5), Label.NOT_SOLVING, "zero", "natural")]
    c = _think_contract()
    assert compute(obs, [probe], tasks, c, "int")["above_gold"]["k"] == 1
    assert compute(obs, [probe], tasks, c, "int", controls={("t", reply): 1.5})["above_gold"]["k"] == 0
    assert compute(obs, [probe], tasks, c, "int", controls={("t", reply): 1.2})["above_gold"]["k"] == 1


def test_a_note_above_the_gold_on_a_composite_rubric_gives_the_signal_and_both_totals():
    probe = next(p for p in all_probes() if p.id == "N3")
    tasks = [Task(id="t", prompt="p", answer="7")]
    obs = [Observation("N3", "N", "A", "t", "A", Verdict(Kind.SCORE, score=0.9, sub_scores={"a": 0.8, "fmt": 0.1}),
                       Label.NOT_SOLVING, "zero", probe.tier)]
    c = Contract(full_score=0.5, accepted=["bare"], primary="bare", wrapper_rates={"bare": 1.0}, self_pass={"t": True},
                 gold_score={"t": 0.5}, wrong_checked=1, wrong_same=1, signal_keys=["a"], gold_total={"t": 0.7})
    f = compute(obs, [probe], tasks, c, "int")["findings"][0]
    assert f["above_gold"] and "on a: 0.8 against 0.5 on row t (totals 0.9 and 0.7)" in f["note"]


def test_the_contract_reports_the_gold_total(tmp_path):
    runner_golds = [str(10 + i) for i in range(10)]
    tasks = [Task(id=f"t{i}", prompt="p", answer=g) for i, g in enumerate(runner_golds)]

    def score(completion, answer):
        correct = float(f"\\boxed{{{answer}}}" in completion)
        return correct + 0.2, {"correct": correct, "fmt": 0.2}

    c = infer_contract(FakeRunner(score), tasks)
    assert c.signal_keys == ["correct"] and c.gold_total["t0"] == pytest.approx(1.2) and c.gold_score["t0"] == 1.0


# --- a malformed reply the checker accepts -----------------------------------------------------------------------------

def test_a_malformed_reply_the_checker_accepts_is_a_finding():
    import re

    from rewardaudit import audit

    def reads_the_reply_as_a_pattern(completion, answer):
        m = re.search(r"\\boxed\{(.*)\}", completion)
        return float(bool(m and re.fullmatch(m.group(1), answer)))

    rows = [{"prompt": f"Add {i} and 1. Put the answer in \\boxed{{}}.", "answer": str(i + 1)} for i in range(12)]
    r = audit(reads_the_reply_as_a_pattern, rows, workers=1, probes=["P"])
    p2 = next(f for f in r.findings if f["probe_id"] == "P2" and f["category"] != "robustness")
    assert p2["category"] == "non_solving_accepted" and p2["severity"] == "medium" and p2["rate"] == 1.0 and p2["n"] == 5
