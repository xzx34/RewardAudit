"""Two-stage wrapper search: same contract as all wrappers, fewer calls; open-think prompts."""

import json

import pytest

from rewardaudit.contract import COMMON_WRAPPERS, OPEN_THINK, WRAPPERS, ends_open_think, infer_contract
from rewardaudit.runner import RunnerPool
from rewardaudit.types import Kind, Task, Verdict

N_ALL = len(WRAPPERS)


class FakeRunner:
    """In-process runner around a plain `score(completion, answer) -> float`."""

    max_score = None

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def score(self, task: Task, completion: str) -> Verdict:
        self.calls += 1
        return Verdict(Kind.SCORE, score=float(self.fn(completion, task.answer)))


def _tasks(n=10, prompt="Compute it."):
    return [Task(id=f"t{i}", prompt=prompt, answer=str(10 + i)) for i in range(n)]


def _same(a, b):
    for k in ("primary", "self_pass", "full_score", "gold_score", "signal_keys"):
        assert getattr(a, k) == getattr(b, k), k
    assert [w for w in a.accepted if w in b.wrappers_run] == b.accepted


def test_boxed_checker_settles_in_stage_one():
    tasks = _tasks()
    boxed = lambda c, a: f"\\boxed{{{a}}}" in c  # noqa: E731
    full, two = FakeRunner(boxed), FakeRunner(boxed)
    a = infer_contract(full, tasks, two_stage=False)
    b = infer_contract(two, tasks)
    _same(a, b)
    assert b.wrapper_stage == "common+families" and b.primary == "boxed"
    assert full.calls == (N_ALL + 1) * len(tasks)  # one wrong answer per passing task
    # stage 1 plus the rest of the boxed family (boxed_dollar, think_boxed, open_think_boxed, answer_tag_boxed,
    # think_tag_boxed), plus the wrong answers
    assert two.calls == (len(COMMON_WRAPPERS) + 5 + 1) * len(tasks) == b.calls
    assert set(b.wrappers_run) >= {"boxed_dollar", "think_boxed"} and "json" not in b.wrappers_run


def test_expands_to_all_when_no_common_wrapper_passes():
    tasks = _tasks()
    dollar = lambda c, a: c.strip() == f"${a}$"  # noqa: E731
    r = FakeRunner(dollar)
    c = infer_contract(r, tasks)
    assert c.wrapper_stage == "all" and "most tasks" in c.wrapper_note
    assert c.primary == "dollar" and c.self_pass_rate == 1.0
    assert r.calls == (N_ALL + 1) * len(tasks)  # one wrong answer per passing task


def test_unearned_sub_reward_expands():
    class SubRunner(FakeRunner):
        def score(self, task, completion):
            self.calls += 1
            acc = float(task.answer in completion)
            fmt = float(completion.startswith("<think>") and "\\boxed" in completion)
            return Verdict(Kind.SCORE, score=acc + fmt, sub_scores={"acc": acc, "fmt": fmt})

    c = infer_contract(SubRunner(None), _tasks())  # the format reward needs think_boxed, not in stage 1
    assert c.wrapper_stage == "all" and "fmt" in c.wrapper_note


# --- composite rubrics: the signal decides whether the common wrappers settle it -----------------------------

class CompositeRunner(FakeRunner):
    """`fn(completion, answer) -> sub-rewards`; the score is their sum."""

    def score(self, task, completion):
        self.calls += 1
        subs = self.fn(completion, task.answer)
        return Verdict(Kind.SCORE, score=sum(subs.values()), sub_scores=subs)


def _style(c):
    """A continuous term that differs on every completion and has no ceiling the golds share."""
    return (sum(map(ord, c)) % 97) / 100


def _accuracy_and_style(c, a):
    return {"accuracy": float(f"\\boxed{{{a}}}" in c), "style": 0.3 * _style(c)}


def _wrong_calls(c):
    return [comp for w, _, comp, _ in c.observations if w == "wrong_answer"]


def test_continuous_term_does_not_expand_to_every_wrapper():
    tasks = _tasks(30)
    full, two = CompositeRunner(_accuracy_and_style), CompositeRunner(_accuracy_and_style)
    a = infer_contract(full, tasks, two_stage=False)
    b = infer_contract(two, tasks)
    _same(a, b)
    assert b.signal_keys == ["accuracy"] and b.wrapper_stage == "common+families" and b.wrapper_note == ""
    # stage 1, the rest of the boxed family, one wrong answer per task: nothing is sent twice
    assert two.calls == b.calls == (len(COMMON_WRAPPERS) + 5 + 1) * len(tasks)
    assert full.calls == (N_ALL + 1) * len(tasks)
    assert len(_wrong_calls(b)) == len(tasks)


def test_a_declared_total_does_not_expand_once_the_signal_is_known():
    class Declared(CompositeRunner):
        max_score = 2.0  # accuracy + style could reach it in principle, no reply does

    c = infer_contract(Declared(_accuracy_and_style), _tasks(20))
    assert c.signal_keys == ["accuracy"] and c.wrapper_stage != "all"


def test_a_sub_reward_no_common_wrapper_earned_expands_without_a_wrong_pass_first():
    # accuracy needs '$x$', which stage 1 does not send; a wrong pass before the expansion would be wasted
    def fn(c, a):
        return {"accuracy": float(c.strip() == f"${a}$"), "format": float(a in c)}

    tasks = _tasks()
    r = CompositeRunner(fn)
    c = infer_contract(r, tasks)
    _same(infer_contract(CompositeRunner(fn), tasks, two_stage=False), c)
    assert c.wrapper_stage == "all" and "accuracy" in c.wrapper_note and c.primary == "dollar"
    assert r.calls == (N_ALL + 1) * len(tasks) and len(_wrong_calls(c)) == len(tasks)


def test_one_reward_and_a_metric_pay_for_the_wrong_answers_once():
    # the total is the signal here, so nothing is gained by sending the wrong answers before the expansion
    def fn(c, a):
        return {"correct": float(c.strip() == f"${a}$"), "metric:num_turns": 0.0}

    tasks = _tasks()
    r = CompositeRunner(fn)
    c = infer_contract(r, tasks)
    assert c.wrapper_stage == "all" and c.primary == "dollar" and c.signal_keys == ["correct"]
    assert r.calls == (N_ALL + 1) * len(tasks) and len(_wrong_calls(c)) == len(tasks)


def test_expansion_redoes_the_wrong_answer_only_where_the_best_wrapper_changed():
    # three golds earn accuracy only in '$\\boxed{x}$', which arrives with the boxed family after the first wrong pass
    late = {"10", "11", "12"}

    def fn(c, a):
        form = f"$\\boxed{{{a}}}$" if a in late else f"\\boxed{{{a}}}"
        return {"accuracy": float(form in c), "style": 0.3 * _style(c)}

    tasks = _tasks(10)
    r = CompositeRunner(fn)
    c = infer_contract(r, tasks)
    _same(infer_contract(CompositeRunner(fn), tasks, two_stage=False), c)
    assert c.signal_keys == ["accuracy"] and c.self_pass_rate == 1.0 and c.wrapper_stage == "common+families"
    sent = _wrong_calls(c)
    assert len(sent) == len(tasks) + len(late) and r.calls == c.calls
    # the second round went in the format those golds pass in, so the comparison holds: no wrong answer reaches them
    assert sum(comp.startswith("$\\boxed") for comp in sent) == len(late)
    assert c.wrong_same == 0 and c.wrong_checked == len(tasks) and c.wrong_signal == {t.id: 0.0 for t in tasks}


def test_a_stage_one_signal_that_settles_nothing_still_expands():
    # accuracy is earned on 6 of 20 tasks in stage 1 and on the rest only in a wrapper outside it
    def fn(c, a):
        return {"accuracy": float(f"\\boxed{{{a}}}" in c if int(a) < 16 else c.strip() == f"**{a}**"), "style": 0.1 * _style(c)}

    tasks = _tasks(20)
    r = CompositeRunner(fn)
    c = infer_contract(r, tasks)
    _same(infer_contract(CompositeRunner(fn), tasks, two_stage=False), c)
    assert c.wrapper_stage == "all" and c.self_pass_rate == 1.0 and c.signal_keys == ["accuracy"]
    assert "most tasks" in c.wrapper_note


def test_composite_cases_match_every_wrapper_in_contract_and_comparison():
    def graded(c, a):
        return {"correct": float(c.strip().endswith(f"Answer: {a}")), "length": min(1.0, len(c) / 500)}

    def weighted(c, a):
        return {"accuracy": 0.8 * (f"<answer>{a}</answer>" in c), "format": 0.2 * ("<answer>" in c), "metric:n": 0.0}

    def think(c, a):
        return {"correct": float(f"<answer>{a}</answer>" in c), "think": float(c.startswith("<think>"))}

    for fn in (_accuracy_and_style, graded, weighted, think):
        tasks = _tasks(12)
        a, b = infer_contract(CompositeRunner(fn), tasks, two_stage=False), infer_contract(CompositeRunner(fn), tasks)
        _same(a, b)
        assert (a.wrong_checked, a.wrong_same, a.wrong_mean) == (b.wrong_checked, b.wrong_same, b.wrong_mean), fn


def test_a_format_the_user_gave_is_the_primary_when_it_is_accepted():
    tasks = _tasks(prompt="Solve it.\n<|im_start|>assistant\n<think>")
    def fn(c, a):
        return c.endswith(a) or c.endswith(f"{a}</sol>") or c.endswith(f"<answer>{a}</answer>")

    assert infer_contract(FakeRunner(fn), tasks).primary == "open_think_tag"
    c = infer_contract(FakeRunner(fn), tasks, formats=["<sol>{answer}</sol>"])
    assert c.primary == "<sol>{answer}</sol>" and "<sol>{answer}</sol>" in c.accepted and c.wrapper_stage != "all"


def test_open_think_prompts_join_stage_one():
    tasks = _tasks(prompt="Solve it.\n<|im_start|>assistant\n<think>")
    assert ends_open_think(tasks) and not ends_open_think(_tasks())
    # the checker reads prompt + completion: exactly one <think> and one </think> before the answer tag
    def fn(c, a):
        return c.count("<think>") == 0 and c.count("</think>") == 1 and c.endswith(f"<answer>{a}</answer>")

    r = FakeRunner(fn)
    c = infer_contract(r, tasks)
    assert c.primary == "open_think_tag" and c.self_pass_rate == 1.0
    assert set(OPEN_THINK) <= set(c.wrappers_run) and c.wrapper_stage != "all"
    assert r.calls < N_ALL * len(tasks)
    full = infer_contract(FakeRunner(fn), tasks, two_stage=False)
    _same(full, c)


# --- through real workers: checkers written the way training code writes them -------------------------------------

LAST_NUMBER = r"""
import re

def compute_score(solution_str, ground_truth):
    m = re.findall(r"#### (\-?[0-9.,]+)", solution_str)
    return 1.0 if m and m[-1].replace(",", "") == ground_truth else 0.0
"""
LAST_BOX = r"""
def compute_score(solution_str, ground_truth):
    i = solution_str.rfind("\\boxed{")
    return float(i >= 0 and solution_str[i + 7:].split("}")[0] == ground_truth)
"""
ANSWER_LINE = r"""
import re

def score(completion, answer):
    m = re.search(r"[Aa]nswer(?: is)?:? *([ABCD])\b", completion)
    return float(bool(m) and m.group(1) == answer)
"""
ACCURACY_AND_FORMAT = r"""
import re

def compute_score(reward_input, format_weight=0.1):
    response = reward_input["response"]
    boxed = re.search(r"\\boxed\{(.*?)\}", response)
    accuracy = float(bool(boxed) and boxed.group(1) == reward_input["ground_truth"])
    fmt = float(bool(re.fullmatch(r"<think>.*</think>.*\\boxed\{.*\}.*", response, re.S)))
    return {"overall": (1 - format_weight) * accuracy + format_weight * fmt, "format": fmt, "accuracy": accuracy}
"""
SCORE_WITH_ACC = r"""
def compute_score(solution_str, ground_truth):
    tail = solution_str[-300:]
    pred = tail.rsplit("Answer:", 1)[-1].strip() if "Answer:" in tail else ""
    acc = pred == ground_truth
    return {"score": 1.0 if acc else -1.0, "acc": float(acc), "pred": pred, "length": min(1.0, len(solution_str) / 200)}
"""
NUMBERS = [str(10 + i) for i in range(12)]


def _worker_spec(tmp_path, code, golds=NUMBERS, prompt="Compute item {i}.", **spec):
    (tmp_path / "checker.py").write_text(code)
    rows = [{"prompt": prompt.format(i=i), "answer": g, "reference": json.dumps({"solution": g})} for i, g in enumerate(golds)]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return {"kind": "function", "path": str(tmp_path / "checker.py"), "dataset": str(tmp_path / "data.jsonl"), **spec}


@pytest.mark.parametrize("code, golds", [(LAST_NUMBER, NUMBERS), (LAST_BOX, NUMBERS), (ANSWER_LINE, list("ABCD") * 3)],
                         ids=["last_number", "last_box", "answer_line"])
def test_checkers_in_workers_get_identical_contracts(tmp_path, code, golds):
    with RunnerPool(_worker_spec(tmp_path, code, golds), workers=2) as r:
        tasks = r.tasks(12, 0)
        a = infer_contract(r, tasks, two_stage=False)
        b = infer_contract(r, tasks)
    _same(a, b)
    assert b.accepted == a.accepted
    assert b.calls < a.calls


@pytest.mark.parametrize("code, signal", [(ACCURACY_AND_FORMAT, "accuracy"), (SCORE_WITH_ACC, "acc")], ids=["easyr1", "verl_dict"])
def test_composite_checkers_in_workers_get_identical_contracts(tmp_path, code, signal):
    with RunnerPool(_worker_spec(tmp_path, code), workers=2) as r:
        tasks = r.tasks(12, 0)
        a = infer_contract(r, tasks, two_stage=False)
        b = infer_contract(r, tasks)
    _same(a, b)
    assert signal in b.signal_keys and (a.wrong_checked, a.wrong_same) == (b.wrong_checked, b.wrong_same)
    assert len(_wrong_calls(b)) == len(tasks) and b.calls <= a.calls


READS_PROMPT_AND_REPLY = r"""
import re

def compute_score(solution_str, ground_truth):
    reply = solution_str.split("<|im_start|>assistant", 1)[-1]
    if reply.count("<think>") != 1 or reply.count("</think>") != 1:  # the prompt opens the block, the reply closes it
        return 0.0
    m = re.search(r"<answer>(.*?)</answer>", reply, re.S)
    return float(bool(m) and m.group(1).strip() == ground_truth["solution"])
"""


def test_a_checker_that_reads_prompt_and_reply_passes_with_an_open_think_wrapper(tmp_path):
    golds = [f"item {i}" for i in range(10)]
    spec = _worker_spec(tmp_path, READS_PROMPT_AND_REPLY, golds, prompt="Who did it in case {i}?\n<|im_start|>assistant\n<think>",
                        reference_col="reference", ground_truth_json=True, prompt_in_completion=True)
    with RunnerPool(spec, workers=2) as r:
        c = infer_contract(r, r.tasks(10, 0))
    assert c.primary == "open_think_tag" and c.self_pass_rate == 1.0
