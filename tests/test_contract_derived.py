"""Wrappers derived from the environment and the prompts; signal keys over every scored gold."""

import random

import pytest

from rewardaudit.contract import (
    COMMON_WRAPPERS,
    REASONING,
    WRAPPERS,
    Contract,
    derive_wrappers,
    infer_contract,
    wrapper_fn,
)
from rewardaudit.probes.base import ProbeContext
from rewardaudit.probes.order import o4_retract_to_gold, single_block
from rewardaudit.types import Kind, Task, Verdict


class FakeRunner:
    """In-process runner around `score(completion, answer) -> float | (float, sub_scores)`."""

    max_score = None

    def __init__(self, fn, meta=None):
        self.fn, self.calls, self.meta = fn, 0, meta or {}

    def score(self, task: Task, completion: str) -> Verdict:
        self.calls += 1
        out = self.fn(completion, task.answer)
        if isinstance(out, tuple):
            return Verdict(Kind.SCORE, score=float(out[0]), sub_scores=out[1])
        return Verdict(Kind.SCORE, score=float(out))


def _tasks(n=10, prompt="Compute it.", system=None):
    msgs = (lambda p: [{"role": "system", "content": system}, {"role": "user", "content": p}]) if system else None
    return [Task(id=f"t{i}", prompt=prompt, answer=str(10 + i), messages=msgs(prompt) if msgs else None)
            for i in range(n)]


def _parse_tag(c: str, tag: str) -> str | None:
    if f"<{tag}>" in c and f"</{tag}>" in c:
        return c.split(f"<{tag}>", 1)[1].split(f"</{tag}>", 1)[0].strip()
    return None


# --- ids ------------------------------------------------------------------------------------------------

def test_wrapper_ids_resolve_from_the_id_alone():
    assert wrapper_fn("boxed")("7") == "\\boxed{7}"
    assert wrapper_fn("tag:masked_output")("x") == "<masked_output>x</masked_output>"
    assert wrapper_fn("reasoning_tag:masked_output")("x") == REASONING + "<masked_output>x</masked_output>"
    assert wrapper_fn("prefix:Final")("7") == "Final: 7"
    assert wrapper_fn("reasoning_prefix:Final Result")("7") == REASONING + "Final Result: 7"
    for bad in ("nope", "tag:", "tag:a b", "prefix:a:b", "other:x"):
        with pytest.raises(KeyError):
            wrapper_fn(bad)


# --- derivation -----------------------------------------------------------------------------------------

def test_parser_answer_tag_comes_first():
    tasks = _tasks(system="Put your answer in <reversed_text> tags.")
    d = derive_wrappers(tasks, {"answer_tag": "reversed_text"})
    assert list(d) == ["tag:reversed_text", "reasoning_tag:reversed_text"]
    assert d["tag:reversed_text"] == "the parser's answer tag"


@pytest.mark.parametrize("system, expected", [
    ("Put your answer in <reversed_text> tags.", ["tag:reversed_text"]),
    ("Bethink thee by degrees within the <ponder>...</ponder> tokens; thereafter, render thy final answer "
     "within <andswaru>...</andswaru>.", ["tag:andswaru"]),
    ("Return ONLY the masked text wrapped in XML tags:\n<masked_output>\n[Your masked text here]\n</masked_output>",
     ["tag:masked_output"]),
    ("Solve it. End with a line of the form\nFinal: <answer>", ["prefix:Final"]),
    ("The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes)",
     ["prefix:ANSWER"]),
    # formats the fixed wrappers already write, reasoning tags and markup are not derived
    ("The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes)", []),
    ("Think inside <think> </think>, then give the answer inside <answer> </answer>.", []),
    ("Call a search engine by <search> query </search>; it will return the top searched results between "
     "<information> and </information>.", []),
    ("Here is some html: <div>hello</div>. Compute 2 + 2 and put the final answer in \\boxed{}.", []),
    ("Let $N$ be the count. Answer: $N$ is what we want.", []),
    ("Write it as Final answer: <answer>42</answer>", []),
])
def test_formats_named_in_prompts(system, expected):
    assert [w for w in derive_wrappers(_tasks(system=system)) if not w.startswith("reasoning_")] == expected


def test_derivation_is_capped_and_needs_a_recurring_instruction():
    many = " ".join(f"Put the answer to part {i} in <part{i}> tags." for i in range(8))
    d = derive_wrappers(_tasks(prompt=many))
    assert len(d) == 8 and list(d)[:2] == ["tag:part0", "reasoning_tag:part0"]
    tasks = _tasks(10)
    tasks[0] = Task(id="t0", prompt="Put your answer in <odd_one> tags.", answer="10")
    assert derive_wrappers(tasks) == {}  # one prompt in ten is task content, not the format


def test_user_given_wrappers_add_nothing():
    r = FakeRunner(lambda c, a: _parse_tag(c, "x") == a, meta={"answer_tag": "x"})
    c = infer_contract(r, _tasks(), wrappers=[("bare", lambda a: a)])
    assert c.wrappers_run == ["bare"] and c.derived == {}


# --- contracts with derived wrappers --------------------------------------------------------------------

def test_parser_tag_settles_in_stage_one():
    tasks = _tasks(system="Reverse the text. Put your answer in <reversed_text> tags.")
    r = FakeRunner(lambda c, a: _parse_tag(c, "reversed_text") == a, meta={"answer_tag": "reversed_text"})
    c = infer_contract(r, tasks)
    assert c.primary == "tag:reversed_text" and c.self_pass_rate == 1.0
    assert c.accepted == ["tag:reversed_text", "reasoning_tag:reversed_text"]
    assert c.wrapper_stage == "common" and r.calls == (len(COMMON_WRAPPERS) + 2 + 1) * len(tasks)  # + wrong answers
    assert c.to_dict()["derived_wrappers"]["tag:reversed_text"] == "the parser's answer tag"


def test_prompt_answer_line_becomes_primary():
    tasks = _tasks(prompt="Compute it. End with a line\nFinal: <answer>")
    r = FakeRunner(lambda c, a: c.strip().splitlines()[-1] == f"Final: {a}")
    c = infer_contract(r, tasks)
    assert c.primary == "prefix:Final" and "reasoning_prefix:Final" in c.accepted and c.self_pass_rate == 1.0


def test_derived_tag_the_checker_ignores_is_not_primary():
    # the prompt names a tag, but the checker reads the last box anywhere: the tag earns nothing on its own
    tasks = _tasks(prompt="Compute it. Put the final answer in <final> tags.")
    r = FakeRunner(lambda c, a: c.rsplit("\\boxed{", 1)[-1].split("}")[0] == a if "\\boxed{" in c else False)
    c = infer_contract(r, tasks)
    assert "tag:final" in c.wrappers_run and "tag:final" not in c.accepted
    assert c.primary == "boxed"


def test_fixed_contracts_unchanged_without_derived_formats():
    tasks = _tasks()
    boxed = lambda c, a: f"\\boxed{{{a}}}" in c  # noqa: E731
    c = infer_contract(FakeRunner(boxed, meta={"answer_tag": None}), tasks)
    assert c.derived == {} and all(w in dict(WRAPPERS) for w in c.wrappers_run) and c.primary == "boxed"


# --- signal keys ----------------------------------------------------------------------------------------

def _style(c: str) -> float:
    """A continuous style score that differs on every completion."""
    return (sum(map(ord, c)) % 97) / 100


def test_continuous_style_term_never_joins_the_signal():
    tasks = _tasks(30, system="Think within <ponder>...</ponder>; then render thy final answer within "
                              "<andswaru>...</andswaru>.")

    def fn(c, a):
        correct = float(_parse_tag(c, "andswaru") == a)
        fmt = 0.5 if "<andswaru>" in c else 0.0
        style = _style(c)
        return correct + 0.5 * fmt + style, {"correct": correct, "format": fmt, "style": style}

    c = infer_contract(FakeRunner(fn, meta={"answer_tag": "andswaru"}), tasks)
    assert c.signal_keys == ["correct"] and c.self_pass_rate == 1.0 and c.primary == "tag:andswaru"


def test_continuous_term_that_separates_is_left_out():
    tasks = _tasks(20, prompt="Put your answer in <out> tags.")

    def fn(c, a):
        got = _parse_tag(c, "out")
        correct = float(got == a)
        closeness = (1.0 - min(1.0, abs(float(got) - float(a)) / 10)) * (0.6 + _style(c) / 3) if got else 0.0
        return correct + closeness, {"correct": correct, "closeness": closeness}

    c = infer_contract(FakeRunner(fn), tasks)
    assert c.signal_keys == ["correct"] and "closeness left out" in c.signal_note


def test_signal_falls_back_to_self_passing_golds():
    # 7 of 10 golds are rejected outright: over all golds 'acc' does not separate, over the passing ones it does
    tasks = _tasks(10)
    bad = {"10", "11", "12", "13", "14", "15", "16"}

    def fn(c, a):
        acc = float(f"\\boxed{{{a}}}" in c and a not in bad)
        fmt = float("\\boxed{" in c)
        return acc + fmt, {"acc": acc, "fmt": fmt}

    c = infer_contract(FakeRunner(fn), tasks)
    assert c.signal_keys == ["acc"] and c.self_pass_rate == pytest.approx(0.3)


# --- probes on derived ids ------------------------------------------------------------------------------

def test_derived_tags_are_single_block():
    assert single_block("tag:masked_output") and single_block("reasoning_tag:x") and single_block("answer_tag")
    assert not single_block("prefix:Final") and not single_block("boxed")
    t = Task("t", "Compute it.", "42")
    for primary, sent in (("tag:out", False), ("reasoning_tag:out", False), ("prefix:Final", True)):
        contract = Contract(full_score=1.0, accepted=[primary], primary=primary, wrapper_rates={primary: 1.0},
                            self_pass={"t": True}, gold_score={"t": 1.0}, task_wrappers={"t": [primary]})
        ctx = ProbeContext(contract=contract, gold_type="int", tasks=[t], task=t)
        assert bool(o4_retract_to_gold(t, random.Random(0), ctx)) == sent, primary


@pytest.mark.parametrize("primary", ["tag:masked_output", "reasoning_tag:masked_output", "prefix:Final",
                                     "reasoning_prefix:Final"])
def test_probe_labels_hold_in_derived_formats(primary):
    import re

    from rewardaudit.equiv import contains_equiv, equiv
    from rewardaudit.goldtype import infer_gold_type
    from rewardaudit.probes.registry import all_probes
    from rewardaudit.types import Label

    tasks = [Task("int", "Compute 2^5.", "32"), Task("text", "Name the capital of France.", "Paris"),
             Task("mcq", "What is 6 times 7?\nA. 41\nB. 42\nC. 43\nD. 44", "B"),
             Task("long", "Mask the names.", "Hello [PII], are you willing to be our new founder?")]
    ids = [t.id for t in tasks]
    contract = Contract(full_score=1.0, accepted=[primary], primary=primary, wrapper_rates={primary: 1.0},
                        self_pass={i: True for i in ids}, gold_score={i: 1.0 for i in ids},
                        task_wrappers={i: [primary] for i in ids})
    ctx = ProbeContext(contract=contract, gold_type="int", tasks=tasks,
                       gold_types={t.id: infer_gold_type(t.answer) for t in tasks})
    inner = re.compile(r"<masked_output>(.*?)</masked_output>|^Final: (.*)$", re.S | re.M)
    n = 0
    for probe in all_probes():
        for task in tasks:
            ctx.task = task
            for _, comp in probe.generate(task, random.Random(0), ctx):
                n += 1
                gt = ctx.gtype(task)
                if probe.family == "M" and probe.label == Label.NOT_SOLVING:
                    answers = [a or b for a, b in inner.findall(comp)] or [comp]  # some M variants are unwrapped
                    assert not any(equiv(a.strip(), task.answer, gt) for a in answers), (probe.id, comp)
                if probe.family == "Q" and probe.label == Label.SOLVING and probe.id not in ("Q23", "Q24", "Q25", "Q26", "Q27", "Q30"):
                    assert contains_equiv(comp, task.answer, "set" if probe.id == "Q10" else gt), (probe.id, comp)
    assert n > 100
