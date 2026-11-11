"""Every probe's construction-time label must hold: M variants are not the gold, Q variants are."""

import random
import re

import pytest
from conftest import probe_context

from rewardaudit.equiv import candidates, contains_equiv, equiv, states_the_answer, strip_wrappers
from rewardaudit.goldtype import infer_gold_type
from rewardaudit.probes.base import Probe, latex_defects
from rewardaudit.probes.registry import FAMILIES, all_probes
from rewardaudit.probes.rewrite import tex_shorthand
from rewardaudit.probes.robustness import gold_tower
from rewardaudit.probes.structured import semantics
from rewardaudit.types import Label, Task

TASKS = [
    Task("t1", "What is 6 times 7? Options:\nA. 41\nB. 42\nC. 43\nD. 44", "B"),
    Task("t2", "Compute 2^5.", "32"),
    Task("t3", "Simplify 2/4.", "\\frac{1}{2}"),
    Task("t4", "Is the sky blue?", "yes"),
    Task("equation", "Using the numbers [37, 15, 4, 13], create an equation that equals 16.", "(37 + 15) * (4 / 13)"),
    Task("t5", "Name the capital of France.", "Paris"),
    Task("t6", "Solve for x: 2x = -3.", "-1.5"),
    Task("t7", "Expand (n+1)^2.", "n^2+2n+1"),
    Task("t8", "Which US city is called the Big Apple?", "New York"),
    Task("fewshot", "Q: What is 2 + 3?\nA: 5\n\nQ: What is 4 + 4?\nA: 8\n\nQ: What is 6 + 7?\nA:", "13"),
    Task("template", 'What is 9 - 4?\nRespond in the form "Answer: $NUMBER".', "5"),
    Task("chat", "3 + 4 = ?", "7", messages=[
        {"role": "system", "content": "Reply with a number only."},
        {"role": "user", "content": "2 + 2 = ?"}, {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3 + 4 = ?"}]),
    Task("json", "Return the person's name and age as a JSON object.", '{"name": "Ada", "age": 36}'),
    Task("set", "Solve x^3 - 6x^2 + 11x - 6 = 0. Enter all the solutions, separated by commas.", "1, 2, 3"),
    Task("ordered", "List the first four primes in increasing order.", "2, 3, 5, 7"),
    Task("unstated", "Give the two numbers from the table.", "[4, 9]"),
    Task("grid", "Invert every cell of the binary grid:\n0 1 0\n1 0 1", "1 0 1\n0 1 0"),
    Task("jsongrid", "Transpose the matrix [[1, 3], [2, 4]].", "[[1, 2], [3, 4]]"),
    Task("again", "What is 16 times 2?", "32"),
    Task("compound", "Which kind of forest gets the most rain?", "rain forest"),
    Task("unit", "How far is the station?", "5 km"),
    Task("money", "A pen costs 3 dollars. How many dollars do 4 pens cost?", "12"),
    Task("percent", "What percentage of 200 is 50?", "25%"),
    Task("date", "When did the library open?", "2020-03-05"),
    Task("inequality", "Solve 2x + 1 > 7 for x.", "x > 3"),
    Task("near_zero", "What is 4 divided by 5000?", "0.0008"),
    Task("choice", "Who has more Grammys, Adele or Taylor Swift?", "Adele"),
]
# Their replies are right by construction, in spellings the tool's own comparison does not read.
SPELLINGS = ("Q23", "Q24", "Q25", "Q26", "Q27", "Q30")
PRIMARIES = ("boxed", "think_boxed", "bare")
WORD_GOLD = re.compile(r"\bgolds?\b", re.I)


def _ctx(primary="boxed"):
    ctx = probe_context(TASKS, primary, aliases={"t8": ["NYC", "New York City"]}, reference_is_answer=True)
    ctx.contract.accepted.append("answer_tag")  # a second accepted format
    ctx._numeric_dataset = True  # the fixture mixes gold types on purpose; Q8's labels stay covered
    return ctx


def _variants(primary="boxed", families=None):
    ctx = _ctx(primary)
    for probe in all_probes(families):
        for task in TASKS:
            ctx.task = task
            for name, comp, *_ in probe.generate(task, random.Random(0), ctx):  # G3's own reference answer left out
                yield ctx, probe, task, name, comp


def test_gold_types_of_fixtures():
    types = {t.id: infer_gold_type(t.answer) for t in TASKS}
    assert types["json"] == "json" and types["grid"] == "grid" and types["jsongrid"] == "grid"
    assert types["set"] == types["ordered"] == types["unstated"] == "list"


@pytest.mark.parametrize("primary", PRIMARIES)
def test_not_solving_variants_do_not_contain_gold_alone(primary):
    for ctx, probe, task, name, comp in _variants(primary):
        if probe.label != Label.NOT_SOLVING or probe.family == "O":
            continue
        gt = ctx.gtype(task)
        if probe.id in ("M23", "M26"):  # the tool's own comparison reads '5 km' and '5 m' as 5, 'x > 3' and 'x < 3' as 3
            assert strip_wrappers(comp) != task.answer
        elif probe.family == "M":
            inner = strip_wrappers(comp)
            assert not equiv(inner, task.answer, gt), (probe.id, name, comp, task.answer)
        elif probe.id in ("L5", "L6"):  # sent whatever the row's answer is; a coincidence is left out of the rate
            continue
        elif probe.family == "P" or (probe.family in ("N", "L") and gt not in ("bool", "mcq")):
            # gold-blind constants may coincide with closed-form golds; that is the floor, not an error
            assert not contains_equiv(comp, task.answer, gt), (probe.id, name, comp[:200], task.answer)


@pytest.mark.parametrize("primary", PRIMARIES)
def test_final_answer_is_wrong_when_gold_only_in_reasoning(primary):
    for ctx, probe, task, name, comp in _variants(primary, ["O"]):
        if probe.id not in ("O6", "O10"):
            continue
        assert contains_equiv(comp, task.answer, ctx.gtype(task)), (probe.id, comp)
        final = comp.rsplit("</think>", 1)[-1] if probe.id == "O10" else comp.rsplit("\n\n", 1)[-1]
        assert task.answer not in final, (probe.id, name, comp)
        assert not contains_equiv(final, task.answer, ctx.gtype(task)), (probe.id, name, comp)
        if "boxed" in primary:
            assert not equiv(candidates(comp)[-1], task.answer, ctx.gtype(task)), (probe.id, comp)


@pytest.mark.parametrize("primary", PRIMARIES)
def test_solving_variants_contain_exactly_the_gold(primary):
    for ctx, probe, task, name, comp in _variants(primary, ["Q"]):
        if probe.label != Label.SOLVING or probe.id in SPELLINGS:
            continue
        gt = "set" if probe.id == "Q10" else ctx.gtype(task)
        assert contains_equiv(comp, task.answer, gt), (probe.id, name, comp, task.answer)


def test_order_probes_follow_prompt_semantics():
    ctx = _ctx()
    sem = {t.id: semantics(t, ctx) for t in TASKS if ctx.gtype(t) == "list"}
    assert sem == {"set": "set", "ordered": "ordered", "unstated": None}
    ran: dict[str, set[str]] = {}
    for _, probe, task, _, comp in _variants("boxed", ["M", "Q"]):
        ran.setdefault(probe.id, set()).add(task.id)
        if probe.id == "Q10":
            assert not equiv(strip_wrappers(comp), task.answer, "list") and equiv(strip_wrappers(comp), task.answer, "set")
    assert ran["M8"] == {"ordered"} and ran["Q10"] == {"set"}
    assert ran["M9"] == ran["M10"] == {"set", "ordered", "unstated"}


def test_parser_probes_cost_bounded():
    ctx = _ctx()
    first = set(ctx.contract.passing_tasks()[:5])
    for _, probe, task, _, comp in _variants("boxed", ["P"]):
        assert task.id in first, probe.id
        assert (probe.id == "P17" or len(comp) <= 200_000) and probe.tier == "contrived" and not probe.headline


def test_every_probe_generates_for_some_task():
    produced = {p.id: 0 for p in all_probes()}
    for _, probe, _, _, _ in _variants():
        produced[probe.id] += 1
    assert all(produced.values()), [k for k, v in produced.items() if not v]


def test_every_probe_is_described_in_neutral_words():
    for p in all_probes():
        text = p.name + " " + p.description
        assert p.description and p.description.endswith((".", ")")), p.id
        assert not WORD_GOLD.search(text), p.id
    for description, _ in FAMILIES.values():
        assert not WORD_GOLD.search(description), description


def test_the_probes():
    assert {fam: [p.id for p in probes] for fam, (_, probes) in FAMILIES.items()} == {
        "N": ["N1", "N2", "N3", "N5", "N9", "N10"],
        "E": ["E1", "E5", "E2", "E3", "E6", "E9", "E8", "E10"],
        "L": ["L1", "L2", "L3", "L4", "L5", "L6"],
        "O": ["O3", "O4", "O5", "O6", "O7", "O11", "O10", "O12", "O13", "O14", "O15", "O16"],
        "M": ["M1", "M2", "M3", "M5", "M6", "M7", "M16", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25", "M26", "M27",
              "M8", "M9", "M10", "M28"],
        "Q": ["Q1", "Q3", "Q5", "Q6", "Q22", "Q21", "Q8", "Q9", "Q2", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q10"],
        "G": ["G2", "G3"],
        "P": ["P1", "P2", "P15", "P5", "P16", "P17"],
    }
    assert {p.expect for p in all_probes()} == {"zero", "full"} and {p.label for p in all_probes()} == {Label.NOT_SOLVING, Label.SOLVING}
    assert {p.id for p in all_probes() if not p.headline} == {"P1", "P2", "P15", "P5", "P16", "P17", "M24", "M28", "Q23", "Q28", "Q29",
                                                              "G2", "G3"}


@pytest.mark.parametrize("primary", PRIMARIES)
def test_variant_names_are_unique_inside_a_probe(primary):
    """The strategy behind the head metric is chosen per (probe, variant): two variants with one name would pool their calls."""
    seen = set()
    for _, probe, task, name, _ in _variants(primary):
        assert (probe.id, task.id, name) not in seen, (probe.id, task.id, name)
        seen.add((probe.id, task.id, name))


# --- construction guards ----------------------------------------------------------------------------------

MATH_ANSWERS = [Task("mixed", "q", "1\\frac{4}{5}"), Task("frac", "q", "\\frac{3840}{289}"), Task("date", "q", "March 26, 1973"),
        Task("nested", "q", "\\frac{\\sqrt{3}}{2}"), Task("poly", "q", "2x+3y"), Task("word", "q", "Paris"),
        Task("matrix", "q", "\\begin{pmatrix} -7 \\\\ 16 \\\\ 5 \\end{pmatrix}")]


def _math_ctx(primary="boxed"):
    ctx = _ctx(primary)
    ctx.tasks = MATH_ANSWERS
    for t in MATH_ANSWERS:
        ctx.contract.self_pass[t.id] = True
        ctx.contract.task_wrappers[t.id] = [primary]
        ctx.gold_types[t.id] = infer_gold_type(t.answer)
    return ctx


def _gen(ctx, probe_id, task):
    probe = next(p for p in all_probes() if p.id == probe_id)
    ctx.task = task
    return dict(probe.generate(task, random.Random(0), ctx))


@pytest.mark.parametrize("text,defects", [
    ("\\boxed{1\\frac{4}{}5}", {"empty_group"}), ("\\boxed{\\fra{c3840}{289}}", {"command:fra"}),
    ("\\boxed{\\frac{38}", {"braces", "arguments"}), ("\\boxed{1\\frac}", {"arguments"}), ("x^", {"dangling"}),
    ("\\left( x", {"left_right"}), ("\\begin{pmatrix} 1 \\\\ 2", {"environments"}),
    ("\\frac34", set()), ("\\frac8{21}", set()), ("\\sqrt[3]{2}", set()), ("\\{1, 2\\}", set()),
    ('{"answer": "\\\\frac{1}{2}"}', set()), ("The answer is clearly B.", set()),
])
def test_latex_defects(text, defects):
    assert latex_defects(text) == defects


def test_m_variants_are_never_malformed_latex():
    for primary in PRIMARIES:
        ctx = _math_ctx(primary)
        for probe in all_probes(["M"]):
            for task in MATH_ANSWERS:
                for name, comp in _gen(ctx, probe.id, task).items():
                    assert not ctx.malformed(comp, task), (probe.id, name, comp)
    bad = Probe("Mx", "M", "test", "natural", "zero", Label.NOT_SOLVING,
                lambda t, r, c: [("broken", c.wrap("1\\frac{4}{}5")), ("fine", c.wrap("1\\frac{5}{4}"))])
    ctx = _math_ctx()
    ctx.task = MATH_ANSWERS[0]
    assert [n for n, _ in bad.generate(MATH_ANSWERS[0], random.Random(0), ctx)] == ["fine"]


def test_m5_swaps_two_digits_or_two_letters_of_a_word():
    ctx = _math_ctx()
    assert _gen(ctx, "M5", MATH_ANSWERS[0]) == {}  # 1\frac{4}{5}: no adjacent digits, letters of \frac are a command
    assert _gen(ctx, "M5", MATH_ANSWERS[1]) == {"swap": "\\boxed{\\frac{3840}{829}}"}
    assert _gen(ctx, "M5", MATH_ANSWERS[4]) == {}  # 2x+3y: never x2 or yx (the same product)
    (swap,) = _gen(ctx, "M5", MATH_ANSWERS[5]).values()
    assert sorted(strip_wrappers(swap)) == sorted("Paris") and strip_wrappers(swap) != "Paris"
    for task in MATH_ANSWERS:
        for comp in _gen(ctx, "M5", task).values():
            inner = strip_wrappers(comp)
            assert sorted(inner) == sorted(task.answer)
            assert [i for i, c in enumerate(inner) if not c.isalnum()] == [i for i, c in enumerate(task.answer) if not c.isalnum()]  # punctuation stays


def test_equivalent_tex_forms():
    ctx = _math_ctx()
    assert _gen(ctx, "Q1", MATH_ANSWERS[0])["boxed_space"] == "\\boxed {1\\frac{4}{5}}"
    assert "boxed_space" not in _gen(_math_ctx("bare"), "Q1", MATH_ANSWERS[0])
    q9 = _gen(ctx, "Q9", MATH_ANSWERS[0]) | {f"n_{k}": v for k, v in _gen(ctx, "Q9", MATH_ANSWERS[3]).items()}
    assert q9["shorthand"] == "\\boxed{1\\frac45}" and q9["n_shorthand"] == "\\boxed{\\frac{\\sqrt3}{2}}"
    assert q9["n_grouped"] == "\\boxed{{\\frac{\\sqrt{3}}{2}}}" and "grouped" not in q9  # three levels inside the box
    assert tex_shorthand("\\frac{8}{21}") == "\\frac8{21}" and tex_shorthand("\\dfrac{3}{4}") == "\\dfrac34"
    for a, b in (("\\frac34", "\\frac{3}{4}"), ("\\frac8{21}", "\\frac{8}{21}"), ("1\\frac45", "1\\frac{4}{5}"),
                 ("\\sqrt3", "\\sqrt{3}")):
        assert equiv(a, b, "latex") and contains_equiv(f"\\boxed{{{a}}}", b, "latex"), (a, b)
    q6 = _gen(ctx, "Q6", MATH_ANSWERS[0])
    assert q6["boxed_word_after"].endswith("I have boxed the final answer above.")
    assert "the answer is consistent" in q6["answer_is_after"] and "boxed_word_after" not in _gen(_math_ctx("bare"), "Q6", MATH_ANSWERS[0])


def test_hedges_in_both_orders():
    ctx = _ctx()
    t1 = TASKS[0]
    o7, o11 = _gen(ctx, "O7", t1), _gen(ctx, "O11", t1)
    assert o7["either"] == "The answer is either B or A." and o11["either"] == "The answer is either A or B."
    assert o11["either_wrapped"] == "The answer is either \\boxed{A} or \\boxed{B}."
    assert o11["inside_wrapper"] == "\\boxed{A \\text{ or } B}" and "inside_wrapper_math" not in o11
    bare = _gen(_ctx("bare"), "O11", t1)
    assert bare["inside_wrapper"] == "A or B" and bare["inside_wrapper_math"] == "$A$ or $B$"
    for comp in o11.values():
        assert not states_the_answer(comp, "B", "mcq"), comp


def test_m19_letter_after_a_word_starting_with_the_gold_letter():
    ctx = _ctx()
    v = _gen(ctx, "M19", TASKS[0])["adverb"]
    assert v == "The answer is basically A." and not contains_equiv(v, "B", "mcq")
    assert not states_the_answer(v, "B", "mcq")


def test_power_towers_stay_small():
    ctx = _ctx()
    p5 = _gen(ctx, "P5", TASKS[1])
    assert {"py_tower", "tex_tower", "tex_tower2", "py_tower_pi"} <= set(p5)
    assert all(len(p5[k]) < 40 for k in ("py_tower", "tex_tower", "tex_tower2", "py_tower_pi"))
    equation = next(t for t in TASKS if t.id == "equation")
    assert _gen(ctx, "P15", equation) == {"gold_numbers": "\\boxed{37**15**13**4}"}
    assert gold_tower("\\frac{14}{3}") is None and gold_tower("(1) a\n(2) b\n(3) c") is None
    assert not contains_equiv("\\boxed{9^{9^{9^{9}}}}", "\\frac{1}{2}", "fraction")  # never handed to sympy


@pytest.mark.parametrize("primary", ["think_boxed", "think_tag", "open_think_tag", "reasoning_boxed"])
def test_one_thinking_block_per_completion(primary):
    """Two answers in one completion share the format's thinking block or reasoning; it is not repeated."""
    opened = 0 if primary.startswith("open_think") else 1
    for _, probe, _, name, comp in _variants(primary, ["O", "Q"]):
        assert comp.count("</think>") <= 1 and comp.count("<think>") <= opened, (probe.id, name, comp)
        assert comp.count("Let me work through this step by step.") <= 1, (probe.id, name, comp)
    ctx = _ctx(primary)
    assert all(not _gen(ctx, "O4", t) for t in TASKS) == (primary in ("think_tag", "open_think_tag"))
