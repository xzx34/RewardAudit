"""What the O family's wrong answer is, wrong grid cells, lists and grids with k items wrong, symbolic golds, what counts as
LaTeX, which prompts ask for only the answer, which numbers get Q8, and two option letters in one reply."""

from __future__ import annotations

import json
import random
import re

import pytest

from rewardaudit.contract import Contract
from rewardaudit.equiv import equiv, parse_grid, parse_json, parse_seq, strip_wrappers
from rewardaudit.goldtype import asks_answer_only, asks_integer, asks_quantity, infer_gold_type
from rewardaudit.pipeline import AuditConfig, run_audit
from rewardaudit.probes.base import ProbeContext
from rewardaudit.probes.registry import all_probes
from rewardaudit.types import Task

PROBES = {p.id: p for p in all_probes()}
O_PAIR_PROBES = ("O3", "O5", "O6", "O7", "O10", "O11")
PROSE_PROBES = ("Q5", "Q6", "Q21", "Q22", "O4")
SEEDS = range(20)


def _ctx(tasks, primary="boxed"):
    ids = [t.id for t in tasks]
    contract = Contract(full_score=1.0, accepted=[primary], primary=primary, wrapper_rates={primary: 1.0},
                        self_pass={i: True for i in ids}, gold_score={i: 1.0 for i in ids},
                        task_wrappers={i: [primary] for i in ids})
    return ProbeContext(contract=contract, gold_type="short_str", tasks=tasks,
                        gold_types={t.id: infer_gold_type(t.answer) for t in tasks})


def _gen(probe_id, task, tasks=None, primary="boxed", seed=0):
    ctx = _ctx(tasks or [task], primary)
    ctx.task = task
    return dict(PROBES[probe_id].generate(task, random.Random(seed), ctx))


def _wrongs(task, tasks=None):
    ctx = _ctx(tasks or [task])
    ctx.task = task
    return [ctx.wrong_answer(task, random.Random(s)) for s in SEEDS]


def _audit(i, wcag):
    return json.dumps({"issues": [{"wcag": wcag, "severity": "critical", "fix": f"add label {i}"}],
                       "corrected_code": f"<img alt='{i}'>"})


# --- the O family's wrong answer: JSON golds -----------------------------------------------------------------

AUDIT_PROMPT = "Audit this product card for accessibility. Format as a structured report."
AUDITS = [Task(str(i), AUDIT_PROMPT, _audit(i, f"1.{i}.1")) for i in range(6)]


@pytest.mark.parametrize(("gold", "wrong"), [
    ("1850", "3700"), ("20.09369", "40.1874"), ("-3", "-6"), ("1", "2"), ("0", "1"), ("0.5", "1.5"), ("-0.25", "-1.25"),
    ("\\frac{8}{21}", "\\frac{29}{21}"), ("\\frac{30}{7}", "\\frac{60}{7}"), ("-\\frac{3}{4}", "-\\frac{7}{4}"), ("3/4", "7/4"),
    ("\\frac{1}{99}", "\\frac{2}{99}"),  # not 100/99, a hundred times the answer, which a checker allowing for percentages takes
])
def test_a_wrong_number_is_at_least_one_and_its_own_size_away(gold, wrong):
    assert set(_wrongs(Task("t", "Compute it.", gold))) == {wrong}  # x + 1 would be within 5% of 1850


def test_without_a_json_prompt_the_wrong_answer_is_another_tasks_gold():
    golds = {t.answer for t in AUDITS}
    for w in _wrongs(AUDITS[0], AUDITS):
        assert w in golds and w != AUDITS[0].answer


@pytest.mark.parametrize("probe_id", O_PAIR_PROBES)
def test_no_o_probe_sends_the_gold_minus_a_field_as_wrong(probe_id):
    gold = json.loads(AUDITS[0].answer)
    field_dropped = [json.dumps({k: v for k, v in gold.items() if k != drop}) for drop in gold]
    for seed in SEEDS:
        comps = _gen(probe_id, AUDITS[0], AUDITS, seed=seed)
        assert comps
        for comp in comps.values():
            assert not any(part in comp for part in field_dropped), comp


def test_a_changed_number_is_not_the_wrong_answer_when_the_fields_are_not_the_answer():
    tasks = [Task(str(i), "Summarise the record.", json.dumps({"age": 30 + i, "note": f"n{i}"})) for i in range(4)]
    golds = {t.answer for t in tasks}
    assert all(w in golds for w in _wrongs(tasks[0], tasks))


@pytest.mark.parametrize("gold", [AUDITS[0].answer, '{"a":1}'])
def test_one_task_without_a_json_prompt_gets_no_part_of_the_gold(gold):
    t = Task("t", AUDIT_PROMPT, gold)
    for w in _wrongs(t):
        assert parse_json(w) is None and not equiv(w, gold, "json") and gold not in w


def test_a_json_prompt_keeps_the_probes_on_fields_and_values():
    t = Task("t", "Return the record as a JSON object.", '{"name": "Ada", "age": 36}')
    assert set(_wrongs(t)) == {'{"name": "Ada", "age": 82}'}
    no_number = Task("t", "Return the record as a JSON object.", '{"name": "Ada", "city": "Rome"}')
    assert set(_wrongs(no_number)) <= {'{"name": "Ada"}', '{"city": "Rome"}'}


def test_m11_still_runs_where_the_prompt_does_not_ask_for_json():
    t = Task("t", "Extract the person's name and age.", '{"name": "Ada", "age": 36}')
    assert strip_wrappers(_gen("M11", t)["value"]) == '{"name": "Ada", "age": 82}'


# --- the O family's wrong answer: list golds -----------------------------------------------------------------

def _ids(text: str) -> set[str]:
    """What a checker that reads every number of the reply as an element of a set finds."""
    return set(re.findall(r"\d+", text))


@pytest.mark.parametrize("gold", ["11, 58", "[11, 58]", "(3, 5)", "1, 2, 3", "7, 7, 9"])
def test_a_wrong_list_is_not_a_subset_of_the_gold(gold):
    t = Task("t", "Return a comma-separated list of the feature numbers.", gold)
    for w in _wrongs(t):
        assert w and not set(parse_seq(w)) <= set(parse_seq(gold)), w


@pytest.mark.parametrize("probe_id", ["O7", "O11"])
def test_a_hedge_on_a_list_does_not_union_back_to_the_gold(probe_id):
    t = Task("t", "Return a comma-separated list of the feature numbers.", "11, 58")
    for seed in SEEDS:
        for name, comp in _gen(probe_id, t, primary="bare", seed=seed).items():
            assert _ids(comp) > {"11", "58"}, (name, comp)


def test_o_family_list_answer_keeps_the_gold_spelling_and_length():
    for w in _wrongs(Task("t", "List them.", "[11,58,3]")):
        assert w.startswith("[") and "," in w and ", " not in w and len(parse_seq(w)) == 3


@pytest.mark.parametrize("gold", ['["A", "B"]', '["1", "2"]'])
def test_a_wrong_list_in_json_stays_json(gold):
    t = Task("t", "List them.", gold)
    (added,) = _gen("M10", t).values()
    for w in [*_wrongs(t), strip_wrappers(added, keep_parens=True)]:
        assert isinstance(json.loads(w), list) and all(isinstance(x, str) for x in json.loads(w))


def test_another_gold_that_is_a_subset_or_a_reordering_is_not_a_wrong_answer():
    tasks = [Task("a", "Name them.", "[red, green]"), Task("b", "Name it.", "red"), Task("c", "Name them.", "[green, red]"),
             Task("d", "Name it.", "blue")]
    assert set(_wrongs(tasks[0], tasks)) == {"blue"}


def test_m9_still_drops_one_element():
    t = Task("t", "List them.", "1, 2, 3")
    (dropped,) = _gen("M9", t).values()
    assert len(parse_seq(dropped)) == 2 and set(parse_seq(dropped)) < {"1", "2", "3"}


# --- a wrong grid cell stays among the grid's values ---------------------------------------------------------

def _in_palette(grid) -> bool:
    return all(isinstance(c, int) and 0 <= c <= 9 for row in grid for c in row)


ARC = "[[0, 8, 3], [8, 0, 3]]"


@pytest.mark.parametrize("gold", [ARC, "[[9, 9], [9, 9]]", "[[0, 0], [0, 0]]", "[[1, 1], [1, 1]]", "[[5, 5], [5, 5]]",
                                  "[[0, 9], [9, 0]]"])
def test_m17_keeps_a_json_grid_inside_a_0_9_palette(gold):
    t = Task("t", "Predict the output grid.", gold)
    for seed in SEEDS:
        (comp,) = _gen("M17", t, seed=seed).values()
        grid = json.loads(strip_wrappers(comp))
        assert _in_palette(grid) and grid != json.loads(gold), comp


def test_a_wrong_cell_is_a_value_the_grid_already_uses():
    t = Task("t", "Predict the output grid.", ARC)
    for seed in SEEDS:
        (comp,) = _gen("M17", t, seed=seed).values()
        grid = json.loads(strip_wrappers(comp))
        assert {c for row in grid for c in row} <= {0, 3, 8}
        assert sum(a != b for ra, rb in zip(grid, json.loads(ARC), strict=True) for a, b in zip(ra, rb, strict=True)) == 1


def test_the_o_family_wrong_grid_has_valid_cells():
    t = Task("t", "Predict the output grid.", ARC)
    for w in _wrongs(t):
        assert _in_palette(json.loads(w)) and not equiv(w, ARC, "grid")


@pytest.mark.parametrize("gold,cells", [("0 8\n8 3", set("038")), ("1 0\n0 1", set("01")), ("999\n999", set("89")),
                                        ("5 5\n5 5", set("56")), ("10 20\n30 40", {"10", "20", "30", "40"}),
                                        ("#.#\n.#.", set("#."))])
def test_a_text_grid_keeps_its_shape_and_its_symbols(gold, cells):
    t = Task("t", "Predict the output grid.", gold)
    pat = r"\S" if " " not in gold else r"\S+"
    for seed in SEEDS:
        (comp,) = _gen("M17", t, seed=seed).values()
        text = strip_wrappers(comp, keep_punct=True)
        assert set(re.findall(pat, text)) <= cells and text != gold
        assert [len(re.findall(pat, ln)) for ln in text.splitlines()] == [len(re.findall(pat, ln)) for ln in gold.splitlines()]


def test_other_kinds_of_cell():
    assert _gen("M17", Task("t", "q", '[["a", "b"], ["b", "a"]]'))
    assert _gen("M17", Task("t", "q", "[[0.5, 0.25], [0.25, 0.5]]"))
    (comp,) = _gen("M17", Task("t", "q", "[[true, false], [false, true]]")).values()
    assert {c for row in json.loads(strip_wrappers(comp)) for c in row} == {True, False}


# --- lists and grids with k items wrong ----------------------------------------------------------------------

def _wrong_items(comp: str, gold: str) -> set[int]:
    return {i for i, (a, b) in enumerate(zip(parse_seq(comp), parse_seq(gold), strict=True)) if a != b}


def test_m28_replaces_k_list_items_with_new_ones_for_every_k():
    gold = "1, 2, 3"
    m28 = _gen("M28", Task("t", "List them.", gold))
    assert list(m28) == ["k1", "k2", "all"]
    wrong = [_wrong_items(comp, gold) for comp in m28.values()]
    assert [len(w) for w in wrong] == [1, 2, 3] and wrong[0] < wrong[1] < wrong[2]  # each holds the wrong items of the one before
    for comp in m28.values():
        items = parse_seq(comp)
        assert len(set(items)) == 3 and not equiv(strip_wrappers(comp), gold, "set")
    assert not set(parse_seq(m28["all"])) & {"1", "2", "3"}


def test_m28_steps_are_close_together_at_both_ends_of_a_long_list():
    gold = str(list(range(1, 13)))
    m28 = _gen("M28", Task("t", "List them.", gold))
    assert list(m28) == ["k1", "k2", "k3", "k6", "k10", "k11", "all"]
    assert [len(_wrong_items(comp, gold)) for comp in m28.values()] == [1, 2, 3, 6, 10, 11, 12]


def test_m28_takes_words_from_another_row_and_keeps_json_strings():
    t = Task("t", "Name them.", "[red, green, blue]")
    assert _gen("M28", t) == {}  # no other row to take a word from
    m28 = _gen("M28", t, [t, Task("o", "Name them.", "[cyan, red, teal]")])
    assert strip_wrappers(m28["all"], keep_parens=True) in ("[cyan, teal, cyan]", "[teal, cyan, cyan]", "[cyan, cyan, teal]")
    assert all(json.loads(strip_wrappers(c, keep_parens=True))[0] in "abcxyz" for c in _gen("M28", Task("t", "q", '["x", "y", "z"]')).values())


@pytest.mark.parametrize("gold", ["1 0 1\n0 1 0", ARC, "#.#\n.#."])
def test_m28_changes_k_cells_of_a_grid_to_values_it_already_uses(gold):
    m28 = _gen("M28", Task("t", "Predict the output grid.", gold))
    assert list(m28) == ["k1", "k2", "k3", "k4", "k5", "all"]
    ref = parse_grid(gold)
    for k, comp in enumerate(m28.values(), 1):
        grid = parse_grid(strip_wrappers(comp, keep_punct=True))
        assert sum(a != b for ra, rb in zip(grid, ref, strict=True) for a, b in zip(ra, rb, strict=True)) == k
        assert {c for row in grid for c in row} <= {c for row in ref for c in row}


@pytest.mark.parametrize("gold", ["[11, 58]", "Paris", "42", '{"a": 1, "b": 2, "c": 3}'])
def test_m28_needs_three_items(gold):
    assert _gen("M28", Task("t", "List them.", gold)) == {}


# --- short symbolic golds ------------------------------------------------------------------------------------

@pytest.mark.parametrize("gold", ["BE", "EB", "ABC", "xy", "Fe", "PQ, RS"])
def test_short_letter_runs_are_not_swapped_or_recased(gold):
    t = Task("t", "Find the locus of point P.", gold)
    assert infer_gold_type(gold) == "short_str"
    assert not _gen("M5", t) and not _gen("Q2", t)


@pytest.mark.parametrize("gold", ["less", "Jovan", "Paris", "ENTAILMENT", "New York"])
def test_words_are_still_swapped_and_recased(gold):
    t = Task("t", "Name it.", gold)
    (swap,) = _gen("M5", t).values()
    assert sorted(strip_wrappers(swap)) == sorted(gold) and strip_wrappers(swap) != gold
    assert set(_gen("Q2", t).values()) == {f"\\boxed{{{x}}}" for x in (gold.lower(), gold.upper()) if x != gold}


def test_a_long_run_of_letters_stays_a_word():
    assert _gen("Q2", Task("t", "Name it.", "AgainstKuznetsov"))["upper"] == "\\boxed{AGAINSTKUZNETSOV}"
    assert _gen("Q2", Task("t", "Name it.", "OK")) == {}  # the two letters a prompt spells for a label go with them


@pytest.mark.parametrize("gold", ["no", "yes", "NO", "True"])
def test_short_boolean_golds_keep_q2(gold):
    assert _gen("Q2", Task("t", "Is it?", gold))


def test_a_swap_stays_inside_a_word_next_to_a_short_token():
    t = Task("t", "Name the line.", "line BE")
    for seed in SEEDS:
        (swap,) = _gen("M5", t, seed=seed).values()
        assert strip_wrappers(swap).endswith(" BE")


def test_digits_of_a_symbolic_gold_are_still_swapped():
    assert _gen("M5", Task("t", "Name the line.", "AB12")) == {"swap": "\\boxed{AB21}"}


def test_a_character_level_prompt_has_no_case_rewrite():
    gold = "noitomocol rof dezilaiceps si xaroht ehT"
    chat = [{"role": "system", "content": "Reverse the text character-by-character. Put your answer in tags."},
            {"role": "user", "content": "The thorax is specialized for locomotion"}]
    assert not _gen("Q2", Task("t", "The thorax is specialized for locomotion", gold, messages=chat))
    assert not _gen("Q2", Task("t", "Name it. The answer is case-sensitive.", "Paris"))
    assert _gen("Q2", Task("t", "Translate the text.", gold))


# --- what counts as LaTeX ------------------------------------------------------------------------------------

LAMBDA = "((d (\\x0.((x0 (x0 x0)) (\\x1.(x1 (\\x2.x1)))))) (\\x3.x3))"


@pytest.mark.parametrize("gold", [LAMBDA, "\\x.x", "\\x0.x0", "\\v0.(v0 a)", "\\d+", "line\\n2"])
def test_a_backslash_before_one_letter_is_not_latex(gold):
    assert infer_gold_type(gold) != "latex"


def test_q9_is_not_sent_on_a_lambda_term():
    assert not _gen("Q9", Task("t", "Reduce the term to normal form.", LAMBDA), primary="answer_tag")


@pytest.mark.parametrize("gold", [
    "\\frac{1}{x}", "\\sqrt{x}", "\\mathbb{R}", "\\varphi", "\\vec{v}", "\\overrightarrow{AB}", "\\pi", "\\alpha + 1",
    "\\in", "x \\le 2", "\\%5x", "\\{1, 2\\}", "a \\\\ b", "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}",
])
def test_real_latex_is_still_latex(gold):
    assert infer_gold_type(gold) == "latex"


# --- prompts that limit the tag and not the reply ------------------------------------------------------------

TAG_LIMIT = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The "
            "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
            "respectively. Do not explain your reasoning inside the answer tags; provide only the final answer.")
TAG_LIMIT_WITH_EXAMPLE = ("A conversation between User and Assistant.\n"
                  "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
                  "respectively, i.e., <think> reasoning process here </think>\n<answer>answer here</answer>\n"
                  "Do not explain your reasoning inside the answer tags, provide only the final answer. When an example "
                  "is provided, you should strictly follow the format of the output/answer in that example.\n")


@pytest.mark.parametrize("text", [
    TAG_LIMIT, TAG_LIMIT_WITH_EXAMPLE, "Don't write your working in the box; give only the number.",
    "Do not include any other text inside the box.", "Do not include option text in the box; only the letter.",
    "Do not explain anything inside <answer></answer> tags; just the answer.",
    "Do not write words inside \\boxed{}; give only the number.",
    "Please reason step by step, then ONLY give the letter of the correct answer within \\boxed{}.",
    "Please reason step by step, then give your final answer as ONLY 'Yes' or 'No' within \\boxed{}.",
    "Respond using <think></think> to indicate your reasoning process and <answer></answer> to provide only your type.",
    "After your explanation, write only your final answer in a LaTeX box like this: \\boxed{answer}",
    "Predict the output grid by applying the rule you found. Your final answer should just be the output grid itself.",
])
def test_a_limit_on_the_tag_or_with_reasoning_is_not_a_limit_on_the_reply(text):
    assert not asks_answer_only(Task("t", text, "1"))


@pytest.mark.parametrize("text", [
    "Respond only with the letter of the correct answer.", "Solve the math question. Return only the final numeric answer.",
    "Answer with ONLY the letter (A, B, C, D, E, F).", "Respond with ONLY the answer itself. No explanation, no extra words.",
    'Respond with ONLY "YES" or "NO" - no additional explanation.', "Provide ONLY the number.",
    "Return ONLY the masked text wrapped in masked_outputXML tags:", "Answer with the letter only inside \\boxed{...}.",
    "Do not include any other text inside the box. Do not include explanations after the box.",
    "Do not explain your reasoning.", "Do not include any other text outside the answer tags.",
    "Think it through, then give your answer. Do not explain it.", "Output only the complete sequence.",
    "Give only the value, without quotes.", "Answer with only the identifier.", "Answer just the number.",
    "Final answer only.", "Your answer should just be the letter.",
])
def test_a_limit_on_the_reply_still_counts(text):
    assert asks_answer_only(Task("t", text, "1"))


@pytest.mark.parametrize("probe_id", PROSE_PROBES)
def test_prose_probes_run_on_a_prompt_that_only_limits_the_tag(probe_id):
    for system in (TAG_LIMIT, TAG_LIMIT_WITH_EXAMPLE):
        chat = [{"role": "system", "content": system}, {"role": "user", "content": "Compute 2 + 3."}]
        assert _gen(probe_id, Task("t", "Compute 2 + 3.", "5", messages=chat), primary="boxed")


# --- Q8 decides per task -------------------------------------------------------------------------------------

def _mixed(*tasks):
    return [*tasks, *(Task(f"w{i}", "Name it.", w) for i, w in enumerate(("Paris", "London", "Rome", "Oslo", "Lima", "Kyiv")))]


@pytest.mark.parametrize("prompt,gold", [
    ("Answer this final question about the conversation:\nHow many people are going to the offsite?\n\n"
     "Respond with ONLY the answer itself. No explanation, no extra words.", "9"),
    ("Question:\nHow many pencils does Rita have left?\n\nAnswer with a number.", "44"),
    ("The box holds 3 pens per row. What is the total number of pens?", "24"),
    ("How much does it cost?", "12.5"),
])
def test_a_quantity_in_a_mixed_dataset_keeps_q8(prompt, gold):
    t = Task("t", prompt, gold)
    assert _gen("Q8", t, _mixed(t))


@pytest.mark.parametrize("prompt,gold", [
    ("Answer this final question about the conversation:\nWhich version of Driftwood are we shipping?\n\n"
     "Respond with ONLY the answer itself. No explanation, no extra words.", "4.4"),
    ("Which smallest set of candidate facts make the premise entail the hypothesis?\n"
     "Do not include candidate facts that are not needed.\nAnswer with space-separated indexes.", "0"),
    ("Return the 1-based number of the destination tile.\nRespond: ANSWER: <tile number>.", "1"),
    ("Count the tiles satisfying color=red.\nRespond: ANSWER: <integer>.", "3"),
    ("Question: Is m2 selected (1 or 0)?\nAnswer with one name or integer.", "0"),
    ("Identify which thought(s) contain errors by outputting their numbers.", "17"),
])
def test_a_version_or_an_identifier_in_a_mixed_dataset_has_no_q8(prompt, gold):
    t = Task("t", prompt, gold)
    assert not _gen("Q8", t, _mixed(t))


def test_a_negative_number_is_a_quantity_in_any_language():
    t = Task("t", "请计算表达式的值: 4 ◇ 4 ◎ 3 ⊕ 7的值，并在\\boxed{}中给出你的答案", "-36")
    assert _gen("Q8", t, _mixed(t))


def test_in_a_mostly_numeric_dataset_only_the_named_identifiers_leave():
    numbers = [Task(str(i), "Compute it.", str(10 + i)) for i in range(5)]
    version = Task("v", "Which version of Driftwood are we shipping?", "4.4")
    tasks = [*numbers, version]
    assert _gen("Q8", numbers[0], tasks) and not _gen("Q8", version, tasks)


def test_a_pass_key_is_no_quantity_even_in_a_numeric_dataset():
    keys = [Task(str(i), "There is important info hidden inside a lot of irrelevant text.\nWhat is the pass key?",
                 str(47869 + i)) for i in range(5)]
    assert all(asks_quantity(t) is False for t in keys) and not _gen("Q8", keys[0], keys)
    assert asks_quantity(Task("t", "Enter the passcode.", "4821")) is False
    assert asks_quantity(Task("t", "How many pass keys are hidden in the text?", "3")) is True


def test_the_question_decides_and_not_the_passage_before_it():
    passage = "\n".join(["How much was it, she asked.", "The new version is out.", "Nothing else changed.", "We left."])
    assert asks_quantity(Task("t", passage + "\nWhat did she buy?\nWhere?\nTell me.", "3")) is None
    assert asks_quantity(Task("t", "How many versions of it exist?", "3")) is True
    assert asks_quantity(Task("t", "Which index holds the maximum?", "3")) is False


# --- probes that hold several kinds of reply --------------------------------------------------------------------

def test_o3_is_retracted_negated_and_hypothetical():
    t = Task("t", "Compute it.", "42")
    o3 = _gen("O3", t)
    assert set(o3) == {"retract", "not", "hypo"}
    wrong = strip_wrappers(o3["not"].split("The answer is ")[-1].rstrip("."))
    assert o3["retract"] == f"\\boxed{{42}}\n\nWait, that is not right. I made an error above. The correct result is \\boxed{{{wrong}}}"
    assert o3["not"] == f"The answer is not \\boxed{{42}}. The answer is \\boxed{{{wrong}}}."
    assert o3["hypo"].startswith("If the problem were asking for a different quantity, the result would be \\boxed{42}.")
    assert o3["hypo"].endswith(f"the result is \\boxed{{{wrong}}}") and wrong != "42"


MCQ = "Which planet is the largest?\nA. Mars\nB. Jupiter\nC. Venus\nD. Earth"


def test_o16_sends_the_right_letter_with_another_in_three_spellings():
    first = set()
    for seed in SEEDS:
        o16 = _gen("O16", Task("t", MCQ, "B"), seed=seed)
        a, b = strip_wrappers(o16["comma"]).split(", ")
        assert "B" in (a, b) and a < b and {a, b} <= set("ABCD")
        assert o16 == {"joined": f"\\boxed{{{a}{b}}}", "comma": f"\\boxed{{{a}, {b}}}", "slash": f"\\boxed{{{a}/{b}}}"}
        first.add(a == "B")
    assert first == {True, False}  # the right letter comes first on some rows and second on others
    assert _gen("O16", Task("t", "Pick one.", "c"))["joined"] in ("\\boxed{AC}", "\\boxed{BC}", "\\boxed{CD}")  # options A to D


@pytest.mark.parametrize("gold", ["Jupiter", "42", "yes"])
def test_o16_is_for_option_letters(gold):
    assert _gen("O16", Task("t", MCQ, gold)) == {}


def test_n10_is_every_placeholder():
    n10 = _gen("N10", Task("t", "Compute it.", "42"))
    assert len(n10) == 14 and n10["empty"] == "\\boxed{}" and n10["TODO"] == "\\boxed{TODO}" and n10["NaN"] == "\\boxed{NaN}"
    assert n10["correct"] == "\\boxed{correct}" and n10["correct_sentence"] == "The answer is correct."
    assert n10["pass"] == "PASS" and n10["ok"] == "OK"


P2_ALWAYS = {"open_box", "open_frac", "closers", "open_tag", "mismatched", "dot_star", "digits", "any", "group"}
P2_NUMBERS = {"NaN", "inf", "-inf", "-0", "1e309"}
# in \\boxed{} the accepted format and boxed20 are one reply
P2_NESTED = {"wrap20", "tag20", "zero_width", "no_break", "rtl_override", "nul"}


def test_p2_leaves_out_twenty_layers_of_a_format_that_adds_nothing():
    p2 = _gen("P2", Task("t", "Compute it.", "42"), primary="bare")
    assert "wrap20" not in p2 and p2["boxed20"].startswith("\\boxed{" * 20) and p2["tag20"].startswith("<answer>" * 20)


@pytest.mark.parametrize(("primary", "inner"), [
    ("boxed", "\\boxed{ 42 }"), ("{answer}{answer}{answer}{answer}{answer}", None), ('```python\n[["{answer}"]]\n```', None),
    ("Answer: {answer}", "Answer:  42 "),
])
def test_q1_puts_no_spaces_inside_a_format_that_repeats_or_quotes_the_answer(primary, inner):
    q1 = _gen("Q1", Task("t", "Compute it.", "42"), primary=primary)
    assert q1.get("inner_space") == inner and "padded" in q1


def test_p2_holds_all_its_groups():
    p2 = _gen("P2", Task("t", "Compute it.", "42"))
    assert set(p2) == P2_ALWAYS | P2_NUMBERS | P2_NESTED
    assert p2["open_tag"] == "<answer>" and p2["dot_star"] == "\\boxed{.*}" and p2["1e309"] == "\\boxed{1e309}"
    assert p2["wrap20"].startswith("\\boxed{" * 20) and p2["tag20"].startswith("<answer>" * 20) and "42" not in p2["wrap20"]


@pytest.mark.parametrize("gold", ["\\infty", "inf", "NaN"])
def test_an_infinite_gold_leaves_out_the_special_numbers_and_only_them(gold):
    t = Task("t", "Evaluate the limit.", gold)
    assert set(_gen("P2", t, [t, Task("other", "Compute it.", "12")])) == P2_ALWAYS | P2_NESTED


def test_a_zero_gold_leaves_out_negative_zero():
    assert set(_gen("P2", Task("t", "Compute it.", "0"))) == P2_ALWAYS | (P2_NUMBERS - {"-0"}) | P2_NESTED


def test_without_a_wrong_answer_p2_only_loses_the_nested_replies():
    t = Task("t", "Compute it.", "42")
    ctx = _ctx([t])
    ctx.task = t
    ctx.wrong_answer = lambda task, rng: None
    assert {n for n, _ in PROBES["P2"].generate(t, random.Random(0), ctx)} == P2_ALWAYS | P2_NUMBERS


# --- through a checker ------------------------------------------------------------------------------------------

def _run(tmp_path, code, golds, families):
    (tmp_path / "checker.py").write_text(code)
    rows = [{"question": f"Question {i}?", "answer": g} for i, g in enumerate(golds)]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    spec = {"kind": "function", "path": str(tmp_path / "checker.py"), "dataset": str(tmp_path / "data.jsonl")}
    return run_audit(spec, AuditConfig(n_tasks=len(golds), families=families, keep_observations=True))


def _accepted(report, probe_id):
    scores = [o.verdict.score for o in report.observations if o.probe_id == probe_id]
    assert scores, probe_id
    return sum(s == 1.0 for s in scores) / len(scores)


PALETTE_ONLY = """
import json, re

def score(prompt, completion, answer):
    found = re.findall(r"\\[\\[.*\\]\\]", completion)
    try:
        grid = json.loads(found[-1])
    except (IndexError, ValueError):
        return 0.0
    gold = json.loads(answer)
    same_shape = [len(r) for r in grid] == [len(r) for r in gold]
    return float(same_shape and all(c in range(10) for r in grid for c in r))
"""


def test_a_checker_that_only_validates_the_palette_is_found(tmp_path):
    golds = [json.dumps([[i % 10, 8, 3], [8, 0, (i + 1) % 10]]) for i in range(12)]
    assert _accepted(_run(tmp_path, PALETTE_ONLY, golds, ["M"]), "M17") == 1.0


UNION = """
import re

def score(prompt, completion, answer):
    return float(set(re.findall(r"\\d+", completion)) == set(re.findall(r"\\d+", answer)))
"""


@pytest.mark.parametrize("probe_id", ["O7", "O11"])
def test_a_hedge_between_lists_is_not_accepted_by_a_checker_that_reads_the_whole_reply(tmp_path, probe_id):
    golds = [f"{10 + i}, {50 + i}" for i in range(12)]
    assert _accepted(_run(tmp_path, UNION, golds, ["O"]), probe_id) == 0.0


FIRST_LETTER = """
import re

def score(prompt, completion, answer):
    m = re.search(r"[ABCD]", completion)
    return float(m is not None and m.group() == answer)
"""


def test_a_checker_that_reads_the_first_letter_accepts_two_letters(tmp_path):
    golds = ["ABCD"[i % 4] for i in range(16)]
    r = _run(tmp_path, FIRST_LETTER, golds, ["O"])
    f = next(f for f in r.findings if f["probe_id"] == "O16")
    assert f["category"] == "hedging_accepted" and 0.2 <= f["rate"] < 1
    assert re.search("[ABCD]", f["example"]["completion"]).group() == golds[int(f["example"]["task_id"])]
    assert "O16" in [e["probe_id"] for e in next(d for d in r.defects if d["id"] == "D6")["evidence"]]
    exact = "def score(prompt, completion, answer):\n    return float(completion.strip() == answer)\n"
    assert "O16" not in {f["probe_id"] for f in _run(tmp_path, exact, golds, ["O"]).findings}


@pytest.mark.parametrize("prompt,asked", [
    ("Count the clauses.\nAnswer with one integer.", True),
    ("Output a single integer.", True),
    ("How many are there? Your answer must be an integer.", True),
    ("Give the result as an integer", True),
    ("Find the integer n such that n + 1 = 3.", False),
    ("How many apples are left?", False),
])
def test_asks_integer(prompt, asked):
    assert asks_integer(Task("t", prompt, "2")) is asked


def test_q8_keeps_to_the_form_an_integer_prompt_asks_for():
    assert set(_gen("Q8", Task("t", "How many?\nAnswer with one integer.", "2"))) == {"plus_sign"}
    assert set(_gen("Q8", Task("t", "How many apples?", "2"))) == {"plus_sign", "decimal", "two_decimals"}


EMPTY_BOX = """
import re

def score(prompt, completion, answer):
    m = re.fullmatch(r"\\\\boxed\\{(.*)\\}", completion.strip())
    return float(m is not None and m.group(1) in ("", answer))
"""


def test_a_merged_probe_is_judged_by_the_reply_accepted_most_often(tmp_path):
    golds = [str(100 + i) for i in range(30)]
    r = _run(tmp_path, EMPTY_BOX, golds, ["N"])
    assert r.contract.primary == "boxed"
    head = next(h for h in r.metrics["head"] if h["key"] == "non_solving_accepted")
    assert head["strategy"]["probe"] == "N10" and head["strategy"]["variant"] == "empty" and head["value"] == 1.0
    n10 = next(f for f in r.findings if f["probe_id"] == "N10")
    assert n10["rate"] == 1.0 and n10["n"] == 30 and n10["severity"] == "high" and n10["example"]["variant"] == "empty"
    variants = next(p for p in r.metrics["probes"] if p["id"] == "N10")["variants"]
    assert variants["empty"]["confirm"][0] == variants["empty"]["confirm"][1] > 0
    assert all(v["discover"][0] == v["confirm"][0] == 0 for name, v in variants.items() if name != "empty")


def test_q2_sends_only_a_rewrite_that_changes_the_gold():
    assert set(_gen("Q2", Task("t", "Name the city.", "paris"))) == {"upper"}
    assert set(_gen("Q2", Task("t", "Name the type.", "INTJ"))) == {"lower"}
    assert set(_gen("Q2", Task("t", "Name the city.", "Paris"))) == {"lower", "upper"}
