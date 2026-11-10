"""The probes on how answers are compared: attached characters, units, relation signs, tolerance, other spellings, fixed
answers."""

import random
import re

import pytest
from conftest import probe_context

from rewardaudit import audit
from rewardaudit.contract import Contract
from rewardaudit.equiv import equiv
from rewardaudit.metrics import Tally, _sweep_severity
from rewardaudit.probes.registry import all_probes
from rewardaudit.probes.rewrite import parse_date
from rewardaudit.types import Kind, Label, Observation, Task, Verdict

PROBES = {p.id: p for p in all_probes()}


def _gen(probe_id, answer, prompt="What is the answer?", others=()):
    task = Task("t", prompt, answer)
    tasks = [task, *(Task(f"o{i}", "p", a) for i, a in enumerate(others))]
    ctx = probe_context(tasks, "bare")
    ctx.task = task
    return dict(PROBES[probe_id].generate(task, random.Random(0), ctx))


@pytest.mark.parametrize(("answer", "expected"), [
    ("42", {"append_0": "420", "prepend_1": "142"}),
    ("-42", {"append_0": "-420", "prepend_1": "-142"}),
    ("3.5", {"prepend_1": "13.5"}),  # 3.50 is the same number
])
def test_m21_attaches_digits_and_keeps_the_sign(answer, expected):
    assert _gen("M21", answer) == expected


def test_m21_attaches_letters_and_a_word_from_another_row():
    variants = _gen("M21", "Rome", others=["London Bridge"])
    assert variants["suffix_x"] == "Romex" and variants["another_word"] in ("Rome London", "Rome Bridge")
    assert _gen("M21", "Rome.") == {"suffix_x": "Romex"}
    assert "another_word" not in _gen("M21", "Rome", others=["the owl"])  # short words may be articles


@pytest.mark.parametrize("answer", ["yes", "B", "\\frac{1}{2}"])
def test_m21_leaves_other_kinds_of_answer_alone(answer):
    assert _gen("M21", answer) == {}


@pytest.mark.parametrize(("answer", "prompt"), [
    ("$174", "How much tax does Alice have to pay in 2018?"),  # 174 with a stray letter is still 174 dollars
    ("((2,9),(64,625))", "Compute the operation."),
    ("盗窃", "被告人的罪名是？"),  # a Latin letter after a word in another script is a stray mark
    ("ACD", "Which are correct?\nA. one\nB. two\nC. three\nD. four"),  # ACDx still names A, C and D
    ("d) Stafne's bone cavity", "The most likely diagnosis is:\n\na) Radicular cyst\nd) Stafne's bone cavity"),
])
def test_m21_attaches_nothing_where_it_leaves_the_answer_the_same(answer, prompt):
    assert _gen("M21", answer, prompt, others=["London Bridge"]) == {}


def test_m21_still_attaches_to_words_and_letters_that_are_no_option_set():
    assert _gen("M21", "Rome", "What is the capital of Italy?") == {"suffix_x": "Romex"}
    assert _gen("M21", "XYZ", "Which are correct?\nA. one\nB. two") == {"suffix_x": "XYZx"}


def test_one_digit_changed_in_a_long_answer_is_another_answer():
    gold = "E_1 = \\\\frac{\\\\sqrt{3}}{8}Mc^2 + \\\\frac{\\\\sqrt{5}}{7}"
    assert equiv(gold.replace("{8}", "{9}"), gold, "latex") is False
    assert equiv("the treaty of 1982 between them", "the treaty of 1983 between them", "long_text") is False
    assert equiv("a long sentence about the river", "a long sentence about the rivers", "long_text") is True


def test_m6_leaves_the_number_of_a_line_alone():
    gold = "(1) Zoey is a knave\n(2) Oliver is a knight"
    assert _gen("M6", gold) == {}
    assert _gen("M6", "*1*: Ice machines are rated.\n*2*: A new machine makes less ice.") == {}
    assert re.fullmatch(r"(\*1\*: Sam has )\d( apples)", next(iter(_gen("M6", "*1*: Sam has 5 apples").values())))
    assert re.fullmatch(r"(1\. Sam has )\d( apples)", next(iter(_gen("M6", "1. Sam has 5 apples").values())))
    task = Task("t", "What is it?", "20.09369")  # a decimal point is not a line number: every digit can change
    ctx = probe_context([task], "bare")
    ctx.task = task
    positions = set()
    for seed in range(60):
        for _, reply in PROBES["M6"].generate(task, random.Random(seed), ctx):
            positions |= {i for i, (a, b) in enumerate(zip(reply, task.answer, strict=True)) if a != b}
    assert {0, 1} <= positions


def test_m22_writes_a_positive_number_with_a_unicode_minus():
    assert _gen("M22", "42") == {"unicode_minus": "−42"}
    assert _gen("M22", "-42") == _gen("M22", "0") == _gen("M22", "Paris") == {}


CHOICE_PROMPT = ("Where would Ricky look for the earphones given the story?\n\nPick one of the following choices:\n1 - piano\n"
        "2 - producer's desk\n3 - recording booth\n\nYou must pick one option.")


def test_a_minus_before_the_number_of_a_listed_choice_is_not_sent():
    assert _gen("M2", "2", CHOICE_PROMPT) == _gen("M22", "2", CHOICE_PROMPT) == {}
    assert _gen("M2", "5", CHOICE_PROMPT) == {"neg": "-5"} and _gen("M22", "5", CHOICE_PROMPT) == {"unicode_minus": "−5"}  # no such choice
    rules = "1. a#b = a + b\n2. a#b = a - b if a > b\n3. otherwise 7\nCompute 1#2."  # numbered rules are no choices
    assert _gen("M2", "2", rules) == {"neg": "-2"}
    assert _gen("M3", "2", CHOICE_PROMPT) == {"x10": "20", "div10": "0.2"}  # 20 is no choice either, and the sign is not involved


@pytest.mark.parametrize(("answer", "expected"), [
    ("5 km", "5 m"), ("5km", "5m"), ("2.5 kg", "2.5 g"), ("10 °C", "10 K"), ("3 mL", "3 L"), ("5\\text{ km}", "5\\text{ m}"),
    ("5\\,\\mathrm{s}", "5\\,\\mathrm{ms}"),
])
def test_m23_changes_the_unit_only(answer, expected):
    assert _gen("M23", answer) == {"other_unit": expected}


@pytest.mark.parametrize("answer", ["42", "5 apples", "km", "Paris"])
def test_m23_needs_a_number_and_a_unit(answer):
    assert _gen("M23", answer) == {}


def test_m24_sweeps_and_skips_what_prints_the_same():
    assert _gen("M24", "100") == {"0.1%": "100.1", "1%": "101", "2%": "102", "5%": "105", "10%": "110"}
    assert _gen("M24", "1000000") == {"0.1%": "1001000", "1%": "1010000", "2%": "1020000", "5%": "1050000", "10%": "1100000"}
    assert _gen("M24", "0") == _gen("M24", "Paris") == {}
    assert PROBES["M24"].sweep and not PROBES["M24"].headline


@pytest.mark.parametrize(("variant", "rate", "severity"), [
    ("0.1%", 1.0, "low"), ("1%", 1.0, "medium"), ("2%", 0.5, "medium"), ("5%", 1.0, "high"), ("10%", 0.5, "high"), ("10%", 0.4, "low"),
])
def test_how_wide_the_accepted_error_is_decides_its_severity(variant, rate, severity):
    assert _sweep_severity(Label.NOT_SOLVING, variant, rate) == severity


def test_a_tolerance_is_reported_as_the_widest_error_accepted_and_stays_out_of_the_rate():
    def within_two_percent(completion, answer):
        try:
            return float(abs(float(completion) - float(answer)) <= 0.025 * abs(float(answer)))
        except ValueError:
            return 0.0

    rows = [{"prompt": f"Add {i} and 10.", "answer": str(i + 10)} for i in range(12)]  # small enough that off by one is not within
    r = audit(within_two_percent, rows, workers=1)
    f = next(f for f in r.findings if f["probe_id"] == "M24")
    assert f["severity"] == "medium" and f["rate"] == 1.0 and f["example"]["variant"] == "2%"
    assert f["note"] == "accepted: 0.1% 100%, 1% 100%, 2% 100%, 5% 0%, 10% 0%"
    assert next(d for d in r.defects if d["id"] == "D3")["severity"] == "medium"
    probes = {p["id"]: p for p in r.metrics["probes"]}
    wrong = next(h for h in r.metrics["head"] if h["key"] == "wrong_accepted")
    assert wrong["n"] == sum(p["n_scored"] for p in probes.values() if p["family"] == "M" and p["headline"])


@pytest.mark.parametrize(("answer", "expected"), [
    ("x > 3", {"flipped": "x < 3", "adjacent": "x \\ge 3"}),
    ("x \\le \\frac{1}{2}", {"flipped": "x \\ge \\frac{1}{2}", "adjacent": "x < \\frac{1}{2}"}),
    ("1 \\leq n \\leq 10", {"flipped": "1 \\geq n \\geq 10", "adjacent": "1 < n < 10"}),  # every sign of the kind
    ("-2 < x \\le 3", {"flipped": "-2 > x \\le 3", "adjacent": "-2 \\le x \\le 3"}),  # the first kind only
    ("x < 1 or x > 1", {"flipped": "x > 1 or x > 1", "adjacent": "x \\le 1 or x > 1"}),  # both kinds at once: x ≠ 1 again
    ("x ≥ 2", {"flipped": "x ≤ 2", "adjacent": "x > 2"}),
    ("a <= b", {"flipped": "a >= b", "adjacent": "a < b"}),
    ("x<y", {"flipped": "x>y", "adjacent": "x\\le y"}),  # not \ley
    ("x>-1", {"flipped": "x<-1", "adjacent": "x\\ge-1"}),
    ("a \\neq 0", {"flipped": "a = 0"}),
    ("<b>x</b> > 2", {"flipped": "<b>x</b> < 2", "adjacent": "<b>x</b> \\ge 2"}),
])
def test_m26_turns_one_kind_of_relation_sign_around(answer, expected):
    assert _gen("M26", answer) == expected


@pytest.mark.parametrize("answer", ["f: A -> B", "p => q", "x <=> y", "a << 2", "\\left< x \\right>", "the <em>red</em> one",
                                    "x \\leqslant 2", "\\neg p", "\\leftarrow", "42", "Paris", "def f(x):\n    return x > 0"])
def test_m26_needs_a_relation_sign(answer):
    assert _gen("M26", answer) == {}


def test_a_checker_that_keeps_only_the_digits_accepts_a_relation_turned_around():
    def digits(completion, answer):
        return float(re.findall(r"\d+", completion) == re.findall(r"\d+", answer))

    rows = [{"prompt": f"Solve x - {i} > 0 for x.", "answer": f"x > {i}"} for i in range(2, 14)]
    r = audit(digits, rows, workers=1)
    f = next(f for f in r.findings if f["probe_id"] == "M26")
    assert f["severity"] == "high" and f["rate"] == 1.0 and f["category"] == "wrong_accepted"
    assert next(d for d in r.defects if d["id"] == "D1")["severity"] == "high"
    assert "M26" not in {f["probe_id"] for f in audit(lambda completion, answer: float(completion == answer), rows, workers=1).findings}


@pytest.mark.parametrize(("answer", "expected"), [
    ("0", {"5e-5": "0.00005", "9e-5": "0.00009"}),  # not 5e-05, which a checker reading digits takes for 5
    ("0.0", {"5e-5": "0.00005", "9e-5": "0.00009"}),
    ("0.000216", {"5e-5": "0.000266", "9e-5": "0.000306"}),
    ("-0.0003", {"5e-5": "-0.00025", "9e-5": "-0.00021"}),
    ("0.0008", {"9e-5": "0.00089"}),  # 0.00085 is less than a tenth off
    ("0.00005", {"5e-5": "0.0001", "9e-5": "0.00014"}),
    ("-0.00005", {"5e-5": "0", "9e-5": "0.00004"}),
    ("0.005", {}),  # 0.00505 is 1% off, which M24 sends
])
def test_m27_adds_less_than_1e_4_where_that_is_more_than_a_tenth_of_the_answer(answer, expected):
    assert _gen("M27", answer) == expected


def test_m27_is_for_numbers_under_a_hundredth():
    assert _gen("M27", "0.01") == _gen("M27", "-0.5") == _gen("M27", "42") == _gen("M27", "Paris") == {}


def test_an_absolute_tolerance_is_found_and_a_relative_one_is_not():
    # 0.0002 apart, so that another row's answer, which the tool sends as a wrong one first, is outside the tolerance
    rows = [{"prompt": f"What is {i} times 0.0002?", "answer": f"{i * 0.0002:.4f}"} for i in range(-4, 5)]

    def absolute(completion, answer):
        try:
            return float(abs(float(completion) - float(answer)) < 1e-4)
        except ValueError:
            return 0.0

    def relative(completion, answer):  # within 4%, as some checkers allow
        try:
            return float(abs(float(completion) - float(answer)) <= 0.04 * abs(float(answer)))
        except ValueError:
            return 0.0

    r = audit(absolute, rows, workers=1)
    f = next(f for f in r.findings if f["probe_id"] == "M27")
    assert f["severity"] == "high" and f["rate"] == 1.0 and f["n"] == 14  # 0.00005 only where the answer is under 0.0005
    assert "M27" in [e["probe_id"] for e in next(d for d in r.defects if d["id"] == "D3")["evidence"]]
    assert "M27" not in {f["probe_id"] for f in audit(relative, rows, workers=1).findings}


@pytest.mark.parametrize(("answer", "expected"), [
    ("42", {"words": "forty-two", "full_width": "４２"}),
    ("7", {"words": "seven", "full_width": "７"}),
    ("20", {"words": "twenty", "full_width": "２０"}),
    ("12345", {"scientific": "1.2345e4", "full_width": "１２３４５"}),
    ("10000", {"scientific": "1e4", "full_width": "１００００"}),
    ("-5", {"full_width": "-５", "unicode_minus": "−5"}),
    ("0.001234", {"scientific": "1.234e-3", "full_width": "０.００１２３４"}),
])
def test_q23_spells_a_number_other_ways(answer, expected):
    assert _gen("Q23", answer, "How many are there?") == expected


def test_m25_sends_a_hundred_times_and_a_hundredth():
    assert _gen("M25", "42") == {"x100": "4200", "div100": "0.42"}
    assert _gen("M25", "42", "What percentage of the class passed?") == {}  # 0.42 and 42 can both be right there
    assert _gen("M25", "0") == _gen("M25", "Paris") == {}


def test_a_checker_that_allows_for_percentages_is_found():
    def allows_percent(completion, answer):  # as some real checkers do, in every comparison
        try:
            return float(any(abs(float(completion) - float(answer) * k) < 1e-9 for k in (1, 100, 0.01)))
        except ValueError:
            return 0.0

    rows = [{"prompt": f"Add {i} and 10.", "answer": str(i + 10)} for i in range(12)]
    r = audit(allows_percent, rows, workers=1)
    assert next(f for f in r.findings if f["probe_id"] == "M25")["severity"] == "high"
    assert next(d for d in r.defects if d["id"] == "D3")["severity"] == "high"
    assert "D2" not in {d["id"] for d in r.defects}


def test_q23_stays_low_and_out_of_the_rate():
    def digits_only(completion, answer):
        return float(completion == answer)

    rows = [{"prompt": f"How many are {i} and 10?", "answer": str(i + 10)} for i in range(12)]
    r = audit(digits_only, rows, workers=1, probes=["Q"])
    assert next(f for f in r.findings if f["probe_id"] == "Q23")["severity"] == "low" and not PROBES["Q23"].headline


def test_q23_is_for_quantities():
    assert _gen("Q23", "42", "Which version of the library added this?") == {}
    assert _gen("Q23", "Paris") == {}


def test_q24_drops_the_unit_or_adds_the_one_the_question_names():
    assert _gen("Q24", "5 km") == {"without_unit": "5"}
    assert _gen("Q24", "12", "A pen costs 3 dollars. How many dollars do 4 pens cost?") == {"with_unit": "12 dollars"}
    assert _gen("Q24", "12", "How many dollars? Put the final answer in a box.") == {"with_unit": "12 dollars"}
    assert _gen("Q24", "12", "How many pens are there?") == _gen("Q24", "Paris") == {}
    assert _gen("Q24", "12", "How many dollars? Reply with only the number.") == {}


def test_q25_swaps_percent_and_decimal():
    assert _gen("Q25", "25%") == {"as_decimal": "0.25"}
    assert _gen("Q25", "12.5\\%") == {"as_decimal": "0.125"}
    assert _gen("Q25", "50", "What percentage of 200 is 100?") == {"percent_sign": "50%"}
    assert _gen("Q25", "12", "He ate 20 percent of the apples. How many apples are left?") == {}  # percent in passing
    assert _gen("Q25", "0.5", "What percentage of 200 is 1?") == {}  # 0.5 or 50%: it could mean either
    assert _gen("Q25", "Paris") == {}


@pytest.mark.parametrize(("answer", "names"), [
    ("2020-03-05", {"long": "March 5, 2020", "short": "Mar 5, 2020"}),
    ("March 5, 2020", {"iso": "2020-03-05", "short": "Mar 5, 2020"}),
    ("5 March 2020", {"iso": "2020-03-05", "long": "March 5, 2020", "short": "Mar 5, 2020"}),
    ("Mar. 5th, 2020", {"iso": "2020-03-05", "long": "March 5, 2020", "short": "Mar 5, 2020"}),
])
def test_q26_writes_a_date_other_ways(answer, names):
    assert _gen("Q26", answer) == names


@pytest.mark.parametrize("answer", ["3/5/2020", "2020-13-05", "March 2020", "Foo 5, 2020", "42"])
def test_q26_needs_a_date_that_says_which_number_is_the_month(answer):
    assert parse_date(answer) is None and _gen("Q26", answer) == {}


@pytest.mark.parametrize(("answer", "expected"), [
    ("Beyoncé", {"no_accents": "Beyonce"}),
    ("the chair", {"no_article": "chair"}),
    ("a green apple", {"no_article": "green apple"}),
    ("apples", {"article": "the apples"}),
    ("New York", {}),  # no one writes a name with a hyphen
    ("rain forest", {"article": "the rain forest", "hyphen": "rain-forest"}),
    ("well-known", {"article": "the well-known", "no_hyphen": "well known"}),
    ("Romeo & Juliet", {"and_word": "Romeo and Juliet"}),
    ("Salt and Pepper", {"ampersand": "Salt & Pepper"}),
])
def test_q27_changes_the_spelling_of_words(answer, expected):
    assert _gen("Q27", answer) == expected


@pytest.mark.parametrize(("answer", "prompt"), [
    ("BE", "Name the line."), ("42", "How many?"), ("New York", "Rewrite it with a hyphen."), ("Paris", "Copy it character by character."),
    ("calculate_triangle_area(base=10, height=5)", "Call the function for a triangle of base 10 and height 5."),
    ("THOSE FROM MARS", "Decrypt this Caesar cipher text: MAHLX YKHF FTKL. Provide only the decrypted text."),
])
def test_q27_leaves_symbols_and_prompts_on_characters_alone(answer, prompt):
    assert _gen("Q27", answer, prompt) == {}


@pytest.mark.parametrize("prompt", ["Decrypt this Caesar cipher text: MAHLX YKHF.", "Encrypt the plaintext with the rule below.",
                                    "Below are the rules for a specific cipher."])
def test_q2_leaves_the_case_of_a_cipher_answer_alone(prompt):
    assert _gen("Q2", "LXOVE", prompt) == {}
    assert _gen("Q2", "Paris", "What is the capital of France?") == {"lower": "paris", "upper": "PARIS"}


def test_q3_adds_no_period_to_a_function_call():
    assert _gen("Q3", "triangle_properties.get(side1=5, side2=4, side3=3)") == {
        "newline": "triangle_properties.get(side1=5, side2=4, side3=3)\n"}
    assert set(_gen("Q3", "sin(x)")) == {"period", "newline"}  # an expression, not a call with keyword arguments


def test_l5_sends_the_most_common_answer_on_every_row_with_a_common_answer():
    assert _gen("L5", "7", others=["3", "3", "5"]) == {"most_common": "3"}
    assert _gen("L5", "3", others=["3", "5"]) == {"most_common": "3"}  # the rate leaves this row out
    assert _gen("L5", "7", others=["3", "5"]) == {}  # no two rows share one


def test_l5_counts_only_rows_where_it_is_not_the_right_answer():
    probe = PROBES["L5"]
    tasks = [Task("a", "p", "3"), Task("b", "p", "3"), Task("c", "p", "7")]
    ids = [t.id for t in tasks]
    contract = Contract(full_score=1.0, accepted=["bare"], primary="bare", wrapper_rates={"bare": 1.0},
                        self_pass={i: True for i in ids}, gold_score={i: 1.0 for i in ids})
    tally = Tally([probe], tasks, contract, "int")
    verdict = Verdict(Kind.SCORE, score=1.0)
    tally.add([Observation("L5", "L", "most_common", t.id, "3", verdict, Label.NOT_SOLVING, "zero", "natural") for t in tasks])
    assert tally.stats["L5"].coincident == 2 and tally.stats["L5"].obs_hits == 1


def test_l6_sends_the_last_number_of_the_prompt_where_answers_are_numbers():
    assert _gen("L6", "9", "Add 4 and 5.") == {"last_number": "5"}
    assert _gen("L6", "9", "Add these.") == {}
    assert _gen("L6", "Paris", "Name the 1 city of 2.") == {}
