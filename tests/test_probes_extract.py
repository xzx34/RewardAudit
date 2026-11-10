"""The probes on which answer a checker reads: two formats, the thinking block, option letters, ranges, long text around it."""

import random
import re

import pytest

from rewardaudit import audit
from rewardaudit.contract import Contract
from rewardaudit.equiv import contains_equiv
from rewardaudit.goldtype import infer_task_type, option_texts
from rewardaudit.metrics import _sweep_severity
from rewardaudit.probes.base import _FILLER, ProbeContext
from rewardaudit.probes.registry import all_probes
from rewardaudit.probes.rewrite import LENGTHS
from rewardaudit.report import render_markdown
from rewardaudit.types import Label, Task

PROBES = {p.id: p for p in all_probes()}
MCQ = "What is 6 times 7?\nA. 41\nB. 42\nC. 43\nD. 44"


def _gen(probe_id, answer, prompt="What is the answer?", primary="boxed", accepted=("answer_tag",)):
    task = Task("t", prompt, answer)
    contract = Contract(full_score=1.0, accepted=[primary, *accepted], primary=primary, wrapper_rates={primary: 1.0},
                        self_pass={"t": True}, gold_score={"t": 1.0}, task_wrappers={"t": [primary]})
    ctx = ProbeContext(contract=contract, gold_type="int", tasks=[task], gold_types={"t": infer_task_type(task)})
    ctx.task = task
    return dict(PROBES[probe_id].generate(task, random.Random(0), ctx))


def test_o12_writes_a_right_and_a_wrong_answer_in_two_accepted_formats_both_ways():
    assert _gen("O12", "42") == {"right_in_accepted": "\\boxed{42}\n<answer>84</answer>", "right_in_other": "\\boxed{84}\n<answer>42</answer>"}
    assert _gen("O12", "42", primary="answer_tag", accepted=("boxed",))["right_in_other"] == "<answer>84</answer>\n\\boxed{42}"
    assert _gen("O12", "42", primary="hash", accepted=("final_answer",))["right_in_accepted"] == "#### 42\nFinal Answer: 84"


@pytest.mark.parametrize(("primary", "accepted"), [("boxed", ()), ("boxed", ("reasoning_boxed", "boxed_dollar")), ("final_answer", ("answer_colon",)),
                                                   ("bare", ("boxed",))])
def test_o12_needs_a_second_accepted_format_with_a_mark_of_its_own(primary, accepted):
    variants = _gen("O12", "42", primary=primary, accepted=accepted)
    assert bool(variants) == (primary == "bare")  # any mark is its own next to a bare answer


def test_a_tolerance_does_not_make_the_wrong_half_of_a_hedge_a_right_answer():
    import math

    def last_number_within_5_percent(completion, answer):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
        return float(bool(numbers) and math.isclose(float(numbers[-1]), float(answer), rel_tol=0.05))

    rows = [{"prompt": f"Add {i} and 100.", "answer": str(i + 100)} for i in range(12)]  # 101 is within 5% of 100
    found = {f["probe_id"] for f in audit(last_number_within_5_percent, rows, workers=1, probes=["O"]).findings}
    assert "O11" in found and not found & {"O3", "O5", "O6", "O7", "O10"}  # it reads the last answer, and nothing else


def test_o13_leaves_the_answer_only_in_the_thinking_block():
    assert _gen("O13", "42") == {"think_only": "<think>\nLet me work this out.\nSo the value is \\boxed{42}\n</think>"}
    assert _gen("O13", "42", primary="think_tag")["think_only"].startswith("<think>\n")
    reply = _gen("O13", "42", primary="open_think_tag")["think_only"]  # the prompt has opened the block
    assert reply.endswith("</think>") and "<think>" not in reply and "<answer>42</answer>" in reply


def test_o13_is_serious_only_where_the_format_has_a_thinking_block():
    rows = [{"prompt": f"Add {i} and 10.", "answer": str(i + 10)} for i in range(12)]

    def reads_any_box(completion, answer):
        return float(f"\\boxed{{{answer}}}" in completion)

    def needs_think_format(completion, answer):
        return float("</think>" in completion and f"\\boxed{{{answer}}}" in completion)

    assert next(f for f in audit(reads_any_box, rows, workers=1).findings if f["probe_id"] == "O13")["severity"] == "low"
    assert next(f for f in audit(needs_think_format, rows, workers=1).findings if f["probe_id"] == "O13")["severity"] == "high"


def test_o14_pairs_a_letter_with_another_options_text():
    variants = _gen("O14", "B", MCQ)
    assert re.fullmatch(r"\\boxed\{B\) 4[134]\}", variants["right_letter"])
    assert re.fullmatch(r"\\boxed\{[ACD]\) 42\}", variants["right_text"])


@pytest.mark.parametrize("ask", ["The last line of your response should be of the form 'Answer: $LETTER'.",
                                 "Answer with the option's letter from the given choices directly.",
                                 'Output the answer in the format of "The answer is (X)" at the end.'])
def test_o14_sends_only_the_wrong_letter_where_the_prompt_asks_for_the_letter(ask):
    variants = _gen("O14", "B", f"{MCQ}\n\n{ask}")
    assert set(variants) == {"right_text"} and re.fullmatch(r"\\boxed\{[ACD]\) 42\}", variants["right_text"])


@pytest.mark.parametrize("text", ["Let f(x) = 6x. What is f(7)?", "She receives a letter from her mother. What is 6 times 7?"])
def test_o14_sends_both_where_the_prompt_does_not_ask_for_the_letter(text):
    assert set(_gen("O14", "B", text + MCQ[len("What is 6 times 7?"):])) == {"right_letter", "right_text"}


@pytest.mark.parametrize(("answer", "prompt"), [("B", "Pick B or C."), ("42", MCQ), ("B", "A. same\nB. same")])
def test_o14_needs_options_with_different_texts(answer, prompt):
    assert _gen("O14", answer, prompt) == {}


def test_an_option_letter_f_is_a_letter_where_the_prompt_lists_it():
    ten = "\n".join(f"{c}. option {c}" for c in "ABCDEFGHIJ")
    assert infer_task_type(Task("t", f"Pick one.\n{ten}", "F")) == "mcq"
    assert infer_task_type(Task("t", "True or false: the sky is green.", "F")) == "bool"
    variants = _gen("O7", "F", f"Pick one.\n{ten}")
    assert not any("T" in re.sub(r"\\text\{ or \}|The answer is either|\\boxed", "", v) for v in variants.values())


def test_option_texts_takes_the_last_question_of_a_prompt_with_examples():
    assert option_texts("A. 1\nB. 2\nAnswer: B\n\nA. 3\nB. 4") == {"A": "3", "B": "4"}
    assert option_texts("No options here.") == option_texts("A. only one") == {}


def test_o15_sends_a_range_around_the_answer():
    assert _gen("O15", "42") == {"between": "\\boxed{between 41 and 43}", "plus_minus": "\\boxed{42 ± 1}", "inequality": "\\boxed{40 < x < 45}"}
    assert _gen("O15", "Paris") == _gen("O15", "\\frac{1}{2}") == {}


@pytest.mark.parametrize("probe_id", ["Q28", "Q29"])
def test_long_text_comes_in_four_lengths_around_the_answer_and_holds_no_answer(probe_id):
    variants = _gen(probe_id, "42")
    assert list(variants) == [name for name, _ in LENGTHS]
    for (name, size), reply in zip(LENGTHS, variants.values(), strict=True):
        assert size < len(reply) <= size + 40, name
        assert (reply.startswith("\\boxed{42}") if probe_id == "Q28" else reply.endswith("\\boxed{42}"))
        assert reply.count("42") == 1 and not re.search(r"\d", reply.replace("\\boxed{42}", ""))
    assert PROBES[probe_id].sweep and not PROBES[probe_id].headline


def test_the_filler_holds_no_answer_of_common_kinds():
    for sentence in _FILLER:
        for gold, kind in (("A", "mcq"), ("B", "mcq"), ("yes", "bool"), ("no", "bool"), ("1", "int"), ("0", "int")):
            assert not contains_equiv(sentence, gold, kind), (sentence, gold)


def test_long_text_is_not_sent_when_the_prompt_asks_for_only_the_answer():
    assert _gen("Q28", "42", "What is it? Reply with only the number.") == {}


def test_long_text_leaves_out_a_sentence_holding_the_reference_answer():
    assert all("step" not in reply.replace("\\boxed{step}", "") for reply in _gen("Q28", "step").values())


def test_q30_sends_the_option_text_unless_the_prompt_asks_for_a_letter():
    assert _gen("Q30", "B", MCQ) == {"text": "\\boxed{42}", "letter_and_text": "\\boxed{B) 42}"}
    assert _gen("Q30", "B", MCQ + "\nAnswer with the letter.") == {"letter_and_text": "\\boxed{B) 42}"}
    assert _gen("Q30", "B", "Pick B or C.") == {}


def test_q30_reads_a_template_that_asks_for_a_letter():
    assert "text" not in _gen("Q30", "B", MCQ + '\nEnd with "The answer is (X)".')


def test_q31_writes_latex_markup_inside_a_box():
    assert _gen("Q31", "Paris") == {"text": "\\boxed{\\text{Paris}}"}
    assert _gen("Q31", "45", "What is the angle ABC?")["degrees"] == "\\boxed{45^\\circ}"
    assert "degrees" not in _gen("Q31", "45", "How many apples are there?")
    assert _gen("Q31", "\\frac{1}{2}") == _gen("Q31", "(3,4]") == _gen("Q31", "42") == _gen("Q31", "p - q") == {}  # not what goes in \\text{}
    assert "text" not in _gen("Q31", "Paris", primary="answer_tag")
    assert _gen("Q31", "Saint-Tropez") == {"text": "\\boxed{\\text{Saint-Tropez}}"}


@pytest.mark.parametrize(("variant", "rate", "severity"), [("0.1k", 1.0, "medium"), ("1k", 1.0, "medium"), ("4k", 0.5, "medium"), ("16k", 1.0, "low"), ("64k", 1.0, "low"), ("1k", 0.2, "low")])
def test_a_right_answer_lost_at_an_ordinary_length_is_the_serious_case(variant, rate, severity):
    assert _sweep_severity(Label.SOLVING, variant, rate) == severity


def test_a_checker_that_reads_the_last_characters_is_found():
    def tail(completion, answer):  # reads only the last 300 characters, as some real checkers do
        return float(f"\\boxed{{{answer}}}" in completion[-300:])

    rows = [{"prompt": f"Add {i} and 10.", "answer": str(i + 10)} for i in range(12)]
    r = audit(tail, rows, workers=1)
    f = next(f for f in r.findings if f["probe_id"] == "Q28")
    assert f["severity"] == "medium" and f["example"]["variant"] == "1k" and f["rate"] == 1.0
    assert f["note"] == "accepted: 0.1k 100%, 1k 0%, 4k 0%, 16k 0%, 64k 0%"
    assert "Q29" not in {g["probe_id"] for g in r.findings}
    assert next(d for d in r.defects if d["id"] == "D8")["severity"] == "medium"


def test_a_checker_that_loses_the_answer_next_to_any_text_has_no_window():
    def whole_reply(completion, answer):
        return float(completion in (answer, f"\\boxed{{{answer}}}"))

    rows = [{"prompt": f"Add {i} and 10.", "answer": str(i + 10)} for i in range(12)]
    r = audit(whole_reply, rows, workers=1, probes=["Q"])
    f = next(f for f in r.findings if f["probe_id"] == "Q28")
    assert f["example"]["variant"] == "0.1k" and f["severity"] == "low"  # it takes a reply to be its answer field
    assert "D8" not in {d["id"] for d in r.defects} and "Q28" in {e["probe_id"] for d in r.defects if d["id"] == "D9" for e in d["evidence"]}


def test_scoring_time_measured_on_one_long_reply_is_not_rated():
    from rewardaudit.metrics import ProbeStats, _timing_finding

    s = ProbeStats(PROBES["Q29"])
    s.variant_ms["1k"], s.variant_ms["64k"] = [10.0], [900.0]
    f = _timing_finding(s)
    assert f["severity"] == "low" and f["rated"] == "medium" and f["note"].endswith("too few calls to rate")
    s.variant_ms["1k"], s.variant_ms["64k"] = [10.0, 12.0], [900.0, 950.0]
    assert _timing_finding(s)["severity"] == "medium"


def test_scoring_time_that_grows_with_the_reply_is_a_finding():
    def slow(completion, answer):
        import time

        time.sleep(len(completion) / 64_000 * 0.7)
        return float(completion.endswith(f"\\boxed{{{answer}}}") or completion.startswith(f"\\boxed{{{answer}}}"))

    rows = [{"prompt": f"Add {i} and 10.", "answer": str(i + 10)} for i in range(6)]
    r = audit(slow, rows, workers=4, probes=["Q"])
    f = next(f for f in r.findings if f["name"] == "scoring time grows with reply length")
    assert f["category"] == "robustness" and f["severity"] == "medium" and "at 64k" in f["note"]
    assert next(d for d in r.defects if d["id"] == "D13")["severity"] == "medium"
    assert "The checker takes much longer on a long reply." in render_markdown(r)
