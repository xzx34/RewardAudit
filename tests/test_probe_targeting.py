"""Which tasks the JSON, prose and numeral probes go to, and how they are written."""

from __future__ import annotations

import json
import random
import re

import pytest
from conftest import probe_context

from rewardaudit.contract import REASONING, WRAPPERS
from rewardaudit.equiv import contains_equiv
from rewardaudit.errors import AuditError, NotAuditable
from rewardaudit.pipeline import AuditConfig, run_audit
from rewardaudit.probes.hedge import SINGLE_BLOCK, single_block
from rewardaudit.probes.mutate import m7_insert_word
from rewardaudit.probes.registry import all_probes
from rewardaudit.probes.structured import _changed
from rewardaudit.prompt import asks_answer_first, asks_answer_last, asks_answer_only
from rewardaudit.types import Task

PROBES = {p.id: p for p in all_probes()}
PROSE_PROBES = ("Q5", "Q6", "Q21", "Q22", "O4")


def _gen(probe_id, task, tasks=None, primary="boxed"):
    ctx = probe_context(tasks or [task], primary)
    ctx.task = task
    return dict(PROBES[probe_id].generate(task, random.Random(0), ctx))


def _tasks(*golds):
    return [Task(str(i), "Compute it.", g) for i, g in enumerate(golds)]


# --- JSON probes ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("x", [0, 1, 36, -36, 0.5, -0.5, 99999, 1e6, -1e6])
def test_a_changed_number_is_beyond_any_tolerance(x):
    y = _changed(x)
    assert abs(y - x) >= 10 and abs(y - x) >= abs(x) and type(y) is type(x)


def test_changed_keeps_the_other_kinds():
    assert _changed(True) is False and _changed("a") == "ax" and _changed([1]) is None and _changed(None) is None


# --- O4 and formats with one answer ---------------------------------------------------------------------------

@pytest.mark.parametrize("wid", ["hash", "reasoning_hash", "answer_colon", "final_answer", "the_final_answer_is",
                                 "reasoning_final"])
def test_line_formats_have_one_answer_block(wid):
    t = Task("t", "Compute it.", "42")
    assert single_block(wid) and not _gen("O4", t, primary=wid)
    assert _gen("O3", t, primary=wid)


def test_every_single_block_id_is_a_real_wrapper():
    assert set(SINGLE_BLOCK) <= {w for w, _ in WRAPPERS}


@pytest.mark.parametrize("wid", ["boxed", "bare", "reasoning_boxed", "think_boxed", "prefix:Final"])
def test_other_formats_still_get_the_retraction(wid):
    assert not single_block(wid) and _gen("O4", Task("t", "Compute it.", "42"), primary=wid)


@pytest.mark.parametrize("primary, line", [("hash", "#### 42"), ("answer_colon", "Answer: 42"), ("final_answer", "Final Answer: 42"),
                                           ("prefix:Final", "Final: 42"), ("RESULT>> {answer}", "RESULT>> 42")])
def test_a_format_that_opens_a_line_stays_on_a_line_of_its_own(primary, line):
    t = Task("t", "Compute it.", "42")
    assert _gen("Q5", t, primary=primary)["before"].endswith(f"we arrive at the result.\n\n{line}")
    assert _gen("O5", t, primary=primary)["two_gold_first"].startswith(f"{line}\n")


@pytest.mark.parametrize("primary, end", [("boxed", "Therefore, \\boxed{42}"), ("answer_tag", "Therefore, <answer>42</answer>"),
                                          ("<sol>{answer}</sol>", "Therefore, <sol>42</sol>"), ("bare", "Therefore, 42")])
def test_other_formats_can_sit_inside_a_sentence(primary, end):
    assert _gen("Q5", Task("t", "Compute it.", "42"), primary=primary)["before"].endswith(end)


def test_a_format_the_user_gave_is_one_block_and_probes_write_in_it():
    t = Task("t", "Compute it.", "42")
    assert single_block("<sol>{answer}</sol>") and not _gen("O4", t, primary="<sol>{answer}</sol>")
    assert _gen("O3", t, primary="<sol>{answer}</sol>")["retract"].startswith("<sol>42</sol>")


def test_o3_still_shows_a_parser_that_keeps_the_first_answer():
    comp = _gen("O3", Task("t", "Compute it.", "42"), primary="hash")["retract"]
    assert re.search(r"#### (\S+)", comp).group(1) == "42" and re.findall(r"#### (\S+)", comp)[-1] != "42"


# --- prose and the thinking block -----------------------------------------------------------------------------

def test_prose_goes_after_the_thinking_block():
    t = Task("t", "Compute it.", "42")
    think = f"<think>\n{REASONING.strip()}\n</think>\n"
    assert _gen("Q21", t, primary="think_boxed") == {"sentence": think + "The final answer is \\boxed{42}",
                                                     "thus": think + "Thus, the answer is \\boxed{42}"}
    q5 = _gen("Q5", t, primary="think_tag")["before"]
    assert q5.startswith(think + "First, identify") and q5.endswith("Therefore, <answer>42</answer>")
    open_think = f"{REASONING.strip()}\n</think>\n"
    assert _gen("Q21", t, primary="open_think_boxed")["thus"] == open_think + "Thus, the answer is \\boxed{42}"
    assert _gen("Q5", t, primary="open_think_tag")["before"].startswith(open_think + "First, identify")


def test_prose_goes_after_the_reasoning_preamble():
    t = Task("t", "Compute it.", "42")
    assert _gen("Q5", t, primary="reasoning_hash")["before"].startswith(REASONING + "First, identify")
    assert _gen("Q21", t, primary="reasoning_hash")["thus"] == REASONING + "Thus, the answer is #### 42"


@pytest.mark.parametrize("wid", ["bare", "boxed", "answer_tag", "hash", "json", "code_block"])
def test_formats_without_a_preamble_start_with_the_prose(wid):
    t = Task("t", "Compute it.", "42")
    assert _gen("Q5", t, primary=wid)["before"].startswith("First, identify")
    assert _gen("Q21", t, primary=wid)["sentence"].startswith("The final answer is ")
    assert "42" in _gen("Q5", t, primary=wid)["before"]


# --- Q21 and the period ---------------------------------------------------------------------------------------

Q21_GOLDS = ["42", "-1.5", "B", "yes", "Paris", "New York", "\\frac{1}{2}", "2, 3, 5", '{"a": 1}', "1 0\n0 1", "line one\nline two"]
SENTENCE_FORMATS = ("the_final_answer_is", "reasoning_boxed")


@pytest.mark.parametrize("wid", [w for w, _ in WRAPPERS if w not in SENTENCE_FORMATS])
@pytest.mark.parametrize("gold", Q21_GOLDS)
def test_q21_adds_no_period_of_its_own(wid, gold):
    comps = _gen("Q21", Task("t", "Compute it.", gold), primary=wid)
    assert set(comps) == {"sentence", "thus"}
    for comp in comps.values():
        assert not comp.endswith("."), comp


@pytest.mark.parametrize("wid", SENTENCE_FORMATS)
def test_a_format_that_already_is_a_sentence_gets_no_q21(wid):
    assert not _gen("Q21", Task("t", "Compute it.", "42"), primary=wid)
    assert not _gen("Q21", Task("t", "Name it.", "Paris"), primary=wid)


def test_a_gold_keeps_its_own_period_in_a_sentence():
    t = Task("t", "Say it.", "The sky is blue.")
    assert _gen("Q21", t, primary="bare") == {"sentence": "The final answer is The sky is blue.",
                                              "thus": "Thus, the answer is The sky is blue."}
    assert _gen("Q21", t)["sentence"] == "The final answer is \\boxed{The sky is blue.}"


LAST_WORD = """
def score(prompt, completion, answer):
    words = completion.split()
    return float(bool(words) and words[-1] == answer)
"""


def test_a_checker_that_refuses_a_trailing_period_fails_q3_and_not_q21(tmp_path):
    spec = _spec(tmp_path, [str(10 + i) for i in range(12)])
    (tmp_path / "checker.py").write_text(LAST_WORD)
    found = {f["probe_id"]: f for f in run_audit(spec, AuditConfig(n_tasks=12, families=["Q"])).findings}
    assert found["Q3"]["rate"] == 0.5 and "Q21" not in found  # of Q3's variants only the period is refused


# --- prompts that ask for only the answer ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Return only the answer", "Give just the number", "no explanation", "RETURN ONLY THE ANSWER.",
    "Reply with a number only.", "Output only the final value.", "Answer with the letter and nothing else.",
    "Print the result, no other text.", "Write it without any explanation.", "Do not explain.", "Don't explain it.",
    "Do not include any other text.", "Solve it.\n\nGive only the final answer.", "Your only output is the number.",
    "(C) $12,000\n(D) $9,000\n\nAnswer with the option's letter from the given choices directly.", "Answer directly.",
])
def test_prompts_asking_for_only_the_answer(text):
    assert asks_answer_only(Task("t", text, "1"))


@pytest.mark.parametrize("text", [
    "only one repeated number", "The list has only one repeated number.", "just below this number",
    "The value lies just below this number.", "only\n\nselect", "Consider the first case only\n\nSelect the best option.",
    "Use the first row only.\nThe answer is a number.", "Compute it and explain your reasoning.", "",
    "what is the price of the item? Answer this question using the text in the image directly.",  # where to read it, not how
])
def test_prompts_that_do_not_ask_for_it(text):
    assert not asks_answer_only(Task("t", text, "1"))


def test_answer_only_reads_every_message_of_the_chat():
    chat = [{"role": "system", "content": "Reply with a number only."}, {"role": "user", "content": "2 + 2 = ?"}]
    assert asks_answer_only(Task("t", "2 + 2 = ?", "4", messages=chat))
    assert not asks_answer_only(Task("t", "Return only the answer.", "4", messages=chat[1:]))  # chat() is the prompt here
    assert not asks_answer_only(Task("t", "2 + 2 = ?", "4", messages=[{"role": "user", "content": None}]))


@pytest.mark.parametrize("probe_id", PROSE_PROBES)
def test_prose_probes_are_not_sent_when_the_prompt_asks_for_only_the_answer(probe_id):
    assert _gen(probe_id, Task("t", "Compute 2 + 3.", "5"))
    assert not _gen(probe_id, Task("t", "Compute 2 + 3. Return only the answer.", "5"))
    chat = [{"role": "system", "content": "No explanation."}, {"role": "user", "content": "Compute 2 + 3."}]
    assert not _gen(probe_id, Task("t", "Compute 2 + 3.", "5", messages=chat))


@pytest.mark.parametrize("probe_id", ["Q1", "Q3", "O3", "O5", "O7"])
def test_probes_about_the_answer_itself_still_run_on_such_a_prompt(probe_id):
    assert _gen(probe_id, Task("t", "Compute 2 + 3. Return only the answer.", "5"))


# --- prompts that say where the answer goes -------------------------------------------------------------------

AMOUNT_FIRST = "Question: How much tax does Alice have to pay in 2018?. State the amount first.\nAnswer:"
ANSWER_LAST = [
    'Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" '
    '(without quotes) where $ANSWER is the answer to the problem.\n\nCompute 2 + 3.',
    'Solve the following problem. The final line of your response MUST be of the following format:\n"ANSWER: $ANSWER"',
    'Compute 2 + 3.\n\nThink step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.',
    'Think step by step and then output the answer in the format of "The answer is (X)" at the end.\n\nQuestion: 2 + 3?',
    'Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: '
    '(your answer here, including the choice number)"',
    'Problem: Compute 2 + 3.\nYour response should end with "The final answer is [answer]" where [answer] is the response.',
    "Please provide your answer at the end using a Python markdown code block.",
]
BEFORE, AFTER = ("Q5", "Q21", "Q22", "Q29"), ("Q6", "Q28")


def test_a_prompt_that_asks_for_the_answer_first():
    task = Task("t", AMOUNT_FIRST, "5")
    assert asks_answer_first(task) and not asks_answer_only(task) and not asks_answer_last(task)
    assert all(not _gen(p, task) for p in (*BEFORE, "O4")) and all(_gen(p, task) for p in AFTER)
    assert not asks_answer_first(Task("t", "First write the first line, from left to right.", "5"))


@pytest.mark.parametrize("text", ANSWER_LAST)
def test_a_prompt_that_asks_for_the_answer_at_the_end(text):
    task = Task("t", text, "5")
    assert asks_answer_last(task) and not asks_answer_only(task)
    assert not _gen("Q6", task) and all(_gen(p, task) for p in (*BEFORE, "O4"))
    assert _gen("Q28", task)  # its 64k reply is the one call that times the checker on text after the answer


@pytest.mark.parametrize("text", [
    "At the end of the concert, she also performs a 15-minute encore. How long is it?",
    "Pad the plaintext with '#' at the end to make the total a multiple of 6.", "What is the answer at the end?",
])
def test_the_end_of_something_else_is_not_where_the_answer_goes(text):
    assert not asks_answer_last(Task("t", text, "5"))


R1V = ("The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The "
       "reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
       "<think> reasoning process here </think><answer> answer here </answer>")


def test_text_outside_the_two_blocks_a_prompt_writes_out_is_not_sent():
    chat = [{"role": "user", "content": "Compute 2 + 3. " + R1V}]
    task = Task("t", chat[0]["content"], "5", messages=chat)
    assert all(not _gen(p, task, primary="think_tag") for p in (*BEFORE[:2], "Q29", *AFTER))
    assert _gen("Q22", task, primary="think_tag")["think"].startswith("<think>\n")  # inside the thinking block
    assert all(_gen(p, task, primary="boxed") for p in (*BEFORE, *AFTER))  # the checker takes another format
    plain = Task("t", "Compute 2 + 3. Put the reasoning in <think> </think> and the answer in <answer> </answer> tags.", "5")
    assert all(_gen(p, plain, primary="think_tag") for p in (*BEFORE, *AFTER))


# --- Q8 and the dataset's kind --------------------------------------------------------------------------------

def test_a_numeric_dataset_is_a_strict_majority():
    assert probe_context(_tasks("4.4", "12", "7", "Paris", "3")).numeric_dataset()
    assert not probe_context(_tasks("1", "2", "Paris", "Rome")).numeric_dataset()
    assert not probe_context(_tasks("Paris", "London", "4.4", "Rome", "Oslo")).numeric_dataset()


def test_q8_leaves_a_version_number_in_a_word_dataset_alone():
    words = _tasks("Paris", "London", "4.4", "Rome", "Oslo")
    assert not _gen("Q8", words[2], words)
    numbers = _tasks("4.4", "12", "7", "Paris", "3")
    assert _gen("Q8", numbers[0], numbers)["trailing_zero"] == "\\boxed{4.40}"


def test_q8_puts_the_trailing_zero_on_the_number_not_after_its_percent_sign():
    numbers = _tasks("21.46%", "0.5%", "12", "7")
    assert _gen("Q8", numbers[0], numbers) == {"trailing_zero": "\\boxed{21.460%}"}
    assert _gen("Q8", numbers[1], numbers) == {"no_leading_zero": "\\boxed{.5%}", "trailing_zero": "\\boxed{0.50%}"}


@pytest.mark.parametrize("primary", ["boxed", "think_boxed", "bare"])
def test_q8_variants_still_contain_the_gold(primary):
    tasks = _tasks("32", "-1.5", "13", "1000", "0.25")
    ctx = probe_context(tasks, primary)
    for t in tasks:
        ctx.task = t
        comps = [c for _, c in PROBES["Q8"].generate(t, random.Random(0), ctx)]
        assert comps and all(contains_equiv(c, t.answer, ctx.gtype(t)) for c in comps), t.answer


# --- M7 -------------------------------------------------------------------------------------------------------

def test_the_inserted_word_is_never_last():
    t = Task("t", "Name the city.", "New York City")
    ctx = probe_context([t], "bare")
    ctx.task = t
    places = set()
    for seed in range(60):
        words = dict(m7_insert_word(t, random.Random(seed), ctx))["insert_not"].split()
        assert words[-1] != "not" and len(words) == 4
        places.add(words.index("not"))
    assert places == {0, 1, 2}


# --- empty golds ----------------------------------------------------------------------------------------------

CHECKER = """
def score(prompt, completion, answer):
    return float(bool(answer) and completion.strip() == answer)
"""


def _spec(tmp_path, golds):
    (tmp_path / "checker.py").write_text(CHECKER)
    rows = [{"question": f"Question {i}?", "answer": g} for i, g in enumerate(golds)]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return {"kind": "function", "path": str(tmp_path / "checker.py"), "dataset": str(tmp_path / "data.jsonl")}


def test_an_empty_gold_is_left_out_and_not_a_rejected_answer(tmp_path):
    golds = [f"{i} plums" for i in range(23)] + [""] * 6 + ["   "]
    r = run_audit(_spec(tmp_path, golds), AuditConfig(n_tasks=30, families=["N", "M"], keep_observations=True))
    empty = {str(i) for i in range(23, 30)}
    assert len(r.tasks) == r.n_rows == 23 and not empty & {t.id for t in r.tasks}
    assert len(r.contract.self_pass) == 23 and r.contract.self_pass_rate == 1.0 and r.contract.rejected_golds() == []
    assert not [f for f in r.metrics["findings"] if f["probe_id"] == "G1"]
    assert next(i for i in r.dataset_issues if i["id"] == "empty_rows_left_out")["count"] == 7
    assert not empty & {o.task_id for o in r.observations}
    assert r.metrics["n_tasks"] == 23 and r.metrics["floor"]["value"] == pytest.approx(1 / 23)  # '' is no constant answer


def test_empty_golds_do_not_change_the_kind_of_the_dataset(tmp_path):
    golds = [str(10 + i) for i in range(5)] + ["Paris", "Rome", "Oslo"] + [""] * 4
    r = run_audit(_spec(tmp_path, golds), AuditConfig(n_tasks=12, families=["Q"], keep_observations=True))
    assert "Q8" in {o.probe_id for o in r.observations}  # five numbers are most of the eight answerable golds


def test_a_gold_that_is_refused_still_counts_next_to_empty_ones(tmp_path):
    code = """
def score(prompt, completion, answer):
    return float(bool(answer) and answer != "7 plums" and completion.strip() == answer)
"""
    spec = _spec(tmp_path, [f"{i} plums" for i in range(20)] + [""] * 5)
    (tmp_path / "checker.py").write_text(code)
    r = run_audit(spec, AuditConfig(n_tasks=25, families=["N"]))
    g1 = next(f for f in r.metrics["findings"] if f["probe_id"] == "G1")
    assert g1["n"] == 20 and g1["rate"] == pytest.approx(1 / 20)


def test_a_dataset_of_only_empty_golds_cannot_be_audited(tmp_path):
    with pytest.raises(AuditError, match="All 5 rows of .* have an empty reference answer"):
        run_audit(_spec(tmp_path, [""] * 5), AuditConfig(n_tasks=5, families=["N"]))
    with pytest.raises(NotAuditable, match="5 of 5 reference answers are empty or placeholders"):
        run_audit(_spec(tmp_path, ["n/a"] * 5), AuditConfig(n_tasks=5, families=["N"]))
