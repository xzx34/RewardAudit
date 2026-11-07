"""The five dataset checks: each fires on a planted issue and stays quiet on look-alikes."""

import pytest

from rewardaudit.checks.dataset import run_dataset_checks
from rewardaudit.types import Task


def checks(tasks):
    return {i["id"]: i for i in run_dataset_checks(tasks)}


def _plain(k: int) -> list[Task]:
    """Rows with nothing wrong: distinct prompts and distinct answers that the prompts do not hold."""
    return [Task(f"p{i}", f"Add the two numbers on card {'abcdefghijklmnopqrstuvwxyz'[i % 26]}{i // 26 * 'x'}.", str(1000 + 7 * i)) for i in range(k)]


def _mcq(i: int, gold: str, letters: str = "ABCD") -> Task:
    options = "\n".join(f"{ch}. option {ch.lower()}" for ch in letters)
    return Task(f"m{i}", f"Question {i}: pick one.\n{options}", gold)


def test_a_clean_dataset_has_no_issues():
    assert run_dataset_checks(_plain(20)) == [] and run_dataset_checks([]) == []


def test_every_issue_says_the_same_things():
    tasks = _plain(20)
    tasks[0].answer = ""
    tasks[1].prompt = tasks[2].prompt
    tasks[3].prompt = f"The total is {tasks[3].answer}. What is the total?"
    tasks[4].answer = tasks[5].answer = tasks[6].answer = tasks[7].answer = "7"
    found = run_dataset_checks(tasks)
    assert {i["id"] for i in found} == {"empty_answers", "duplicate_prompts", "answer_in_prompt", "skewed_answers"}
    for issue in found:
        assert {"id", "name", "severity", "count", "note"} <= set(issue) and issue["severity"] in ("high", "medium", "low")


# --- empty_answers ---------------------------------------------------------------------------------------------------

def test_empty_answers():
    tasks = _plain(20)
    tasks[3].answer, tasks[7].answer = "", "   \n"
    issue = checks(tasks)["empty_answers"]
    assert issue["count"] == 2 and issue["tasks"] == ["p3", "p7"] and issue["severity"] == "high"


def test_a_zero_is_an_answer():
    tasks = _plain(20)
    tasks[3].answer = "0"
    assert "empty_answers" not in checks(tasks)


def test_an_issue_names_at_most_ten_rows_and_counts_them_all():
    tasks = _plain(40)
    for t in tasks[:15]:
        t.answer = ""
    issue = checks(tasks)["empty_answers"]
    assert issue["count"] == 15 and issue["tasks"] == [f"p{i}" for i in range(10)]


# --- duplicate_prompts -----------------------------------------------------------------------------------------------

def test_duplicate_prompts_ignore_case_and_spacing():
    tasks = _plain(20)
    tasks.append(Task("again", "  " + tasks[3].prompt.upper().replace(" ", "  \n"), "1"))
    tasks.append(Task("once_more", tasks[3].prompt, "2"))
    issue = checks(tasks)["duplicate_prompts"]
    assert issue["count"] == 2 and issue["tasks"] == [["p3", "again"], ["p3", "once_more"]] and issue["severity"] == "medium"


def test_prompts_that_differ_in_a_number_are_not_duplicates():
    tasks = [Task(f"t{i}", f"Tom has {i} apples and Ann gives him 3 more. How many apples does Tom have?", str(i + 3)) for i in range(20)]
    assert "duplicate_prompts" not in checks(tasks)


# --- answer_in_prompt ------------------------------------------------------------------------------------------------

def test_a_number_in_the_prompt_is_low_until_most_rows_have_it():
    some = _plain(20)
    for t in some[:5]:
        t.prompt = f"The cards show {t.answer} and 3. What do they show together?"
    issue = checks(some)["answer_in_prompt"]
    assert issue["count"] == 5 and issue["severity"] == "low" and "figures" in issue["note"]
    most = _plain(20)
    for t in most[:15]:
        t.prompt = f"The cards show {t.answer} and 3. What do they show together?"
    assert checks(most)["answer_in_prompt"]["severity"] == "high"


def test_text_in_the_prompt_is_high_above_a_fifth_of_the_rows():
    cities = ["Paris", "Tokyo", "Lima", "Oslo", "Cairo", "Quito", "Rome", "Hanoi", "Dakar", "Doha"]
    few = [Task(f"t{i}", f"Name the capital of country {i}.", c) for i, c in enumerate(cities)]
    few[0].prompt = "Is Paris the capital of country 0? Name it."
    issue = checks(few)["answer_in_prompt"]
    assert issue["count"] == 1 and issue["severity"] == "medium" and "inside the prompt" in issue["note"]
    many = [Task(f"t{i}", f"Is {c} the capital of country {i}? Name it.", c) for i, c in enumerate(cities)]
    assert checks(many)["answer_in_prompt"]["severity"] == "high"


@pytest.mark.parametrize("prompt, gold", [
    ("The cards show 420 and 3.", "42"),  # a longer number
    ("The word parisian is on the card.", "Paris"),  # inside a longer word
])
def test_a_longer_token_does_not_hold_the_answer(prompt, gold):
    tasks = _plain(20)
    tasks[0].prompt, tasks[0].answer = prompt, gold
    assert "answer_in_prompt" not in checks(tasks)


def test_closed_answers_are_expected_in_the_prompt():
    letters = [_mcq(i, "ABCD"[i % 4]) for i in range(20)]
    assert "answer_in_prompt" not in checks(letters)
    yes_no = [Task(f"t{i}", f"Is {i} even? Answer yes or no.", "no" if i % 2 else "yes") for i in range(20)]
    assert "answer_in_prompt" not in checks(yes_no)


def test_an_empty_answer_is_not_in_every_prompt():
    tasks = _plain(20)
    tasks[0].answer = ""
    assert "answer_in_prompt" not in checks(tasks)


# --- skewed_answers --------------------------------------------------------------------------------------------------

def test_skewed_answers():
    tasks = _plain(20)
    for t in tasks[:4]:
        t.answer = "1000"
    issue = checks(tasks)["skewed_answers"]
    assert issue["count"] == 4 and issue["severity"] == "medium" and "'1000' would be right on 20%" in issue["note"]
    for t in tasks[:8]:
        t.answer = "1000"
    assert checks(tasks)["skewed_answers"]["severity"] == "high"


def test_skew_is_read_on_normalised_answers():
    tasks = _plain(20)
    for t, form in zip(tasks[:4], ("Yes", "yes.", " YES ", "yes"), strict=True):
        t.answer = form
    assert checks(tasks)["skewed_answers"]["count"] == 4


def test_below_a_fifth_and_one_answer_for_all_rows_are_not_skew():
    tasks = _plain(20)
    for t in tasks[:3]:
        t.answer = "1000"
    assert "skewed_answers" not in checks(tasks)
    same = [Task(f"t{i}", f"Card {i}: what is on it?", "7") for i in range(10)]
    assert "skewed_answers" not in checks(same)


# --- answer_not_an_option --------------------------------------------------------------------------------------------

def test_answer_not_an_option():
    tasks = [_mcq(i, "ABCD"[i % 4]) for i in range(12)]
    tasks[1].answer = "E"
    tasks[2].answer = "H"
    issue = checks(tasks)["answer_not_an_option"]
    assert issue["count"] == 2 and issue["tasks"] == ["m1", "m2"] and issue["severity"] == "high"


def test_an_option_list_of_any_length_counts_and_no_list_says_nothing():
    assert "answer_not_an_option" not in checks([_mcq(i, "E", "ABCDE") for i in range(4)] + [_mcq(9, "A", "ABCDE")])
    assert "answer_not_an_option" not in checks([Task(f"t{i}", f"Which one for {i}?", "ABCD"[i % 4]) for i in range(8)])


def test_options_are_not_read_on_a_dataset_of_other_answers():
    tasks = _plain(20)
    tasks[0].prompt = "A. one\nB. two\nC. three"
    assert "answer_not_an_option" not in checks(tasks)
