import pytest

from rewardaudit.equiv import candidates, contains_equiv, equiv, flip_bool, to_number
from rewardaudit.labels import states_the_answer


def test_numeric_equivalence():
    assert equiv("42", "42.0")
    assert equiv("1/2", "0.5")
    assert equiv("\\frac{1}{2}", "0.5")
    assert equiv("1,000", "1000")
    assert equiv("50%", "0.5")
    assert equiv("-3", "−3")
    assert not equiv("42", "43")
    assert not equiv("42", "420")


def test_wrappers_and_letters():
    assert equiv("\\boxed{42}", "42")
    assert equiv("$\\boxed{C}$", "(C)")
    assert equiv("yes", "Yes.")
    assert not equiv("C", "D")
    assert to_number("$42 dollars") == 42


def test_contains_and_candidates():
    assert contains_equiv("so the answer is 42.", "42")
    assert contains_equiv("Answer: forty-two", "42")
    assert not contains_equiv("the answer is 420", "42")
    assert candidates("\\boxed{1} and \\boxed{2}") == ["1", "2"]
    assert candidates("The answer is either 3 or 4.") == ["3", "4"]


def test_derived_answer_fields():
    derived = ["tag:masked_output", "reasoning_tag:masked_output", "prefix:Final", "boxed"]
    assert candidates("<masked_output>42</masked_output>") == []
    assert candidates("<masked_output>42</masked_output>", derived) == ["42"]
    assert candidates("Some thoughts.\nFinal: 42.", derived) == ["42"]
    assert candidates("<think>41</think>\nfinally: 41", derived) == []
    assert not states_the_answer("<masked_output>42</masked_output>", "42")
    assert states_the_answer("<masked_output>42</masked_output>", "42", None, derived)
    assert states_the_answer("Final: 84/2", "42", None, derived)
    both = "<masked_output>41</masked_output><masked_output>42</masked_output>"
    assert not states_the_answer(both, "42", None, derived)
    assert not states_the_answer("<masked_output>41</masked_output>", "42", None, derived)
    assert contains_equiv("<masked_output>1/2</masked_output>", "0.5", None, derived)
    assert candidates("x\n<sol>42</sol>", ["<sol>{answer}</sol>"]) == ["42"]
    assert candidates("x\nResult is 42\ny", ["Result is {answer}"]) == ["42"]
    assert states_the_answer("<sol>84/2</sol>", "42", None, ["<sol>{answer}</sol>"])
    assert not states_the_answer("<sol>41</sol><sol>42</sol>", "42", None, ["<sol>{answer}</sol>"])


def test_labels():
    assert not states_the_answer("", "42")
    assert states_the_answer("\\boxed{42}", "42")
    assert not states_the_answer("\\boxed{41}\n\\boxed{42}", "42")
    assert not states_the_answer("either 42 or 43", "42")
    assert states_the_answer("<answer>yes</answer>", "yes")
    assert not states_the_answer("forty-two", "42")


@pytest.mark.parametrize("gold,other", [("yes", "no"), ("no", "yes"), ("NO", "YES"), ("Yes", "No"), ("true", "false"), ("False", "True"),
                                        ("Y", "N"), ("correct", "incorrect"), ("right", "wrong")])
def test_a_boolean_is_flipped_in_its_own_words(gold, other):
    assert flip_bool(gold) == other and flip_bool(other) == gold and not equiv(gold, other)
