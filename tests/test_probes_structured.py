"""Structured golds: type detection, parsed-value equivalence, labels, and the order semantics read from prompts."""

import pytest

from rewardaudit.equiv import contains_equiv, equiv, parse_grid, parse_seq
from rewardaudit.goldtype import infer_gold_type, order_semantics
from rewardaudit.labels import states_the_answer
from rewardaudit.probes.structured import dumps_like, parse_list_gold


@pytest.mark.parametrize("gold,kind", [
    ('{"a": 1}', "json"), ('[{"a": 1}, {"a": 2}]', "json"), ("[1, 2, 3]", "list"), ('["x", "y"]', "list"),
    ("1, 2, 3", "list"), ("(1, 2)", "list"), ("A, C", "list"), ("[red, green]", "list"),
    ("[[1, 0], [0, 1]]", "grid"), ("1 0\n0 1", "grid"), ("#.#\n.#.", "grid"),
    ("[1, 2)", "short_str"), ("(-2, 1]", "short_str"), ("Paris, France", "short_str"), ("1,000", "int"),
    ("\\frac{1}{2}, 3", "latex"), ("line one\nline two", "long_text"),
    ("sql_injection", "short_str"), ("Epiglottitis", "short_str"), ("pizza", "short_str"), ("{[()]}", "short_str"),
    ("< { } >", "short_str"), ("{0, 1, 2}", "list"),
    ("x^2", "latex"), ("a_{1}", "latex"), ("x_1", "latex"), ("2pi", "latex"), ("3pi/4", "latex"), ("sqrt(2)", "latex"),
])
def test_gold_type_detection(gold, kind):
    assert infer_gold_type(gold) == kind


def test_json_equivalence_is_parsed_value_equality():
    g = '{"name": "Ada", "age": 36, "tags": [1, 2]}'
    assert equiv('{"tags": [1, 2], "age": 36.0, "name": "Ada"}', g, "json")  # key order, 36 vs 36.0
    assert equiv('{"name":"Ada","age":36.0000001,"tags":[1,2]}', g, "json")  # numeric tolerance
    assert equiv('```json\n{"name": "Ada", "age": 36, "tags": [1, 2]}\n```', g, "json")
    assert not equiv('{"name": "Ada", "age": 37, "tags": [1, 2]}', g, "json")
    assert not equiv('{"name": "Ada", "tags": [1, 2]}', g, "json")
    assert not equiv('{"name": "Ada", "age_x": 36, "tags": [1, 2]}', g, "json")
    assert not equiv('{"name": "Ada", "age": 36, "tags": [1, 2], "note": "x"}', g, "json")
    assert not equiv('{"name": "Ada", "age": 36, "tags": [2, 1]}', g, "json")
    assert not equiv('{"ok": true}', '{"ok": 1}', "json")
    assert equiv('{"ok": true}', '{"ok": "yes"}', "json")


def test_list_order_matters_unless_set():
    assert equiv("[1, 2, 3]", "1, 2, 3", "list") and equiv("1,2,3", "1, 2.0, 3", "list")
    assert not equiv("3, 2, 1", "1, 2, 3", "list")
    assert equiv("3, 2, 1", "1, 2, 3", "set") and equiv("[3, 1, 2]", "1, 2, 3", "set")
    assert not equiv("1, 2", "1, 2, 3", "set") and not equiv("1, 1, 2", "1, 2, 2", "set")
    assert parse_seq("(1, 2), (3, 4)") == ["(1, 2)", "(3, 4)"] and parse_seq("[1, 2)") is None


def test_grid_equivalence_is_cell_wise():
    g = "1 0 1\n0 1 0"
    assert equiv("1,0,1\n0,1,0", g, "grid") and equiv("[[1, 0, 1], [0, 1, 0]]", g, "grid")
    assert equiv("```\n1 0 1\n0 1 0\n```", g, "grid")
    assert not equiv("1 0 1\n1 1 0", g, "grid") and not equiv("1 0\n0 1", "1 0 1\n0 1 0", "grid")
    assert parse_grid("#.#\n.#.") == [["#", ".", "#"], [".", "#", "."]]


def test_contains_finds_structures_in_prose():
    assert contains_equiv('Here it is: {"b": 2, "a": 1}. Done.', '{"a": 1, "b": 2}', "json")
    assert not contains_equiv('\\boxed{{"a": 1, "b": 3}}', '{"a": 1, "b": 2}', "json")
    assert contains_equiv("Therefore, the grid is\n1 0\n0 1", "1 0\n0 1", "grid")
    assert not contains_equiv("1 1\n0 1", "1 0\n0 1", "grid")


def test_labels_leave_semantics_dependent_answers_uncertain():
    assert states_the_answer("\\boxed{1, 2, 3}", "1, 2, 3", "list")
    assert not states_the_answer("\\boxed{3, 2, 1}", "1, 2, 3", "list")
    assert not states_the_answer("\\boxed{1, 2, 4}", "1, 2, 3", "list")
    g = '{"a": 1, "b": 2}'
    assert not states_the_answer('{"a": 1, "b": 2, "c": 3}', g, "json")
    assert not states_the_answer('{"a": 1, "a": 5, "b": 2}', g, "json")
    assert not states_the_answer('{"a": 5, "b": 2}', g, "json")


@pytest.mark.parametrize("prompt,sem", [
    ("Enter all the solutions, separated by commas.", "set"),
    ("Give the roots in any order.", "set"),
    ("Find all possible values of k.", "set"),
    ("Find all values of a and b.", None),
    ("Find the ordered pair (a, b).", "ordered"),
    ("List them in increasing order.", "ordered"),
    ("In order to win, list the numbers.", None),
    ("Tom ordered 3 pizzas and 2 sodas.", None),
    ("List all solutions in increasing order.", None),
    ("A set of 5 books costs $20.", None),
])
def test_order_semantics(prompt, sem):
    assert order_semantics(prompt) == sem


def test_variants_keep_the_gold_spelling():
    seq = parse_list_gold("[1,2, 3]")
    assert seq and seq.tokens == ["1", "2", "3"] and seq.write(["3", "2", "1"]) == "[3, 2, 1]"
    assert dumps_like({"a": 1, "b": [2, 3]}, '{"x":1,"y":2}') == '{"a":1,"b":[2,3]}'
    assert dumps_like({"a": 1}, '{\n  "x": 1\n}') == '{\n  "a": 1\n}'
