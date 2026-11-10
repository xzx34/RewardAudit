"""G2: the other accepted answers of a row, which the checker receives as a list, each sent in the accepted format. One that
loses the score is a finding of D16, apart from the head metrics."""

import random

import pytest
from typer.testing import CliRunner

from rewardaudit import audit, cli
from rewardaudit.adapters.function_adapter import reference_aliases
from rewardaudit.contract import Contract
from rewardaudit.defects import BY_PROBE
from rewardaudit.goldtype import infer_gold_type
from rewardaudit.metrics import REJECTED_GOLD_NAME
from rewardaudit.pipeline import exit_code
from rewardaudit.probes.base import ProbeContext
from rewardaudit.probes.reference import MAX_ALIASES
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import render_markdown
from rewardaudit.types import Task

G2 = next(p for p in all_probes() if p.id == "G2")
ROWS = [{"question": f"Name the thing on card {i}.", "answers": {"text": [f"thing{i}", f"item{i}", f"object{i}"]}} for i in range(12)]
LISTED = {"answer_col": "answers.text.0", "reference_col": "answers.text"}


def _task(answer, info):
    return Task("t", "Name it.", answer, info=info)


def test_the_other_answers_are_the_list_the_checker_receives_without_the_answer_blanks_and_repeats():
    t = _task("Paris", {"answers": {"text": ["Paris", " Paris ", "paris", "", "  ", "City of Light", "paris"]}})
    assert reference_aliases(LISTED, t) == ["paris", "City of Light"]
    assert reference_aliases({"reference_col": "answers.text"}, _task("Paris", {"answers": {"text": ["Paris"]}})) == []


@pytest.mark.parametrize("spec, info", [
    ({"answer_col": "answers.text.0"}, {"answers": {"text": ["Paris", "City of Light"]}}),  # the checker gets "Paris" alone
    ({"reference_template": "{answers}"}, {"answers": ["Paris", "City of Light"]}),  # a template is one string
    ({"reference_col": "annotations"}, {"annotations": [{"answer": ["Paris", "City of Light"]}]}),  # no list of strings
    ({"reference_col": "groundtruth"}, {"groundtruth": ["2.5", 2.5]}),
    ({"reference_col": "solution"}, {"solution": "Paris, the City of Light"}),
    ({"reference_col": "missing"}, {"answers": ["Paris", "City of Light"]}),
    ({}, {"answers": ["Paris", "City of Light"]}),  # an environment's spec names no column
])
def test_no_other_answers_when_the_checker_receives_no_list_of_them(spec, info):
    assert reference_aliases(spec, _task("Paris", info)) == []


def _ctx(tasks, aliases, primary="boxed"):
    ids = [t.id for t in tasks]
    contract = Contract(full_score=1.0, accepted=[primary], primary=primary, wrapper_rates={primary: 1.0},
                        self_pass=dict.fromkeys(ids, True), gold_score=dict.fromkeys(ids, 1.0),
                        task_wrappers={i: [primary] for i in ids})
    return ProbeContext(contract=contract, gold_type="short_str", tasks=tasks, aliases=aliases,
                        gold_types={t.id: infer_gold_type(t.answer) for t in tasks})


def _replies(task, aliases, primary="boxed"):
    ctx = _ctx([task], {task.id: aliases} if aliases else {}, primary)
    ctx.task = task
    return dict(G2.generate(task, random.Random(0), ctx))


def test_g2_sends_each_other_answer_in_the_accepted_format():
    city = Task("city", "Which US city is called the Big Apple?", "New York")
    assert _replies(city, ["NYC", "New York City"]) == {"alias1": "\\boxed{NYC}", "alias2": "\\boxed{New York City}"}
    assert _replies(city, ["NYC"], "answer_tag") == {"alias1": "<answer>NYC</answer>"}
    many = [f"name {i}" for i in range(20)]
    assert list(_replies(city, many).values()) == [f"\\boxed{{name {i}}}" for i in range(MAX_ALIASES)]


def test_g2_sends_only_letters_where_the_prompt_asks_for_a_letter():
    asks = Task("q", "Only one of the following options is correct, tell me the answer using one single letter (A, B, C, or D).",
                "A")
    assert _replies(asks, ["Walking Georgie", "A"]) == {"alias1": "\\boxed{A}"}  # the option's text is not what it asks for
    assert _replies(asks, ["Walking Georgie"]) == {}
    free = Task("q", "Which option is correct?\nA. Walking Georgie\nB. Methodist", "A")
    assert _replies(free, ["Walking Georgie"]) == {"alias1": "\\boxed{Walking Georgie}"}


def test_g2_sends_nothing_without_other_answers_or_next_to_an_answer_made_of_parts():
    assert _replies(Task("city", "Which city?", "Paris"), []) == {}
    assert _replies(Task("primes", "List the prime factors of 30.", "2, 3, 5"), ["2", "3", "5"]) == {}
    assert _replies(Task("point", "Return the point.", '{"x": 3, "y": 4}'), ["3", "4"]) == {}


def test_g2_belongs_to_d16_and_does_not_count_in_the_rate():
    assert BY_PROBE["G2"] == BY_PROBE["G1"] == "D16"
    assert G2.label.value == "solving" and G2.expect == "full" and not G2.headline and not G2.gold_blind and not G2.by_variant


def _first_only(completion, answer):
    return float(completion.strip() == answer[0])


def _any(completion, answer):
    return float(completion.strip() in answer)


def test_a_checker_that_reads_only_the_first_answer_rejects_the_others():
    r = audit(_first_only, ROWS, n="all", probes=["N", "G"], workers=1, **LISTED)
    f = next(f for f in r.findings if f["probe_id"] == "G2")
    assert f["category"] == "right_rejected" and f["severity"] == "medium" and f["rate"] == 1.0 and f["n"] == 24
    assert f["note"] == "rejected: 'item0', 'item1', 'item2'" and f["example"]["completion"] == "item0"
    d16 = next(d for d in r.defects if d["id"] == "D16")
    assert d16["severity"] == "medium" and [e["probe_id"] for e in d16["evidence"]] == ["G2"] and exit_code(r) == 0
    head = next(h for h in r.metrics["head"] if h["key"] == "right_rejected")
    assert head["n"] == 0  # G2 counts in no rate
    md = render_markdown(r)
    assert "### MEDIUM: G2 other accepted answers of the row" in md and "Rejected: 'item0', 'item1', 'item2'" in md


def test_a_checker_that_takes_any_of_them_has_no_finding():
    r = audit(_any, ROWS, n="all", probes=["G"], workers=1, **LISTED)
    g2 = next(p for p in r.metrics["probes"] if p["id"] == "G2")
    assert g2["n_scored"] == 24 and g2["obs_hits"] == 0 and not r.findings and not r.defects


def test_below_a_tenth_of_the_answers_it_is_low():
    def score(completion, answer):
        return float(completion.strip() in answer and completion.strip() not in ("item0", "item1"))

    f = next(f for f in audit(score, ROWS, n="all", probes=["G"], workers=1, **LISTED).findings if f["probe_id"] == "G2")
    assert f["rate"] == pytest.approx(2 / 24) and f["severity"] == "low" and f["note"] == "rejected: 'item0', 'item1'"


def test_the_note_names_the_answer_without_the_format():
    def score(completion, answer):
        reply = completion.strip()
        return float(reply.startswith("<answer>") and reply.endswith("</answer>") and reply[8:-9] == answer[0])

    r = audit(score, ROWS, n="all", probes=["G"], workers=1, **LISTED)
    f = next(f for f in r.findings if f["probe_id"] == "G2")
    assert r.contract.primary == "answer_tag" and f["example"]["completion"] == "<answer>item0</answer>"
    assert f["note"] == "rejected: 'item0', 'item1', 'item2'"


def test_without_the_list_the_checker_gets_one_answer_and_g2_sends_nothing():
    r = audit(lambda completion, answer: float(completion.strip() == answer), ROWS, n="all", probes=["G"], workers=1,
              answer_col="answers.text.0")
    assert "G2" not in {p["id"] for p in r.metrics["probes"]}
    assert [f["probe_id"] for f in r.findings] == ["G3"]  # that answer with a period is another string to this checker


@pytest.mark.parametrize("primary, reply, answer", [
    ("bare", "Paris", "Paris"), ("boxed", "\\boxed{Paris}", "Paris"), ("answer_colon", "Answer: Paris", "Paris"),
    ("think_tag", "<think>\nLet me work through this step by step.\n\nFirst I identify what is asked, then I carry out the "
                  "computation carefully.\n</think>\n<answer>Paris</answer>", "Paris"),
    ("json", '{"answer": "Paris"}', "Paris"), ("boxed", "Paris", "Paris"),  # not written in the format: as it is
])
def test_the_answer_of_a_reply_in_the_accepted_format(primary, reply, answer):
    assert _ctx([Task("t", "q", "x")], {}, primary).contract.answer_of(reply, "t") == answer


def test_the_g_family_lists_g1_with_g2():
    res = CliRunner().invoke(cli.app, ["probes", "G"])
    assert res.exit_code == 0 and "G1" in res.output and REJECTED_GOLD_NAME in res.output and "G2" in res.output
