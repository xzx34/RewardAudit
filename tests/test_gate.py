"""Whether a checker can be audited: what the adapter reports, what the reference answers are, what the calls with them
show. A reference answer that is no answer, or a checker the probes cannot measure, must stop the audit and not become
a finding."""

import json

import pytest

from rewardaudit.contract import Contract
from rewardaudit.errors import AuditError, NotAuditable, short_error
from rewardaudit.gate import (
    FAILED,
    MIN_PASSING,
    check_contract,
    check_loaded,
    classify_contract,
    classify_meta,
    label_gold_reason,
    low_gold_share_reason,
    no_reference_reason,
)
from rewardaudit.types import Kind, Task, Verdict

WRAPPERS = ["bare", "boxed", "final_answer"]


def _contract(scores: dict[str, float], wrong: float | None = None, **fields) -> Contract:
    """Every wrapper of a task's gold gets `scores[task]`; `wrong` (default the same) goes to the wrong answer."""
    obs = [(w, tid, f"{w} {tid}", Verdict(Kind.SCORE, score=s)) for w in WRAPPERS for tid, s in scores.items()]
    obs += [("wrong_answer", tid, "x", Verdict(Kind.SCORE, score=s if wrong is None else wrong)) for tid, s in scores.items()]
    full = max(scores.values())
    self_pass = {tid: full > 0 and s >= full for tid, s in scores.items()}
    return Contract(full_score=full, accepted=WRAPPERS if full > 0 else [], primary="bare", wrapper_rates={w: 1.0 for w in WRAPPERS},
                    self_pass=self_pass, gold_score=dict(scores), observations=obs, wrappers_run=list(WRAPPERS), **fields)


def _tasks(scores: dict[str, float]) -> list[Task]:
    return [Task(tid, f"Question {tid} ?", f"gold answer {tid}") for tid in scores]


def _gold_calls(n_tasks: int, passing: int, verdict) -> Contract:
    """One call per task in one format, with the verdict of `verdict(i)`; the first `passing` tasks pass."""
    obs = [("boxed", f"t{i}", "c", verdict(i)) for i in range(n_tasks)]
    return Contract(full_score=1.0, accepted=["boxed"], primary="boxed", wrapper_rates={},
                    self_pass={f"t{i}": i < passing for i in range(n_tasks)}, gold_score={}, observations=obs,
                    wrappers_run=["boxed"])


def _stops(contract: Contract, tasks: list[Task], meta: dict | None = None) -> NotAuditable:
    with pytest.raises(NotAuditable) as e:
        check_contract(contract, tasks, meta or {})
    return e.value


ZEROS = {f"t{i}": 0.0 for i in range(10)}


# --- what the adapter reports after loading ------------------------------------------------------------------------------

@pytest.mark.parametrize("meta, cls", [
    ({"framework": "verifiers 0.1.14 (v0 API)", "env_class": "ToolEnv", "single_turn": False}, "multi_turn_or_tools"),
    ({"framework": "verifiers 0.1.14 (v0 API)", "single_turn": True, "tools": 3}, "multi_turn_or_tools"),
    ({"framework": "verifiers 0.1.14 (v0 API)", "single_turn": None, "max_turns": 10}, "multi_turn_or_tools"),
    ({"framework": "verifiers 0.4.0", "needs_container": True}, "multi_turn_or_tools"),
    ({"framework": "verifiers 0.4.0", "toolsets": ["SearchToolset"]}, "multi_turn_or_tools"),
    ({"framework": "verifiers 0.4.0", "env_classes": ["TextArenaEnv"]}, "multi_turn_or_tools"),
    ({"framework": "verifiers 0.4.0", "judges": ["ReferenceJudge"]}, "llm_judge"),
    ({"framework": "verifiers 0.1.14 (v0 API)", "single_turn": True, "judges": ["JudgeRubric"]}, "llm_judge"),
])
def test_classify_meta(meta, cls):
    assert classify_meta(meta)[0] == cls


def test_classify_meta_auditable():
    assert classify_meta({"framework": "verifiers 0.4.0", "judges": [], "toolsets": [], "env_classes": []}) is None
    # a custom single-turn Environment subclass (IFEval-style) is not multi-turn
    assert classify_meta({"framework": "verifiers 0.1.14 (v0 API)", "single_turn": None, "max_turns": None}) is None
    assert classify_meta({"style": "bare", "function": "score", "path": "reward.py"}) is None  # a plain scoring function


def test_a_stock_rubric_of_metrics_only_can_pay_nothing():
    only_metrics = {"framework": "verifiers 0.1.14 (v0 API)", "rubric_classes": ["Rubric"], "reward_funcs": ["turns", "length"],
                    "reward_weights": [0.0, 0.0]}
    cls, reason, _ = classify_meta(only_metrics)
    assert cls == "gold_never_passes" and "turns, length" in reason
    assert classify_meta({**only_metrics, "rubric_classes": ["RubricGroup", "Rubric", "MultiTurnMonitorRubric"]})[0] == "gold_never_passes"
    assert classify_meta({**only_metrics, "reward_weights": [0.0, 1.0]}) is None
    # a rubric class of its own may compute its reward itself
    assert classify_meta({**only_metrics, "rubric_classes": ["RubricGroup", "MyRubric"]}) is None
    assert classify_meta({**only_metrics, "framework": "verifiers 0.4.0"}) is None


def test_check_loaded_stops_before_any_call_and_says_what_to_do():
    with pytest.raises(NotAuditable) as e:
        check_loaded({"single_turn": False, "env_class": "ToolEnv"}, [Task("a", "q", "1")])
    assert e.value.cls == "multi_turn_or_tools" and "ToolEnv" in e.value.reason and "one reply" in e.value.advice
    spec = json.dumps({"func_name": "validate_quotation"})
    with pytest.raises(NotAuditable) as e:
        check_loaded({}, [Task(str(i), "Follow the constraints.", spec) for i in range(6)])
    assert e.value.cls == "no_reference" and "--answer-col" in e.value.advice
    assert isinstance(e.value, AuditError) and e.value.reason in str(e.value) and e.value.advice in str(e.value)
    assert check_loaded({}, [Task("a", "q", "1")]) is None


def test_no_reference_reason():
    spec = json.dumps({"func_name": "validate_quotation", "N": None})
    assert "specifications" in no_reference_reason([Task(str(i), "Write a poem.", spec) for i in range(6)])
    assert "empty" in no_reference_reason([Task(str(i), "q", "" if i % 4 else "7") for i in range(8)])
    assert "placeholders" in no_reference_reason([Task(str(i), "q", "None.") for i in range(8)])
    assert "the row's id" in no_reference_reason([Task(str(i), "q", str(i)) for i in range(8)])
    assert no_reference_reason([Task(str(i), "q", str(i + 1)) for i in range(8)]) is None
    assert no_reference_reason([Task(str(i), "q", json.dumps({"x": i})) for i in range(8)]) is None
    assert "no rows" in no_reference_reason([])


# --- what the calls with the reference answer show ---------------------------------------------------------------------

def test_classify_contract():
    tasks = [Task(f"t{i}", "q", "1") for i in range(10)]
    ok = lambda i: Verdict(Kind.SCORE, score=1.0)  # noqa: E731
    assert classify_contract(_gold_calls(10, 10, ok), tasks) is None
    net = lambda i: Verdict(Kind.ERROR, error="NetworkBlocked", network=True)  # noqa: E731
    assert classify_contract(_gold_calls(10, 0, net), tasks)[0] == "llm_judge"
    slow = lambda i: Verdict(Kind.TIMEOUT)  # noqa: E731
    assert classify_contract(_gold_calls(10, 0, slow), tasks)[0] == "timeout"
    dead = lambda i: Verdict(Kind.CRASH, error="worker hit the memory limit")  # noqa: E731
    assert classify_contract(_gold_calls(10, 0, dead), tasks) == (
        "audit_failed", "10 of 10 calls with the reference answer crashed: worker hit the memory limit", FAILED)
    err = lambda i: Verdict(Kind.ERROR, error="Traceback\nKeyError: 'answer'")  # noqa: E731
    cls, reason, _ = classify_contract(_gold_calls(10, 0, err), tasks)
    assert cls == "gold_never_passes"
    assert reason == "The checker raised an error on 10 of 10 calls with the reference answer: KeyError: 'answer'"
    zero = lambda i: Verdict(Kind.SCORE, score=0.0)  # noqa: E731
    assert classify_contract(_gold_calls(10, 0, zero), tasks)[0] == "gold_never_passes"
    assert "Only 2 of 10" in classify_contract(_gold_calls(10, 2, ok), tasks)[1]
    same = _gold_calls(10, 2, ok)
    same.wrong_checked = same.wrong_same = 2
    assert classify_contract(same, tasks)[1].endswith("reference and wrong answers score the same on those")


def test_too_few_passing_golds_is_judged_against_the_size_of_the_dataset():
    ok = lambda i: Verdict(Kind.SCORE, score=1.0)  # noqa: E731
    tasks = [Task(f"t{i}", "q", "1") for i in range(3)]
    assert classify_contract(_gold_calls(3, 3, ok), tasks) is None  # a small dataset where all pass
    assert classify_contract(_gold_calls(3, 2, ok), tasks)[0] == "gold_never_passes"
    many = [Task(f"t{i}", "q", "1") for i in range(50)]
    assert classify_contract(_gold_calls(50, MIN_PASSING - 1, ok), many)[0] == "gold_never_passes"
    assert classify_contract(_gold_calls(50, MIN_PASSING, ok), many) is None


def test_a_contract_with_no_gold_call_scored_says_so():
    empty = Contract(full_score=0.0, accepted=[], primary="bare", wrapper_rates={}, self_pass={}, gold_score={})
    assert classify_contract(empty, [])[:2] == ("gold_never_passes", "No reference answer could be scored")


def test_the_wrong_answer_pass_does_not_count_as_network_use():
    calls = [("boxed", f"t{i}", "c", Verdict(Kind.SCORE, score=1.0)) for i in range(5)]
    calls += [("wrong_answer", f"t{i}", "x", Verdict(Kind.ERROR, error="NetworkBlocked", network=True)) for i in range(5)]
    contract = Contract(full_score=1.0, accepted=["boxed"], primary="boxed", wrapper_rates={}, self_pass={f"t{i}": True for i in range(5)},
                        gold_score={}, observations=calls, wrappers_run=["boxed"])
    assert classify_contract(contract, [Task(f"t{i}", "q", "1") for i in range(5)]) is None


# --- a framework failure is not a gold failure -----------------------------------------------------------------------------

TASK_ERROR = ("Traceback (most recent call last):\n  File \"/venv/lib/python3.12/site-packages/verifiers/v1/task.py\", line 279, in score\n"
              "    raise error_cls(msg)\n"
              "verifiers.v1.errors.TaskError: task LegalTask scoring: TypeError: An asyncio.Future, a coroutine or an "
              "awaitable is required\n")


def _errors(n: int, error: str, n_errors: int | None = None) -> Contract:
    obs = [("bare", f"t{i}", "c", Verdict(Kind.ERROR, error=error) if i < (n if n_errors is None else n_errors)
            else Verdict(Kind.SCORE, score=0.0)) for i in range(n)]
    return Contract(full_score=0.0, accepted=[], primary="bare", wrapper_rates={}, self_pass={f"t{i}": False for i in range(n)},
                    gold_score={}, observations=obs, wrappers_run=["bare"])


def _four() -> list[Task]:
    return [Task(f"t{i}", "q", "1") for i in range(4)]


def test_a_typed_v1_error_on_the_gold_calls_is_load_failed_with_its_cause():
    cls, reason, advice = classify_contract(_errors(4, TASK_ERROR), _four())
    assert cls == "load_failed" and advice == ""  # the versions OLD_VERIFIERS names predate tasksets
    assert reason == ("4 of 4 rollouts with the reference answer failed on this verifiers: TaskError: task LegalTask scoring: "
                      "TypeError: An asyncio.Future, a coroutine or an awaitable is required")
    other = TASK_ERROR.replace("verifiers.v1.errors.TaskError", "verifiers.v1.errors.SandboxError")
    assert classify_contract(_errors(4, other), _four())[0] == "load_failed"


def test_the_adapters_rollout_error_gives_its_cause():
    error = "Traceback (most recent call last):\nrewardaudit.adapters.verifiers_adapter.RolloutError: ValueError: x\n"
    _, reason, advice = classify_contract(_errors(4, error), _four())
    assert reason == "4 of 4 rollouts with the reference answer failed on this verifiers: ValueError: x"
    assert "--verifiers '<0.1.11'" in advice


def test_the_cause_keeps_its_place_in_an_environment_of_ones_own():
    """An environment outside site-packages gets the place of the error in front; the class name is still shortened."""
    error = ("Traceback (most recent call last):\n  File \"/home/me/my_env.py\", line 3, in load\n    raise ValueError('x')\n"
             "ValueError: x\n\nThe above exception was the direct cause of the following exception:\n\n"
             "rewardaudit.adapters.verifiers_adapter.RolloutError: ValueError: x\n")
    assert classify_contract(_errors(4, error), _four())[1].endswith("on this verifiers: my_env.py, line 3: ValueError: x")


@pytest.mark.parametrize("error", [
    "Traceback (most recent call last):\n  File \"r.py\", line 1\nKeyError: 'answer'\n",
    "Traceback (most recent call last):\nmy_env.errors.TaskError: the package's own class\n",  # not the framework's
    "Traceback (most recent call last):\n  File \"r.py\", line 1\n    raise TaskError('x')\nValueError: y\n",  # a line of source
])
def test_a_reward_that_raises_stays_a_gold_failure(error):
    cls, reason, _ = classify_contract(_errors(4, error), _four())
    assert cls == "gold_never_passes" and reason.startswith("The checker raised an error on 4 of 4")


def test_a_few_typed_errors_are_not_a_framework_failure():
    assert classify_contract(_errors(4, TASK_ERROR, n_errors=1), _four())[0] == "gold_never_passes"


def test_golds_that_fail_on_an_environment_suggest_another_verifiers():
    on_env = _stops(_contract(ZEROS), _tasks(ZEROS), {"framework": "verifiers 0.1.14 (v0 API)"})
    assert on_env.cls == "gold_never_passes" and "--verifiers '<0.1.11'" in on_env.advice and "--format" in on_env.advice
    on_function = _stops(_contract(ZEROS), _tasks(ZEROS), {"style": "bare"})
    assert on_function.cls == "gold_never_passes" and "--verifiers" not in on_function.advice


@pytest.mark.parametrize("framework", ["verifiers 0.4.1", "verifiers 0.1.15.dev170 (v0 API, taskset + harness)"])
def test_golds_that_fail_on_a_taskset_do_not_suggest_a_verifiers_that_predates_it(framework):
    stop = _stops(_contract(ZEROS), _tasks(ZEROS), {"framework": framework})
    assert stop.cls == "gold_never_passes" and "--format" in stop.advice and "--verifiers" not in stop.advice


def test_an_environment_is_not_told_to_name_a_column_or_wrap_its_checker():
    """An environment brings its own rows and rubric: --answer-col, --reference-col and a wrapper do nothing for it."""
    env = {"framework": "verifiers 0.1.14 (v0 API)"}
    with pytest.raises(NotAuditable) as e:
        check_loaded(env, [Task(str(i), "Return the json and nothing else.", "") for i in range(6)])
    assert e.value.cls == "no_reference" and "environment" in e.value.advice and "--answer-col" not in e.value.advice
    lists = {f"t{i}": 0.0 for i in range(6)}
    tasks = [Task(t, "q", json.dumps([f"a{t}", f"b{t}"])) for t in lists]
    stop = _stops(_contract(lists), tasks, env)
    assert stop.cls == "gold_never_passes" and "--answer-col" not in stop.advice and "wrap" not in stop.advice
    assert "--answer-col" in _stops(_contract(lists), tasks, {}).advice
    scores = {f"t{i}": 0.15 for i in range(10)}
    low_share = _stops(_contract(scores, wrong=0.0, gold_total=dict(scores)), _tasks(scores), {**env, "reward_weights": [0.3, 0.7]})
    assert low_share.cls == "no_reference" and "--answer-col" not in low_share.advice


# --- golds that label the row --------------------------------------------------------------------------------------------

BIX = json.dumps({"gold": "", "options": [], "gold_index": 0, "gold_letter": "A", "question_id": "bix-4", "explanation": ""})


@pytest.mark.parametrize("tasks,expect", [
    ([Task(n, f"Solve it.\n\nFunction signature: def {n}(nums: list[int]) -> int:\n", n) for n in ("two_sum", "is_valid", "max_subarray")],
     "identifiers the prompt prints"),
    ([Task(str(i), f"Do x.\nConstraint Type: {c}\nConstraint ID: {c}\nFollow it.", c)
      for i, c in enumerate(("sentence:keyword", "words:consonants", "format:sub-bullets", "count:numbers"))], "identifiers the prompt prints"),
    ([Task(str(i), "Fill in the template.", f"plate-{i:02d}", info={"question": "q", "info": {"spec_id": f"plate-{i:02d}", "tier": "L0"}})
      for i in range(4)], "repeat an id field"),
    ([Task(str(i), "What is shown?", f"item-{i}", info={"item_id": f"item-{i}"}) for i in range(4)], "repeat an id field"),
    ([Task(str(i), "q", BIX) for i in range(4)], "empty answer field"),
    ([Task(str(i), "q", json.dumps({"answer": None, "other": i})) for i in range(4)], "empty answer field"),
])
def test_label_golds(tasks, expect):
    assert expect in label_gold_reason(tasks)


@pytest.mark.parametrize("tasks", [
    # an MCQ letter, in a prompt that prints every letter
    [Task(str(i), "Pick one.\nA) x\nB) y\nC) z\nD) w", "ABCD"[i], info={"option_id": "ABCD"[i]}) for i in range(4)],
    # short answers the prompt prints
    [Task(str(i), f"{g} is named in this passage. Who is named?", g, info={"wiki_id": g})
     for i, g in enumerate(("Paris", "Marie Curie", "mitochondria", "carbon dioxide"))],
    [Task(str(i), f"The {g} matters here. What matters?", g) for i, g in enumerate(("T-cell", "COVID-19", "x-ray", "well-known"))],
    # numbers, also when the row has them as an id
    [Task(str(i), f"What is {i} plus 40? Note: {i + 40}", str(i + 40), info={"answer_id": str(i + 40)}) for i in range(4)],
    [Task(str(i), f"The time is 12:3{i}. What time is it?", f"12:3{i}") for i in range(4)],
    [Task(str(i), f"Ratio 1.{i}", f"1.{i}") for i in range(4)],
    # identifiers offered as a choice: the prompt prints other rows' golds as well
    [Task(str(i), "Label it: self_harm, hate_speech, no_label.", g) for i, g in enumerate(("self_harm", "hate_speech", "no_label", "self_harm"))],
    # an identifier the prompt does not print
    [Task(str(i), "Name the function that sums.", f"sum_{i}") for i in range(4)],
    # JSON with an answer in it, or with no field that holds one
    [Task(str(i), "q", json.dumps({"answer": g})) for i, g in enumerate((0, False, "x", [1]))],
    [Task(str(i), "q", json.dumps({"gold_ids": [i], "x": ""})) for i in range(4)],
    [Task(str(i), "q", json.dumps([{"answer": ""}])) for i in range(4)],
    # a prompt's identifier below half of the rows
    [Task("0", "def two_sum(x): ...", "two_sum"), Task("1", "q", "7"), Task("2", "q", "8"), Task("3", "q", "9"), Task("4", "q", "1")],
    # what no simple rule covers: prose about the reply, and scoring keys in another shape
    [Task(str(i), "Write unit tests for this function.", "tests for add: normal(1+2=3), zero(0+0=0)") for i in range(4)],
    [Task(str(i), "Return STRICT JSON.", json.dumps({"findings": [{"doc_id": "d"}], "questions": [], "question_count": 3})) for i in range(4)],
    [Task(str(i), "Return JSON with selected_event_ids.", json.dumps({"gold_ids": ["e1"]})) for i in range(4)],
])
def test_not_label_golds(tasks):
    assert label_gold_reason(tasks) is None


LABELS = {f"row{i}": 1.0 for i in range(10)}


def _label_tasks() -> list[Task]:
    return [Task(tid, f"Write it.\ndef fn_{tid}(x):", f"fn_{tid}") for tid in LABELS]


def test_a_label_gold_the_checker_cannot_tell_from_a_wrong_answer_is_no_reference():
    same = _contract(LABELS, wrong_checked=10, wrong_same=10, wrong_mean=1.0)
    stop = _stops(same, _label_tasks())
    assert stop.cls == "no_reference" and "10 of 10 reference answers are identifiers" in stop.reason


def test_a_label_gold_that_fails_is_no_reference():
    assert _stops(_contract(dict.fromkeys(LABELS, 0.0)), _label_tasks()).cls == "no_reference"


def test_an_identifier_gold_the_checker_separates_stays_in_the_audit():
    """A question whose right answer is a name the prompt prints: its checker tells the gold from a wrong answer."""
    separated = _contract(LABELS, wrong=0.0, wrong_checked=10, wrong_same=0)
    assert check_contract(separated, _label_tasks(), {}) is None


def test_other_classes_come_before_a_label_gold():
    net = _contract(LABELS, wrong_checked=10, wrong_same=10)
    net.observations = [(w, t, c, Verdict(Kind.ERROR, error="NetworkBlocked", network=True)) for w, t, c, _ in net.observations]
    assert _stops(net, _label_tasks()).cls == "llm_judge"


def test_no_reference_reason_says_nothing_of_labels_before_the_contract():
    """A label gold reaches the gold calls first, so a checker that does separate it is seen doing so."""
    assert no_reference_reason(_label_tasks()) is None
    assert no_reference_reason([Task(str(i), "q", BIX) for i in range(4)]) is None


def test_a_checker_that_does_not_separate_ordinary_golds_goes_on_to_be_reported():
    """no_discrimination is the report's to say, with what it measured; only a label gold stops the audit here."""
    scores = {f"t{i}": 0.5 for i in range(10)}
    assert check_contract(_contract(scores, wrong_checked=10, wrong_same=10), _tasks(scores), {}) is None


# --- golds that earn a small share of what the rubric pays -------------------------------------------------------------------

def test_passing_golds_that_earn_a_small_share_of_the_rubric_are_labels():
    scores = {f"t{i}": 0.15 for i in range(10)}
    contract = _contract(scores, wrong=0.0, gold_total=dict(scores))
    weights = {"reward_weights": [0.3, 0.1, 0.2, 0.25, 0.15, 0.0]}
    assert "15% of what the rubric pays" in low_gold_share_reason(contract, weights)
    assert low_gold_share_reason(contract, {"reward_weights": [0.15]}) is None  # the whole of a smaller rubric
    assert low_gold_share_reason(contract, {}) is None
    apart = _contract(scores, wrong=0.0, gold_total={k: 1.0 for k in scores})  # mutually exclusive terms: 1 of 3.3
    assert low_gold_share_reason(apart, {"reward_weights": [1.0, 1.0, 0.3, 1.0]}) is None


def test_such_a_checker_is_no_reference_and_gets_no_findings():
    scores = {f"t{i}": 0.15 for i in range(10)}
    contract = _contract(scores, wrong=0.0, gold_total=dict(scores))
    stop = _stops(contract, _tasks(scores), {"reward_weights": [0.3, 0.7]})
    assert stop.cls == "no_reference" and "of what the rubric pays" in stop.reason


def test_a_low_share_says_nothing_when_the_gold_does_not_separate():
    """There the report says that the gold does not separate; the share of a rubric adds nothing to it."""
    scores = {f"t{i}": 0.15 for i in range(10)}
    contract = _contract(scores, gold_total=dict(scores), wrong_checked=10, wrong_same=10)
    assert check_contract(contract, _tasks(scores), {"reward_weights": [0.3, 0.7]}) is None


# --- errors cut down to one line ---------------------------------------------------------------------------------------------

def test_short_error_of_an_installer():
    uv = ("installing a/b failed:\nResolved 108 packages in 8.03s\n  × Failed to build `pkg==0.6.10`\n"
          "  ├─▶ The build backend returned an error\n  ╰─▶ Call to `build_wheel` failed (exit status: 1)\n"
          "hint: Build failures usually indicate a problem with the package\n")
    assert short_error(uv) == "Failed to build `pkg==0.6.10`: Call to `build_wheel` failed (exit status: 1)"


def test_short_error_says_what_a_hub_http_error_says():
    gated = "huggingface_hub.errors.HfHubHTTPError: (Request ID: Root=1-6aec)\n\n403 Forbidden: this dataset is gated.\nUrl: x"
    assert short_error(gated) == "huggingface_hub.errors.HfHubHTTPError: 403 Forbidden: this dataset is gated."


def test_short_error_of_a_traceback_names_the_users_file_and_no_library():
    tb = ('Traceback (most recent call last):\n  File "/a/rewardaudit/worker.py", line 120, in main\n'
          '  File "/home/me/reward.py", line 3, in score\n    return x\n'
          '  File "/venv/lib/python3.12/site-packages/lib.py", line 9, in f\n    raise NameError("x")\nNameError: name \'x\' is not defined\n')
    assert short_error(tb) == "reward.py, line 3: NameError: name 'x' is not defined"
    assert short_error("") == "unknown error" and short_error("just one line") == "just one line"
