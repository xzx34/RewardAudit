"""The verifiers adapters, on fake `verifiers` modules: the taskset form of the 0.1.15 pre-releases (`load_environment(config:
vf.EnvConfig)` returns `vf.Env(taskset, harness)`, whose rewards are scored by the harness runtime), the plain v0 rollout,
the v1 taskset's rows, and how an environment gets the verifiers it was written for. No Hub package runs here."""

import abc
import copy
import inspect
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from rewardaudit import envs
from rewardaudit.adapters.verifiers_adapter import (
    REPLY_MODEL,
    RolloutError,
    VerifiersV0Adapter,
    VerifiersV1Adapter,
    _content_text,
    _taskset_runtime,
    _v0_load,
    gold_text,
)
from rewardaudit.api import _environment_spec
from rewardaudit.errors import AuditError, NotAuditable
from rewardaudit.gate import classify_contract, classify_meta
from rewardaudit.pipeline import AuditConfig, run_audit
from rewardaudit.report import direct_scoring_line, render_markdown, render_terminal
from rewardaudit.runner import Runner
from rewardaudit.types import Kind, Task, Verdict


class FakeEnvConfig:
    def __init__(self, taskset=None, harness=None):
        self.taskset, self.harness = taskset or {}, harness or {}


class FakeState(dict):
    @classmethod
    def for_task(cls, task):
        return cls(prompt=task["prompt"])


class FakeRuntime:
    """Rewards as the taskset's @vf.reward signals; num_turns is a zero-weight metric."""

    rollout_signals = [{"name": "exact", "kind": "reward", "weight": 1.0, "stage": "rollout"},
                       {"name": "num_turns", "kind": "metric", "weight": 0.0, "stage": "rollout"}]
    group_signals: list = []

    def __init__(self):
        self.seen = []

    async def score_rollout(self, task, state):
        self.seen.append((task, dict(state)))
        reply = state["completion"][-1]["content"]
        exact = float(reply.strip() == task["answer"])
        state["metrics"] = {"exact": exact, "num_turns": 0.0}
        state["reward"] = exact * 1.0
        return state


ROWS = [{"prompt": [{"role": "user", "content": f"What is {i} + {i}?"}], "answer": str(2 * i),
         "info": {"answer": str(2 * i)}, "example_id": i} for i in range(4)]


def _fake_vf(toolsets=()):
    runtime = FakeRuntime()
    taskset = SimpleNamespace(config=SimpleNamespace(toolsets=list(toolsets), user=None), to_task=lambda row: dict(row))
    program = SimpleNamespace(fn=None, command=None, sandbox=None)
    harness = SimpleNamespace(runtime=runtime, config=SimpleNamespace(toolsets=[], program=program, sandbox=None,
                                                                       user=None, max_turns=-1))

    class Env:  # an Environment with an empty rubric; the taskset carries the rewards
        rubric = SimpleNamespace(funcs=[])

        def __init__(self):
            self.taskset, self.harness = taskset, harness

        def get_dataset(self):
            return list(ROWS)

    def load_environment(config: FakeEnvConfig):  # required, as in aime2024 0.2.0
        assert isinstance(config, FakeEnvConfig)
        return Env()

    vf = types.ModuleType("verifiers")
    vf.__version__ = "0.1.15.dev189"
    vf.EnvConfig, vf.State = FakeEnvConfig, FakeState
    vf.load_environment = lambda env_id, **kw: None
    mod = types.ModuleType("fake_aime")
    mod.load_environment = load_environment
    return vf, mod, runtime


@pytest.fixture
def adapter(monkeypatch):
    def make(**kw):
        vf, mod, runtime = _fake_vf(**kw)
        monkeypatch.setitem(sys.modules, "verifiers", vf)
        monkeypatch.setitem(sys.modules, "fake_aime", mod)
        return VerifiersV0Adapter({"kind": "verifiers", "env": "o/fake_aime"}), runtime
    return make


def test_taskset_form_scores_through_the_harness_runtime(adapter):
    ad, runtime = adapter()
    assert ad.meta["reward_funcs"] == ["exact", "num_turns"] and ad.meta["reward_weights"] == [1.0, 0.0]
    assert ad.meta["single_turn"] is True and "taskset + harness" in ad.meta["framework"]
    assert classify_meta(ad.meta) is None  # single-turn, no tools, no judge: goes on to the gold calls
    tasks = ad.tasks(10, 0)
    assert [t.answer for t in tasks] == ["0", "2", "4", "6"] and tasks[1].prompt == "What is 1 + 1?"
    gold = ad.score_raw(tasks[1], "2")
    assert gold.score == 1.0 and gold.sub_scores == {"exact": 1.0, "metric:num_turns": 0.0} and gold.parsed is None
    assert ad.score_raw(tasks[1], "3").score == 0.0
    task, state = runtime.seen[0]
    assert task["answer"] == "2" and state["completion"] == [{"role": "assistant", "content": "2"}]
    assert state["prompt"] == ROWS[1]["prompt"]  # the state the runtime gets is built from the task


def test_taskset_form_with_toolsets_is_multi_turn(adapter):
    ad, _ = adapter(toolsets=[SimpleNamespace(name="search")])
    assert ad.meta["toolsets"] == ["search"] and ad.meta["single_turn"] is False
    assert classify_meta(ad.meta)[0] == "multi_turn_or_tools"


def test_v0_load_passes_env_config_only_when_required():
    vf = SimpleNamespace(EnvConfig=FakeEnvConfig)
    seen = {}

    def required(config: FakeEnvConfig):
        seen["required"] = config
        return "env"

    def plain(num_examples: int = -1, config: dict | None = None):
        seen["plain"] = (num_examples, config)
        return "env"

    assert _v0_load(vf, required) == "env" and isinstance(seen["required"], FakeEnvConfig)
    _v0_load(vf, plain)
    assert seen["plain"] == (-1, None)  # an optional dict `config` of a plain v0 loader is left alone
    seen.clear()
    _v0_load(SimpleNamespace(), plain)  # verifiers without EnvConfig: the loader's own defaults
    assert seen["plain"] == (-1, None)
    assert _taskset_runtime(SimpleNamespace(rubric=None)) is None


def test_xml_answer_tag():
    from rewardaudit.adapters.verifiers_adapter import _xml_answer_tag

    xml = SimpleNamespace(answer_field="reversed_text", _fields=[("reversed_text", ["reversed_text"])])
    plain = SimpleNamespace(answer_field=None)
    assert _xml_answer_tag([None, plain, xml]) == "reversed_text"
    assert _xml_answer_tag([plain, SimpleNamespace(answer_field="answer")]) is None  # not an XMLParser


# --- the gold reaches a plain v0 rubric in the form the dataset holds it ------------------------------------------

STRUCTURED = [{"prompt": [{"role": "user", "content": "Review statement A."}], "answer": {"defects": []},
               "info": '{"company": "A"}'},
              {"prompt": [{"role": "user", "content": "Review statement B."}], "answer": {"defects": ["cash_mismatch"]},
               "info": {"company": "B"}},
              {"prompt": [{"role": "user", "content": "How many?"}], "answer": 7}]


class StructuredRubric:
    """Reads `answer` as verifiers hands it over; verifiers turns an exception into a reward of 0."""

    parser = None

    def __init__(self):
        self.seen = []

    def _get_reward_funcs(self):
        def f1():
            pass

        return [f1]

    def _get_reward_weights(self):
        return [1.0]

    async def score_rollout(self, state):
        self.seen.append(state)
        try:
            found = state["completion"][-1]["content"].split()
            ok = float(set(found) == set(state["answer"].get("defects", [])))
        except AttributeError:
            ok = 0.0
        state["metrics"], state["reward"] = {"f1": ok}, ok


@pytest.fixture
def structured(monkeypatch):
    rubric = StructuredRubric()

    class Env:
        def __init__(self):
            self.rubric = rubric

        def get_dataset(self):
            return list(STRUCTURED)

    vf = types.ModuleType("verifiers")
    vf.load_environment = lambda env_id, **kw: None
    mod = types.ModuleType("fake_review")
    mod.load_environment = lambda: Env()
    monkeypatch.setitem(sys.modules, "verifiers", vf)
    monkeypatch.setitem(sys.modules, "fake_review", mod)
    return VerifiersV0Adapter({"kind": "verifiers", "env": "o/fake_review"}), rubric


def test_structured_gold_stays_structured_for_the_checker(structured):
    ad, rubric = structured
    a, b, _ = ad.tasks(10, 0)
    assert a.answer == '{"defects": []}' and json.loads(b.answer) == {"defects": ["cash_mismatch"]}  # text: JSON
    assert ad.score_raw(b, "cash_mismatch").score == 1.0  # a string gold would make `answer.get` raise: always 0
    assert ad.score_raw(b, "nothing").score == 0.0
    assert ad.score_raw(a, "").score == 1.0
    state = rubric.seen[0]
    assert state["answer"] == {"defects": ["cash_mismatch"]} and state["input"]["answer"] == state["answer"]


def test_other_gold_values_keep_their_type_and_info_is_decoded(structured):
    ad, rubric = structured
    a, _, count = ad.tasks(10, 0)
    assert count.answer == "7"
    ad.score_raw(count, "7")
    assert rubric.seen[-1]["answer"] == 7
    ad.score_raw(a, "")
    assert rubric.seen[-1]["info"] == {"company": "A"}  # a JSON string in the dataset, as `init_state` decodes it


def test_a_caller_may_score_against_another_gold(structured):
    ad, rubric = structured
    a = ad.tasks(10, 0)[0]
    a.answer = "something else"
    ad.score_raw(a, "")
    assert rubric.seen[-1]["answer"] == "something else"


@pytest.mark.parametrize("value, text", [(None, ""), ("x", "x"), (3, "3"), (True, "True"), ([1, "a"], '[1, "a"]'),
                                         ({"k": ["é"]}, '{"k": ["é"]}')])
def test_gold_text(value, text):
    assert gold_text(value) == text


# --- plain v0: scores go through the environment's own rollout, as in a real run -----------------------------------

class FakeError(Exception):  # vf.Error: the rollout loop records these in state["error"] and goes on to the rubric
    pass


class FakeEmptyReply(FakeError):
    pass


class Message:
    """A pydantic message of verifiers 0.1.14: attributes, not keys, and a `str()` that shows the class and every field."""

    def __init__(self, content):
        self.role, self.content = type(self).ROLE, content

    def __repr__(self):
        return f"{type(self).__name__}(role={self.role!r}, content={self.content!r}, tool_calls=None)"


class SystemMessage(Message):
    ROLE = "system"


class UserMessage(Message):
    ROLE = "user"


class AssistantMessage(Message):
    ROLE = "assistant"


MESSAGE_TYPES = {"system": SystemMessage, "user": UserMessage, "assistant": AssistantMessage}


class FakeClient(abc.ABC):
    def __init__(self, client_or_config):
        self._client = client_or_config

    @property
    def client(self):
        return self._client

    @abc.abstractmethod
    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs): ...

    @abc.abstractmethod
    async def close(self): ...


class FakeRolloutEnv:
    """`run_group` of verifiers 0.1.14 cut to what a single-turn rollout does: `init_state`, one model request, the rubric."""

    def __init__(self, rows, funcs, weights=None, system_prompt=None, setup=None):
        self.rows = [{**r, "prompt": [{"role": "system", "content": system_prompt}, *r["prompt"]] if system_prompt else r["prompt"],
                      "example_id": i} for i, r in enumerate(rows)]
        self.rubric = SimpleNamespace(funcs=funcs, _get_reward_funcs=lambda: funcs,
                                      _get_reward_weights=lambda: weights or [1.0] * len(funcs))
        self.parser = SimpleNamespace(parse_answer=lambda completion: completion[-1].content if completion else None)
        self.sampling_args, self.setup, self.inputs, self.requests = {"n": 1}, setup, [], []

    def get_dataset(self):
        return self.rows

    async def run_group(self, group_inputs, client, model, sampling_args, **kwargs):
        return [await self._one(inp, client, model, sampling_args) for inp in group_inputs]

    async def _one(self, inp, client, model, sampling_args):
        self.inputs.append(inp)
        input_ = copy.deepcopy(inp)
        if "task" in input_:
            raise ValueError("Plain string task routes are no longer supported")
        prompt = [MESSAGE_TYPES[m["role"]](m["content"]) for m in input_["prompt"]]
        state = {"input": input_, "task": input_, "prompt": prompt, "client": client, "model": model, "trajectory": [],
                 "completion": [], "error": None}
        try:
            if self.setup:
                self.setup(state)
            response = await client.get_response(prompt=prompt, model=model, sampling_args=sampling_args, tools=None, state=state)
            self.requests.append(prompt)
            state["completion"] = [AssistantMessage(response.message.content)]
            state["trajectory"].append({"response": response})
        except FakeError as e:
            state["error"] = e
        extras = {k: v for k, v in input_.items() if k not in ("prompt", "answer", "info", "example_id")}
        objects = {**extras, "prompt": prompt, "completion": state["completion"], "answer": input_.get("answer", ""),
                   "state": state, "info": input_.get("info", {}), "task": input_, "parser": self.parser}
        metrics = {}
        for f in self.rubric.funcs:
            try:
                metrics[f.__name__] = float(await _maybe(f(**objects)))
            except Exception:  # noqa: BLE001  (verifiers logs it and gives the function 0)
                metrics[f.__name__] = 0.0
        reward = sum(metrics[f.__name__] * w for f, w in zip(self.rubric.funcs, self.rubric._get_reward_weights(), strict=True))
        return {"reward": reward, "metrics": metrics, "error": repr(state["error"]) if state["error"] else None}


async def _maybe(x):
    return await x if inspect.isawaitable(x) else x


CHAT_ROWS = [{"prompt": [{"role": "user", "content": f"Solve {i}."}], "answer": str(i)} for i in range(3)]


def _rollout_vf(env_factory, *, with_client=True, name="fake_rollout"):
    vf = types.ModuleType("verifiers")
    vf.load_environment = lambda env_id, **kw: None
    vf.SingleTurnEnv = FakeRolloutEnv
    vf.MultiTurnEnv = type("MultiTurnEnv", (), {})
    vf.EmptyModelResponseError = FakeEmptyReply
    if with_client:
        vf.Client = FakeClient
        vf.Response = lambda **kw: SimpleNamespace(**kw)
        vf.ResponseMessage = lambda **kw: SimpleNamespace(**kw)
    mod = types.ModuleType(name)
    mod.load_environment = env_factory
    return vf, mod


@pytest.fixture
def rollout(monkeypatch):
    """(adapter, env) for a FakeRolloutEnv made from the keyword arguments."""
    def make(funcs, rows=CHAT_ROWS, with_client=True, env_class=None, **kw):
        env = (env_class or FakeRolloutEnv)(rows, funcs, **kw)
        vf, mod = _rollout_vf(lambda: env, with_client=with_client)
        monkeypatch.setitem(sys.modules, "verifiers", vf)
        monkeypatch.setitem(sys.modules, "fake_rollout", mod)
        if with_client:  # the helper that makes real messages comes with the versions whose rollout hands them over
            utils = types.ModuleType("verifiers.utils.message_utils")
            utils.normalize_messages = lambda ms: [MESSAGE_TYPES[m["role"]](m["content"]) for m in ms]
            monkeypatch.setitem(sys.modules, "verifiers.utils.message_utils", utils)
        return VerifiersV0Adapter({"kind": "verifiers", "env": "o/fake_rollout"}), env
    return make


def _reads_real_message(completion, answer, **kw):
    return float(completion[-1].content == answer)  # a dict has no `.content`: verifiers would turn that into 0


def test_a_reward_that_only_works_with_the_real_message_type(rollout):
    ad, env = rollout([_reads_real_message])
    assert ad.meta["scoring_path"] == "rollout" and "scoring_path_reason" not in ad.meta
    gold, wrong = ad.tasks(10, 0)[1], "nope"
    assert ad.score_raw(gold, gold.answer).score == 1.0 and ad.score_raw(gold, wrong).score == 0.0
    assert env.inputs[-1] == env.rows[1]  # the dataset's row, as `generate` hands it over


def test_a_reward_on_str_of_the_message_sees_what_a_real_run_shows(rollout):
    """A similarity over `str(completion[0])` compares the gold with the message's repr, not with the reply text."""
    def whole_message(completion, answer, **kw):
        text = completion[0]["content"] if isinstance(completion[0], dict) else str(completion[0])
        return float(text == answer)

    ad, _ = rollout([whole_message])
    gold = ad.tasks(10, 0)[1]
    assert ad.score_raw(gold, gold.answer).score == 0.0


def test_a_reward_that_looks_up_prompt_zero_gets_the_system_prompt(rollout):
    """verifiers puts the system prompt first, so a table keyed by the user's text misses, and the fallback pays any reply."""
    table = {r["prompt"][0]["content"]: r["answer"] for r in CHAT_ROWS}
    seen = []

    def lookup(prompt, completion, **kw):
        seen.append(prompt)
        key = prompt[0].content
        return float(completion[-1].content == table[key]) if key in table else 0.5

    ad, _ = rollout([lookup], system_prompt="You are careful.")
    gold = ad.tasks(10, 0)[0]
    assert ad.score_raw(gold, gold.answer).score == 0.5 and ad.score_raw(gold, "anything at all").score == 0.5
    assert [type(m) for m in seen[0]] == [SystemMessage, UserMessage]
    ad, _ = rollout([lookup])
    assert ad.score_raw(gold, "no").score == 0.0  # without a system prompt the same lookup finds its row


def test_reward_functions_get_the_rows_other_columns_and_the_real_state(rollout):
    def uses_them(question, task, state, **kw):
        assert isinstance(task, dict) and task["question"] == question and state["model"] == REPLY_MODEL
        assert state["trajectory"][0]["response"].message.finish_reason == "stop"
        return 1.0

    ad, _ = rollout([uses_them], rows=[{**r, "question": f"q{i}"} for i, r in enumerate(CHAT_ROWS)])
    assert ad.score_raw(ad.tasks(10, 0)[2], "2").score == 1.0


def test_sub_scores_and_zero_weight_metrics_in_a_rollout(rollout):
    def exact(completion, answer, **kw):
        return float(completion[-1].content == answer)

    def num_turns(state, **kw):
        return len(state["trajectory"])

    ad, _ = rollout([exact, num_turns], weights=[1.0, 0.0])
    raw = ad.score_raw(ad.tasks(10, 0)[1], "1")
    assert raw.score == 1.0 and raw.sub_scores == {"exact": 1.0, "metric:num_turns": 1.0} and raw.parsed == "1"
    assert raw.network is False


def test_an_empty_reply_is_scored_as_the_real_clients_leave_it(rollout):
    seen = []

    def records(completion, state, **kw):
        seen.append((completion, state["error"]))
        return 0.25

    ad, env = rollout([records])
    raw = ad.score_raw(ad.tasks(10, 0)[0], "")
    assert raw.score == 0.25 and seen[0][0] == [] and isinstance(seen[0][1], FakeEmptyReply) and not env.requests


def test_scoring_against_another_gold_still_works_in_a_rollout(rollout):
    ad, env = rollout([_reads_real_message])
    t = ad.tasks(10, 0)[0]
    t.answer = "something else"
    assert ad.score_raw(t, "something else").score == 1.0 and env.rows[0]["answer"] == "0"


def test_a_row_without_an_answer_column_reaches_the_reward_as_a_real_run_does(rollout):
    seen = []

    def records(answer, target, **kw):
        seen.append((answer, target))
        return 1.0

    ad, _ = rollout([records], rows=[{"prompt": r["prompt"], "target": r["answer"]} for r in CHAT_ROWS])
    t = ad.tasks(10, 0)[1]
    assert t.answer == "1"  # the gold our probes use comes from `target`
    ad.score_raw(t, "1")
    assert seen == [("", "1")]


@pytest.mark.parametrize("call", [
    lambda state: state["client"].client.chat.completions.create(model="m", messages=[]),
    lambda state: _run(state["client"].get_response(prompt=[], model="m", sampling_args={})),
    lambda state: _run(state["client"].get_native_response([], "m", {})),
], ids=["native client", "get_response", "abstract method"])
def test_a_checker_that_calls_a_model_counts_as_a_network_attempt(rollout, call):
    def judge(state, **kw):
        call(state)
        return 1.0

    ad, _ = rollout([judge])
    raw = ad.score_raw(ad.tasks(10, 0)[0], "0")
    assert raw.network is True and raw.score == 0.0  # the attempt raised, and verifiers turned that into 0


def _run(coro):
    if inspect.isawaitable(coro):  # run inside a reward function: step it by hand, the loop is busy
        coro.send(None)


def test_the_next_call_starts_clean_after_a_model_call(rollout):
    calls = []

    def judge_once(state, **kw):
        if not calls:
            calls.append(1)
            _run(state["client"].get_response(prompt=[], model="m", sampling_args={}))
        return 1.0

    ad, _ = rollout([judge_once])
    t = ad.tasks(10, 0)[0]
    assert ad.score_raw(t, "0").network is True and ad.score_raw(t, "0").network is False


def test_a_rollout_that_cannot_start_raises_and_is_classified_as_such(rollout):
    def never(**kw):
        raise AssertionError

    ad, _ = rollout([never], rows=[{**r, "task": "named"} for r in CHAT_ROWS])
    with pytest.raises(RolloutError, match="Plain string task routes"):
        ad.score_raw(ad.tasks(10, 0)[0], "0")
    error = "Traceback (most recent call last):\nrewardaudit.adapters.verifiers_adapter.RolloutError: ValueError: x\n"
    contract = SimpleNamespace(observations=[("boxed", "t", "c", Verdict(Kind.ERROR, error=error))] * 4, passing_tasks=lambda: [],
                               no_discrimination=False, wrappers_run=["boxed"])
    cls, reason, advice = classify_contract(contract, [Task("t", "p", "a")])
    assert cls == "load_failed" and reason == "4 of 4 rollouts with the reference answer failed on this verifiers: ValueError: x"
    assert "--verifiers" in advice


def test_a_rollout_error_before_the_reply_has_no_score(rollout):
    def setup(state):
        raise FakeError("sandbox did not start")

    ad, _ = rollout([lambda **kw: 1.0], setup=setup)
    with pytest.raises(RolloutError, match="sandbox did not start"):
        ad.score_raw(ad.tasks(10, 0)[0], "0")


def test_a_rollout_that_never_asks_for_the_reply_has_no_score(rollout):
    class Stops(FakeRolloutEnv):
        async def run_group(self, group_inputs, client, model, sampling_args, **kwargs):
            return [{"reward": 0.0, "metrics": {}, "error": None}]

    ad, _ = rollout([lambda **kw: 1.0], env_class=Stops)
    with pytest.raises(RolloutError, match="without asking"):
        ad.score_raw(ad.tasks(10, 0)[0], "0")


def test_without_the_client_api_the_direct_form_scores_and_says_so(rollout):
    def reads_dict(completion, answer, **kw):
        return float(completion[-1]["content"] == answer)

    class OldRubric(SimpleNamespace):
        async def score_rollout(self, state):
            ok = reads_dict(state["completion"], state["answer"])
            state["timing"]["scoring_ms"] = 1.0
            state["timing"]["total_ms"] += state["timing"]["scoring_ms"]  # as verifiers 0.1.10 does
            state["metrics"], state["reward"] = {"reads_dict": ok}, ok

    class OldEnv(FakeRolloutEnv):
        def __init__(self, rows, funcs, **kw):
            super().__init__(rows, funcs, **kw)
            self.rubric = OldRubric(funcs=funcs, _get_reward_funcs=lambda: funcs, _get_reward_weights=lambda: [1.0])

    ad, _ = rollout([reads_dict], with_client=False, env_class=OldEnv)
    assert ad.meta["scoring_path"] == "direct" and "no client API" in ad.meta["scoring_path_reason"]
    assert ad.score_raw(ad.tasks(10, 0)[1], "1").score == 1.0
    assert "direct rubric call" in direct_scoring_line(ad.meta) and direct_scoring_line({"scoring_path": "rollout"}) is None


def test_a_multi_turn_environment_is_not_rolled_out(monkeypatch):
    class MultiEnv:
        def __init__(self):
            self.rubric, self.parser = SimpleNamespace(funcs=[]), None

        def get_dataset(self):
            return CHAT_ROWS

        async def run_group(self, *a, **k):
            raise AssertionError("one reply is not a rollout of a multi-turn environment")

    vf, mod = _rollout_vf(MultiEnv)
    vf.MultiTurnEnv = MultiEnv
    monkeypatch.setitem(sys.modules, "verifiers", vf)
    monkeypatch.setitem(sys.modules, "fake_rollout", mod)
    ad = VerifiersV0Adapter({"kind": "verifiers", "env": "o/fake_rollout"})
    assert ad.meta["single_turn"] is False and ad.meta["scoring_path"] == "direct" and "multi-turn" in ad.meta["scoring_path_reason"]


def test_a_report_says_when_it_was_scored_outside_the_rollout():
    result = run_audit({"kind": "demo", "name": "substring"}, AuditConfig(n_tasks=6, seed=1))
    assert "direct rubric call" not in render_markdown(result)
    result.meta.update(scoring_path="direct", scoring_path_reason="because")
    assert "direct rubric call, not through the environment's rollout (because)" in render_markdown(result)
    for full in (False, True):
        console = Console(file=io.StringIO(), width=200)
        render_terminal(result, console, full=full)
        assert "direct rubric call" in console.file.getvalue()


def test_rows_with_a_repeated_id_stay_apart_in_a_rollout(rollout):
    ad, env = rollout([_reads_real_message], rows=[{**r, "id": "same"} for r in CHAT_ROWS])
    tasks = ad.tasks(10, 0)
    assert [t.id for t in tasks] == ["0", "1", "2"]
    assert ad.score_raw(tasks[2], "2").score == 1.0 and env.inputs[-1]["answer"] == "2"


def test_rows_with_a_repeated_id_stay_apart_in_a_v1_taskset(monkeypatch):
    class Data:
        system_prompt, image, idx = None, None, None

        def __init__(self, gold):
            self.id, self.prompt, self.answer = "same", "Which one?", gold

        def model_dump(self, exclude=()):
            return {"id": self.id, "prompt": self.prompt, "answer": self.answer}

    class VfTask:
        config = None

        def __init__(self, gold):
            self.data = Data(gold)

        @classmethod
        def toolsets(cls, config):
            return []

        def hooks(self, kind):
            return []

        def plugged_judges(self):
            return []

    class Taskset:
        config = None

        @classmethod
        def toolsets(cls, config):
            return []

        def load(self):
            return [VfTask(g) for g in ("a", "b", "c")]

    v1 = types.ModuleType("verifiers.v1")
    v1.taskset_config_type = lambda tid: lambda id: SimpleNamespace(id=id)
    v1.load_taskset = lambda cfg: Taskset()
    package = types.ModuleType("verifiers")
    package.v1 = v1
    monkeypatch.setitem(sys.modules, "verifiers", package)
    monkeypatch.setitem(sys.modules, "verifiers.v1", v1)
    ad = VerifiersV1Adapter({"kind": "verifiers", "env": "o/fake_v1"})
    tasks = ad.tasks(None, 0)
    assert [(t.id, t.answer) for t in tasks] == [("0", "a"), ("1", "b"), ("2", "c")] and ad.meta["n_rows"] == 3
    assert [ad._vf_tasks[t.id].data.answer for t in tasks] == ["a", "b", "c"]


# --- the same through a real worker process: the guard sees a checker's network use during a rollout ------------------

FAKE_VERIFIERS = '''
import abc
from types import SimpleNamespace
class Client(abc.ABC):
    def __init__(self, c): self._client = c
    @abc.abstractmethod
    async def close(self): ...
Response = ResponseMessage = lambda **kw: SimpleNamespace(**kw)
def load_environment(*a, **k): ...
class SingleTurnEnv: ...
class MultiTurnEnv: ...
'''

FAKE_ENV = '''
import socket
import verifiers as vf
class Env(vf.SingleTurnEnv):
    parser = None
    rubric = None
    sampling_args = {}
    def get_dataset(self):
        return [{"prompt": [{"role": "user", "content": "q"}], "answer": "a", "mode": m} for m in ("plain", "socket", "model")]
    async def run_group(self, inputs, client, model, args, **k):
        inp = inputs[0]
        state = {"client": client}
        response = await client.get_response(prompt=inp["prompt"], model=model, sampling_args=args)
        reward = 0.0
        try:
            reward = float(response.message.content == inp["answer"])
            if inp["mode"] == "socket":
                socket.create_connection(("192.0.2.1", 443), timeout=1)
            if inp["mode"] == "model":
                state["client"].client.chat
        except Exception:
            reward = 0.0  # verifiers swallows what a reward function raises; the guard must have seen it anyway
        return [{"reward": reward, "metrics": {}, "error": None}]
def load_environment():
    return Env()
'''


def test_the_network_guard_sees_a_checker_inside_a_rollout(tmp_path):
    (tmp_path / "verifiers.py").write_text(FAKE_VERIFIERS, encoding="utf-8")
    (tmp_path / "fake_guarded.py").write_text(FAKE_ENV, encoding="utf-8")
    with Runner({"kind": "verifiers", "env": "o/fake_guarded", "cwd": str(tmp_path)}, python=sys.executable) as runner:
        assert runner.meta["scoring_path"] == "rollout"
        plain, socket_user, model_user = runner.tasks(3, 0)
        ok = runner.score(plain, "a")
        assert ok.score == 1.0 and ok.network is False
        by_socket = runner.score(socket_user, "a")
        assert by_socket.score == 0.0 and by_socket.network is True
        by_model = runner.score(model_user, "a")
        assert by_model.score == 0.0 and by_model.network is True
        assert runner.score(plain, "a").network is False


def test_a_package_that_does_not_import_says_why(monkeypatch):
    import importlib

    def fail(name, *a, **k):
        if name == "pkg_x":
            raise ImportError("cannot import name 'ChatMessage' from 'verifiers.types'", name="verifiers.types")
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setitem(sys.modules, "verifiers", types.ModuleType("verifiers"))
    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(ImportError, match="does not import: cannot import name 'ChatMessage'"):
        VerifiersV0Adapter({"env": "o/pkg-x", "taskset_id": "pkg-x"})
    with pytest.raises(ImportError, match="has no load_environment"):
        VerifiersV0Adapter({"env": "o/absent", "taskset_id": "absent"})


@pytest.mark.parametrize("content,text", [("q", "q"), (None, ""), ([{"type": "text", "text": "q"}, {"type": "image_url", "image_url": {}}], "q")])
def test_a_multimodal_prompt_is_text_to_the_tool(content, text):
    assert _content_text(content) == text


# --- an environment through the whole audit: a real worker, the fake modules on its path ---------------------------------

ENV_TEMPLATE = '''
import verifiers as vf
class Env(vf.{base}):
    parser = None
    rubric = None
    sampling_args = {{}}
    def get_dataset(self):
        return [{{"prompt": [{{"role": "user", "content": f"Item {{i}}?"}}], "answer": str(10 + i)}} for i in range(12)]
    async def run_group(self, inputs, client, model, args, **k):
        inp = inputs[0]
        {rollout}
        return [{{"reward": reward, "metrics": {{}}, "error": None}}]
def load_environment():
    return Env()
'''
REPLY = '''try:
            reply = (await client.get_response(prompt=inp["prompt"], model=model, sampling_args=args)).message.content
        except Exception:
            reply = ""
        '''


def _env_spec(tmp_path, base="SingleTurnEnv", rollout=REPLY + 'reward = float(reply.strip().endswith(inp["answer"]))'):
    (tmp_path / "verifiers.py").write_text(FAKE_VERIFIERS, encoding="utf-8")
    (tmp_path / "fake_env_full.py").write_text(ENV_TEMPLATE.format(base=base, rollout=rollout), encoding="utf-8")
    return {"kind": "verifiers", "env": "o/fake_env_full", "cwd": str(tmp_path)}


def test_an_environment_goes_through_the_whole_audit(tmp_path):
    r = run_audit(_env_spec(tmp_path), AuditConfig(n_tasks=6, families=["N"]))
    assert r.meta["scoring_path"] == "rollout" and r.n_rows == 12 and len(r.tasks) == 6
    assert r.contract.self_pass_rate == 1.0 and not r.contract.no_discrimination
    assert r.spec["env"] == "o/fake_env_full" and "dataset" not in r.spec


def test_an_empty_answer_of_an_environment_is_a_dataset_issue_and_not_a_rejection(tmp_path):
    """An environment loads its own rows, so an empty answer is not left out as a dataset file's would be."""
    spec = _env_spec(tmp_path)
    env = tmp_path / "fake_env_full.py"
    env.write_text(env.read_text(encoding="utf-8").replace('"answer": str(10 + i)', '"answer": "" if i == 0 else str(10 + i)'),
                   encoding="utf-8")
    r = run_audit(spec, AuditConfig(n_tasks=None, families=["N"]))
    assert r.n_rows == len(r.tasks) == 12 and r.tasks[0].answer == "" and r.tasks[0].id not in r.contract.self_pass
    assert [i["tasks"] for i in r.dataset_issues if i["id"] == "empty_answers"] == [[r.tasks[0].id]]
    assert "empty_rows_left_out" not in {i["id"] for i in r.dataset_issues} and "G1" not in {f["probe_id"] for f in r.findings}


@pytest.mark.parametrize("kw, cls, advice", [
    ({"base": "MultiTurnEnv"}, "multi_turn_or_tools", "one reply"),
    ({"rollout": "raise ValueError('Plain string task routes are no longer supported')"}, "load_failed", "--verifiers"),
    ({"rollout": REPLY + "reward = 0.0"}, "gold_never_passes", "--verifiers"),  # a reward that reads no reply on this verifiers
    ({"rollout": "__import__('socket').create_connection(('192.0.2.1', 443), timeout=1)"}, "llm_judge", "remote service"),
], ids=["multi_turn", "refused_rollout", "dead_reward", "network_in_the_rollout"])
def test_an_environment_that_cannot_be_audited_says_so(tmp_path, kw, cls, advice):
    with pytest.raises(NotAuditable) as e:
        run_audit(_env_spec(tmp_path, **kw), AuditConfig(n_tasks=6, families=["N"]))
    assert e.value.cls == cls and advice in e.value.advice


# --- which verifiers an environment gets ----------------------------------------------------------------------------------

def test_a_package_whose_module_has_another_name_is_found_by_it(monkeypatch):
    installed = {}
    monkeypatch.setattr(envs, "install_env", lambda env, verifiers=None, log=None: (Path(sys.executable), installed))
    installed.update(module="renamed_pkg")
    assert _environment_spec("x/other-name", None, None) == (
        {"kind": "verifiers", "env": "x/other-name", "taskset_id": "renamed_pkg"}, sys.executable, None)
    installed.update(module="other_name")
    assert _environment_spec("x/other-name", None, None)[0] == {"kind": "verifiers", "env": "x/other-name"}


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """`install_env` on a venv already there, whose package `detect_api` says what `info` says. Returns the pins it asks of uv."""
    py = tmp_path / "envs" / "o__n" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setattr(envs, "CACHE", tmp_path)
    monkeypatch.setattr(envs, "_uv", lambda: "uv")

    def run(info, verifiers=None):
        pins = []
        monkeypatch.setattr(envs, "_run", lambda cmd, what, timeout_s: pins.append(cmd[-1]))
        monkeypatch.setattr(envs, "detect_api", lambda *a, **k: dict(info))
        python, out = envs.install_env("o/n", verifiers=verifiers)
        assert python == py
        return pins, out["verifiers"]
    return run


@pytest.mark.parametrize("info, pin", [
    ({"api": "v0", "verifiers": "0.1.14"}, None),
    ({"api": "v0", "verifiers": "0.4.1"}, envs.V0_PIN),
    ({"api": "v0", "verifiers": "0.4.1", "verifiers_req": ["verifiers>=0.3"]}, None),  # it says it runs on that line
    ({"api": "v0", "verifiers": "0.4.1", "verifiers_req": ["verifiers @ git+https://github.com/o/v.git"]}, None),
    ({"api": "v0_taskset", "verifiers": "0.1.14"}, envs.TASKSET_PIN),
    ({"api": "v0_taskset", "verifiers": "0.1.15.dev170"}, None),
    ({"api": "v1", "verifiers": "0.4.1"}, None),
])
def test_the_verifiers_a_packages_api_needs(installed, info, pin):
    pins, version = installed(info)
    assert pins == ([pin] if pin else []) and version == (pin or info["verifiers"])


@pytest.mark.parametrize("asked, pin", [("<0.1.13", "verifiers<0.1.13"), ("verifiers==0.1.10", "verifiers==0.1.10")])
def test_verifiers_given_by_the_user_decides(installed, asked, pin):
    assert installed({"api": "v0", "verifiers": "0.4.1", "verifiers_req": ["verifiers>=0.3"]}, verifiers=asked) == ([pin], pin)


def test_a_version_asked_for_gets_a_venv_of_its_own(tmp_path, monkeypatch):
    """A venv is reused once installed: `o/n@0.1.4` put into the venv of `o/n` would audit the version that one holds."""
    monkeypatch.setattr(envs, "CACHE", tmp_path)
    assert envs.env_dir("o/n@0.1.4") not in (envs.env_dir("o/n"), envs.env_dir("o/n@0.1.5"))
    assert envs.env_dir("o/n@latest") == envs.env_dir("o/n")
    cmds = []

    def run(cmd, what, timeout_s):
        cmds.append(cmd)
        if cmd[1] == "venv":
            (Path(cmd[-1]) / "bin").mkdir(parents=True)
            (Path(cmd[-1]) / "bin" / "python").write_text("")

    monkeypatch.setattr(envs, "_uv", lambda: "uv")
    monkeypatch.setattr(envs, "_run", run)
    monkeypatch.setattr(envs, "detect_api", lambda *a, **k: {"api": "v1", "verifiers": "0.4.1"})
    run(["uv", "venv", "--python", "3.12", str(envs.env_dir("o/n"))], "", None)  # o/n is installed already
    python, _ = envs.install_env("o/n@0.1.4")
    assert python == envs.env_dir("o/n@0.1.4") / "bin" / "python" and "n==0.1.4" in cmds[-1]


def test_versions_are_ordered_well_enough_to_compare_bounds():
    order = ["0.1.7", "0.1.10", "0.1.20", "0.2.0.dev1", "0.2.0a1", "0.2.0", "0.2.0.post1", "1.0"]
    assert sorted(reversed(order), key=envs._version_key) == order


def test_the_verifiers_a_package_declares():
    assert envs.declared_min_verifiers(["verifiers[openenv]>=0.3.1"]) == "0.3.1"
    assert envs.declared_min_verifiers(["verifiers>=0.1.15.dev152"]) == "0.1.15.dev152"
    assert envs.declared_min_verifiers(['verifiers>=0.4; extra == "dev"', "verifiers"]) is None
    assert envs.declared_min_verifiers([]) is None


def test_taskset_window():
    assert envs._in_taskset_window("0.1.15.dev152") and envs._in_taskset_window("0.1.15.dev189")
    assert not any(envs._in_taskset_window(v) for v in ("0.1.14", "0.1.15.dev151", "0.1.15.dev357", "0.4.0", ""))


TASKSET_SOURCE = '''
import verifiers as vf

class AimeTaskset(vf.Taskset[vf.TasksetConfig]):
    pass

def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
'''


def test_detect_api_reads_what_a_package_is_written_for(tmp_path, monkeypatch):
    files = {
        "aime_x.py": TASKSET_SOURCE,
        "plain_v0.py": "import verifiers as vf\ndef load_environment():\n    return vf.SingleTurnEnv()\n",
        "v1_x.py": "import verifiers.v1 as vf\nclass T(vf.Taskset[vf.Task, vf.TasksetConfig]):\n    pass\n",
        "renamed_pkg.py": "import verifiers as vf\ndef load_environment():\n    return vf.SingleTurnEnv()\n",
        "verifiers-0.1.14.dist-info/METADATA": "Metadata-Version: 2.1\nName: verifiers\nVersion: 0.1.14\n",
        "plain_v0-0.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: plain-v0\nVersion: 0.1\nRequires-Dist: verifiers>=0.1.9\nRequires-Dist: other\n",
        "plain_v0-0.1.dist-info/RECORD": "plain_v0.py,,\nplain_v0-0.1.dist-info/METADATA,,\n",
        "other_name-0.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: other-name\nVersion: 0.1\n",
        "other_name-0.1.dist-info/RECORD": "renamed_pkg.py,,\nother_name-0.1.dist-info/METADATA,,\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    py = Path(sys.executable)
    assert envs.detect_api(py, "aime_x")["api"] == "v0_taskset" and envs.detect_api(py, "v1_x")["api"] == "v1"
    info = envs.detect_api(py, "plain_v0")
    assert info["api"] == "v0" and info["dist"] == "plain-v0" and info["verifiers"] == "0.1.14" and info["verifiers_req"] == ["verifiers>=0.1.9"]
    assert envs.detect_api(py, "other-name")["module"] == "renamed_pkg"
    assert envs.detect_api(py, "not_installed_here")["api"] == "unknown"


def test_installing_without_uv_or_with_a_failing_step_says_what_failed(monkeypatch):
    monkeypatch.setattr(envs.shutil, "which", lambda *a, **k: None)
    with pytest.raises(AuditError, match="needs uv: pip install uv"):
        envs.install_env("o/n")
    with pytest.raises(AuditError, match="Expected a Hub environment as owner/name"):
        envs.install_env("no-slash")
    with pytest.raises(AuditError, match="Installing o/n failed"):
        envs._run([sys.executable, "-c", "raise SystemExit(1)"], "Installing o/n", None)
