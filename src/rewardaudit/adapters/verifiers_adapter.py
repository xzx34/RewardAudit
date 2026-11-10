"""Audit a Prime Intellect `verifiers` environment: its own scoring, given exactly one assistant reply, the completion
under test. `VerifiersAdapter` picks one of two by what is installed: `VerifiersV1Adapter` for a v1 taskset, whose
reward hooks score a trace, and `VerifiersV0Adapter` for an Environment from `load_environment()`, scored through its own
rollout.

Runs inside the worker, under the interpreter where the environment's package is installed.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
from typing import Any

from rewardaudit.adapters.base import Adapter, RawScore, unique_ids
from rewardaudit.types import Task

ANSWER_FIELDS = ("answer", "target", "solution", "ground_truth", "expected", "label", "final_answer", "reference",
                 "answer_key")


def _taskset_id(env: str) -> str:
    return env.split("@")[0].split("/")[-1]


def gold_text(answer: Any) -> str:
    """The gold as text, for probes and wrappers. A structured gold becomes JSON, the form a reply could carry
    (a Python repr such as "{'a': 1}" is not parseable as one)."""
    if answer is None:
        return ""
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False, default=str)
    return str(answer)


class VerifiersAdapter(Adapter):
    """Dispatches to the v1 (Taskset) or v0 (Environment) implementation by what is installed."""

    def __new__(cls, spec: dict[str, Any]):  # type: ignore[override]
        if cls is VerifiersAdapter:
            import verifiers

            if hasattr(verifiers, "load_environment"):
                return object.__new__(VerifiersV0Adapter)
            return object.__new__(VerifiersV1Adapter)
        return object.__new__(cls)


class VerifiersV1Adapter(VerifiersAdapter):
    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        import verifiers.v1 as vf

        self.vf = vf
        self.spec = spec
        tid = spec.get("taskset_id") or _taskset_id(spec["env"])
        last_err: Exception | None = None
        self.cfg_cls = None
        for candidate in (tid, tid.replace("-", "_"), tid.replace("_", "-")):
            try:
                self.cfg_cls = vf.taskset_config_type(candidate)
                tid = candidate
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if self.cfg_cls is None:
            raise ImportError(f"taskset {tid!r} is not importable: {last_err}")
        self.taskset_id = tid
        cfg = self.cfg_cls(id=tid)
        self.taskset = vf.load_taskset(cfg)
        self.name = spec["env"]
        self._vf_tasks: dict[str, Any] = {}
        self._tasks: list[Task] | None = None
        self._loop = asyncio.new_event_loop()
        self._runtime = None
        self._setup_done: set[str] = set()
        self._parser: Any = None
        self.meta = {
            "framework": f"verifiers {getattr(__import__('verifiers'), '__version__', '?')}",
            "taskset": type(self.taskset).__name__,
            "taskset_id": tid,
            "scoring_path": "trace",
        }

    # --- tasks -----------------------------------------------------------------

    def _load_all(self) -> None:
        if self._tasks is not None:
            return
        raw = list(self.taskset.load())
        out: list[Task] = []
        answer_field = None
        reward_names: list[str] = []
        ids = unique_ids([str(t.data.id if t.data.id is not None else (t.data.idx if t.data.idx is not None else i))
                          for i, t in enumerate(raw)])
        for t, tid in zip(raw, ids, strict=True):
            d = t.data
            dump = d.model_dump(exclude={"skills"})
            prompt_val = d.prompt
            messages = None
            if isinstance(prompt_val, str) or prompt_val is None:
                prompt = prompt_val or ""
                if d.system_prompt:
                    messages = [{"role": "system", "content": d.system_prompt}, {"role": "user", "content": prompt}]
            else:
                msgs = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in prompt_val]
                users = [m.get("content") for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
                prompt = users[-1] if users else "\n".join(str(m.get("content", "")) for m in msgs)
                messages = ([{"role": "system", "content": d.system_prompt}] if d.system_prompt else []) + msgs
            if answer_field is None:
                answer_field = next((f for f in ANSWER_FIELDS if f in dump), None)
            answer = dump.get(answer_field, "") if answer_field else ""
            info = {k: v for k, v in dump.items() if k not in ("prompt", "system_prompt", answer_field)
                    and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            out.append(Task(id=tid, prompt=prompt, answer=gold_text(answer), messages=messages, info=info))
            self._vf_tasks[tid] = t
            if not reward_names:
                reward_names = [f.__name__ for f in t.hooks("reward")]
                self._parser, parser_name = _v1_parser(self.vf, t, self.taskset)
                self.meta.update({
                    "task_class": type(t).__name__,
                    "reward_hooks": reward_names,
                    "metric_hooks": [f.__name__ for f in t.hooks("metric")],
                    "stop_hooks": [f.__name__ for f in t.hooks("stop")],
                    "judges": [type(j).__name__ for j in t.plugged_judges()],
                    "needs_container": bool(getattr(t, "NEEDS_CONTAINER", False)),
                    "image": d.image,
                    "toolsets": _v1_toolsets(t, self.taskset),
                    "env_classes": _v1_env_classes(self.vf, self.taskset),
                    "parser": parser_name,
                    "answer_field": answer_field,
                })
        self._tasks = out

    def all_tasks(self) -> list[Task]:
        self._load_all()
        assert self._tasks is not None
        return self._tasks

    # --- scoring ---------------------------------------------------------------

    async def _ensure_runtime(self) -> Any:
        if self._runtime is None:
            runtimes = self.vf.runtimes
            if hasattr(runtimes, "make_backend"):  # verifiers 0.4.1: a backend lends out Runtime views of itself
                backend = runtimes.make_backend(self.vf.SubprocessConfig())
                await backend.start()
                self._runtime = backend.bind()
            else:
                rt = runtimes.make_runtime(self.vf.SubprocessConfig())
                rt.env = {}
                await rt.start()
                self._runtime = rt
        return self._runtime

    async def _score(self, task: Task, completion: str) -> RawScore:
        vf = self.vf
        if task.id not in self._vf_tasks:
            self._load_all()  # parallel workers score tasks they did not sample themselves
        t = self._vf_tasks[task.id]
        runtime = await self._ensure_runtime()
        trace = vf.Trace(task=vf.TraceTask(type=type(t).__name__, data=t.data, key=t.key, hash=t.hash),
                         agent=vf.AgentInfo(config=vf.AgentConfig()))
        if task.id not in self._setup_done:
            params = list(inspect.signature(t.setup).parameters)
            await (t.setup(trace, runtime) if "trace" in params else t.setup(runtime))
            self._setup_done.add(task.id)
        parent = None
        for m in task.chat():
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                msg = vf.SystemMessage(content=content)
            elif role == "assistant":
                msg = vf.AssistantMessage(content=content)
            else:
                msg = vf.UserMessage(content=content)
            trace.nodes.append(vf.MessageNode(parent=parent, message=msg))
            parent = len(trace.nodes) - 1
        trace.nodes.append(vf.MessageNode(parent=parent, message=vf.AssistantMessage(content=completion), sampled=True))
        await t.score(trace, runtime)
        if trace.errors:
            err = trace.errors[-1]
            raise RuntimeError(f"scoring error: {getattr(err, 'message', err)}")
        subs = {k: (v.score if v is not None else float("nan")) for k, v in trace.rewards.items()}
        subs.update({f"metric:{k}": (v if v is not None else float("nan")) for k, v in trace.metrics.items()})
        info = trace.info if isinstance(trace.info, dict) else {}
        recorded = next((info[k] for k in PARSED_KEYS if info.get(k) is not None), None)
        parsed = str(recorded) if recorded is not None else self.parse(completion)
        return RawScore(float(trace.reward), sub_scores=subs, parsed=parsed)

    def score_raw(self, task: Task, completion: str) -> RawScore:
        return self._loop.run_until_complete(self._score(task, completion))

    def parse(self, completion: str) -> str | None:
        if self._parser is None:
            return None
        try:
            out = self._parser(completion)
            if inspect.isawaitable(out):
                out = self._loop.run_until_complete(out)
        except Exception:  # noqa: BLE001
            return None
        return None if out is None else str(out)


PARSED_KEYS = ("parsed", "parsed_answer", "extracted_answer", "extracted", "prediction", "pred")
BOXED_HELPERS = ("verify_boxed_math_answer", "extract_boxed_answer")


def _v1_parser(vf: Any, task: Any, taskset: Any) -> tuple[Any, str]:
    """The extraction a v1 taskset applies to the reply: an explicit parser or `parse_answer` / `extract_answer`
    method, else the boxed-answer helper its reward hook calls. (None, "") when there is no such hook."""
    for owner in (task, taskset):
        parser = getattr(owner, "parser", None)
        fn = getattr(parser, "parse_answer", None)
        if callable(fn):
            return fn, f"{type(parser).__name__}.parse_answer"
        for name in ("parse_answer", "extract_answer"):
            fn = getattr(owner, name, None)
            if callable(fn):
                return fn, f"{type(owner).__name__}.{name}"
    extract = getattr(vf, "extract_boxed_answer", None)
    for hook in task.hooks("reward"):
        try:
            src = inspect.getsource(hook)
        except (OSError, TypeError):
            continue
        if extract is not None and any(h in src for h in BOXED_HELPERS):
            def boxed(text: str) -> str:  # what verify_boxed_math_answer compares
                if "<think>" in text and "</think>" not in text:
                    return ""
                return extract(text.rsplit("</think>", 1)[-1], strict=True).strip()

            return boxed, f"extract_boxed_answer (called by reward hook {hook.__name__})"
    return None, ""


def _v1_toolsets(task: Any, taskset: Any) -> list[str]:
    out: list[str] = []
    for owner, cfg in ((type(task), task.config), (type(taskset), getattr(taskset, "config", None))):
        try:
            out += [type(x).__name__ for x in owner.toolsets(cfg) or []]
        except Exception:  # noqa: BLE001
            continue
    return out


def _v1_env_classes(vf: Any, taskset: Any) -> list[str]:
    """Env subclasses the taskset's package defines; a custom `run()` drives its own turns."""
    import sys

    base = getattr(vf, "Env", None)
    mod = sys.modules.get(type(taskset).__module__.split(".")[0])
    if base is None or mod is None:
        return []
    names = []
    for sub in {type(taskset).__module__, mod.__name__}:
        for obj in vars(sys.modules.get(sub, mod)).values():
            if isinstance(obj, type) and issubclass(obj, base) and obj is not base and obj.__name__ not in names:
                names.append(obj.__name__)
    return names


class RolloutError(RuntimeError):
    """The environment's own rollout failed, as a real run on this verifiers version would: there is no score to audit."""


REPLY_MODEL = "rewardaudit-reply"  # `state["model"]` in a rollout; no model is behind it


def _reply_client(vf: Any) -> Any:
    """A `vf.Client` that answers a rollout's one model request with the completion under test (`.text`), so the
    environment's own code builds what its reward functions receive. None on a verifiers without that client API.

    It holds no connection. A further request (`.extra_calls`), or a reach for the native client, can only be the checker
    calling a model, which a real client would send to the network: it counts as a network attempt."""
    base, response, message = (getattr(vf, n, None) for n in ("Client", "Response", "ResponseMessage"))
    if not (isinstance(base, type) and response and message):
        return None

    def refuse(self: Any, *a: Any, **k: Any) -> Any:
        self.extra_calls += 1
        raise RuntimeError("a model call beyond the reply under test")

    async def get_response(self: Any, prompt: Any, model: str, sampling_args: Any = None, tools: Any = None, **k: Any) -> Any:
        if self.answered:
            refuse(self)
        self.answered = True
        if not self.text:  # what the OpenAI clients raise on an empty reply: the rubric then scores `completion == []`
            raise getattr(vf, "EmptyModelResponseError", RuntimeError)("Model returned no content")
        return response(id="rewardaudit", created=0, model=model,
                        message=message(content=self.text, finish_reason="stop", is_truncated=False))

    def init(self: Any) -> None:
        base.__init__(self, None)
        self.reset("")

    def reset(self: Any, text: str) -> None:
        self.text, self.answered, self.extra_calls = text, False, 0

    stubs = dict.fromkeys(getattr(base, "__abstractmethods__", ()), refuse)
    return type("ReplyClient", (base,), {**stubs, "__init__": init, "reset": reset, "get_response": get_response,
                                         "client": property(refuse)})()


class VerifiersV0Adapter(VerifiersAdapter):
    """verifiers < 0.2: `load_environment()` returns an Environment with a dataset, parser and rubric.
    Scores go through the environment's own rollout, as in a real run, with a client whose one reply is the completion;
    a direct `rubric.score_rollout` call remains only where that rollout cannot run (`meta["scoring_path"]`).

    The 0.1.15 pre-releases add `load_environment(config: vf.EnvConfig)` returning `vf.Env(taskset, harness)`,
    an Environment whose rubric is empty: its rewards are the taskset's `@vf.reward` signals, scored by
    `harness.runtime.score_rollout(task, state)`."""

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        import importlib

        import verifiers as vf

        self.vf = vf
        self.spec = spec
        name = spec.get("taskset_id") or _taskset_id(spec["env"])
        mod, cause = None, None
        for candidate in (name, name.replace("-", "_")):
            try:
                mod = importlib.import_module(candidate)
                break
            except ImportError as e:
                if e.name != candidate:  # the package is there, something it imports is not
                    cause = e
        if mod is None and cause is not None:
            raise ImportError(f"environment package {name!r} does not import: {cause}") from cause
        if mod is None or not hasattr(mod, "load_environment"):
            raise ImportError(f"environment package {name!r} has no load_environment()")
        self.env = _v0_load(vf, mod.load_environment)
        self.name = spec["env"]
        self._loop = asyncio.new_event_loop()
        self._tasks: list[Task] | None = None
        self._rows: dict[str, dict[str, Any]] = {}
        self._runtime = _taskset_runtime(self.env)
        self._vf_tasks: dict[str, Any] = {}
        if self._runtime is not None:
            self._init_taskset_meta()
            return
        rubric = getattr(self.env, "rubric", None)
        funcs: list[Any] = []
        for attr in ("_get_reward_funcs", "get_reward_funcs", "reward_funcs"):
            f = getattr(rubric, attr, None)
            if f is None:
                continue
            try:
                funcs = list(f() if callable(f) else f)
            except TypeError:
                funcs = list(f) if not callable(f) else []
            if funcs:
                break
        rubrics = _rubric_tree(rubric)
        try:
            weights = list(rubric._get_reward_weights()) if rubric is not None else []
        except Exception:  # noqa: BLE001
            weights = []
        names = [getattr(x, "__name__", str(x)) for x in (funcs or [])]
        # zero-weight functions are metrics (num_turns and the like), not part of the reward
        self._metric_names = {n for n, w in zip(names, weights, strict=False) if w == 0}
        self.meta = {
            "framework": f"verifiers {getattr(vf, '__version__', '?')} (v0 API)",
            "env_class": type(self.env).__name__,
            "env_classes": [c.__name__ for c in type(self.env).__mro__ if c is not object],
            "single_turn": _v0_single_turn(vf, self.env),
            "tools": len(getattr(self.env, "tools", None) or getattr(self.env, "oai_tools", None) or []),
            "rubric": type(rubric).__name__ if rubric is not None else None,
            "rubric_classes": [type(r).__name__ for r in rubrics],
            "reward_funcs": names,
            "reward_weights": weights if len(weights) == len(names) else [],
            "parser": type(getattr(self.env, "parser", None)).__name__,
            # the XML tag the reward reads the answer from (XMLParser answer_field), e.g. <reversed_text>
            "answer_tag": _xml_answer_tag([getattr(self.env, "parser", None)] + [getattr(r, "parser", None) for r in rubrics]),
            "judges": [type(r).__name__ for r in rubrics if any("Judge" in c.__name__ for c in type(r).__mro__)],
            "max_turns": getattr(self.env, "max_turns", None),
        }
        multi_turn = self.meta["single_turn"] is False  # one reply is not a rollout of these; the audit stops on them anyway
        self._client = None if multi_turn or not callable(getattr(self.env, "run_group", None)) else _reply_client(vf)
        self.meta["scoring_path"] = "direct" if self._client is None else "rollout"
        if self._client is None:
            self.meta["scoring_path_reason"] = ("a multi-turn rollout takes more than one reply" if multi_turn else
                                                "this verifiers version has no client API to drive the rollout with")

    def _init_taskset_meta(self) -> None:
        env, vf = self.env, self.vf
        signals = list(getattr(self._runtime, "rollout_signals", None) or [])
        names = [str(s.get("name")) for s in signals]
        self._metric_names = {str(s.get("name")) for s in signals if s.get("kind") != "reward"}
        ts_cfg, h_cfg = getattr(env.taskset, "config", None), getattr(env.harness, "config", None)
        toolsets = [_name_of(t) for c in (ts_cfg, h_cfg) for t in (getattr(c, "toolsets", None) or [])]
        program = getattr(h_cfg, "program", None)
        sandbox = getattr(h_cfg, "sandbox", None) or getattr(program, "sandbox", None)
        custom_program = bool(getattr(program, "fn", None) or getattr(program, "command", None))
        user = getattr(ts_cfg, "user", None) or getattr(h_cfg, "user", None)
        self.meta = {
            "framework": f"verifiers {getattr(vf, '__version__', '?')} (v0 API, taskset + harness)",
            "env_class": type(env).__name__,
            "env_classes": [c.__name__ for c in type(env).__mro__ if c is not object],
            "taskset": type(env.taskset).__name__,
            "harness": type(env.harness).__name__,
            "single_turn": not (toolsets or user or sandbox or custom_program),
            "tools": len(toolsets),
            "toolsets": toolsets,
            "needs_container": bool(sandbox),
            "user_simulator": bool(user),
            "custom_program": custom_program,
            "rubric": None,
            "rubric_classes": [],
            "reward_funcs": names,
            "reward_weights": [float(s.get("weight") or 0.0) for s in signals],
            "group_signals": [str(s.get("name")) for s in getattr(self._runtime, "group_signals", None) or []],
            "parser": None,
            "judges": [],
            "max_turns": getattr(h_cfg, "max_turns", None),
            "scoring_path": "harness_runtime",
        }

    async def _score_taskset(self, task: Task, completion: str) -> RawScore:
        row = self._rows[task.id]
        if task.id not in self._vf_tasks:
            self._vf_tasks[task.id] = self.env.taskset.to_task(dict(row))
        vtask = self._vf_tasks[task.id]
        state = self.vf.State.for_task(vtask)
        state["completion"] = [{"role": "assistant", "content": completion}]
        await self._runtime.score_rollout(vtask, state)
        metrics = state.get("metrics") or {}
        subs = {(f"metric:{k}" if k in self._metric_names else k): float(v)
                for k, v in metrics.items() if isinstance(v, (int, float))}
        return RawScore(float(state.get("reward") or 0.0), sub_scores=subs or None, parsed=None)

    def _dataset(self):
        for attr in ("get_dataset", "get_eval_dataset"):
            f = getattr(self.env, attr, None)
            if callable(f):
                try:
                    ds = f()
                    if ds is not None:
                        return ds
                except Exception:  # noqa: BLE001
                    continue
        ds = getattr(self.env, "dataset", None) or getattr(self.env, "eval_dataset", None)
        if ds is None:
            raise RuntimeError("environment exposes no dataset")
        return ds

    def _load_all(self) -> None:
        if self._tasks is not None:
            return
        out: list[Task] = []
        rows = [dict(row) for row in self._dataset()]
        for row, tid in zip(rows, unique_ids([str(r.get("id", i)) for i, r in enumerate(rows)]), strict=True):
            prompt_val = row.get("prompt") or row.get("question") or ""
            messages = None
            if isinstance(prompt_val, list):
                users = [_content_text(m.get("content")) for m in prompt_val if m.get("role") == "user"]
                prompt = users[-1] if users else ""
                messages = prompt_val
            else:
                prompt = str(prompt_val)
            answer = _row_gold(row)
            info = {k: v for k, v in row.items() if k not in ("prompt", "answer")
                    and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            out.append(Task(id=tid, prompt=prompt, answer=gold_text(answer), messages=messages, info=info))
            self._rows[tid] = row
        self._tasks = out

    def all_tasks(self) -> list[Task]:
        self._load_all()
        assert self._tasks is not None
        return self._tasks

    async def _score(self, task: Task, completion: str) -> RawScore:
        if task.id not in self._rows:
            self._load_all()
        if self._runtime is not None:
            return await self._score_taskset(task, completion)
        if self._client is not None:
            return await self._score_rollout(task, completion)
        return await self._score_direct(task, completion)

    async def _score_rollout(self, task: Task, completion: str) -> RawScore:
        """What `generate` does with one example: the row goes in as it is (the caller may score against another gold),
        and `init_state`, the rollout loop and the rubric build everything a reward function receives."""
        row, client = self._rows[task.id], self._client
        client.reset(completion)
        rollout_input = row if gold_text(_row_gold(row)) == task.answer else {**row, "answer": task.answer}
        args = copy.deepcopy(getattr(self.env, "sampling_args", None) or {})
        try:
            out = (await self.env.run_group([rollout_input], client, REPLY_MODEL, args))[0]
        except Exception as e:  # reward functions cannot raise this far: verifiers turns those into a reward of 0
            raise RolloutError(f"{type(e).__name__}: {e}") from e
        network = client.extra_calls > 0
        if out.get("error") and completion and not network:
            raise RolloutError(str(out["error"]))
        if not client.answered:  # a stop condition ended it first: the rubric never saw the reply under test
            raise RolloutError("the rollout ended without asking for a reply")
        return self._raw_score(out.get("reward"), out.get("metrics"), completion, network)

    def _raw_score(self, reward: Any, metrics: Any, completion: str, network: bool = False) -> RawScore:
        subs = {(f"metric:{k}" if k in self._metric_names else k): float(v)
                for k, v in (metrics or {}).items() if isinstance(v, (int, float))}
        return RawScore(float(reward or 0.0), sub_scores=subs or None, parsed=self.parse(completion), network=network)

    async def _score_direct(self, task: Task, completion: str) -> RawScore:
        row = self._rows[task.id]
        rubric = self.env.rubric
        completion_msgs = _as_messages([{"role": "assistant", "content": completion}])
        prompt = _as_messages(task.chat())
        task_name = row.get("task") or "default"
        info = _row_info(row)
        answer = _row_answer(task, row)
        state: dict[str, Any] = {
            "prompt": prompt, "completion": completion_msgs, "answer": answer, "task": task_name,
            "info": info, "responses": [], "trajectory": [], "turn": 1, "reward": 0.0, "metrics": {},
            "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},  # score_rollout adds to total_ms
            "input": {"prompt": prompt, "answer": answer, "task": task_name, "info": info},
        }
        sig = inspect.signature(rubric.score_rollout)
        params = sig.parameters
        if list(params) == ["state"] or ("state" in params and "completion" not in params):
            res = rubric.score_rollout(state)  # verifiers >= 0.1.6: in-place on the state
        else:
            kwargs = {"prompt": prompt, "completion": completion_msgs, "answer": answer, "state": state,
                      "task": task_name, "info": info}
            res = rubric.score_rollout(**{k: v for k, v in kwargs.items() if k in params})
        if inspect.isawaitable(res):
            res = await res
        reward = getattr(res, "reward", None)
        metrics = getattr(res, "metrics", None)
        if reward is None and isinstance(res, dict):
            reward, metrics = res.get("reward"), res.get("metrics")
        if reward is None:
            reward, metrics = state.get("reward"), state.get("metrics")
        if reward is None:
            raise RuntimeError(f"unexpected score_rollout result: {type(res).__name__}")
        return self._raw_score(reward, metrics, completion)

    def score_raw(self, task: Task, completion: str) -> RawScore:
        return self._loop.run_until_complete(self._score(task, completion))

    def parse(self, completion: str) -> str | None:
        if self._runtime is not None:
            return None  # taskset rewards parse inside each signal; the Env's default parser is not theirs
        parser = getattr(self.env, "parser", None)
        fn = getattr(parser, "parse_answer", None)
        if not callable(fn):
            return None
        try:
            out = fn(_as_messages([{"role": "assistant", "content": completion}]))
        except Exception:  # noqa: BLE001
            try:
                out = fn(completion)
            except Exception:  # noqa: BLE001
                return None
        return None if out is None else str(out)


def _as_messages(messages: list[dict[str, Any]]) -> Any:
    """Chat messages in the form the installed verifiers hands to reward functions and parsers."""
    try:
        from verifiers.utils.message_utils import normalize_messages
    except ImportError:
        return messages
    return normalize_messages(messages)


def _content_text(content: Any) -> str:
    """A message's text; a multimodal one is a list of parts."""
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return str(content or "")


def _row_gold(row: dict[str, Any]) -> Any:
    answer = row.get("answer")
    return answer if answer is not None else next((row[f] for f in ANSWER_FIELDS if f in row), "")


def _row_answer(task: Task, row: dict[str, Any]) -> Any:
    """The gold the reward function reads: the dataset's own value, as verifiers hands it over (a dict stays a
    dict, an int an int), unless the caller scores against another gold than the row's."""
    raw = row.get("answer")
    return raw if raw is not None and gold_text(raw) == task.answer else task.answer


def _row_info(row: dict[str, Any]) -> Any:
    """`info` as verifiers hands it over: a dataset may keep it as a JSON string, which `init_state` decodes."""
    info = row.get("info") or {}
    if isinstance(info, str):
        try:
            return json.loads(info)
        except ValueError:
            return info
    return info


def _v0_load(vf: Any, load: Any) -> Any:
    """Call `load_environment`; the 0.1.15 pre-release form takes one required `config: vf.EnvConfig`."""
    env_config = getattr(vf, "EnvConfig", None)
    try:
        param = inspect.signature(load).parameters.get("config")
    except (TypeError, ValueError):
        param = None
    if env_config is not None and param is not None and (
            param.default is inspect.Parameter.empty or "EnvConfig" in str(param.annotation)):
        return load(config=env_config())
    return load()


def _taskset_runtime(env: Any) -> Any:
    """The harness runtime that scores a `vf.Env(taskset, harness)`, or None for a plain v0 Environment."""
    harness = getattr(env, "harness", None)
    runtime = getattr(harness, "runtime", None)
    if getattr(env, "taskset", None) is None or not callable(getattr(runtime, "score_rollout", None)):
        return None
    return runtime


def _xml_answer_tag(parsers: list[Any]) -> str | None:
    for parser in parsers:
        tag = getattr(parser, "answer_field", None)
        if isinstance(tag, str) and tag and hasattr(parser, "_fields"):
            return tag
    return None


def _name_of(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get("name") or x.get("id") or "toolset")
    return str(getattr(x, "name", None) or type(x).__name__)


def _rubric_tree(rubric: Any) -> list[Any]:
    """The rubric and every rubric nested in it (RubricGroup and friends)."""
    if rubric is None:
        return []
    out = [rubric]
    for child in getattr(rubric, "rubrics", None) or []:
        out += _rubric_tree(child)
    return out


def _v0_single_turn(vf: Any, env: Any) -> bool | None:
    """True for SingleTurnEnv, False for other MultiTurnEnv subclasses (tool, sandbox, game envs), None for a
    custom Environment subclass that is neither."""
    single, multi = getattr(vf, "SingleTurnEnv", None), getattr(vf, "MultiTurnEnv", None)
    subs = getattr(env, "envs", None)
    if subs and not (single and isinstance(env, single)):  # EnvGroup: single-turn only if every member is
        flags = [_v0_single_turn(vf, e) for e in subs]
        return False if False in flags else (True if all(flags) else None)
    if single is not None and isinstance(env, single):
        return True
    if multi is not None and isinstance(env, multi):
        return False
    return None
