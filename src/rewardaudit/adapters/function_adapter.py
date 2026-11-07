"""Audit a plain Python scoring function.

Recognised signatures (detected from parameter names, override with spec["style"]):

  bare       : score(prompt, completion, answer, **kw)        parameters bound by name, see below
  verl       : compute_score(solution_str, ground_truth, ...) or (data_source, solution_str, ground_truth, extra_info)
  trl        : reward(prompts, completions, **columns) -> list[float]   (TRL GRPO; open-r1 style uses this too)
  openrlhf   : reward_func(queries, prompts, labels) -> {"rewards": [...]}   (queries = prompt + response)
  areal      : reward_fn(prompt, completions, prompt_ids, completion_ids, **data)   (one completion string;
               token ids are not available and are passed as None)
  rllm       : reward_fn(task_info: dict, action: str) -> RewardOutput
  easyr1     : compute_score(reward_input: dict) or compute_score(reward_inputs: list[dict]) -> {"overall", ...}
               (response_length is the whitespace word count; there is no tokenizer)
  positional : fn(completion, reference) or fn(prompt, completion, reference)

bare and verl bind each parameter by name: the completion to solution_str / completion / response / predict_str /
model_output / model_response / prediction / ..., the reference to ground_truth / answer / gold / reference /
target / solution / label / gt_answer / ..., plus prompt, messages, info, metadata, data_source, extra_info.
Leading parameters without a default that no name matches take the completion, then the reference, by position.
slime's (response, label, metadata) and understand-r1-zero's (model_response, gt_answer) are bare.

The reference (ground truth) is the gold answer, a column (`reference_col`, may be a dotted path such as
reward_model.ground_truth) or a template (`reference_template`; {answer} and any column, non-string columns as
JSON). Structured values (dict / list) are passed as they are. A JSON string is parsed when the spec sets
`ground_truth_json`, or when the reference parameter is annotated as a dict or list. `prompt_in_completion`
passes prompt + completion as the completion, as verl's reward manager does for TinyZero and Logic-RL (the prompt
as stored in the dataset; no chat template is applied).

The function may return a number or bool; a dict (main value from score / overall / reward / rewards / total /
accuracy, the first present; other numeric keys become sub-scores); a tuple (its first numeric element, as verl's
dispatcher takes res[0] and understand-r1-zero returns (info, reward)); an object with a score / reward attribute
(rLLM RewardOutput); or a one-element list or tensor of any of these.
"""

from __future__ import annotations

import base64
import collections.abc
import copy
import importlib
import importlib.util
import inspect
import json
import sys
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rewardaudit.adapters.base import Adapter, RawScore
from rewardaudit.adapters.dataset import MISSING, answer_text, get_path, load_tasks
from rewardaudit.errors import AuditError
from rewardaudit.types import Task

CANDIDATE_NAMES = ("score", "compute_score", "reward", "reward_fn", "reward_func", "accuracy_reward", "verify")
COMPLETION_NAMES = ("solution_str", "completion", "response", "predict_str", "pred_str", "model_output",
                    "model_response", "prediction", "pred", "llm_solution", "generation", "response_str",
                    "completion_str")
REFERENCE_NAMES = ("ground_truth", "answer", "gold", "reference", "target", "solution", "label", "gt", "gt_answer",
                   "gold_answer", "ground_truth_answer", "reference_answer", "ref")
PROMPT_NAMES = ("prompt", "question", "problem")
MAIN_KEYS = ("score", "overall", "reward", "rewards", "total", "accuracy")
STYLES = ("bare", "verl", "trl", "openrlhf", "areal", "rllm", "easyr1", "positional")
VERL_GROUND_TRUTH = "reward_model.ground_truth"


def load_callable(path: str) -> tuple[Callable[..., Any], str]:
    """`file.py`, `file.py:func`, or `pkg.module:func`."""
    target, _, func_name = path.partition(":")
    if target.endswith(".py") or Path(target).exists():
        p = Path(target).resolve()
        if not p.exists():
            raise AuditError(f"Scoring file not found: {target}. Check the path; a scoring function is given as reward.py or reward.py:function")
        mod_name = f"_rewardaudit_user_{p.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, p)
        if spec is None or spec.loader is None:
            raise AuditError(f"{target} is not a Python file; a scoring function is given as reward.py or reward.py:function")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(p.parent))
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    if func_name:
        fn = getattr(module, func_name, None)
        if not callable(fn):
            raise AuditError(f"{target} has no function {func_name!r}. Name one it defines, as {target}:function_name")
        return fn, func_name
    for name in CANDIDATE_NAMES:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, name
    public = [
        (n, f)
        for n, f in vars(module).items()
        if inspect.isfunction(f) and not n.startswith("_") and f.__module__ == module.__name__
    ]
    for n, f in public:
        if "score" in n or "reward" in n:
            return f, n
    if len(public) == 1:
        return public[0][1], public[0][0]
    raise AuditError(f"Cannot tell which function in {path} does the scoring. Name it `score`, or use {target}:function_name")


def detect_style(fn: Callable[..., Any]) -> str:
    names = set(inspect.signature(fn).parameters)
    if {"queries", "prompts", "labels"} <= names:
        return "openrlhf"
    if names & {"prompt_ids", "completion_ids"}:
        return "areal"
    if {"task_info", "action"} <= names:
        return "rllm"
    if names & {"reward_input", "reward_inputs"}:
        return "easyr1"
    if "completions" in names:
        return "trl"
    if names & {"solution_str", "ground_truth", "data_source"}:
        return "verl"
    if names & set(COMPLETION_NAMES + REFERENCE_NAMES):
        return "bare"
    return "positional"


def _is_conversational(fn: Callable[..., Any]) -> bool | None:
    try:
        ann = inspect.signature(fn).parameters["completions"].annotation
    except (KeyError, ValueError):
        return None
    text = str(ann)
    if "dict" in text:
        return True
    if "str" in text:
        return False
    return None


def _number(x: Any) -> float | None:
    """A number, bool, or one-element list / tuple / tensor / array as a float; None otherwise."""
    if isinstance(x, (bool, int, float)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return _number(x[0]) if len(x) == 1 else None
    if hasattr(x, "item") and not isinstance(x, (str, bytes, dict)):
        try:
            return float(x.item())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _coerce(result: Any) -> RawScore:
    if result is None:
        raise ValueError("scoring function returned None")
    if isinstance(result, list):
        if len(result) != 1:
            raise ValueError(f"expected a single score, got {len(result)} values")
        return _coerce(result[0])
    x = _number(result)
    if x is not None:
        return RawScore(x)
    if isinstance(result, tuple):  # verl's dispatcher reads res[0]; understand-r1-zero returns (info, reward)
        for item in result:
            if (x := _number(item)) is not None:
                return RawScore(x)
        raise ValueError(f"tuple result has no numeric element: {result!r:.200}")
    if isinstance(result, dict):
        subs = {str(k): v for k, v in ((k, _number(v)) for k, v in result.items()) if v is not None}
        main = next((subs[k] for k in MAIN_KEYS if k in subs), None)
        if main is None:
            if not subs:
                raise ValueError(f"dict result has no numeric fields: {result!r:.200}")
            main = sum(subs.values())
        return RawScore(main, sub_scores=subs)
    for attr in ("score", "reward", "overall"):
        if (x := _number(getattr(result, attr, None))) is not None:
            return RawScore(x)
    raise ValueError(f"unsupported score type: {type(result).__name__}")


def _structured_hint(fn: Callable[..., Any], names: list[str]) -> tuple[type, ...]:
    """dict and/or list if a reference parameter is annotated as one (Dict[str, str], list[str]), else ()."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 (unresolvable annotations)
        return ()
    out: list[type] = []
    for name in names:
        h = hints.get(name)
        origin = typing.get_origin(h) or h
        if origin in (dict, collections.abc.Mapping, collections.abc.MutableMapping):
            out.append(dict)
        elif origin in (list, collections.abc.Sequence) and h is not str:
            out.append(list)
    return tuple(out)


class FunctionAdapter(Adapter):
    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        self.spec = spec
        if spec.get("pickled"):  # a function handed to the Python API
            import cloudpickle

            self.fn = cloudpickle.loads(base64.b64decode(spec["pickled"]))
            self.fn_name = getattr(self.fn, "__name__", type(self.fn).__name__)
        else:
            self.fn, self.fn_name = load_callable(spec["path"])
        self.style = spec.get("style") or detect_style(self.fn)
        if self.style not in STYLES:
            raise AuditError(f"Unknown function style {self.style!r}; one of {', '.join(STYLES)}")
        self.sig = inspect.signature(self.fn)
        self.params = list(self.sig.parameters)
        self.named = [n for n, p in self.sig.parameters.items()
                      if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
        self.var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in self.sig.parameters.values())
        self.conversational = _is_conversational(self.fn)
        self.kwargs = dict(spec.get("kwargs") or {})
        # What the checker receives as ground truth. By default the gold answer itself; a column (e.g. the full
        # worked solution) or a template such as "\\boxed{{{answer}}}" can be used instead.
        self.reference_col = spec.get("reference_col")
        self.reference_template = spec.get("reference_template")
        self.answer_col = spec.get("answer_col")
        self.ground_truth_json = bool(spec.get("ground_truth_json"))
        self.prompt_in_completion = bool(spec.get("prompt_in_completion"))
        self.hint = _structured_hint(self.fn, [n for n in self.named if n in REFERENCE_NAMES])
        path = spec.get("path") or ""
        self.name = (f"{Path(path.split(':')[0]).stem}:{self.fn_name}" if path else self.fn_name)
        self.meta = {"style": self.style, "function": self.fn_name, "path": path}
        self.meta.update({k: True for k in ("ground_truth_json", "prompt_in_completion") if spec.get(k)})
        self._tasks: list[Task] | None = None

    def reference(self, task: Task) -> Any:
        """The ground truth handed to the checker (see the module docstring)."""
        ref: Any = MISSING
        if self.reference_col:
            ref = get_path(task.info, self.reference_col)
        if ref is MISSING and self.reference_template:
            fields = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in task.info.items()}
            ref = self.reference_template.format_map({**fields, "answer": task.answer})
        if ref is MISSING:
            raw = get_path(task.info, self.answer_col if self.answer_col and "." in self.answer_col
                           else VERL_GROUND_TRUTH)
            ref = raw if isinstance(raw, (dict, list)) and answer_text(raw) == task.answer else task.answer
        if isinstance(ref, (dict, list)):
            return copy.deepcopy(ref)
        if isinstance(ref, str) and (self.ground_truth_json or self.hint):
            try:
                parsed = json.loads(ref)
            except ValueError:
                if self.ground_truth_json:
                    raise ValueError(f"ground_truth_json is set but the reference is not JSON: {ref[:80]!r}") from None
                return ref
            if self.ground_truth_json or isinstance(parsed, self.hint):
                return parsed
            return ref
        return answer_text(ref)

    def all_tasks(self) -> list[Task]:
        if self._tasks is None:
            self._tasks = load_tasks(
                self.spec["dataset"],
                prompt_col=self.spec.get("prompt_col"),
                answer_col=self.spec.get("answer_col"),
                split=self.spec.get("split"),
            )
        return self._tasks

    # --- calling conventions -------------------------------------------------

    def _extra(self) -> dict[str, Any]:
        """spec kwargs the function accepts."""
        return {k: v for k, v in self.kwargs.items() if k in self.named or self.var_kw}

    def _columns(self, task: Task) -> dict[str, Any]:
        """Dataset columns plus the reference under its usual names (TRL and AReaL pass the row as keywords)."""
        columns = copy.deepcopy(task.info)
        ref = self.reference(task)
        for alias in ("answer", "solution", "ground_truth", "target"):
            columns[alias] = ref
        return columns

    def _call_trl(self, task: Task, completion: str, conversational: bool) -> Any:
        comp: Any = [{"role": "assistant", "content": completion}] if conversational else completion
        prompt: Any = task.chat() if conversational else task.prompt
        columns = {k: [v] for k, v in self._columns(task).items()}
        columns.update({k: [v] for k, v in self.kwargs.items()})
        if not self.var_kw:
            columns = {k: v for k, v in columns.items() if k in self.params}
        kw: dict[str, Any] = {"completions": [comp], **columns}
        if "prompts" in self.params:
            kw["prompts"] = [prompt]
        return self.fn(**kw)

    def _bind(self, task: Task, completion: str) -> dict[str, Any]:
        """Keyword arguments for bare / verl functions: by name, then the completion and the reference by
        position for leading parameters no name matched."""
        kw: dict[str, Any] = {}
        roles = {"completion": False, "reference": False}
        for name in self.named:
            if name in COMPLETION_NAMES:
                kw[name], roles["completion"] = completion, True
            elif name in REFERENCE_NAMES:
                kw[name], roles["reference"] = self.reference(task), True
            elif name in PROMPT_NAMES:
                kw[name] = task.prompt
            elif name == "messages":
                kw[name] = task.chat()
            elif name == "data_source":
                kw[name] = task.info.get("data_source", "audit")
            elif name == "extra_info":
                kw[name] = copy.deepcopy(task.info.get("extra_info", {}))
            elif name == "metadata":
                meta = task.info.get("metadata")
                kw[name] = copy.deepcopy(meta if isinstance(meta, dict) else task.info)
            elif name == "info":
                kw[name] = copy.deepcopy(task.info)
        missing = [r for r, done in roles.items() if not done]
        for name in self.named:
            p = self.sig.parameters[name]
            if not missing or p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD or p.default is not p.empty:
                break
            if name not in kw and name not in self.kwargs:
                kw[name] = completion if missing.pop(0) == "completion" else self.reference(task)
        kw.update(self._extra())
        return kw

    def _call(self, task: Task, completion: str) -> Any:
        style = self.style
        if self.prompt_in_completion and style != "openrlhf":
            completion = task.prompt + completion
        if style == "trl":
            if self.conversational is None:
                try:
                    out = self._call_trl(task, completion, True)
                    self.conversational = True
                    return out
                except (TypeError, KeyError, AttributeError, IndexError):
                    self.conversational = False
            return self._call_trl(task, completion, bool(self.conversational))
        if style in ("bare", "verl"):
            return self.fn(**self._bind(task, completion))
        if style == "openrlhf":
            return self.fn(queries=[task.prompt + completion], prompts=[task.prompt], labels=[self.reference(task)],
                           **self._extra())
        if style == "areal":  # reward_fn(prompt, completions, prompt_ids, completion_ids, **data), positionally
            fixed = [n for n, p in self.sig.parameters.items() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD][:4]
            data = {k: v for k, v in (self._columns(task) | self.kwargs).items()
                    if k not in fixed and (self.var_kw or k in self.named)}
            return self.fn(task.prompt, completion, None, None, **data)
        if style == "rllm":
            info = copy.deepcopy(task.info)
            task_info = {**info, "problem": task.prompt, "ground_truth": self.reference(task),
                         "data_source": info.get("data_source", "audit")}
            return self.fn(task_info, completion, **self._extra())
        if style == "easyr1":
            item = {"response": completion, "response_length": len(completion.split()),
                    "ground_truth": self.reference(task)}
            return self.fn([item] if "reward_inputs" in self.params else item, **self._extra())
        n = len(self.params)
        if n == 2:
            return self.fn(completion, self.reference(task))
        if n >= 3:
            return self.fn(task.prompt, completion, self.reference(task))
        raise ValueError(f"cannot call {self.fn_name} with {n} parameters; use a recognised signature")

    def score_raw(self, task: Task, completion: str) -> RawScore:
        return _coerce(self._call(task, completion))
