"""The Python interface, which the command line uses too."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from rewardaudit.errors import AuditError
from rewardaudit.pipeline import AuditConfig, AuditResult, Log, run_audit
from rewardaudit.probes.registry import FAMILIES

DEFAULT_N = 100
_SPEC_OPTIONS = ("style", "kwargs", "reference_col", "reference_template", "ground_truth_json", "prompt_in_completion", "split")


def _is_function_target(target: str) -> bool:
    return target.split(":")[0].endswith(".py") or Path(target.split(":")[0]).is_file() or (":" in target and "/" not in target)


def _function_spec(checker: Callable[..., Any] | str, source: str) -> dict[str, Any]:
    if isinstance(checker, str):
        return {"kind": "function", "path": checker, "dataset": source}
    import cloudpickle

    try:
        pickled = cloudpickle.dumps(checker)
    except Exception as e:  # noqa: BLE001  (cloudpickle raises whatever the function's globals raise)
        raise AuditError(f"Cannot pass this function to a worker process ({type(e).__name__}: {e}). "
                         "Put it in a file and pass the file's path") from None
    return {"kind": "function", "pickled": base64.b64encode(pickled).decode("ascii"), "dataset": source,
            "sys_path": [p for p in sys.path if p]}


def _write_rows(rows: Iterable[Mapping[str, Any]], directory: str) -> str:
    path = os.path.join(directory, "rows.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
    return path


def _data_source(data: Any, directory: str) -> str:
    """A path or hf: name as it is; rows in memory (a list of dicts, a Dataset, a DataFrame) go to a file the workers read."""
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, str):
        return data
    if hasattr(data, "to_dict") and hasattr(data, "columns"):  # a pandas DataFrame
        data = data.to_dict("records")
    try:
        return _write_rows(data, directory)
    except (TypeError, ValueError):  # what is not iterable, and rows that are not mappings
        raise AuditError(f"Cannot read a dataset from {type(data).__name__}: pass a path, hf:org/name, or a list of dicts") from None


def _environment_spec(target: str, verifiers: str | None, log: Log | None) -> tuple[dict[str, Any], str | None, str | None]:
    """The spec, the interpreter and any verifiers pin for `owner/name` (installed from the Hub) or `name` (installed in
    this Python)."""
    spec: dict[str, Any] = {"kind": "verifiers", "env": target}
    if "/" not in target:
        if importlib.util.find_spec("verifiers") is None:
            raise AuditError(f"{target} is not a file. A scoring function is a .py file; an environment from the Hub is "
                             "owner/name; an environment of your own needs verifiers and itself installed in this Python")
        return spec, None, None
    from rewardaudit.envs import install_env

    python, info = install_env(target, verifiers=verifiers, log=log)
    name = target.split("@")[0].split("/")[-1]
    if info.get("module") and info["module"] not in (name, name.replace("-", "_"), name.replace("_", "-")):
        spec["taskset_id"] = info["module"]  # the package's module is named differently from the environment
    pinned = str(info.get("verifiers", "")).startswith("verifiers")  # install_env records a pin it applied
    return spec, str(python), info["verifiers"] if pinned else None


def audit(
    checker: Callable[..., Any] | str,
    data: Any = None,
    *,
    n: int | str | None = DEFAULT_N,
    seed: int = 0,
    prompt_col: str | None = None,
    answer_col: str | None = None,
    workers: int | None = None,
    timeout: float = 10.0,
    probes: list[str] | None = None,
    reply_format: str | None = None,
    verifiers: str | None = None,
    log: Log | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    **options: Any,
) -> AuditResult:
    """Audit one checker and return the result.

    checker: a scoring function; a file (`reward.py` or `reward.py:function`); a `verifiers` environment installed in this
      Python (`my-env`); or one from the Prime Intellect Hub (`owner/name`), which is installed into its own venv.
    data: for a scoring function, a file (jsonl, json, csv, parquet), `hf:org/name[:config[:split]]`, or rows in memory.
    n: how many rows to draw at random from the whole dataset; `"all"` or None for every row.
    probes: probe families to run, such as ["N", "O"]; all of them by default.
    reply_format: how the checker wants a reply written, if it is none of the common formats: `<sol>{answer}</sol>`.
    verifiers: for an environment, a constraint on the verifiers version, such as "<0.1.13".
    options: for a scoring function, one of style, kwargs, reference_col, reference_template, ground_truth_json,
      prompt_in_completion, split.
    """
    if isinstance(checker, Path):
        checker = str(checker)
    if not (callable(checker) or isinstance(checker, str)):
        raise AuditError(f"Cannot audit a {type(checker).__name__}: pass a scoring function, a .py file or an environment's name")
    if n == "all":
        n = None
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
        raise AuditError(f"n must be a positive whole number or 'all', got {n!r}")
    unknown = sorted(set(options) - set(_SPEC_OPTIONS))
    if unknown:
        raise AuditError(f"Unknown option {unknown[0]!r}; the options are {', '.join(_SPEC_OPTIONS)}")
    families = [f.strip().upper() for f in probes] if probes else None
    if families and not set(families) <= set(FAMILIES):  # none of an unknown family's probes would run, and the audit would read as clean
        raise AuditError(f"Unknown probe family {sorted(set(families) - set(FAMILIES))[0]!r}; the families are {', '.join(FAMILIES)}")
    if reply_format is not None and "{answer}" not in reply_format:
        raise AuditError(f"A reply format holds {{answer}} where the answer goes, as in <sol>{{answer}}</sol>; got {reply_format!r}")
    is_function = callable(checker) or _is_function_target(checker)

    with tempfile.TemporaryDirectory(prefix="rewardaudit-") as tmp:
        python = pin = None
        if isinstance(checker, str) and checker.startswith("demo:"):
            spec: dict[str, Any] = {"kind": "demo", "name": checker.split(":", 1)[1]}
        elif is_function:
            if data is None:
                raise AuditError("A scoring function needs a dataset: a jsonl, json, csv or parquet file, or hf:org/name")
            spec = _function_spec(checker, _data_source(data, tmp))
            spec.update({k: v for k, v in {"prompt_col": prompt_col, "answer_col": answer_col, **options}.items() if v})
        else:
            if data is not None:
                raise AuditError(f"{checker} is an environment and brings its own dataset; a dataset goes with a scoring "
                                 "function, as in reward.py --dataset data.jsonl")
            spec, python, pin = _environment_spec(checker, verifiers, log)
        config = AuditConfig(n_tasks=n, seed=seed, timeout_s=timeout, python=python, families=families,
                             formats=[reply_format] if reply_format else [],
                             workers=workers if workers is not None else min(4, os.cpu_count() or 1))
        result = run_audit(spec, config, log=log, progress=progress)
        if pin:
            result.meta["verifiers_pin"] = pin
        return result
