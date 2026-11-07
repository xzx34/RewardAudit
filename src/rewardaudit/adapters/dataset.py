"""Load every row of a dataset as tasks: jsonl / json / csv / parquet files or Hugging Face datasets.

Columns may be dotted paths into nested values (verl parquet files keep the gold in reward_model.ground_truth), and a number
in a path picks an item of a list (answers.text.0).
A structured gold (dict or list) becomes JSON text in Task.answer; its parent column stays in Task.info.
A row whose gold is empty is left out (SQuAD 2.0 keeps its unanswerable questions as rows with no answer).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rewardaudit.adapters.base import unique_ids
from rewardaudit.errors import AuditError
from rewardaudit.types import Task

PROMPT_COLS = ("prompt", "question", "problem", "input", "query", "instruction", "text")
ANSWER_COLS = ("answer", "solution", "ground_truth", "target", "label", "final_answer", "output", "reference",
               "reward_model.ground_truth")  # the last one: verl parquet files
FORMATS = ".jsonl, .json, .csv, .parquet or hf:org/name[:config[:split]]"
MISSING = object()


def get_path(row: dict[str, Any], key: str) -> Any:
    """row[key], or the nested value of a dotted key ('reward_model.ground_truth', 'answers.text.0'); MISSING if absent.
    An item past the end of a list is None: a row whose list of answers is empty has no answer, and that is no missing column."""
    if key in row:
        return row[key]
    cur: Any = row
    for part in key.split("."):
        if isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return MISSING
    return cur


def _no_answer(value: Any) -> bool:
    """A gold that holds no answer: None (an item past the end of a list too), a blank string or an empty list. Read
    before `answer_text`, which writes an empty list as '[]'."""
    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, list) and not value)


def answer_text(value: Any) -> str:
    """The gold as text: strings as they are, structured values (dict / list) as JSON, None as ''."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value if isinstance(value, str) else str(value)


def _json_lines(p: Path) -> list[dict[str, Any]]:
    rows = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    raise AuditError(f"{p}: line {i} is not valid JSON. Fix that line: a .jsonl file holds one JSON object per line") from None
    return rows


def _rows_from_path(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in (".jsonl", ".json", ".csv", ".parquet"):
        raise AuditError(f"Cannot read {path}: the dataset must be {FORMATS}")
    if not p.exists():
        raise AuditError(f"Dataset not found: {path}. Check the path, or name a Hugging Face dataset as hf:org/name")
    if suffix == ".jsonl":
        return _json_lines(p)
    if suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            raise AuditError(f"{p} is not valid JSON; if it holds one object per line, name it .jsonl") from None
        if isinstance(data, dict):
            data = next((data[k] for k in ("data", "rows", "examples", "tasks") if isinstance(data.get(k), list)), None)
        if not isinstance(data, list):
            raise AuditError(f"{p} must hold a list of rows, or an object with a 'data' list")
        return data
    if suffix == ".csv":
        with p.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise AuditError("Reading parquet needs pyarrow: pip install 'rewardaudit[data]'") from None
    return pq.read_table(path).to_pylist()


def _rows_from_hf(spec: str, split: str | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise AuditError("Loading Hugging Face datasets needs datasets: pip install 'rewardaudit[data]'") from None
    parts = spec.split(":")
    config = parts[1] if len(parts) > 1 and parts[1] else None
    return [dict(r) for r in load_dataset(parts[0], config, split=parts[2] if len(parts) > 2 else (split or "train"))]


def _pick(rows: list[dict[str, Any]], explicit: str | None, candidates: tuple[str, ...], flag: str, what: str) -> str:
    row = rows[0]
    if explicit:
        if get_path(row, explicit) is MISSING:
            raise AuditError(f"The dataset has no column {explicit!r}. Its columns: {', '.join(map(str, row))}")
        return explicit
    for c in candidates:
        if get_path(row, c) is not MISSING:
            return c
    raise AuditError(f"Could not tell which column holds the {what}. The dataset's columns: {', '.join(map(str, row))}. "
                     f"Name it with {flag}")


def _render_prompt(value: Any) -> tuple[str, list[dict[str, Any]] | None]:
    """A prompt column may be a string or a chat message list."""
    if isinstance(value, list) and value and isinstance(value[0], dict) and "role" in value[0]:
        user_turns = [m.get("content", "") for m in value if m.get("role") == "user"]
        text = user_turns[-1] if user_turns else "\n".join(str(m.get("content", "")) for m in value)
        return str(text), value
    return str(value), None


def load_tasks(
    source: str,
    *,
    prompt_col: str | None = None,
    answer_col: str | None = None,
    split: str | None = None,
) -> tuple[list[Task], int]:
    """The rows that have a reference answer, as tasks, and how many rows were left out for having none. A row keeps its
    number (or id) in the file either way."""
    rows = _rows_from_hf(source[3:], split) if source.startswith("hf:") else _rows_from_path(source)
    if not rows:
        raise AuditError(f"The dataset {source} is empty")
    odd = next((r for r in rows if not isinstance(r, dict)), None)
    if odd is not None:
        raise AuditError(f"The rows of {source} must be objects with named columns, got {type(odd).__name__}")
    pc = _pick(rows, prompt_col, PROMPT_COLS, "--prompt-col", "prompt")
    ac = _pick(rows, answer_col, ANSWER_COLS, "--answer-col", "reference answer")
    tasks: list[Task] = []
    ids = unique_ids([str(r.get("id", i)) for i, r in enumerate(rows)])
    for i, (row, tid) in enumerate(zip(rows, ids, strict=True)):
        prompt_value, answer = get_path(row, pc), get_path(row, ac)
        if prompt_value is MISSING or answer is MISSING:
            raise AuditError(f"Row {i} of {source} has no column {pc if prompt_value is MISSING else ac!r}")
        if _no_answer(answer):
            continue
        prompt, messages = _render_prompt(prompt_value)
        info = {k: v for k, v in row.items() if k not in (pc, ac)}
        tasks.append(Task(id=tid, prompt=prompt, answer=answer_text(answer), messages=messages, info=info))
    if not tasks:
        raise AuditError(f"All {len(rows):,} rows of {source} have an empty reference answer in {ac!r}. "
                         "If another column holds it, name it with --answer-col")
    return tasks, len(rows) - len(tasks)
