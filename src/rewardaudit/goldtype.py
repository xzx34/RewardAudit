"""Infer the kind of answer a dataset expects, from its reference answers; a row's prompt decides only whether a T or
an F is an option letter."""

from __future__ import annotations

import re
from collections import Counter

from rewardaudit.equiv import (
    outer_brackets,
    parse_json,
    split_top,
    strip_wrappers,
    to_bool,
    to_letter,
    to_number,
)
from rewardaudit.prompt import mcq_options
from rewardaudit.types import Task

# A command of two or more letters (\frac, \mathbb) or an escaped symbol (\%, \{, \\). A backslash before one letter is
# something else: a lambda binder (\x0.x0), an escape sequence (\n, \d).
_TEX_BACKSLASH = re.compile(r"\\(?:[A-Za-z]{2,}|[^A-Za-z\s])")


def infer_gold_type(answer: str) -> str:
    a = strip_wrappers(answer)
    if not a:
        return "short_str"
    if to_bool(a) is not None:
        return "bool"
    if to_letter(a) is not None:
        return "mcq"
    if re.fullmatch(r"[+-]?\d+(,\d{3})*", a):
        return "int"
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?", a) or re.fullmatch(r"[+-]?\d+(\.\d+)?%", a):
        return "float"
    if re.fullmatch(r"-?\d+\s*/\s*\d+", a) or re.fullmatch(r"-?\\[dt]?frac\{-?\d+\}\{\d+\}", a):
        return "fraction"
    v = parse_json(a)
    if isinstance(v, dict):
        return "json"
    if isinstance(v, list):
        if _json_grid(v):
            return "grid"
        return "list" if v and all(_scalar(x) for x in v) else "json"
    if _text_grid(strip_wrappers(answer, keep_punct=True)):
        return "grid"
    if "\n" in a and re.search(r"\b(def|class|return|import|function|#include|public static)\b", a):
        return "code"
    if (_TEX_BACKSLASH.search(a) or re.search(r"\^|_\{|(?<![A-Za-z])[A-Za-z0-9]_\d", a)
            or re.search(r"(?<![A-Za-z])(sqrt|\d*pi)(?![A-Za-z])", a) and len(a) < 40):
        return "latex"
    if _text_list(strip_wrappers(answer, keep_parens=True)):
        return "list"
    if len(a.split()) <= 6 and "\n" not in a:
        return "short_str"
    return "long_text"


def _scalar(x: object) -> bool:
    return x is None or isinstance(x, (bool, int, float)) or (isinstance(x, str) and len(x) <= 40)


def _json_grid(v: list) -> bool:
    return (len(v) >= 2 and all(isinstance(r, list) and r and all(_scalar(c) for c in r) for r in v)
            and len({len(r) for r in v}) == 1)


_CELL_CHARS = re.compile(r"[0-9.#*xXoO_+-]+")


def _text_grid(a: str) -> bool:
    """Two or more rows with the same number of short cells, or equal-length rows of single symbols."""
    rows = [ln.strip() for ln in a.strip().splitlines() if ln.strip()]
    if len(rows) < 2:
        return False
    if all(re.search(r"\s", r) for r in rows):
        cells = [r.split() for r in rows]
        return len({len(c) for c in cells}) == 1 and len(cells[0]) >= 2 and all(len(t) <= 3 for c in cells for t in c)
    return len({len(r) for r in rows}) == 1 and len(rows[0]) >= 2 and all(_CELL_CHARS.fullmatch(r) for r in rows)


def _text_list(a: str) -> bool:
    """A one-line comma-separated list of numbers or single symbols, or any short items inside [...]."""
    if "\n" in a or to_number(a) is not None:
        return False
    ob = outer_brackets(a)
    if ob is None:  # unbalanced or mismatched brackets, e.g. an interval [1, 2)
        return False
    items = split_top(ob[1])
    if len(items) < 2 or len(items) > 50 or not all(items) or any(len(x) > 40 for x in items):
        return False
    if all(to_number(x) is not None or re.fullmatch(r"[A-Za-z0-9]", x) for x in items):
        return True
    return ob[0] == "[" and all("," not in x for x in items)


def infer_task_type(task: Task) -> str:
    """The kind of this row's answer, with its prompt to go by: T and F are true and false, unless they are option letters
    listed there (with ten options, F is one)."""
    kind = infer_gold_type(task.answer)
    letter = strip_wrappers(task.answer).strip().upper()
    return "mcq" if kind == "bool" and letter in mcq_options(task.prompt) else kind


def infer_dataset_type(tasks: list[Task]) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter(infer_gold_type(t.answer) for t in tasks)
    major, _ = counts.most_common(1)[0]
    if major in ("int", "float", "fraction") and counts["latex"] >= len(tasks) * 0.3:
        major = "latex"
    return major, counts


def is_numeric(gold_type: str) -> bool:
    return gold_type in ("int", "float", "fraction")
