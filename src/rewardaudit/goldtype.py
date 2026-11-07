"""Infer the kind of answer a dataset expects, from the gold strings alone."""

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


_SET_RE = re.compile(
    r"\b(?:in any order|any order|order (?:does not|doesn't) matter|regardless of (?:the )?order|unordered|"
    r"as a set|the set of (?:all )?(?:the )?(?:solutions|roots|zeros|values)|"
    r"all (?:of )?(?:the )?(?:(?:possible|real|integer|complex|distinct|positive|negative|rational) )*"
    r"(?:solutions|roots|zeros|zeroes)|"
    r"all (?:of )?(?:the )?(?:(?:possible|real|integer|complex|distinct|positive|negative|rational) )*values\b"
    r"(?! of \w+ and\b))", re.I)
_ORDERED_RE = re.compile(
    r"\b(?:ordered (?:pair|triple|tuple|list|sequence|set|quadruple|\w+-tuple)s?|"
    r"in (?!any\b|no\b)(?:\w+ ){0,2}order\b(?!\s+to\b)|sorted (?:list|order|in|by|from)\b|"
    r"sort (?:them|the (?:list|numbers|values|elements))\b|ascending|descending|"
    r"from (?:smallest|least|lowest|largest|greatest|highest) to|"
    r"(?:output|give|write|return|enter|list|print|provide|find) (?:a |the )?(?:\w+ )?sequence\b)", re.I)


def order_semantics(prompt: str) -> str | None:
    """'set' if the prompt says order does not matter, 'ordered' if it says order matters, else None
    (also when it says both)."""
    s, o = bool(_SET_RE.search(prompt)), bool(_ORDERED_RE.search(prompt))
    return "set" if s and not o else "ordered" if o and not s else None


_ANSWER_ONLY_RE = re.compile(
    r"\b(?:return|output|reply|respond|(?<!your final )answer|give|provide|write|state|print)\b"
    r"[^.\n]{0,30}?\b(?:only|just)\b|"
    r"\bonly\s+(?!one\b)(?:\w+\s+){0,3}?(?:answer|result|number|value|letter|code|option|choice)\b|"
    r"\bnothing else\b|\bno (?:other|additional) text\b|\bno explanations?\b|\bwithout (?:any )?explanations?\b|"
    r"\bdo not (?:explain|include (?:any )?(?:explanation|other text|extra))|\bdon't explain", re.I)
# A sentence that limits the inside of a tag or box ('Do not explain your reasoning inside the answer tags; provide only
# the final answer') or asks for reasoning too ('reason step by step, then ONLY give the letter') leaves the reply free.
_REPLY_FREE_RE = re.compile(
    r"\b(?:do not|don't)\b.*\b(?:inside|within|in)\s+(?:the\s+|your\s+)?(?:\S+\s+){0,2}?(?:tags?\b|box\b|\\boxed)|"
    r"\bstep[- ]by[- ]step\b|\breasoning process\b|\bafter your explanation\b", re.I)


def asks_answer_only(task: Task) -> bool:
    """The prompt asks for only the answer, so a right answer with prose around it may rightly be refused. Read
    sentence by sentence, since the pattern's \\s would otherwise join 'only' to an unrelated next line."""
    text = "\n".join(str(m.get("content")) for m in task.chat())
    sentences = (s for line in text.splitlines() for s in re.split(r"(?<=[.!?])\s+", line))
    return any(_ANSWER_ONLY_RE.search(s) and not _REPLY_FREE_RE.search(s) for s in sentences)


_QUANTITY_RE = re.compile(
    r"\bhow (?:many|much|long|far|old|tall|high|fast|often)\b|\b(?:the|total) number of\b(?!\s+the\b)|"
    r"\banswer with (?:a|one|a single) number\b", re.I)
_IDENTIFIER_RE = re.compile(r"\b(?:versions?|index(?:es)?|indices|identifiers?)\b", re.I)


def asks_quantity(task: Task) -> bool | None:
    """True if the question asks for a quantity ('How many people', 'Answer with a number'), False if it names a version
    or an index, None if it says neither. Read from the prompt's last lines, which hold the question and the form of the
    answer; a passage before them can mention anything."""
    tail = "\n".join([ln for ln in task.prompt.splitlines() if ln.strip()][-3:])
    return True if _QUANTITY_RE.search(tail) else False if _IDENTIFIER_RE.search(tail) else None


_INTEGER_RE = re.compile(r"\b(?:answer|respond|reply|output|return|give|provide)\w*\b[^.\n]{0,30}\binteger\b"
                         r"|\b(?:as|must be|should be) an integer\b|\binteger only\b", re.I)


def asks_integer(task: Task) -> bool:
    """The form of the answer is an integer ('Answer with one integer'): 2.0 is then another form than the one asked for."""
    return bool(_INTEGER_RE.search("\n".join([ln for ln in task.prompt.splitlines() if ln.strip()][-3:])))


def infer_dataset_type(tasks: list[Task]) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter(infer_gold_type(t.answer) for t in tasks)
    major, _ = counts.most_common(1)[0]
    if major in ("int", "float", "fraction") and counts["latex"] >= len(tasks) * 0.3:
        major = "latex"
    return major, counts


def is_numeric(gold_type: str) -> bool:
    return gold_type in ("int", "float", "fraction")


def mcq_options(prompt: str) -> list[str]:
    """Letters of options listed in the prompt, e.g. 'A. ...', '(B) ...', 'C) ...'."""
    letters = re.findall(r"(?m)^\s*\(?([A-J])[\.\):]\s+\S", prompt)
    seen: list[str] = []
    for ch in letters:
        if ch not in seen:
            seen.append(ch)
    return seen if len(seen) >= 2 else []
