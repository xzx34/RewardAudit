"""Probes for structured golds: lists (M8 M9 M10 Q10), JSON objects (M11 M12 M13), grids (M17), and lists or grids with
k items wrong (M28).

Every variant is written in the gold's own spelling (separators, spacing, brackets), so the only thing
that differs from the gold is the one change the probe makes. Whether element order matters is read from
the prompt (`order_semantics`); when the prompt does not say, no order change is sent.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from rewardaudit.equiv import (
    equiv,
    outer_brackets,
    parse_json,
    split_top,
    strip_wrappers,
    value_number,
    values_equal,
)
from rewardaudit.probes.base import Probe, ProbeContext, fmt_num
from rewardaudit.prompt import order_semantics
from rewardaudit.types import Label, Task

NS, SOL = Label.NOT_SOLVING, Label.SOLVING


def _raw(task: Task, keep_punct: bool = False) -> str:
    return strip_wrappers(task.answer, keep_parens=True, keep_punct=keep_punct)


def _style(like: str) -> tuple[int | None, str, str]:
    """(indent, item separator, key separator) of a JSON text; with no comma, items follow the key spacing."""
    m = re.search(r"\n([ \t]+)\S", like)
    spaced_kv = bool(re.search(r":\s", like))
    spaced_item = bool(re.search(r",\s", like)) if "," in like else spaced_kv
    return len(m.group(1)) if m else None, "," + " " * spaced_item, ":" + " " * spaced_kv


def dumps_like(obj: object, like: str) -> str:
    """json.dumps in the spacing style of `like`."""
    indent, item, kv = _style(like)
    if indent:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    return json.dumps(obj, separators=(item, kv), ensure_ascii=False)


@dataclass
class Seq:
    """A list gold: raw element texts, their values, and how to write a modified list in the same style."""

    open: str
    tokens: list[str]
    values: list
    sep: str
    close: str
    is_json: bool

    def write(self, tokens: list[str]) -> str:
        return self.open + self.sep.join(tokens) + self.close

    def unique(self, i: int) -> bool:
        return sum(values_equal(self.values[i], v) for v in self.values) == 1


def parse_list_gold(raw: str) -> Seq | None:
    if "\n" in raw.strip():
        return None
    v = parse_json(raw)
    if v is not None and not isinstance(v, list):
        return None
    ob = outer_brackets(raw.strip())
    if ob is None:
        return None
    o, inner, c = ob
    tokens = split_top(inner)
    if len(tokens) < 2 or not all(tokens):
        return None
    if v is not None and len(v) != len(tokens):  # commas inside JSON strings
        return None
    sep = ", " if re.search(r",\s", inner) else ","
    return Seq(o, tokens, v if v is not None else tokens, sep, c, v is not None)


def _seq(task: Task, ctx: ProbeContext) -> Seq | None:
    return parse_list_gold(_raw(task)) if ctx.gtype(task) == "list" else None


def _chat_text(task: Task) -> str:
    parts = [m.get("content") for m in (task.messages or []) if isinstance(m.get("content"), str)]
    return "\n".join([*parts, task.prompt])


def semantics(task: Task, ctx: ProbeContext) -> str | None:
    """'set' / 'ordered' / None from the prompt; a parenthesised gold is a tuple, never a set."""
    s = order_semantics(_chat_text(task))
    seq = _seq(task, ctx)
    if s == "set" and seq is not None and seq.open == "(":
        return None
    return s


def _swap(task: Task, rng: random.Random, ctx: ProbeContext) -> list[tuple[str, str | None]]:
    seq = _seq(task, ctx)
    if seq is None:
        return []
    pairs = [(i, i + 1) for i in range(len(seq.tokens) - 1) if not values_equal(seq.values[i], seq.values[i + 1])]
    if not pairs:
        return []
    i, j = rng.choice(pairs)
    toks = list(seq.tokens)
    toks[i], toks[j] = toks[j], toks[i]
    return [("swap", seq.write(toks))]


def _drop(task: Task, rng: random.Random, ctx: ProbeContext) -> list[tuple[str, str | None]]:
    seq = _seq(task, ctx)
    idx = [i for i in range(len(seq.tokens)) if seq.unique(i)] if seq else []
    if not idx:
        return []
    i = rng.choice(idx)
    return [("drop", seq.write(seq.tokens[:i] + seq.tokens[i + 1 :]))]


def _spelled(seq: Seq, x: str) -> str:
    """`x` as the list spells its elements: a string in a JSON list keeps its quotes."""
    return json.dumps(x) if seq.is_json and isinstance(seq.values[0], str) else x


def _new_element(seq: Seq, task: Task, ctx: ProbeContext) -> str | None:
    nums = [value_number(v) for v in seq.values]
    if all(n is not None for n in nums):
        cand = fmt_num(max(nums) + 1)
        return _spelled(seq, cand) if not any(values_equal(cand, v) for v in seq.values) else None
    if all(isinstance(v, str) and re.fullmatch(r"[A-Za-z]", v) for v in seq.values):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if seq.values[0].isupper() else "abcdefghijklmnopqrstuvwxyz"
        ch = next((ch for ch in alphabet if ch not in seq.values), None)
        return _spelled(seq, ch) if ch else None
    for t in ctx.tasks:  # an element of another task's list gold
        other = parse_list_gold(_raw(t)) if t.id != task.id and ctx.gtype(t) == "list" else None
        if other is None or other.is_json != seq.is_json:
            continue
        for tok, v in zip(other.tokens, other.values, strict=True):
            if not any(values_equal(v, x) for x in seq.values):
                return tok
    return None


def _add(task: Task, rng: random.Random, ctx: ProbeContext) -> list[tuple[str, str | None]]:
    seq = _seq(task, ctx)
    new = _new_element(seq, task, ctx) if seq else None
    return [("add", seq.write([*seq.tokens, new]))] if new else []


def _replace(task: Task, rng: random.Random, ctx: ProbeContext) -> list[tuple[str, str | None]]:
    seq = _seq(task, ctx)
    new = _new_element(seq, task, ctx) if seq else None
    if not new:
        return []
    i = rng.randrange(len(seq.tokens))
    return [("replace", seq.write([*seq.tokens[:i], new, *seq.tokens[i + 1 :]]))]


def _permute(task: Task, rng: random.Random, ctx: ProbeContext) -> list[tuple[str, str | None]]:
    seq = _seq(task, ctx)
    if seq is None:
        return []
    n, out = len(seq.tokens), []
    for name, order in (("reversed", list(range(n))[::-1]), ("rotated", [*range(1, n), 0])):
        if any(not values_equal(seq.values[i], seq.values[j]) for i, j in enumerate(order)):
            comp = seq.write([seq.tokens[i] for i in order])
            if comp not in [c for _, c in out]:
                out.append((name, comp))
    return out


# --- JSON objects --------------------------------------------------------------------------------------------

def _obj(task: Task, ctx: ProbeContext) -> dict | None:
    if ctx.gtype(task) != "json":
        return None
    v = parse_json(_raw(task))
    return v if isinstance(v, dict) and v else None


def _numeric_paths(v: object, path: tuple = ()) -> list[tuple]:
    if isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [path]
    items = v.items() if isinstance(v, dict) else enumerate(v) if isinstance(v, list) else []
    return [p for k, x in items for p in _numeric_paths(x, (*path, k))]


def _changed(x: object) -> object:
    """A value that differs from `x` beyond any numeric tolerance."""
    if isinstance(x, bool):
        return not x
    if isinstance(x, (int, float)):
        return abs(x) * 2 + 10
    if isinstance(x, str):
        return x + "x"
    return None


def _set_path(v: object, path: tuple, value: object) -> object:
    if not path:
        return value
    out = dict(v) if isinstance(v, dict) else list(v)
    out[path[0]] = _set_path(v[path[0]], path[1:], value)
    return out


def _get_path(v: object, path: tuple) -> object:
    for k in path:
        v = v[k]
    return v


def _fresh_key(obj: dict, base: str) -> str:
    k = base
    while k in obj:
        k += "_x"
    return k


def _number(task, rng, ctx):
    obj = _obj(task, ctx)
    paths = _numeric_paths(obj) if obj else []
    if not paths:
        return []
    p = rng.choice(paths)
    return [("value", dumps_like(_set_path(obj, p, _changed(_get_path(obj, p))), _raw(task)))]


def _drop_field(task, rng, ctx):
    obj = _obj(task, ctx)
    if not obj:
        return []
    k = rng.choice(list(obj))
    return [("drop_field", dumps_like({a: b for a, b in obj.items() if a != k}, _raw(task)))]


def _rename_key(task, rng, ctx):
    obj = _obj(task, ctx)
    if not obj:
        return []
    k = rng.choice(list(obj))
    new = _fresh_key(obj, f"{k}_x")
    return [("rename", dumps_like({new if a == k else a: b for a, b in obj.items()}, _raw(task)))]


# --- grids ---------------------------------------------------------------------------------------------------

def _cell_spans(raw: str) -> list[tuple[int, int]]:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    pat = r"[^\s,\[\]()]+" if any(re.search(r"[\s,]", ln.strip()) for ln in lines) else r"\S"
    return [m.span() for m in re.finditer(pat, raw)]


def _other_cell(cell: str, alphabet: list[str]) -> str:
    """Another symbol from the grid's own alphabet (0 <-> 1 for binary grids), so a checker that validates its cells
    still compares this one. In a grid of one symbol: the next integer, which keeps a 9 a single digit."""
    if set(alphabet) <= {"0", "1"}:
        return "1" if cell == "0" else "0"
    others = [a for a in alphabet if a != cell]
    if others:
        return others[0]
    if re.fullmatch(r"-?\d+", cell):
        return "8" if cell == "9" else str(int(cell) + 1)
    return "0"


def _grid(task: Task, ctx: ProbeContext) -> tuple[str, list | None, list]:
    """A grid gold's text, its JSON value (None for a text grid) and its cells: (row, column) pairs or spans of the text."""
    if ctx.gtype(task) != "grid":
        return "", None, []
    raw = _raw(task, keep_punct=True)
    v = parse_json(raw)
    if isinstance(v, list):
        return raw, v, [(r, c) for r, row in enumerate(v) for c in range(len(row))]
    return raw, None, _cell_spans(raw)


def _change_cells(raw: str, v: list | None, cells: list) -> str:
    """The grid with each of `cells` changed to another value of its own kind, in the grid's spelling."""
    if v is not None:
        grid = [list(row) for row in v]
        alphabets = {t: sorted({str(y) for row in v for y in row if type(y) is t}) for t in {type(y) for row in v for y in row}}
        for r, c in cells:
            x = v[r][c]
            grid[r][c] = _changed(x) if x is None or isinstance(x, bool) else type(x)(_other_cell(str(x), alphabets[type(x)]))
        return dumps_like(grid, raw)
    alphabet, out = sorted({raw[s:e] for s, e in _cell_spans(raw)}), raw
    for a, b in sorted(cells, reverse=True):  # from the end, so the spans before it stay where they are
        out = out[:a] + _other_cell(raw[a:b], alphabet) + out[b:]
    return out


def _cell(task, rng, ctx):
    raw, v, cells = _grid(task, ctx)
    return [("cell", _change_cells(raw, v, [rng.choice(cells)]))] if cells else []


# --- partly wrong: k items of a list or grid replaced ----------------------------------------------------------

STEPS = 8  # variants per row at most
NEW = 8  # new elements for a list at most: telling one from each element can take milliseconds where they are strings
ALL = "all"  # the variant with every item replaced


def _items(task: Task, ctx: ProbeContext) -> int:
    seq = _seq(task, ctx)
    return len(seq.tokens) if seq else len(_grid(task, ctx)[2])


def _steps(n: int) -> list[int]:
    """How many items to replace: each number up to n when there are few, else close together at both ends."""
    return list(range(1, n + 1)) if n <= STEPS else sorted({1, 2, 3, n // 2, n - 2, n - 1, n})


def _new_elements(seq: Seq, task: Task, ctx: ProbeContext, k: int) -> list[str]:
    """Up to k elements that are not in the list and differ from each other, by `_new_element` on the list so far."""
    out: list[str] = []
    while len(out) < k and (new := _new_element(seq, task, ctx)) is not None:
        out.append(new)
        seq = Seq(seq.open, [*seq.tokens, new], [*seq.values, json.loads(new) if seq.is_json else new], seq.sep, seq.close,
                  seq.is_json)
    return out


def _partly_wrong(task, rng, ctx):
    """The same random order of items for every k, so each variant holds the wrong items of the one before. A list takes
    new elements as M10 does, used in turn when there are fewer than the items replaced; a grid each cell's other value, as
    M17 does."""
    seq = _seq(task, ctx)
    n = _items(task, ctx)
    steps = [(f"k{k}" if k < n else ALL, k) for k in _steps(n)]
    order = rng.sample(range(n), n)
    if seq is None:
        raw, v, cells = _grid(task, ctx)
        return [(name, _change_cells(raw, v, [cells[i] for i in order[:k]])) for name, k in steps]
    news = _new_elements(seq, task, ctx, min(n, NEW))
    if not news:
        return []
    out = []
    for name, k in steps:
        toks = list(seq.tokens)
        for j, i in enumerate(order[:k]):
            toks[i] = news[j % len(news)]
        out.append((name, seq.write(toks)))
    return out


Maker = Callable[[Task, random.Random, ProbeContext], list[tuple[str, str | None]]]


def _not_gold(make: Maker, gold_type: str | None = None):
    """Wrap each variant; drop any that is equivalent to the gold (as a set for `gold_type='set'`)."""
    def run(task, rng, ctx):
        gt = gold_type or ctx.gtype(task)
        return [(n, ctx.wrap(c)) for n, c in make(task, rng, ctx) if c and not equiv(c, task.answer, gt)]
    return run


def _wrapped(make: Maker):
    return lambda task, rng, ctx: [(n, ctx.wrap(c)) for n, c in make(task, rng, ctx) if c]


def structured_wrong(task: Task, ctx: ProbeContext, rng: random.Random) -> list[str]:
    """Definitely-wrong answers for structured golds: one list element replaced (a dropped one would be a subset of the
    gold), one grid cell changed, and one JSON value changed or field dropped where the prompt asks for JSON. Otherwise
    the fields are not the answer and the gold less one field can still be right: the caller uses another task's gold."""
    gt = ctx.gtype(task)
    makers = {"list": [_replace], "json": [_number, _drop_field] if _obj_asked(task, ctx) else [],
              "grid": [_cell]}.get(gt, [])
    sem = "set" if gt == "list" else gt
    return [c for make in makers for _, c in make(task, rng, ctx) if c and not equiv(c, task.answer, sem)]


def _seq_applies(pred: Callable[[str | None], bool]):
    return lambda t, c: _seq(t, c) is not None and pred(semantics(t, c))


def _obj_asked(task: Task, ctx: ProbeContext) -> bool:
    """A JSON object gold on a task whose prompt asks for JSON: otherwise its keys are not part of the answer."""
    return _obj(task, ctx) is not None and "json" in " ".join(str(m.get("content")) for m in task.chat()).lower()


PROBES_M = [
    Probe("M8", "M", "two list elements swapped where order matters", "natural", "zero", NS, _not_gold(_swap),
          applies=_seq_applies(lambda s: s == "ordered"),
          description="Sends the reference list with two adjacent elements swapped, only on tasks whose prompt "
                      "asks for an ordered answer; a sound checker gives zero."),
    Probe("M9", "M", "one list element dropped", "natural", "zero", NS, _not_gold(_drop, "set"),
          applies=_seq_applies(lambda s: True),
          description="Sends the reference list with one (non-repeated) element removed; a sound checker gives "
                      "zero."),
    Probe("M10", "M", "one list element added", "natural", "zero", NS, _not_gold(_add, "set"),
          applies=_seq_applies(lambda s: True),
          description="Sends the reference list with one extra element that is not already in it; a sound "
                      "checker gives zero."),
    Probe("M11", "M", "one JSON number changed", "natural", "zero", NS, _not_gold(_number),
          applies=lambda t, c: bool(_numeric_paths(_obj(t, c) or {})),
          description="Sends the reference JSON object with one numeric value changed; a checker that only "
                      "validates the keys or the schema accepts it, a sound one gives zero."),
    Probe("M12", "M", "one JSON field dropped", "natural", "zero", NS, _not_gold(_drop_field),
          applies=_obj_asked,
          description="Sends the reference JSON object with one top-level field removed, only on tasks whose "
                      "prompt asks for JSON; a sound checker gives zero."),
    Probe("M13", "M", "one JSON key renamed", "natural", "zero", NS, _not_gold(_rename_key),
          applies=_obj_asked,
          description="Sends the reference JSON object with one key renamed (its value kept), only on tasks whose "
                      "prompt asks for JSON; a checker that compares only values accepts it, a sound one gives "
                      "zero."),
    Probe("M17", "M", "one grid cell changed", "natural", "zero", NS, _not_gold(_cell),
          applies=lambda t, c: c.gtype(t) == "grid",
          description="Sends the reference grid with exactly one cell changed to another value the grid uses; a "
                      "sound checker gives zero."),
    Probe("M28", "M", "k items of a list or grid wrong", "natural", "zero", NS, _not_gold(_partly_wrong),
          applies=lambda t, c: _items(t, c) >= 3, headline=False, sweep=True,
          description="Where the reference answer is a list or a grid of 3 items or more, sends it with k of them replaced "
                      "by wrong ones, as M10 and M17 make them, from one item up to all (at most 8 steps, closer together "
                      "at both ends). Its finding gives the acceptance rate of each k, and the mean share of the score "
                      "where scores fall between nothing and full; its severity is that of the reply with every item "
                      "wrong. A sound checker gives zero."),
]

PROBES_Q = [
    Probe("Q10", "Q", "set elements in another order", "natural", "full", SOL, _wrapped(_permute),
          applies=_seq_applies(lambda s: s == "set"),
          description="Sends the reference list reordered, only on tasks whose prompt says order does not matter "
                      "(\"in any order\", \"all solutions\"); a sound checker gives full credit."),
]
