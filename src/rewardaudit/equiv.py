"""Lenient equivalence between answer strings.

Deliberately generous: anything that *might* be the same answer is treated as the same.
Over-accepting only loses recall (a finding goes unreported); under-accepting would create
false findings, which is the one thing an audit must not do.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections.abc import Iterable
from fractions import Fraction
from functools import lru_cache

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000,
}  # fmt: skip

_BOOL_TRUE = {"true", "yes", "y", "t", "correct", "right"}
_BOOL_FALSE = {"false", "no", "n", "f", "incorrect", "wrong"}
_BOOL_OPPOSITE = {a: b for x, y in (("true", "false"), ("yes", "no"), ("y", "n"), ("t", "f"), ("correct", "incorrect"), ("right", "wrong"))
                  for a, b in ((x, y), (y, x))}

_FRAC_RE = re.compile(r"^(-?)\\?[dt]?frac\{?(-?\d+)\}?\{?(\d+)\}?$")
_SLASH_RE = re.compile(r"^(-?)(\d+)\s*/\s*(\d+)$")
_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_PCT_RE = re.compile(r"^([+-]?\d+(\.\d+)?)\s*%$")
_LETTER_RE = re.compile(r"^\(?([a-j])\)?\.?$")
_UNIT_RE = re.compile(
    r"(cm|mm|km|m|kg|g|s|hours?|hrs?|minutes?|mins?|days?|units?|dollars?|cents?|degrees?|°|deg|percent|"
    r"miles?|feet|ft|inches?|in|pounds?|lbs?|years?|yrs?|people|students|meters?|liters?|ml|mph)$"
)


def strip_wrappers(s: str, keep_parens: bool = False, keep_punct: bool = False) -> str:
    """Remove outer \\boxed{}, $…$, **…**, quotes, <answer>…</answer>, trailing punctuation.
    `keep_parens` keeps an outer (…) pair (tuples, intervals); `keep_punct` keeps trailing punctuation (grids)."""
    s = s.strip()
    patterns = [
        r"\\boxed\s*\{(.*)\}",
        r"\\boxed\s+(.*)",
        r"<answer>(.*)</answer>",
        r"\$+(.*?)\$+",
        r"\*\*(.*)\*\*",
        r"\\\((.*)\\\)",
        r"\\\[(.*)\\\]",
        r"```[a-z]*\n?(.*?)```",
        r"\\text\{(.*)\}",
    ]
    for _ in range(6):
        before = s
        for pat in patterns:
            m = re.fullmatch(pat, s, flags=re.S | re.I)
            if m:
                s = m.group(1).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
            s = s[1:-1].strip()
        if not keep_parens and len(s) >= 2 and s[0] == "(" and s[-1] == ")" and s.count("(") == 1:
            s = s[1:-1].strip()
        if not keep_punct:
            s = s.rstrip(".。;:!,").strip()
        if s == before:
            break
    return s


def _brace_shorthand(s: str) -> str:
    """TeX shorthand arguments braced: \\frac34 -> \\frac{3}{4}, \\frac8{21} -> \\frac{8}{21}, \\sqrt3 -> \\sqrt{3}."""
    s = re.sub(r"\\(frac|binom)\s*(\d)\s*(\d)", r"\\\1{\2}{\3}", s)
    s = re.sub(r"\\(frac|binom)\s*(\d)\s*\{", r"\\\1{\2}{", s)
    s = re.sub(r"\\(frac|binom)\s*(\{[^{}]*\})\s*(\d)", r"\\\1\2{\3}", s)
    return re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)


def normalize(s: str) -> str:
    s = strip_wrappers(str(s))
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = _brace_shorthand(s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\[,;!: ]", "", s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathbf|textbf|mathit)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("\\$", "").replace("$", "")
    s = re.sub(r"\^\{(\w)\}", r"^\1", s)
    s = re.sub(r"_\{(\w)\}", r"_\1", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\s*([+\-*/=^_<>,;])\s*", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.rstrip(".。;:!").strip()
    return s


def to_number(s: str) -> float | None:
    """Parse a scalar from a string; None if it is not a plain number."""
    t = normalize(s).replace(" ", "")
    if not t:
        return None
    t = re.sub(r"^[a-z]{1,2}=", "", t)  # x = 42
    if not _NUM_RE.match(t):
        t = re.sub(r"^(usd|rs\.?|€|£)", "", t)
        t = _UNIT_RE.sub("", t)
    if t in NUMBER_WORDS:
        return float(NUMBER_WORDS[t])
    m = _PCT_RE.match(t)
    if m:
        return float(m.group(1)) / 100.0
    m = _FRAC_RE.match(t) or _SLASH_RE.match(t)
    if m:
        den = int(m.group(3))
        if den == 0:
            return None
        val = float(Fraction(int(m.group(2)), den))
        return -val if m.group(1) == "-" else val
    if _NUM_RE.match(t):
        try:
            return float(t)
        except ValueError:
            return None
    return None


def to_bool(s: str) -> bool | None:
    t = normalize(s)
    if t in _BOOL_TRUE:
        return True
    if t in _BOOL_FALSE:
        return False
    return None


def flip_bool(gold: str) -> str:
    """The opposite of a boolean gold in its own vocabulary and case: a checker that reads yes or no does not know 'false'."""
    s = gold.strip()
    other = _BOOL_OPPOSITE.get(s.lower())
    if other is None:
        return "false" if to_bool(gold) else "true"
    return other.upper() if s.isupper() else other.capitalize() if s[0].isupper() else other


def to_letter(s: str) -> str | None:
    m = _LETTER_RE.match(normalize(s))
    return m.group(1).upper() if m else None


_TOWER_RE = re.compile(r"\d\s*(?:\^|\*\*)\s*\{?\s*\d+\s*(?:\^|\*\*)")  # 9^{9^9}, 9**9**9: evaluation never ends


def _mv_equiv(a: str, b: str) -> bool | None:
    """math-verify, if installed. None means 'could not decide'. Strings of very different length (a 50 KB
    response against a short gold) and numeric power towers are not handed to the symbolic comparator, which
    can take seconds on them."""
    if len(a) > 1000 + 4 * len(b) or len(b) > 1000 + 4 * len(a) or _TOWER_RE.search(a) or _TOWER_RE.search(b):
        return None
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        pa = parse(a) or parse(f"${a}$")
        pb = parse(b) or parse(f"${b}$")
        if not pa or not pb:
            return None
        return bool(verify(pa, pb) or verify(pb, pa))
    except Exception:
        return None


# --- structured answers: json, list (ordered), set (unordered, only when the task says so), grid ------------

STRUCTURED = ("json", "list", "set", "grid")
_OPEN, _CLOSE = "([{", ")]}"
_PAIR = dict(zip(_OPEN, _CLOSE, strict=True))


def split_top(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside brackets."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return [p.strip() for p in parts]


def outer_brackets(s: str) -> tuple[str, str, str] | None:
    """(open, inner, close) if one bracket pair encloses all of `s`, ('', s, '') if none does,
    None if the brackets are unbalanced or mismatched (an interval such as '[1, 2)')."""
    if len(s) < 2 or s[0] not in _OPEN:
        return "", s, ""
    depth = 0
    for i, ch in enumerate(s):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        if depth == 0:
            if i < len(s) - 1:
                return "", s, ""
            return (s[0], s[1:-1], s[-1]) if ch == _PAIR[s[0]] else None
    return None


def parse_json(s: str) -> dict | list | None:
    """A JSON object or array, else None. Duplicate keys keep the last value."""
    s = s.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        v = json.loads(s)
    except (ValueError, RecursionError):
        return None
    return v if isinstance(v, (dict, list)) else None


def parse_seq(s: str) -> list | None:
    """Elements of a JSON array or of a one-line comma-separated list ('1, 2, 3', '[a, b]', '(1, 2)')."""
    s = strip_wrappers(s, keep_parens=True)
    v = parse_json(s)
    if v is not None:
        return v if isinstance(v, list) else None
    if "\n" in s:
        return None
    ob = outer_brackets(s)
    if ob is None:
        return None
    items = split_top(ob[1])
    return items if all(items) else None


def parse_grid(s: str) -> list[list] | None:
    """Rows of cells: a JSON 2-D array, or lines of space/comma-separated cells, or lines of single characters."""
    s = strip_wrappers(s, keep_punct=True)
    v = parse_json(s)
    if isinstance(v, list):
        ok = v and all(isinstance(r, list) and r and not any(isinstance(c, (list, dict)) for c in r) for r in v)
        return v if ok else None
    rows = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not rows or v is not None:
        return None
    if any(re.search(r"[\s,]", r) for r in rows):
        grid = [[c for c in re.split(r"[\s,]+", r.strip("[](), ")) if c] for r in rows]
    else:
        grid = [list(r) for r in rows]
    return grid if all(grid) else None


def value_number(x: object) -> float | None:
    """A JSON number or a numeric string as a float; None for booleans and everything else."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except OverflowError:
            return None
    return to_number(x) if isinstance(x, str) else None


def values_equal(x: object, y: object) -> bool:
    """Parsed-value equality: same keys and shape, numbers within the numeric tolerance, strings by `equiv`."""
    if isinstance(x, dict) or isinstance(y, dict):
        x, y = (parse_json(v) if isinstance(v, str) else v for v in (x, y))
        if not (isinstance(x, dict) and isinstance(y, dict)) or len(x) != len(y):
            return False
        kx, ky = {normalize(k): v for k, v in x.items()}, {normalize(k): v for k, v in y.items()}
        if len(kx) != len(x) or len(ky) != len(y):
            kx, ky = x, y
        return kx.keys() == ky.keys() and all(values_equal(v, ky[k]) for k, v in kx.items())
    if isinstance(x, list) or isinstance(y, list):
        x, y = (parse_seq(v) if isinstance(v, str) else v for v in (x, y))
        if not (isinstance(x, list) and isinstance(y, list)) or len(x) != len(y):
            return False
        return all(values_equal(a, b) for a, b in zip(x, y, strict=True))
    if isinstance(x, bool) or isinstance(y, bool):
        bx, by = (v if isinstance(v, bool) else to_bool(v) if isinstance(v, str) else None for v in (x, y))
        return bx is not None and bx == by
    if x is None or y is None:
        return all(v is None or (isinstance(v, str) and normalize(v) in ("null", "none")) for v in (x, y))
    if not (isinstance(x, str) and isinstance(y, str)):
        nx, ny = value_number(x), value_number(y)
        return nx is not None and ny is not None and math.isclose(nx, ny, rel_tol=1e-6, abs_tol=1e-9)
    return equiv(x, y)


def _multiset_eq(xs: list, ys: list) -> bool:
    if len(xs) != len(ys):
        return False
    left = list(ys)
    for a in xs:
        i = next((i for i, b in enumerate(left) if values_equal(a, b)), None)
        if i is None:
            return False
        del left[i]
    return True


def within_list(a: str, gold: str) -> bool:
    """Every element of `a` is an element of `gold`: read as a list, `a` names nothing the gold does not."""
    xs, ys = parse_seq(a), parse_seq(gold)
    return xs is not None and ys is not None and all(any(values_equal(x, y) for y in ys) for x in xs)


def structured_equiv(a: str, b: str, gold_type: str) -> bool | None:
    """Parsed comparison for json / list / set / grid answers; None if either side does not parse."""
    if gold_type == "json":
        va, vb = parse_json(strip_wrappers(a)), parse_json(strip_wrappers(b))
        return None if va is None or vb is None else values_equal(va, vb)
    if gold_type in ("list", "set"):
        sa, sb = parse_seq(a), parse_seq(b)
        if sa is None or sb is None:
            return None
        return _multiset_eq(sa, sb) if gold_type == "set" else values_equal(sa, sb)
    if gold_type == "grid":
        ga, gb = parse_grid(a), parse_grid(b)
        if ga is None or gb is None:
            return None
        return len(ga) == len(gb) and all(
            len(r) == len(q) and all(values_equal(x, y) for x, y in zip(r, q, strict=True)) for r, q in zip(ga, gb, strict=True))
    return None


def structured_fragments(text: str, gold_type: str, rows: int = 0, limit: int = 20) -> list[str]:
    """Pieces of `text` that may hold a structured answer: the whole text, balanced [...] / {...} spans,
    fenced blocks, and (for grids) the last `rows` non-empty lines."""
    out = [text]
    if gold_type in ("json", "list", "set"):
        i, n = 0, len(text)
        while i < n and len(out) <= limit:
            if text[i] not in "[{":
                i += 1
                continue
            depth, j = 0, i
            while j < n:
                if text[j] in _OPEN:
                    depth += 1
                elif text[j] in _CLOSE:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= n:  # unbalanced from here on
                break
            out.append(text[i : j + 1])
            i = j + 1
    out += re.findall(r"```[a-z]*\n?(.*?)```", text, flags=re.S)[:limit]
    if gold_type == "grid" and rows:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        out.append("\n".join(lines[-rows:]))
    return out


@lru_cache(maxsize=65536)
def equiv(a: str, b: str, gold_type: str | None = None) -> bool:
    """True if `a` and `b` plausibly denote the same answer. For json / list / grid golds both sides are parsed
    and compared as values (lists in order); gold_type 'set' compares lists ignoring order."""
    if a is None or b is None:
        return False
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    if gold_type in STRUCTURED:
        r = structured_equiv(a, b, gold_type)
        if r is not None:
            return r
    xa, xb = to_number(a), to_number(b)
    if xa is not None and xb is not None:
        return math.isclose(xa, xb, rel_tol=1e-6, abs_tol=1e-9)
    if xa is not None or xb is not None:
        if gold_type in ("int", "float", "fraction", "latex") and _mv_equiv(a, b):
            return True
        return False
    ba, bb = to_bool(a), to_bool(b)
    if ba is not None and bb is not None:
        return ba == bb
    la, lb = to_letter(a), to_letter(b)
    if la is not None and lb is not None:
        return la == lb
    if gold_type in ("latex", "fraction", "float", "int", None):
        if _mv_equiv(a, b):
            return True
    if len(na) >= 8 and len(nb) >= 8 and re.findall(r"\d+", na) == re.findall(r"\d+", nb):  # one digit changed is another answer
        return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.92
    return False


_UNITS = {w: v for w, v in NUMBER_WORDS.items() if v < 20}
_TENS = {w: v for w, v in NUMBER_WORDS.items() if 20 <= v < 100}
_WORD_NUM_RE = re.compile(
    r"\b(?:(?P<h>one|two|three|four|five|six|seven|eight|nine)\s+hundred(?:\s+(?:and\s+)?)?)?"
    r"(?:(?P<t>twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-]+(?P<u>one|two|three|four|five|six|seven|eight|nine))?"
    r"|(?P<s>zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|hundred|thousand))\b"
)


def number_words_in(text: str) -> list[float]:
    """Values spelled out in words: 'forty-two', 'two hundred and five', 'seven'."""
    out: list[float] = []
    for m in _WORD_NUM_RE.finditer(text.lower()):
        val = 0
        if m.group("h"):
            val += _UNITS[m.group("h")] * 100
        if m.group("t"):
            val += _TENS[m.group("t")]
            if m.group("u"):
                val += _UNITS[m.group("u")]
        elif m.group("s"):
            val += NUMBER_WORDS[m.group("s")]
        out.append(float(val))
    return out


_NUMBER_TOKEN_RE = re.compile(
    r"-?\\[dt]?frac\s*(?:\{-?\d+\}|\d)\s*(?:\{\d+\}|\d)|-?\d+(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\.\d+"
    r"|-?\d+(?:/\d+)?(?:[eE][+-]?\d+)?"
)


def extract_numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER_TOKEN_RE.finditer(text)]


_BOXED_RE = re.compile(r"\\boxed\s*")


def extract_boxed(text: str) -> list[str]:
    """All \\boxed{...} contents with balanced braces."""
    out: list[str] = []
    for m in _BOXED_RE.finditer(text):
        i = m.end()
        if i >= len(text):
            break
        if text[i] != "{":
            j = i
            while j < len(text) and not text[j].isspace() and text[j] != "$":
                j += 1
            out.append(text[i:j])
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[i + 1 : j])
                    break
    return out


_ANSWER_LINE_RE = re.compile(
    r"\b(?:final answer|the answer|answer|result)\s*(?:is|:|=|：)\s*(?:either\s+)?(.+?)\s*(?:\.\s|\.$|;|$|\n)", re.I
)
_EITHER_OR_RE = re.compile(r"\beither\s+(.+?)\s+or\s+(.+?)(?:[.,;\n]|$)", re.I | re.S)
_X_OR_Y_RE = re.compile(r"(\S+)\s+or\s+(\S+)")
_TEXT_OR_RE = re.compile(r"\s*\\text\s*\{\s*or\s*\}\s*")  # 13 \text{ or } 14


def _derived_fields(text: str, derived: Iterable[str]) -> list[str]:
    """What the answer fields of derived wrapper ids ('tag:masked_output', 'prefix:Final') and of a format the user gave
    ('<sol>{answer}</sol>') hold in `text`."""
    found: list[str] = []
    for wid in derived:
        if "{answer}" in wid:
            head, _, tail = wid.partition("{answer}")
            pattern = f"{re.escape(head)}(.*?){re.escape(tail)}" if tail else f"{re.escape(head)}(.+)$"
            found += [m.group(1) for m in re.finditer(pattern, text, flags=re.I | (re.S if tail else re.M))]
            continue
        kind, _, name = wid.removeprefix("reasoning_").partition(":")
        field = re.escape(name)
        if kind == "tag":
            found += [m.group(1) for m in re.finditer(rf"<{field}>(.*?)</{field}>", text, flags=re.S | re.I)]
        elif kind == "prefix":
            found += [m.group(1) for m in re.finditer(rf"^[ \t]*{field}[ \t]*:[ \t]*(.+)$", text, flags=re.M | re.I)]
    return found


def candidates(text: str, derived: Iterable[str] = ()) -> list[str]:
    """Final-answer candidates a lenient reader would find, in order of appearance. `derived` names the wrapper ids
    of answer fields this checker's own format adds, which no general reader would know about."""
    found: list[str] = []
    found += extract_boxed(text)
    found += [m.group(1) for m in re.finditer(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)]
    found += _derived_fields(text, derived)
    found += [m.group(1).strip() for m in re.finditer(r"^####\s*(.+)$", text, flags=re.M)]
    found += [m.group(1).strip() for m in _ANSWER_LINE_RE.finditer(text)]
    found += [m.group(1).strip() for m in re.finditer(r"\*\*(.+?)\*\*", text)]
    for m in _EITHER_OR_RE.finditer(text):
        found += [m.group(1), m.group(2)]
    expanded: list[str] = []
    for c in found:
        m = _X_OR_Y_RE.fullmatch(_TEXT_OR_RE.sub(" or ", c).strip())
        if m:
            expanded += [m.group(1), m.group(2)]
        else:
            expanded.append(c)
    out: list[str] = []
    for c in (strip_wrappers(c) for c in expanded):
        if c and "<" not in c and ">" not in c and c not in out:
            out.append(c)
    return out


def distinct_candidates(text: str, gold_type: str | None = None, derived: Iterable[str] = ()) -> list[str]:
    out: list[str] = []
    for c in candidates(text, derived):
        if not any(equiv(c, o, gold_type) for o in out):
            out.append(c)
    return out


def contains_equiv(text: str, gold: str, gold_type: str | None = None, derived: Iterable[str] = ()) -> bool:
    """Does any fragment of `text` plausibly equal `gold`?"""
    if not gold.strip():
        return False
    ng = normalize(gold)
    nt = normalize(text)
    g = None if gold_type in STRUCTURED else to_number(gold)  # '1 0\n0 1' is a grid, not the number 1001
    if ng and g is None and to_letter(gold) is None and to_bool(gold) is None:
        if re.search(r"(?<!\w)" + re.escape(ng) + r"(?!\w)", nt):
            return True
    if gold_type in STRUCTURED:
        rows = len([ln for ln in strip_wrappers(gold).splitlines() if ln.strip()])
        if any(structured_equiv(f, gold, gold_type) for f in structured_fragments(text, gold_type, rows)):
            return True
    for c in candidates(text, derived):
        if equiv(c, gold, gold_type):
            return True
    if g is not None:
        if ng.endswith("%") and re.search(r"(?<![\d.])" + re.escape(ng), nt):  # number tokens leave the percent sign out
            return True
        for tok in extract_numbers(text):
            if equiv(tok, gold, gold_type):
                return True
        for val in number_words_in(nt):
            if math.isclose(val, g):
                return True
        return False
    lg = to_letter(gold)
    if lg is not None:
        return re.search(rf"(?<![a-z]){lg.lower()}(?![a-z])", nt) is not None
    bg = to_bool(gold)
    if bg is not None:
        words = ("true", "yes", "correct") if bg else ("false", "no", "incorrect")
        return any(re.search(rf"\b{w}\b", nt) for w in words)
    return False
