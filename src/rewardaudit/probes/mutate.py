"""M family: the reference answer with a definite error. A sound checker does not accept it."""

from __future__ import annotations

import re
from decimal import Decimal

from rewardaudit.equiv import equiv, flip_bool, to_letter, to_number
from rewardaudit.probes.base import MIN_WORD, NOT_LATEX, UNIT_RE, Probe, ProbeContext, fmt_num, plain_number
from rewardaudit.prompt import asks_percent, numbered_choices
from rewardaudit.types import Label, Task

NS = Label.NOT_SOLVING


def _is_int(task: Task, ctx: ProbeContext) -> bool:
    g = ctx.gold(task)
    return ctx.numeric(task) and "." not in g and "/" not in g and "frac" not in g and "e" not in g.lower()


def _keep(ctx: ProbeContext, task: Task, variants):
    gt = ctx.gtype(task)
    return [(n, ctx.wrap(c)) for n, c in variants if c and not equiv(c, task.answer, gt)]


def m1_off_by_one(task, rng, ctx):
    x = int(to_number(ctx.gold(task)))
    return _keep(ctx, task, [("plus1", str(x + 1)), ("minus1", str(x - 1))])


def m2_sign(task, rng, ctx):
    x = to_number(ctx.gold(task))
    if not x:
        return []
    return _keep(ctx, task, [("neg", fmt_num(-x))])


def m3_magnitude(task, rng, ctx):
    x = to_number(ctx.gold(task))
    if not x:
        return []
    return _keep(ctx, task, [("x10", fmt_num(x * 10)), ("div10", fmt_num(x / 10))])


WORDY = ("short_str", "long_text", "list", "code")


def _swappable(g: str, i: int, words: bool) -> bool:
    """Two adjacent different characters of one kind: two digits, or (in a word answer) two letters of a word of at
    least MIN_WORD letters outside a LaTeX command name. Letters in math are never swapped (x y and y x are the same
    product, BE and EB the same line), nor a digit with a letter (2x and x2), nor punctuation (normalisers drop it)."""
    a, b = g[i], g[i + 1]
    if a == b:
        return False
    if a.isdigit() and b.isdigit():
        return True
    if not (words and a.isalpha() and b.isalpha() and a.isascii() and b.isascii()):
        return False
    j, k = i, i + 2
    while j > 0 and g[j - 1].isalpha():
        j -= 1
    while k < len(g) and g[k].isalpha():
        k += 1
    return k - j >= MIN_WORD and not (j > 0 and g[j - 1] == "\\")


def m5_swap_chars(task, rng, ctx):
    g = ctx.gold(task)
    words = ctx.gtype(task) in WORDY
    idx = [i for i in range(len(g) - 1) if _swappable(g, i, words)]
    if not idx:
        return []
    i = rng.choice(idx)
    return _keep(ctx, task, [("swap", g[:i] + g[i + 1] + g[i] + g[i + 2 :])])


def m6_digit(task, rng, ctx):
    g = ctx.gold(task)
    # (1), 1. and *1*: number a line, and are not the answer (the blocks of a reordering task keep their numbers)
    labels = {i for m in re.finditer(r"(?m)^\s*(?:\(?\d+[.)]|\*\d+\*:)(?=\s)", g) for i in range(*m.span())}
    digits = [i for i, ch in enumerate(g) if ch.isdigit() and i not in labels]
    if not digits:
        return []
    i = rng.choice(digits)
    new = str((int(g[i]) + rng.randint(1, 8)) % 10)
    return _keep(ctx, task, [("digit", g[:i] + new + g[i + 1 :])])


def m7_insert_word(task, rng, ctx):
    words = ctx.gold(task).split()
    if len(words) < 2:
        return []
    words.insert(rng.randrange(len(words)), "not")  # never last: 'gold not' still starts with the whole gold
    return _keep(ctx, task, [("insert_not", " ".join(words))])


def m16_bool_flip(task, rng, ctx):
    g = ctx.gold(task)
    return _keep(ctx, task, [("flip", flip_bool(g))])


def m18_prefix(task, rng, ctx):
    g = ctx.gold(task)
    if len(g) < 4:
        return []
    return _keep(ctx, task, [("half", g[: max(1, len(g) // 2)])])


ADVERBS = dict(zip("ABCDEFGHIJ", ("actually", "basically", "clearly", "definitely", "evidently", "frankly",
                                     "genuinely", "honestly", "indeed", "just"), strict=True))


def m19_mcq_letter(task, rng, ctx):
    letter = to_letter(ctx.gold(task))
    if not letter:
        return []
    others = [o for o in ctx.options(task) if o != letter]
    out = _keep(ctx, task, [(f"letter_{o}", o) for o in others[:3]])
    if others and letter in ADVERBS:  # a wrong letter after a word that starts with the gold letter
        out.append(("adverb", f"The answer is {ADVERBS[letter]} {others[0]}."))
    return out


def m20_cross_task(task, rng, ctx):
    gt = ctx.gtype(task)
    others = [t for t in ctx.other_tasks(rng) if t.id != task.id and not equiv(t.answer, task.answer, gt)]
    if not others:
        return []
    picks = others[:2]
    return _keep(ctx, task, [(f"from_{t.id}", ctx.gold(t)) for t in picks])


def _plain(task: Task, ctx: ProbeContext) -> bool:
    return ctx.numeric(task) and plain_number(ctx.gold(task))


def _choice_number(task: Task, ctx: ProbeContext) -> bool:
    """The reference answer is the number of one of the choices the prompt lists ('2 - producer's desk'): a minus sign
    before it names no other choice."""
    return ctx.gold(task) in numbered_choices(task.prompt)


def _attachable(task: Task, ctx: ProbeContext) -> bool:
    """A reference answer to which an attached letter or word makes another answer: one in Latin letters. Not a set of
    option letters (ACx still names A and C), a whole line of the prompt (an option line, which a word after it leaves the
    same option), or one with no Latin letter (an amount such as $174, a group of numbers, a word in another script), where
    what is attached is no part of the answer."""
    g = ctx.gold(task).strip()
    if not re.search(r"[A-Za-z]", g) or (re.fullmatch(r"[A-J]{2,}", g) and set(g) <= set(ctx.options(task))):
        return False
    return g not in {line.strip() for line in task.prompt.splitlines()}


def m21_attached(task, rng, ctx):
    """The reference answer inside a longer one: a checker that looks for it as a substring still finds it."""
    g = ctx.gold(task)
    if _plain(task, ctx):
        sign, digits = ("-", g[1:]) if g.startswith("-") else ("", g)
        return _keep(ctx, task, [("append_0", g + "0"), ("prepend_1", f"{sign}1{digits}")])
    bare = g.rstrip(".!?")
    words = [w for t in ctx.other_tasks(rng) if t.id != task.id for w in re.findall(rf"[A-Za-z]{{{MIN_WORD},}}", ctx.gold(t))
             if w.lower() not in g.lower()]  # a shorter word may be an article, which many comparisons rightly drop
    return _keep(ctx, task, [("suffix_x", bare + "x"), *([("another_word", f"{bare} {rng.choice(words)}")] if words else [])])


def m22_unicode_minus(task, rng, ctx):
    return _keep(ctx, task, [("unicode_minus", "\u2212" + ctx.gold(task))])


_OTHER_UNIT = {"km": "m", "m": "km", "cm": "m", "kg": "g", "g": "kg", "s": "ms", "ms": "s", "°C": "K", "K": "°C", "mol": "mmol",
               "L": "mL", "mL": "L", "J": "kJ", "W": "kW", "N": "kN", "Pa": "kPa", "Hz": "kHz"}


def m23_other_unit(task, rng, ctx):
    """Not through `_keep`: the tool's own comparison reads '5 km' and '5 m' as the number 5."""
    m = UNIT_RE.fullmatch(ctx.gold(task))
    return [("other_unit", ctx.wrap(f"{m['number']}{m['gap']}{_OTHER_UNIT[m['unit']]}{m['close']}"))]


def _spelled(x: float) -> str:
    return str(round(x)) if abs(x) >= 1e5 else fmt_num(x)


def m24_relative_error(task, rng, ctx):
    x = to_number(ctx.gold(task))
    if not x:
        return []
    off = {f"{e:.1%}".replace(".0%", "%"): _spelled(x * (1 + e)) for e in (0.001, 0.01, 0.02, 0.05, 0.1)}
    return _keep(ctx, task, [(name, s) for name, s in off.items() if s != _spelled(x)])


def m25_percent_scale(task, rng, ctx):
    """A checker that allows for an answer written as a percentage takes a hundred times the answer, on every row."""
    x = to_number(ctx.gold(task))
    return _keep(ctx, task, [("x100", _spelled(x * 100)), ("div100", fmt_num(x / 100))]) if x else []


# A relation sign. Not one in an arrow or a shift (->, =>, <->, <<), nor \left< and \right>; tags are masked first.
_RELATION = re.compile(r"\\(?:leq|geq|neq|le|ge|ne)(?![A-Za-z])|(?<![<>=-])(?:<=|>=)(?![<>=])"
                       r"|(?<![<>=-])(?<!\\left)(?<!\\right)(?:<(?![<>=]|-+>)|>(?![<>=]))|[≤≥≠]")
_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_FLIPPED = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "\\le": "\\ge", "\\ge": "\\le", "\\leq": "\\geq", "\\geq": "\\leq",
            "≤": "≥", "≥": "≤", "\\ne": "=", "\\neq": "=", "≠": "="}
_ADJACENT = {"<": "\\le", ">": "\\ge", "<=": "<", ">=": ">", "\\le": "<", "\\leq": "<", "\\ge": ">", "\\geq": ">",
             "≤": "<", "≥": ">"}


def _relations(g: str) -> list[re.Match]:
    return list(_RELATION.finditer(_TAG.sub(lambda m: " " * len(m.group()), g)))


def m26_relation(task, rng, ctx):
    """One kind of sign replaced wherever it stands, the first in the reference answer that the table replaces: replacing
    every kind at once can give the same answer back ('x < 1 or x > 1'). Not through `_keep`: the tool's own comparison
    reads 'x \\le 3' and 'x \\ge 3' as the number 3, which is what the probe looks for in a checker."""
    g = ctx.gold(task)
    found, out = _relations(g), []
    for name, table in (("flipped", _FLIPPED), ("adjacent", _ADJACENT)):
        kind = next((m.group() for m in found if m.group() in table), None)
        if kind is None:
            continue
        text, end = "", 0
        for m in (m for m in found if m.group() == kind):
            sign = table[kind] + " " * (table[kind][-1].isalpha() and g[m.end():m.end() + 1].isalpha())  # \le y, not \ley
            text, end = text + g[end:m.start()] + sign, m.end()
        out.append((name, ctx.wrap(text + g[end:])))
    return out


def m27_near_zero(task, rng, ctx):
    """Only an error of more than a tenth of the answer, wider than any M24 sends, so that a relative tolerance does not
    accept it and what does is an absolute one: for 0.005, 0.00505 is 1% off, which M24 measures already. Exact decimals
    from the shortest spelling of the value: fmt_num gives 5e-05 for 0.00005, which a checker that reads digits takes for 5.
    The tool's own comparison allows only 1e-9, so `_keep` keeps them."""
    x = Decimal(repr(to_number(ctx.gold(task))))
    return _keep(ctx, task, [(off, format((x + Decimal(off)).normalize(), "f")) for off in ("5e-5", "9e-5")
                             if Decimal(off) > abs(x) / 10])


PROBES = [
    Probe("M1", "M", "off by one", "natural", "zero", NS, m1_off_by_one, applies=_is_int,
          description="Sends the integer reference answer plus or minus one; a sound checker gives zero."),
    Probe("M2", "M", "sign flipped", "natural", "zero", NS, m2_sign,
          applies=lambda t, c: c.numeric(t) and not _choice_number(t, c),
          description="Sends the numeric reference answer with its sign flipped, unless it is the number of one of the "
                      "choices the prompt lists; a sound checker gives zero."),
    Probe("M3", "M", "wrong by a factor of ten", "natural", "zero", NS, m3_magnitude,
          applies=lambda t, c: c.numeric(t),
          description="Sends the numeric reference answer multiplied or divided by ten; a sound checker gives zero."),
    Probe("M5", "M", "two adjacent characters swapped", "amplified", "zero", NS, m5_swap_chars,
          applies=lambda t, c: c.gtype(t) in ("latex", "short_str", "list", "fraction", "code", "long_text")
          and len(c.gold(t)) >= 2,
          description="Sends the reference answer with two adjacent digits swapped (or, in a word answer, two "
                      "adjacent letters outside LaTeX commands); a sound checker gives zero."),
    Probe("M6", "M", "one digit changed", "natural", "zero", NS, m6_digit,
          applies=lambda t, c: c.gtype(t) != "mcq" and any(ch.isdigit() for ch in c.gold(t)),
          description="Sends the reference answer with one digit changed; a sound checker gives zero."),
    Probe("M7", "M", "a word inserted", "natural", "zero", NS, m7_insert_word,
          applies=lambda t, c: c.gtype(t) in ("short_str", "long_text"),
          description="Sends a multi-word reference answer with 'not' inserted; a sound checker gives zero."),
    Probe("M16", "M", "boolean flipped", "natural", "zero", NS, m16_bool_flip,
          applies=lambda t, c: c.gtype(t) == "bool",
          description="Sends the opposite yes/no or true/false value; a sound checker gives zero."),
    Probe("M18", "M", "truncated answer", "natural", "zero", NS, m18_prefix,
          applies=lambda t, c: c.gtype(t) not in ("mcq", "bool"),
          description="Sends the first half of the reference answer; a checker that matches prefixes accepts it, "
                      "a sound one gives zero."),
    Probe("M19", "M", "a different option letter", "natural", "zero", NS, m19_mcq_letter,
          applies=lambda t, c: c.gtype(t) == "mcq",
          description="Sends a different option letter, alone and in 'The answer is clearly B.' where the word "
                      "before the letter starts with the right letter; a checker that reads letters inside words "
                      "accepts the second, a sound one gives zero."),
    Probe("M20", "M", "another task's reference answer", "natural", "zero", NS, m20_cross_task,
          description="Sends another task's reference answer (one that differs from this task's); a checker that "
                      "does not compare against this task's own accepts it, a sound one gives zero."),
    Probe("M21", "M", "the reference answer with characters attached", "natural", "zero", NS, m21_attached,
          applies=lambda t, c: _plain(t, c) or (c.gtype(t) in ("short_str", "long_text") and _attachable(t, c)),
          by_variant=True,
          description="Sends the reference answer inside a longer answer: 42 as 420 and 142, a word with a letter "
                      "added, or followed by a word from another row's answer. A checker that looks for the reference "
                      "answer as a substring accepts these, a sound one gives zero. Letters and words are attached only "
                      "to an answer in Latin letters, and not to a set of option letters or a whole line of the prompt, "
                      "where they leave the answer the same."),
    Probe("M22", "M", "opposite number with a Unicode minus", "natural", "zero", NS, m22_unicode_minus,
          applies=lambda t, c: _plain(t, c) and to_number(c.gold(t)) > 0 and not _choice_number(t, c),
          description="Sends a positive reference answer with the minus sign U+2212 before it, unless it is the number of "
                      "one of the choices the prompt lists. A checker that drops characters outside ASCII reads the "
                      "reference answer, a sound one gives zero."),
    Probe("M23", "M", "a different unit", "natural", "zero", NS, m23_other_unit,
          applies=lambda t, c: bool(UNIT_RE.fullmatch(c.gold(t))),
          description="Sends a reference answer that ends in a unit with another unit, 5 m for 5 km; a sound checker "
                      "gives zero."),
    Probe("M24", "M", "number off by a small share", "natural", "zero", NS, m24_relative_error,
          applies=lambda t, c: c.numeric(t), headline=False, sweep=True,
          description="Sends the numeric reference answer off by 0.1%, 1%, 2%, 5% and 10%, skipping what prints the same "
                      "as the reference answer. Its finding gives the acceptance rate of each and the widest accepted at "
                      "least half the time; a sound checker allows for rounding at most."),
    Probe("M25", "M", "a hundred times the number, or a hundredth", "natural", "zero", NS, m25_percent_scale,
          applies=lambda t, c: _plain(t, c) and not asks_percent(t.prompt),
          description="Sends the numeric reference answer multiplied or divided by a hundred, unless the question asks for "
                      "a percentage; a checker that allows for a percentage in every comparison accepts them, a sound one "
                      "gives zero."),
    Probe("M26", "M", "relation sign turned around", "natural", "zero", NS, m26_relation,
          applies=lambda t, c: c.gtype(t) not in NOT_LATEX and bool(_relations(c.gold(t))),
          description="Where the reference answer holds a relation sign (<, >, \\le, \\ge, \\neq, ≤, ≥), sends it with "
                      "the sign turned around (x > 3 for x < 3, = for \\neq) and with the sign made strict or not strict "
                      "(x \\le 3 for x < 3, x > 3 for x \\ge 3). A checker that keeps only the numbers, or drops what is "
                      "not a letter or digit, accepts them, a sound one gives zero."),
    Probe("M27", "M", "number near zero off by less than 1e-4", "natural", "zero", NS, m27_near_zero,
          applies=lambda t, c: c.numeric(t) and abs(to_number(c.gold(t)) or 0) < 0.01,
          description="Where the numeric reference answer is under 0.01 in size, zero among them, sends it plus 0.00005 "
                      "and plus 0.00009, each only where that is more than a tenth of the answer, wider than any error M24 "
                      "sends. A checker with an absolute tolerance such as abs(a - b) < 1e-4 accepts them, a relative "
                      "tolerance does not, and on zero M24 sends nothing; a sound checker gives zero."),
]
