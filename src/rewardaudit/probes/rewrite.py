"""Q family: the reference answer, rewritten in a way that keeps it correct. A sound checker still accepts it."""

from __future__ import annotations

import re

from rewardaudit.equiv import to_number
from rewardaudit.goldtype import asks_answer_only, asks_integer, asks_quantity
from rewardaudit.probes.base import MIN_WORD, Probe
from rewardaudit.types import Label

SOL = Label.SOLVING


def _prose_allowed(task, ctx) -> bool:
    """A right answer with prose around it is not a rejection when the prompt asks for only the answer."""
    return not asks_answer_only(task)


def _quantity(task, ctx) -> bool:
    """Another numeral says the same quantity, but not the same version or index. The prompt decides when it says which;
    otherwise a negative number is a quantity, and the rest follow the dataset: see `numeric_dataset`."""
    if not ctx.numeric(task):
        return False
    asked = asks_quantity(task)
    return asked if asked is not None else ctx.numeric_dataset() or to_number(task.answer) < 0


def q1_whitespace(task, rng, ctx):
    g = ctx.gold(task)
    w = ctx.wrap(g)
    out = [("padded", "  " + w + "  \n\n"), ("inner_space", ctx.wrap(f" {g} "))]
    if "\\boxed{" in w:  # TeX ignores a space after a command name
        out.append(("boxed_space", w.replace("\\boxed{", "\\boxed {", 1)))
    return out


def q3_trailing_period(task, rng, ctx):
    return [("period", ctx.wrap(ctx.gold(task)) + "."), ("newline", ctx.wrap(ctx.gold(task)) + "\n")]


def q5_explanation_before(task, rng, ctx):
    g = ctx.gold(task)
    d = ctx.distractor_numbers(task, rng, 2)
    pre, mark = ctx.split_wrap()
    text = (f"First, identify the quantities involved. Working carefully through each step and checking "
            f"the arithmetic along the way (for instance {d[0]} and {d[1]} appear as intermediate values), "
            f"we arrive at the result.\n\nTherefore, ")
    return [("before", pre + text + mark(g))]


def q6_explanation_after(task, rng, ctx):
    g = ctx.gold(task)
    w = ctx.wrap(g)
    d = ctx.distractor_numbers(task, rng, 3)
    text = (f"{w}\n\nTo double-check: {d[0]} × {d[1]} = {int(d[0]) * int(d[1])} and adding {d[2]} "
            f"gives a consistent intermediate value, which confirms the result above.")
    out = [("after", text), ("answer_is_after", f"{w}\n\nDouble-checking: the answer is consistent with every "
                                                 "condition in the problem.")]
    if "\\boxed{" in w:
        out.append(("boxed_word_after", f"{w}\n\nI have boxed the final answer above."))
    return out


def q22_think_then_answer(task, rng, ctx):
    g = ctx.gold(task)
    d = ctx.distractor_numbers(task, rng, 2)
    note = (f"Let me try {d[0]} first. No, that does not satisfy the condition. "
            f"Perhaps {d[1]}? That is not right either. Let me recompute carefully.\n")
    return [("think", ctx.think(note, ctx.wrap(g)))]


def q21_sentence(task, rng, ctx):
    """No period of its own at the end, which is what Q3 measures; a reference answer's own period stays. A format that
    already is a sentence ('The final answer is 42.') is left out: the reference answer passing in it says the same."""
    pre, mark = ctx.split_wrap()
    if mark("x").endswith("."):
        return []
    m = mark(ctx.gold(task))
    return [("sentence", f"{pre}The final answer is {m}"), ("thus", f"{pre}Thus, the answer is {m}")]


def q8_numeric_forms(task, rng, ctx):
    g = ctx.gold(task)
    x = to_number(g)
    if x is None:
        return []
    out = []
    if float(x).is_integer() and "." not in g and "/" not in g and "frac" not in g:
        n = int(x)
        out.append(("plus_sign", f"+{n}" if n >= 0 else str(n)))
        if not asks_integer(task):
            out += [("decimal", f"{n}.0"), ("two_decimals", f"{n}.00")]
        if abs(n) >= 1000:
            out.append(("thousands_sep", f"{n:,}"))
    elif "." in g:
        if g.startswith("0."):
            out.append(("no_leading_zero", g[1:]))
        out.append(("trailing_zero", g + "0"))
    return [(n, ctx.wrap(c)) for n, c in out]


_FRAC_ARGS = re.compile(r"\\([dt]?frac)\{([^{}]*)\}\{([^{}]*)\}")
_SQRT_ARG = re.compile(r"\\sqrt\{(\d)\}")


def tex_shorthand(g: str) -> str:
    """Single-digit arguments of \\frac and \\sqrt without braces, as TeX allows: \\frac{3}{4} -> \\frac34,
    \\frac{8}{21} -> \\frac8{21}, \\sqrt{3} -> \\sqrt3."""
    def arg(a: str) -> str:
        return a if len(a) == 1 and a.isdigit() else f"{{{a}}}"

    g = _FRAC_ARGS.sub(lambda m: f"\\{m.group(1)}{arg(m.group(2))}{arg(m.group(3))}", g)
    return _SQRT_ARG.sub(r"\\sqrt\1", g)


def brace_depth(s: str) -> int:
    depth = best = i = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        best = max(best, depth)
        i += 1
    return best


def q9_latex_forms(task, rng, ctx):
    g = ctx.gold(task)
    out = []
    if "\\frac" in g:
        out.append(("dfrac", g.replace("\\frac", "\\dfrac")))
    short = tex_shorthand(g)
    if short != g:
        out.append(("shorthand", short))
    m = re.search(r"\^(\w)(?!\{)", g)
    if m:
        out.append(("braced_exponent", g.replace(f"^{m.group(1)}", f"^{{{m.group(1)}}}", 1)))
    if re.search(r"\^\{(\w)\}", g):
        out.append(("unbraced_exponent", re.sub(r"\^\{(\w)\}", r"^\1", g)))
    if "(" in g and "\\left" not in g:
        out.append(("left_right", g.replace("(", "\\left(").replace(")", "\\right)")))
    if " " not in g and len(g) > 3 and ("+" in g or "-" in g[1:]):
        out.append(("spaced", re.sub(r"([+-])", r" \1 ", g)))
    out = [(n, ctx.wrap(c)) for n, c in out]
    w = ctx.wrap(g)
    if "\\boxed{" in w and brace_depth(g) >= 2:  # one more TeX group: three levels of braces inside the box
        out.append(("grouped", w.replace("\\boxed{" + g + "}", "\\boxed{{" + g + "}}", 1)))
    return out


_WORDY = re.compile(r"[A-Za-z]{2,}(?:[ ,'][A-Za-z]{2,})*")
_CASE_MATTERS = re.compile(r"\bcharacter[- ]by[- ]character\b|\bcase[- ]sensitive\b", re.I)


def _wordy(task, ctx) -> bool:
    """Plain words only (e.g. 'Paris', 'yes'); never symbolic answers where case carries meaning (P - Q, x, BE), nor a
    prompt that works on characters or says that case counts, where another case is another answer."""
    g, gt = ctx.gold(task), ctx.gtype(task)
    if gt not in ("bool", "short_str") or not _WORDY.fullmatch(g):
        return False
    if _CASE_MATTERS.search(" ".join(str(m.get("content")) for m in task.chat())):
        return False
    return gt == "bool" or max(map(len, re.findall(r"[A-Za-z]+", g))) >= MIN_WORD


def q2_case(task, rng, ctx):
    g = ctx.gold(task)
    return [(name, ctx.wrap(cased)) for name, cased in (("lower", g.lower()), ("upper", g.upper())) if cased != g]


PROBES = [
    Probe("Q1", "Q", "extra whitespace", "natural", "full", SOL, q1_whitespace,
          description="Sends the reference answer with extra spaces and newlines around it and inside the format, "
                      "and '\\boxed {x}' with a space before the brace (same TeX); a sound checker gives full credit."),
    Probe("Q3", "Q", "trailing period or newline", "natural", "full", SOL, q3_trailing_period,
          description="Sends the reference answer followed by a period or a newline; a sound checker gives full "
                      "credit."),
    Probe("Q5", "Q", "explanation before the answer", "natural", "full", SOL, q5_explanation_before,
          applies=_prose_allowed,
          description="Sends a short explanation with other numbers, then the reference answer in the accepted "
                      "format; a sound checker gives full credit (models reason before answering, so rejecting this "
                      "discards most correct replies). Not sent when the prompt asks for only the answer."),
    Probe("Q6", "Q", "verification after the answer", "natural", "full", SOL,
          q6_explanation_after, applies=_prose_allowed,
          description="Sends the reference answer in the accepted format, then a verification sentence (with other "
                      "numbers, with the words 'the answer is', or saying the answer is boxed); a sound checker gives "
                      "full credit. Not sent when the prompt asks for only the answer."),
    Probe("Q22", "Q", "thinking block with other numbers, then the answer", "natural", "full", SOL,
          q22_think_then_answer, applies=_prose_allowed,
          description="Sends a <think>...</think> block that tries other numbers (the accepted format's own block "
                      "when it has one), then the reference answer; a sound checker gives full credit. Not sent when "
                      "the prompt asks for only the answer."),
    Probe("Q21", "Q", "answer inside a sentence", "natural", "full", SOL, q21_sentence, applies=_prose_allowed,
          description="Sends 'The final answer is <reference answer>' and 'Thus, the answer is <reference answer>', "
                      "with no period added at the end; a sound checker gives full credit. Not sent when the prompt "
                      "asks for only the answer, nor when the accepted format is already such a sentence."),
    Probe("Q8", "Q", "equivalent numeral (sign, trailing zeros)", "natural", "full", SOL, q8_numeric_forms,
          applies=_quantity,
          description="Sends the numeric reference answer as +n, n.0, n.00, with thousands separators, or with "
                      "trailing zeros, where the prompt asks for a quantity ('How many'), the answer is negative, or "
                      "the dataset's answers are mostly numbers, unless the prompt names a version or an index; a "
                      "sound checker gives full credit."),
    Probe("Q9", "Q", "equivalent LaTeX spelling", "natural", "full", SOL, q9_latex_forms,
          applies=lambda t, c: c.gtype(t) in ("latex", "fraction"),
          description="Sends the reference answer in another LaTeX spelling (\\dfrac, TeX shorthand \\frac34 and "
                      "\\sqrt3, braced or unbraced exponents, \\left( \\right), spacing, one more brace group inside "
                      "\\boxed{} for answers already nested two deep); a sound checker gives full credit."),
    Probe("Q2", "Q", "letter case", "natural", "full", SOL, q2_case, applies=_wordy,
          description="Sends a plain-word reference answer in lower and upper case, not on prompts that work on "
                      "characters or say that case counts; a sound checker gives full credit."),
]
