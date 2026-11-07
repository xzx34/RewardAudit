"""Q family: the reference answer, rewritten in a way that keeps it correct. A sound checker still accepts it."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal

from rewardaudit.equiv import NUMBER_WORDS, to_letter, to_number
from rewardaudit.goldtype import asks_answer_only, asks_integer, asks_quantity, option_texts
from rewardaudit.probes.base import MIN_WORD, UNIT_RE, Probe, fmt_num, plain_number
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
            f"we arrive at the result.\n\n{'' if ctx.opens_a_line() else 'Therefore, '}")
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


_WORD_OF = {v: w for w, v in reversed(NUMBER_WORDS.items())}
_FULL_WIDTH = {ord(c): ord(c) + 0xFEE0 for c in "0123456789"}


def _in_words(g: str) -> str | None:
    """A whole number under 100 in English words."""
    n = int(g) if g.isdigit() else -1
    if not 0 <= n < 100:
        return None
    return _WORD_OF.get(n) or f"{_WORD_OF[n - n % 10]}-{_WORD_OF[n % 10]}"


def _scientific(g: str) -> str | None:
    """Exact scientific notation, for a number that a model might write that way: large, or with many digits."""
    sign, digits, exponent = Decimal(g).normalize().as_tuple()
    if not digits or (abs(Decimal(g)) < 10000 and len(digits) < 4):
        return None
    tail = "".join(map(str, digits[1:]))
    return f"{'-' if sign else ''}{digits[0]}{'.' + tail if tail else ''}e{exponent + len(digits) - 1}"


def q23_number_spellings(task, rng, ctx):
    g = ctx.gold(task)
    spelled = {"words": _in_words(g), "scientific": _scientific(g), "full_width": g.translate(_FULL_WIDTH),
               "unicode_minus": g.replace("-", "\u2212") if g.startswith("-") else None}
    return [(name, ctx.wrap(s)) for name, s in spelled.items() if s]


_UNIT_WORDS = ("dollars|cents|meters|metres|kilometers|kilometres|miles|feet|inches|centimeters|kilograms|grams|pounds|ounces|"
               "liters|litres|gallons|seconds|minutes|hours|days|weeks|months|years")
_ASKS_UNIT = re.compile(rf"\b(?:in|how many|number of)\s+({_UNIT_WORDS})\b", re.I)
_ASKS_PERCENT = re.compile(r"\bwhat\s+(?:is\s+the\s+)?percent(?:age)?\b|\b(?:as|in)\s+(?:a\s+)?percent(?:age)?\b"
                           r"|\bhow\s+many\s+percent\b", re.I)


def _question(prompt: str) -> str:
    """The end of a prompt, where the question usually is; a line on the format of the reply may follow it."""
    return " ".join(re.split(r"(?<=[.?!])\s+", prompt.strip())[-2:])


def q24_unit(task, rng, ctx):
    g = ctx.gold(task)
    if m := UNIT_RE.fullmatch(g):
        return [("without_unit", ctx.wrap(m["number"]))]
    asked = _ASKS_UNIT.findall(_question(task.prompt))
    return [("with_unit", ctx.wrap(f"{g} {asked[-1].lower()}"))] if asked and plain_number(g) and _prose_allowed(task, ctx) else []


def q25_percent(task, rng, ctx):
    """A plain number below 1 under a question in percent could mean either, so it stays out."""
    g = ctx.gold(task)
    if m := re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*\\?%", g):
        return [("as_decimal", ctx.wrap(fmt_num(float(m[1]) / 100)))]
    if plain_number(g) and abs(float(g)) >= 1 and _ASKS_PERCENT.search(_question(task.prompt)):
        return [("percent_sign", ctx.wrap(f"{g}%"))]
    return []


_MONTHS = "January February March April May June July August September October November December".split()
_DATE_FORMS = (re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"),
               re.compile(r"(?P<mon>[A-Za-z]{3,9})\.? (?P<d>\d{1,2})(?:st|nd|rd|th)?,? (?P<y>\d{4})"),
               re.compile(r"(?P<d>\d{1,2})(?:st|nd|rd|th)? (?P<mon>[A-Za-z]{3,9})\.?,? (?P<y>\d{4})"))


def parse_date(s: str) -> date | None:
    """A date in a form that says which number is the month; 3/5/2020 does not."""
    for form in _DATE_FORMS:
        if m := form.fullmatch(s):
            parts = m.groupdict()
            month = next((i + 1 for i, name in enumerate(_MONTHS) if name[:3].lower() == parts["mon"][:3].lower()), 0) \
                if "mon" in parts else int(parts["m"])
            try:
                return date(int(parts["y"]), month, int(parts["d"]))
            except ValueError:
                return None
    return None


def q26_date(task, rng, ctx):
    g, d = ctx.gold(task), parse_date(ctx.gold(task))
    month = _MONTHS[d.month - 1]
    forms = {"iso": d.isoformat(), "long": f"{month} {d.day}, {d.year}", "short": f"{month[:3]} {d.day}, {d.year}"}
    return [(name, ctx.wrap(s)) for name, s in forms.items() if s != g]


def _words(task, ctx) -> bool:
    """A short answer in words, on a prompt that does not work on characters or ask about hyphens."""
    prompt = " ".join(str(m.get("content")) for m in task.chat())
    return (ctx.gtype(task) == "short_str" and bool(re.search(rf"[^\W\d_]{{{MIN_WORD},}}", ctx.gold(task)))
            and not _CASE_MATTERS.search(prompt) and not re.search(r"\b(?:hyphens?|dash(?:es)?)\b", prompt, re.I))


def q27_spelling(task, rng, ctx):
    g = ctx.gold(task)
    out = {"no_accents": "".join(c for c in unicodedata.normalize("NFD", g) if not unicodedata.combining(c))}
    article = re.match(r"(?:the|an?)\s+(.+)", g, re.I)
    if article:
        out["no_article"] = article[1]
    elif g[0].islower():
        out["article"] = f"the {g}"
    if re.search(r"[^\W\d_]{3}-[^\W\d_]{3}", g):
        out["no_hyphen"] = g.replace("-", " ")
    elif not article and re.fullmatch(r"[^\W\d_]{3,} [^\W\d_]{3,}", g):
        out["hyphen"] = g.replace(" ", "-")
    out["and_word"] = g.replace("&", "and")
    out["ampersand"] = g.replace(" and ", " & ")
    return [(name, ctx.wrap(s)) for name, s in out.items() if s != g]


LENGTHS = (("1k", 1_000), ("4k", 4_000), ("16k", 16_000), ("64k", 64_000))


def _long_text(task, rng, ctx):
    """Long text around the right answer, in the lengths of a reply, for the checker that reads only part of it."""
    filler = ctx.filler(task, rng)
    return [(name, filler(size)) for name, size in LENGTHS] if filler else []


def q28_long_text_after(task, rng, ctx):
    answer = ctx.wrap(ctx.gold(task))
    return [(name, f"{answer}\n\n{text}") for name, text in _long_text(task, rng, ctx)]


def q29_long_text_before(task, rng, ctx):
    pre, mark = ctx.split_wrap()
    return [(name, f"{pre}{text}\n\n{mark(ctx.gold(task))}") for name, text in _long_text(task, rng, ctx)]


def _few_with_prose(task, ctx) -> bool:
    return task.id in ctx.first_passing() and _prose_allowed(task, ctx)


def q30_option_text(task, rng, ctx):
    """Only the text is not what a prompt that asks for a letter wants; the letter with its text always is."""
    letter, texts = to_letter(ctx.gold(task)), option_texts(task.prompt)
    if letter not in texts:
        return []
    out = [("letter_and_text", ctx.wrap(f"{letter}) {texts[letter]}"))]
    return out if re.search(r"\bletter\b", task.prompt, re.I) else [("text", ctx.wrap(texts[letter])), *out]


def q31_latex_markup(task, rng, ctx):
    g = ctx.gold(task)
    out = [("text", ctx.wrap(f"\\text{{{g}}}"))] if "\\boxed{" in ctx.wrap("x") and "\\" not in g else []
    if plain_number(g) and re.search(r"\b(?:angle|degrees?)\b", _question(task.prompt), re.I):
        out.append(("degrees", ctx.wrap(f"{g}^\\circ")))
    return out


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
    Probe("Q23", "Q", "other spellings of a number", "natural", "full", SOL, q23_number_spellings,
          applies=lambda t, c: _quantity(t, c) and plain_number(c.gold(t)),
          description="Sends a numeric reference answer in English words (under 100), in scientific notation (when large "
                      "or with many digits), in full-width digits, and with the minus sign U+2212; a sound checker gives "
                      "full credit."),
    Probe("Q24", "Q", "with or without the unit", "natural", "full", SOL, q24_unit,
          description="Sends a reference answer that has a unit without it, and adds the unit to one that has none when "
                      "the question names it ('how many dollars'); a sound checker gives full credit."),
    Probe("Q25", "Q", "percent sign or decimal", "natural", "full", SOL, q25_percent,
          description="Sends 25% as 0.25, and adds the percent sign to a number when the question asks for a percentage; "
                      "a sound checker gives full credit."),
    Probe("Q26", "Q", "other spellings of a date", "natural", "full", SOL, q26_date,
          applies=lambda t, c: parse_date(c.gold(t)) is not None,
          description="Sends a date reference answer as 2020-03-05, March 5, 2020 and Mar 5, 2020; a sound checker gives "
                      "full credit."),
    Probe("Q27", "Q", "other spellings of a word", "natural", "full", SOL, q27_spelling, applies=_words,
          description="Sends a short answer in words without accents, with or without a leading article, with a hyphen "
                      "and a space swapped, and with & and 'and' swapped; a sound checker gives full credit."),
    Probe("Q28", "Q", "long text after the answer", "natural", "full", SOL, q28_long_text_after, applies=_few_with_prose,
          headline=False, sweep=True,
          description="Sends the reference answer followed by 1k, 4k, 16k and 64k characters of neutral text. Its finding "
                      "gives the acceptance rate of each and the shortest that loses the score; a sound checker gives "
                      "full credit. Sent on five rows, not when the prompt asks for only the answer."),
    Probe("Q29", "Q", "long text before the answer", "natural", "full", SOL, q29_long_text_before, applies=_few_with_prose,
          headline=False, sweep=True,
          description="Sends 1k, 4k, 16k and 64k characters of neutral text and then the reference answer. Its finding gives "
                      "the acceptance rate of each and the shortest that loses the score; a sound checker gives full "
                      "credit. Sent on five rows, not when the prompt asks for only the answer."),
    Probe("Q30", "Q", "an option's text with or without its letter", "natural", "full", SOL, q30_option_text,
          applies=lambda t, c: c.gtype(t) == "mcq",
          description="Sends the right option's letter with its text, and its text alone unless the prompt asks for a "
                      "letter; a sound checker gives full credit."),
    Probe("Q31", "Q", "an answer in LaTeX markup", "natural", "full", SOL, q31_latex_markup,
          description="Inside \\boxed{} sends the reference answer in \\text{}, and with ^\\circ when the question is about "
                      "an angle; a sound checker gives full credit."),
]
