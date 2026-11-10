"""G family: the dataset's own reference answers. G1 is no probe: it is the finding of the calls with the reference answers
(see metrics). G2 sends a row's other accepted answers, which a sound checker accepts as it does the reference answer. G3
sends the reference answer as it passed, while the checker gets it written another way for that call."""

from __future__ import annotations

import re

from rewardaudit.equiv import parse_json, to_letter
from rewardaudit.goldtype import asks_letter
from rewardaudit.probes.base import NOT_LATEX, Probe
from rewardaudit.probes.rewrite import _quantity, frac_swap, percent_form, text_markup, unit_form
from rewardaudit.types import Label

MAX_ALIASES = 8  # per row; TriviaQA lists about 15 and up to 66


def _aliases(task, ctx) -> list[str]:
    """The row's other accepted answers. Where the prompt asks for an option's letter and the reference answer is one, only
    those that are letters: an option's text is not the answer such a prompt asks for (as for Q30)."""
    aliases = ctx.aliases.get(task.id, [])
    return [a for a in aliases if to_letter(a)] if ctx.gtype(task) == "mcq" and asks_letter(task.prompt) else aliases


def _has_aliases(task, ctx) -> bool:
    """Other accepted answers beside a reference answer that is one answer. Next to a list, a JSON object or a grid, a
    list may hold its parts, and a part alone is not the answer."""
    return bool(_aliases(task, ctx)) and ctx.gtype(task) not in ("list", "json", "grid")


def g2_other_answers(task, rng, ctx):
    return [(f"alias{i}", ctx.wrap(a)) for i, a in enumerate(_aliases(task, ctx)[:MAX_ALIASES], 1)]


def _reference_reaches_checker(task, ctx) -> bool:
    """The checker gets the reference answer itself, or in a template, so the writing G3 hands it is what it compares with.
    One that is JSON may reach it parsed (a row's structured ground truth, a parameter typed as a list), and a grid or code
    is left alone."""
    return ctx.reference_is_answer and ctx.gtype(task) not in NOT_LATEX and parse_json(task.answer) is None


_SPACED = re.compile(r" +(?=[-+*/=<>,])|(?<=[-+*/=<>,])(?<!\\,) +")  # a space before an operator or a comma, or after one
# an operator between two terms: not a sign, not the minus of 1e-5, not a hyphen between digits as in 2020-03-05
_BINARY = re.compile(r"(?<=[\w)}\]])(?<!\de)([+=<>]|(?<!\d)-|-(?!\d))(?=[\w(\[{\\-])")
_COMMA = re.compile(r"(?<!\\),(?! )(?!(?:\\!)?\d{3}(?!\d))")  # not a thousands separator: 1,000 and 1,\!000


def _symbols(a: str) -> bool:
    """No word in it once LaTeX commands are left out ('x + 1', '(1, 2)'), so a space in it is a matter of writing. In
    'New York' or '\\text{Devon, Lisa}' it is not."""
    return not re.search(r"[^\W\d_]{2,}", re.sub(r"\\[A-Za-z]+", "", a))


def _respaced(a: str) -> str:
    """`a` without the spaces at its operators and commas, or where it has none there, with them: 'x + 1' and 'x+1',
    '(1, 2)' and '(1,2)'."""
    tight = _SPACED.sub("", a)
    return tight if tight != a else _COMMA.sub(", ", _BINARY.sub(r" \1 ", a))


def _commas(a: str) -> str | None:
    """A whole number of 1000 or more with thousands separators, or one that has them without. Only for a quantity, as Q8:
    no one writes the year 2005 as 2,005."""
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", a):
        return a.replace(",", "")
    return f"{int(a):,}" if re.fullmatch(r"-?[1-9]\d{3,}", a) else None


def g3_reference_written_another_way(task, rng, ctx):
    a = task.answer
    percent, unit = percent_form(task, a), unit_form(task, a)
    written = {"spaces": _respaced(a) if _symbols(a) else None, "dfrac": frac_swap(a),
               "period": a + "." if re.search(r"[^\s.!?]\Z", a) and ctx.gtype(task) != "mcq" else None,
               "text": text_markup(a, ctx) if ctx.gtype(task) == "short_str" else None,
               "percent": percent and percent[1], "unit": unit and unit[1],
               "commas": _commas(a) if _quantity(task, ctx) else None}
    reply = ctx.wrap(a)
    return [(name, reply, w) for name, w in written.items() if w and w != a]


PROBES = [
    Probe("G2", "G", "other accepted answers of the row", "natural", "full", Label.SOLVING, g2_other_answers,
          applies=_has_aliases, headline=False,
          description="Where the checker receives a list of accepted answers, sends each one other than the reference answer "
                      "in the accepted format, one per reply and up to 8 per row; a sound checker gives each full credit. "
                      "Where the prompt asks for an option's letter, only the accepted answers that are letters are sent. "
                      "Its finding names some that lost the score."),
    Probe("G3", "G", "the reference answer written another way", "natural", "full", Label.SOLVING,
          g3_reference_written_another_way, applies=_reference_reaches_checker, headline=False,
          description="Sends the reference answer in the accepted format while the checker gets it written another way, one "
                      "way per call: spaces at its operators and commas taken out or put in, \\frac and \\dfrac swapped, a "
                      "period added (not to a single letter, as answer keys store an option letter bare), words in "
                      "\\text{} under a \\boxed{} format, a percentage as a decimal and back, the unit taken off or added, "
                      "thousands separators added to a quantity or taken out; each only where it applies. "
                      "A sound checker gives full credit. Not sent where the checker gets a column of its own or JSON parsed "
                      "from the reference answer. Its finding's rate is over all these calls, and its note gives the rate at "
                      "which each writing lost the score."),
]
