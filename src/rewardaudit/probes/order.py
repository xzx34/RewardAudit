"""O family: a right and a wrong answer in one reply: retraction, negation, hedging, which one the checker reads."""

from __future__ import annotations

import random

from rewardaudit.contract import wrapper_fn
from rewardaudit.equiv import normalize, to_letter, to_number
from rewardaudit.goldtype import asks_answer_only, option_texts
from rewardaudit.probes.base import Probe, ProbeContext, fmt_num, plain_number
from rewardaudit.types import Label, Task

NS, SOL = Label.NOT_SOLVING, Label.SOLVING


def _pair(task: Task, rng: random.Random, ctx: ProbeContext):
    return ctx.gold(task), ctx.wrong_answer(task, rng)


def _two(ctx: ProbeContext, text: str, *answers: str) -> str:
    """`text` with {0}, {1}, ... replaced by the answers in the accepted format, after one thinking block or
    reasoning preamble (not one per answer)."""
    pre, mark = ctx.split_wrap()
    return pre + text.format(*(mark(a) for a in answers))


SINGLE_BLOCK = ("answer_tag", "reasoning_tag", "think_tag", "open_think_tag", "answer_tag_boxed", "think_tag_boxed",
                "json", "code_block", "hash", "reasoning_hash", "answer_colon", "final_answer", "the_final_answer_is",
                "reasoning_final")


def single_block(wid: str) -> bool:
    """A format with one answer block, where a second block is a second submission: the fixed ones above and
    every tag derived from the environment or the prompts ('tag:masked_output', 'reasoning_tag:masked_output'), and a
    format the user gave, since nothing shows a second block of it would count."""
    return wid in SINGLE_BLOCK or wid.startswith(("tag:", "reasoning_tag:")) or "{answer}" in wid


RETRACT = "{0}\n\nWait, that is not right. I made an error above. The correct result is {1}"
NEGATED = "The answer is not {0}. The answer is {1}."
HYPOTHETICAL = ("If the problem were asking for a different quantity, the result would be {0}. "
                "But for what is actually asked, the result is {1}")


def o3_not_committed(task, rng, ctx):
    g, w = _pair(task, rng, ctx)
    if w is None:
        return []
    return [("retract", _two(ctx, RETRACT, g, w)), ("not", _two(ctx, NEGATED, g, w)), ("hypo", _two(ctx, HYPOTHETICAL, g, w))]


def o4_retract_to_gold(task, rng, ctx):
    """Skipped for formats with one answer block (<answer> tags, JSON, a code block, a '####' or 'Answer:' line):
    there a second block is a second submission, and rejecting it is a format rule rather than a rejected right
    answer. O3 still reports a parser that keeps the first one."""
    g, w = _pair(task, rng, ctx)
    if w is None or single_block(ctx.contract.wrapper_for(task.id)):
        return []
    return [("retract", _two(ctx, RETRACT, w, g))]


def o5_multiple_wrapped(task, rng, ctx):
    g, w = _pair(task, rng, ctx)
    if w is None:
        return []
    w2 = ctx.wrong_answer(task, random.Random(rng.random())) or w
    return [("three", _two(ctx, "{0}\n{1}\n{2}", w, g, w2)), ("two_gold_first", _two(ctx, "{0}\n{1}" if ctx.opens_a_line() else "{0} {1}", g, w))]


def _hedges(a: str, b: str, ctx: ProbeContext) -> list[tuple[str, str]]:
    """'either a or b': bare, each answer wrapped, inside one wrapper, and (unboxed formats) as two $...$ spans."""
    boxed = "\\boxed{" in ctx.wrap("x")
    out = [("either", f"The answer is either {a} or {b}."),
           ("either_wrapped", _two(ctx, "The answer is either {0} or {1}.", a, b)),
           ("inside_wrapper", ctx.wrap(f"{a} \\text{{ or }} {b}" if boxed else f"{a} or {b}"))]
    if not boxed:
        out.append(("inside_wrapper_math", ctx.wrap(f"${a}$ or ${b}$")))
    return out


def o7_hedge(task, rng, ctx):
    g, w = _pair(task, rng, ctx)
    return [] if w is None else _hedges(g, w, ctx)


def o11_hedge_gold_last(task, rng, ctx):
    g, w = _pair(task, rng, ctx)
    return [] if w is None else _hedges(w, g, ctx)


def o6_gold_in_reasoning(task, rng, ctx):
    g, w = _pair(task, rng, ctx)
    if w is None:
        return []
    return [("reasoning", f"Let me set up the problem. Using only the first condition, so we get {g}. "
                          f"But that ignores the second condition, so the value has to change. "
                          f"Applying both conditions and finishing the computation:\n\n{ctx.wrap(w)}")]


def o10_gold_in_think(task, rng, ctx):
    g, w = _pair(task, rng, ctx)
    if w is None:
        return []
    note = f"My first guess is {g}, but I should check it against every condition before answering.\n"
    return [("think", ctx.think(note, ctx.wrap(w)))]


OTHER_MARKS = ("boxed", "answer_tag", "hash", "final_answer", "answer_colon", "the_final_answer_is")


def _other_mark(task: Task, ctx: ProbeContext) -> str | None:
    """Another format this checker accepts, with a mark of its own. In a format it does not accept, a second answer is only
    text to it, and rightly ignored."""
    written = ctx.split_wrap()[1]("x")
    for wid in OTHER_MARKS:
        other = wrapper_fn(wid)("x")
        if wid in ctx.contract.accepted and other not in written and (written == "x" or written not in other):
            return wid
    return None


def o12_two_formats(task, rng, ctx):
    g, w = _pair(task, rng, ctx)
    other = _other_mark(task, ctx)
    if w is None or other is None:
        return []
    pre, mark = ctx.split_wrap()
    second = wrapper_fn(other)
    return [("right_in_accepted", f"{pre}{mark(g)}\n{second(w)}"), ("right_in_other", f"{pre}{mark(w)}\n{second(g)}")]


def o13_gold_only_in_think(task, rng, ctx):
    written = ctx.wrap("x")
    body = f"Let me work this out.\nSo the value is {ctx.split_wrap()[1](ctx.gold(task))}\n"
    return [("think_only", (body if "</think>" in written and "<think>" not in written else f"<think>\n{body}") + "</think>")]


def o14_letter_and_other_text(task, rng, ctx):
    letter, texts = to_letter(ctx.gold(task)), option_texts(task.prompt)
    others = [o for o in texts if o != letter and normalize(texts[o]) != normalize(texts.get(letter, ""))]
    if letter not in texts or not others:
        return []
    other = rng.choice(others)
    return [("right_letter", ctx.wrap(f"{letter}) {texts[other]}")), ("right_text", ctx.wrap(f"{other}) {texts[letter]}"))]


def o16_two_letters(task, rng, ctx):
    """The right letter and another option's, in the order of the alphabet, as a reply naming several options writes them;
    the other one at random, so the right one comes first on about half the rows."""
    letter = to_letter(ctx.gold(task))
    others = [o for o in ctx.options(task) if o != letter]
    if not letter or not others:
        return []
    a, b = sorted((letter, rng.choice(others)))
    return [("joined", ctx.wrap(a + b)), ("comma", ctx.wrap(f"{a}, {b}")), ("slash", ctx.wrap(f"{a}/{b}"))]


def o15_interval(task, rng, ctx):
    g = ctx.gold(task)
    x = to_number(g)
    return [("between", ctx.wrap(f"between {fmt_num(x - 1)} and {fmt_num(x + 1)}")), ("plus_minus", ctx.wrap(f"{g} ± 1")),
            ("inequality", ctx.wrap(f"{fmt_num(x - 2)} < x < {fmt_num(x + 3)}"))]


PROBES = [
    Probe("O3", "O", "right answer stated but not committed to (retracted, negated, hypothetical)", "natural", "zero", NS,
          o3_not_committed, by_variant=True,
          description="Sends the reference answer and then a wrong final answer, both in the accepted format: "
                      "'wait, that is not right', or 'The answer is not <reference>. The answer is <wrong>.', or the "
                      "reference answer inside an 'if the problem asked something else' clause; a sound checker "
                      "gives zero."),
    Probe("O4", "O", "wrong answer retracted in favour of the right one", "natural", "full", SOL, o4_retract_to_gold,
          applies=lambda t, c: not asks_answer_only(t),
          description="Sends a wrong answer, then 'wait, that is not right' and the reference answer as the final "
                      "answer; a sound checker gives full credit. Not sent for formats with one answer block "
                      "(<answer> tags, JSON, a code block, a '####' or 'Answer:' line), where two blocks are two "
                      "submissions, nor when the prompt asks for only the answer."),
    Probe("O5", "O", "several answers in the accepted format", "natural", "zero", NS, o5_multiple_wrapped,
          description="Sends two or three answers in the accepted format, one of them the reference answer; a "
                      "checker that accepts any match gives credit, a sound one gives zero."),
    Probe("O6", "O", "right answer in the reasoning, wrong final answer", "natural", "zero", NS, o6_gold_in_reasoning,
          description="Sends reasoning that mentions the reference answer as an intermediate value, then a wrong "
                      "final answer in the accepted format; a sound checker reads the final answer and gives zero."),
    Probe("O7", "O", "hedge between two answers", "natural", "zero", NS, o7_hedge,
          description="Sends 'either <reference> or <wrong>', as plain text, with each answer in the accepted "
                      "format, and with both inside one format; a sound checker gives zero."),
    Probe("O11", "O", "hedge between two answers, the right one last", "natural", "zero", NS, o11_hedge_gold_last,
          description="Sends 'either <wrong> or <reference>', as plain text, with each answer in the accepted "
                      "format, and with both inside one format; a checker that reads only the last answer accepts "
                      "it, a sound one gives zero."),
    Probe("O10", "O", "right answer in the thinking block, wrong final answer", "natural", "zero", NS,
          o10_gold_in_think,
          description="Sends the reference answer inside <think>...</think> and a wrong answer in the accepted "
                      "format after it; a sound checker reads only the answer section and gives zero."),
    Probe("O12", "O", "a right and a wrong answer in two accepted formats", "natural", "zero", NS, o12_two_formats,
          description="Where the checker accepts two formats with their own marks, sends the reference answer in one and a "
                      "wrong answer in the other, both ways round; a checker that reads one and ignores the other accepts "
                      "one of them, a sound one gives zero."),
    Probe("O13", "O", "right answer only in the thinking block", "natural", "zero", NS, o13_gold_only_in_think,
          description="Sends the reference answer in a <think>...</think> block with nothing after it, even if the "
                      "accepted format has no such block; a checker that reads the whole reply accepts it, a sound one "
                      "gives zero, since the reply has no answer."),
    Probe("O14", "O", "an option's letter with another option's text", "natural", "zero", NS, o14_letter_and_other_text,
          applies=lambda t, c: c.gtype(t) == "mcq",
          description="Sends the right letter with a wrong option's text, and a wrong letter with the right option's text; "
                      "a checker that reads only one of the two accepts one of them, a sound one gives zero."),
    Probe("O15", "O", "a range around the right answer", "natural", "zero", NS, o15_interval,
          applies=lambda t, c: c.numeric(t) and plain_number(c.gold(t)),
          description="Sends a range that holds the numeric reference answer: 'between 41 and 43', '42 ± 1', '40 < x < 45'; "
                      "a sound checker gives zero, since a range commits to no answer."),
    Probe("O16", "O", "two option letters, one of them right", "natural", "zero", NS, o16_two_letters,
          applies=lambda t, c: c.gtype(t) == "mcq", by_variant=True,
          description="Sends the right option letter together with another one, as BC, B, C and B/C in the accepted "
                      "format; a checker that reads the first or the last letter, or looks for the right one anywhere, "
                      "accepts some of them, a sound one gives zero."),
]
