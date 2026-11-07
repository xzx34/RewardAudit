"""O family: a right and a wrong answer in one reply: retraction, negation, hedging, which one the checker reads."""

from __future__ import annotations

import random

from rewardaudit.goldtype import asks_answer_only
from rewardaudit.probes.base import Probe, ProbeContext
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
    return [("three", _two(ctx, "{0}\n{1}\n{2}", w, g, w2)), ("two_gold_first", _two(ctx, "{0} {1}", g, w))]


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
]
