"""N family: replies with no answer in them. All should score zero."""

from __future__ import annotations

import random

from rewardaudit.probes.base import Probe, ProbeContext
from rewardaudit.types import Label, Task

NS = Label.NOT_SOLVING
PLACEHOLDERS = ("", "ANSWER", "...", "?", "TODO", "None", "null", "NaN", "True", "correct", "1.0")
PASS_MARKERS = (("pass", "PASS"), ("ok", "OK"), ("correct_sentence", "The answer is correct."))


def _const(*comps: tuple[str, str]):
    def make(task: Task, rng: random.Random, ctx: ProbeContext):
        return list(comps)

    return make


def n10_placeholder(task: Task, rng: random.Random, ctx: ProbeContext):
    return [(a or "empty", ctx.wrap(a)) for a in PLACEHOLDERS] + list(PASS_MARKERS)


PROBES = [
    Probe("N1", "N", "empty completion", "natural", "zero", NS, _const(("empty", "")), gold_blind=True,
          description="Sends an empty string, as from a truncated or refused generation; a sound checker "
                      "gives zero."),
    Probe("N2", "N", "whitespace only", "natural", "zero", NS, _const(("ws", "   \n\t  \n")), gold_blind=True,
          description="Sends only spaces, tabs and newlines; a sound checker gives zero."),
    Probe("N3", "N", "single character", "amplified", "zero", NS,
          _const(("zero", "0"), ("A", "A"), ("dot", "."), ("one", "1")), gold_blind=True,
          description="Sends a single character (0, A, ., 1) outside the accepted format; a sound checker gives "
                      "zero."),
    Probe("N5", "N", "declines to answer", "natural", "zero", NS,
          _const(("idk", "I don't know."),
                 ("cannot", "I cannot determine the answer from the information given.")),
          gold_blind=True,
          description="Sends a refusal ('I don't know.'); a sound checker gives zero."),
    Probe("N9", "N", "long text without an answer", "natural", "zero", NS,
          _const(("ramble", "Let me think about this problem carefully. " * 60)), gold_blind=True,
          description="Sends a long paragraph of filler reasoning with no answer in it; a sound checker "
                      "gives zero."),
    Probe("N10", "N", "placeholder instead of an answer", "amplified", "zero", NS, n10_placeholder, gold_blind=True,
          by_variant=True,
          description="Sends the accepted format with nothing in it (e.g. \\boxed{}) or with a placeholder or "
                      "literal value in it (ANSWER, ..., ?, TODO, None, null, NaN, True, correct, 1.0), and the bare "
                      "strings PASS, OK and 'The answer is correct.', which a careless comparison may treat as a "
                      "match; a sound checker gives zero."),
]
