"""P family: parser robustness. Malformed or extreme inputs that should score zero without an error, a timeout
or a crash. A reply the checker accepts is a finding; these calls stay out of the summary's rates (headline=False),
since they run on a few tasks only.

Each call that times out costs the full timeout plus a worker restart, so these probes run on the first five
passing tasks only, with few variants, and every completion stays well under 200 KB.
"""

from __future__ import annotations

import re

from rewardaudit.probes.base import Probe, ProbeContext, safe_not_solving
from rewardaudit.types import Label, Task

NS = Label.NOT_SOLVING
MAX_CHARS = 200_000


def _first_tasks(task: Task, ctx: ProbeContext) -> bool:
    return task.id in ctx.first_passing()


def _safe(variants, task, ctx):
    return [(n, c) for n, c in variants if len(c) <= MAX_CHARS and safe_not_solving(c, task, ctx)]


def p1_slow_regex(task, rng, ctx):
    return _safe([("digit_commas", "1," * 25_000 + "x"),
                  ("open_box_spaces", "\\boxed{" + " " * 50_000),
                  ("spaced_digits", "Final answer: " + "1 " * 25_000 + "!")], task, ctx)


UNBALANCED = (("open_box", "The answer is \\boxed{"), ("open_frac", "\\boxed{\\frac{1}{"), ("closers", "}}}}}}}})])]"),
              ("open_tag", "<answer>"), ("mismatched", "\\boxed{(]}"))
REGEX_PATTERNS = (("dot_star", ".*"), ("digits", "\\d+"), ("any", "[\\s\\S]*"), ("group", "(.*)|^$"))
SPECIAL_FLOATS = ("NaN", "inf", "-inf", "-0", "1e309")


def _nested(wrong: str, ctx: ProbeContext) -> list[tuple[str, str]]:
    """A wrong answer inside twenty wrappers: the accepted format, \\boxed{} and <answer> tags, each once if they coincide."""
    nested = wrong
    for _ in range(20):
        nested = ctx.wrap(nested)
        if len(nested) > MAX_CHARS:
            break
    out: list[tuple[str, str]] = []
    for name, comp in (("wrap20", nested), ("boxed20", "\\boxed{" * 20 + wrong + "}" * 20),
                       ("tag20", "<answer>" * 20 + wrong + "</answer>" * 20)):
        if comp not in [c for _, c in out]:
            out.append((name, comp))
    return out


def p2_malformed(task, rng, ctx):
    """Each group of variants has its own condition; one that does not apply leaves the others in place."""
    out = list(UNBALANCED) + [(n, ctx.wrap(v)) for n, v in REGEX_PATTERNS]
    g = ctx.gold(task).lower()
    if "inf" not in g and "nan" not in g:
        out += [(v, ctx.wrap(v)) for v in SPECIAL_FLOATS]
    wrong = ctx.wrong_answer(task, rng)
    if wrong is not None:
        out += _nested(wrong, ctx)
    return _safe(out, task, ctx)


TOWERS = (("py_tower", "9**9**9"), ("tex_tower", "9^{9^{9^{9}}}"), ("tex_tower2", "2^{2^{2^{2^{2^2}}}}"),
          ("py_tower_pi", "9**9**9\\pi"))  # a few bytes each; evaluating any of them does not finish


def p5_huge_numbers(task, rng, ctx):
    num = "".join(rng.choice("123456789") for _ in range(3000))
    den = "".join(rng.choice("123456789") for _ in range(3000))
    return _safe([("pow", ctx.wrap("10^{10000}")), ("long_frac", ctx.wrap(f"\\frac{{{num}}}{{{den}}}")),
                  ("long_slash", ctx.wrap(f"{num}/{den}"))] + [(n, ctx.wrap(v)) for n, v in TOWERS], task, ctx)


def gold_tower(gold: str) -> str | None:
    """The integers of an expression-valued gold as a Python power tower, largest first, so the tower uses the
    same numbers (e.g. '(37 + 15) * (4 / 13)' -> '37**15**13**4'); None with fewer than three integers >= 2."""
    nums = sorted((int(x) for x in re.findall(r"\d+", gold)), reverse=True)
    if len(nums) > 12 or sum(n >= 2 for n in nums) < 3:
        return None
    return "**".join(map(str, nums))


def p15_gold_tower(task, rng, ctx):
    tower = gold_tower(ctx.gold(task))
    return [] if tower is None else _safe([("gold_numbers", ctx.wrap(tower))], task, ctx)


PROBES = [
    Probe("P1", "P", "inputs that make regexes backtrack", "contrived", "zero", NS, p1_slow_regex,
          applies=_first_tasks, gold_blind=True, headline=False, by_variant=True,
          description="Sends ~50 KB runs (digits and commas, spaces after an open \\boxed{, spaced digits) "
                      "that make some answer-extraction regexes backtrack; a sound checker returns zero "
                      "within the timeout."),
    Probe("P2", "P", "malformed or extreme input", "contrived", "zero", NS, p2_malformed, applies=_first_tasks,
          headline=False, by_variant=True,
          description="Sends an unclosed \\boxed{ or <answer> tag and stray closing braces; NaN, inf, -inf, -0 and "
                      "1e309, and regex patterns such as .* and \\d+, each in the accepted format (the numbers left "
                      "out when the reference answer is infinite or equal to one of them); and a wrong answer inside "
                      "twenty nested formats. A sound checker returns zero without raising."),
    Probe("P15", "P", "the reference answer's numbers as a power tower", "contrived", "zero", NS, p15_gold_tower,
          applies=_first_tasks, headline=False,
          description="Sends the integers of an expression-valued reference answer joined by ** (e.g. 37**15**13**4) "
                      "in the accepted format; a checker that validates the numbers used and then evaluates the "
                      "expression hangs, a sound one returns zero within the timeout."),
    Probe("P5", "P", "huge numbers", "contrived", "zero", NS, p5_huge_numbers, applies=_first_tasks,
          gold_blind=True, headline=False, by_variant=True,
          description="Sends 10^{10000}, a fraction with 3000-digit parts and power towers (9**9**9, "
                      "9^{9^{9^{9}}}, 2^{2^{2^{2^{2^2}}}}, 9**9**9\\pi) in the accepted format; a sound checker "
                      "returns zero without hanging in symbolic comparison or evaluation."),
]
