"""A probe is a generator: given a task it produces completions with a known label."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from rewardaudit.contract import Contract
from rewardaudit.equiv import (
    contains_equiv,
    equiv,
    flip_bool,
    normalize,
    strip_wrappers,
    to_letter,
    to_number,
    within_list,
)
from rewardaudit.goldtype import is_numeric
from rewardaudit.prompt import mcq_options
from rewardaudit.types import Label, Task

# (variant name, completion), or with a third item, the reference answer written another way for that call only (G3)
Variant = tuple[str, str] | tuple[str, str, str]
_SENTINEL = "QQANSWERQQ"
NOT_LATEX = ("json", "grid", "code")  # gold types whose variants are not LaTeX; the LaTeX guard skips them
MIN_WORD = 4  # fewer letters can name a math object, where order or case carries meaning: BE and EB are one line

_CMD_RE = re.compile(r"\\([A-Za-z]+)")
COMMON_COMMANDS = frozenset(
    "frac dfrac tfrac sqrt boxed text textbf mathrm mathbf mathit operatorname left right cdot times div pm mp "
    "pi infty le leq ge geq ne neq approx circ degree in notin cup cap setminus emptyset varnothing subset subseteq "
    "begin end pmatrix bmatrix matrix cases sin cos tan cot sec csc arcsin arccos arctan log ln exp lim sum prod int "
    "alpha beta gamma delta epsilon theta lambda mu sigma phi omega Delta Omega dots ldots cdots quad qquad "
    "displaystyle angle triangle perp parallel overline sim cong equiv mod pmod binom choose".split())


def latex_commands(s: str) -> set[str]:
    return set(_CMD_RE.findall(s))


_ARITY = {"frac": 2, "dfrac": 2, "tfrac": 2, "binom": 2, "sqrt": 1}
_ARGS_RE = re.compile(r"\\(frac|dfrac|tfrac|binom|sqrt)(?![A-Za-z])")


def _group_end(s: str, i: int) -> int | None:
    """Index just past the brace group opening at s[i] ('{'), skipping escaped characters; None if unclosed."""
    depth = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
        if depth == 0:
            return i
    return None


def _missing_argument(s: str) -> bool:
    """A \\frac, \\binom or \\sqrt without all its arguments (a brace group or one token each)."""
    for m in _ARGS_RE.finditer(s):
        i: int | None = m.end()
        if m.group(1) == "sqrt" and s[i:i + 1] == "[":
            i = s.find("]", i) + 1 or None
        for _ in range(_ARITY[m.group(1)]):
            while i is not None and i < len(s) and s[i].isspace():
                i += 1
            if i is None or i >= len(s) or s[i] in "}]^_&$":
                return True
            if s[i] == "{":
                i = _group_end(s, i)
            elif s[i] == "\\":
                k = i + 1
                while k < len(s) and s[k].isalpha():
                    k += 1
                i = max(k, i + 2)
            else:
                i += 1
    return False


def latex_defects(s: str, known: frozenset[str] | set[str] = COMMON_COMMANDS) -> set[str]:
    """Ways `s` is not well-formed LaTeX: unbalanced braces, an empty brace group, a dangling ^, _ or backslash,
    a \\frac or \\sqrt missing an argument, \\left without \\right, \\begin without its \\end, or a command name
    outside `known`."""
    out: set[str] = set()
    depth, i = 0, 0
    while i < len(s):
        if s[i] == "\\":
            i += 2  # an escaped character (\\{ \\} \\\\) or the first letter of a command
            continue
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        if depth < 0:
            out.add("braces")
            depth = 0
        i += 1
    if depth:
        out.add("braces")
    if re.search(r"(?<!\\)\{\s*\}", s):
        out.add("empty_group")
    if re.search(r"(?<!\\)[\^_]\s*(?:$|[}$])", s) or re.search(r"(?<!\\)\\$", s.rstrip()):
        out.add("dangling")
    if _missing_argument(s):
        out.add("arguments")
    if len(re.findall(r"\\left(?![A-Za-z])", s)) != len(re.findall(r"\\right(?![A-Za-z])", s)):
        out.add("left_right")
    if sorted(re.findall(r"\\begin\{([^{}]*)\}", s)) != sorted(re.findall(r"\\end\{([^{}]*)\}", s)):
        out.add("environments")
    out |= {f"command:{c}" for c in latex_commands(s) - set(known)}
    return out


def malformed_latex(variant: str, reference: str, known: frozenset[str] | set[str] = COMMON_COMMANDS) -> bool:
    """True if `variant` has a LaTeX defect that `reference` (the gold, in the same wrapper) does not have.
    Command names in the reference count as known."""
    known = set(known) | latex_commands(reference)
    return bool(latex_defects(variant, known) - latex_defects(reference, known))


def away(x: float, step: float = 1) -> float:
    """`x` moved away from zero by at least `step` and at least its own size: x + 1 below 1, 2x from 1 on. No relative
    tolerance short of 100% takes it for `x`, where x + 1 is within 5% of x from 20 on."""
    return x + math.copysign(max(step, abs(x)), x) if x else step


def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.6g}"


UNIT_RE = re.compile(r"(?P<number>-?\d[\d,.]*)(?P<gap>\s*(?:\\[,; ])?\s*(?:\\(?:text|mathrm)\{\s*)?)"
                     r"(?P<unit>km|cm|kg|ms|mol|mL|Pa|Hz|°C|m|g|s|K|L|J|W|N)(?P<close>\}?)")
_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def plain_number(s: str) -> bool:
    """Digits and at most a sign and a decimal point, so characters can be added to it or spelled in other ways."""
    return bool(_PLAIN_NUMBER.fullmatch(s))


# Neutral text: no numbers, no option letters, no yes or no, nothing a checker reads as an answer.
_FILLER = ("Each step follows from the step before it.", "The reasoning here can be checked line by line.",
           "Nothing in the earlier work needs to change.", "Going over the work again leads to the same place.",
           "The conditions in the problem are all used along the way.", "It helps to write the quantities down before combining them.",
           "Another reading of the problem gives the same outcome.", "The same idea applies when the problem is stated in other words.")

OTHER_TASKS = 30
FIRST_PASSING = 5


@dataclass
class ProbeContext:
    contract: Contract
    gold_type: str
    tasks: list[Task]
    gold_types: dict[str, str] = field(default_factory=dict)  # per-task
    aliases: dict[str, list[str]] = field(default_factory=dict)  # per task: the row's other accepted answers, for G2
    reference_is_answer: bool = False  # the checker gets the row's reference answer itself, or a template around it, for G3
    task: Task | None = None  # the task currently being probed; set by the audit loop
    _commands: frozenset[str] | None = field(default=None, init=False, repr=False)
    _numeric_dataset: bool | None = field(default=None, init=False, repr=False)
    _first_passing: frozenset[str] | None = field(default=None, init=False, repr=False)
    _most_common: str | None = field(default=None, init=False, repr=False)

    def first_passing(self) -> frozenset[str]:
        """The first few tasks whose reference answer passes, for probes whose calls are too costly to run on every task."""
        if self._first_passing is None:
            self._first_passing = frozenset(self.contract.passing_tasks()[:FIRST_PASSING])
        return self._first_passing

    def filler(self, task: Task, rng: random.Random) -> Callable[[int], str] | None:
        """A function giving `size` characters of neutral text that holds no reference answer of this task; None if every
        sentence holds it."""
        pool = [s for s in _FILLER if not contains_equiv(s, self.gold(task), self.gtype(task))]
        return (lambda size: " ".join(rng.choices(pool, k=size // min(map(len, pool)) + 1))[:size]) if pool else None

    def most_common_gold(self) -> str:
        """The reference answer that most rows share, '' if no two rows share one."""
        if self._most_common is None:
            spelled = {normalize(t.answer): self.gold(t) for t in reversed(self.tasks)}
            answer, count = Counter(normalize(t.answer) for t in self.tasks).most_common(1)[0] if self.tasks else ("", 0)
            self._most_common = spelled[answer] if count >= 2 else ""
        return self._most_common

    def wrap(self, answer: str) -> str:
        """Wrap in the format this checker accepts, choosing a wrapper that passed on the current task."""
        return self.contract.wrap(answer, self.task.id if self.task else None)

    def split_wrap(self) -> tuple[str, Callable[[str], str]]:
        """The accepted format as (preamble, marker): the thinking block or reasoning text that the wrapper puts
        before the answer (the same for every answer, '' for most formats) and a function writing one answer
        without it, so a completion with two answers has one thinking block."""
        full = self.wrap(_SENTINEL)
        head = full.split(_SENTINEL, 1)[0]
        pre = head[: head.rfind("\n") + 1]
        if not ("</think>" in pre or pre.endswith("\n\n")):
            pre = ""

        def mark(answer: str) -> str:
            w = self.wrap(answer)
            return w[len(pre):] if w.startswith(pre) else w

        return pre, mark

    def opens_a_line(self) -> bool:
        """The accepted format is a mark and then the answer, with nothing after it ('#### 42', 'Answer: 42'). Checkers often
        read such a line from its start, so a reply writes it on a line of its own."""
        written = self.split_wrap()[1](_SENTINEL)
        return written.endswith(_SENTINEL) and written != _SENTINEL

    def think(self, note: str, answer: str) -> str:
        """`note` as thinking before `answer` (a wrapped answer): inside the wrapper's own thinking block, at the
        start when the prompt has already opened one, else in a new block."""
        if "<think>\n" in answer:
            return answer.replace("<think>\n", "<think>\n" + note, 1)
        if "</think>" in answer:  # open-think format: the completion starts inside the prompt's <think>
            return note + answer
        return f"<think>\n{note}</think>\n\n{answer}"

    def known_commands(self) -> frozenset[str]:
        """LaTeX command names that are real here: common ones plus every name used in a gold of this dataset."""
        if self._commands is None:
            self._commands = COMMON_COMMANDS | frozenset(c for t in self.tasks for c in latex_commands(t.answer))
        return self._commands

    def malformed(self, completion: str, task: Task) -> bool:
        """The completion has a LaTeX defect (unbalanced braces, an empty group, a broken command) that the gold
        in the same wrapper does not have. Never true for json / grid / code golds."""
        if self.gtype(task) in NOT_LATEX:
            return False
        return malformed_latex(completion, self.contract.wrap(self.gold(task), task.id), self.known_commands())

    def gtype(self, task: Task) -> str:
        return self.gold_types.get(task.id, self.gold_type)

    def gold(self, task: Task) -> str:
        """The gold as the dataset gives it, minus answer markup (\\boxed{}, $…$, tags). Trailing punctuation and
        outer parentheses stay: 'Paris.' and '(1, 2)' are what an exact-match checker compares against."""
        return strip_wrappers(task.answer, keep_parens=True, keep_punct=True)

    def numeric(self, task: Task) -> bool:
        return is_numeric(self.gtype(task)) and to_number(task.answer) is not None

    def numeric_dataset(self) -> bool:
        """Most golds here are numbers. Decides only what a task's prompt leaves open: in a word dataset a number the
        prompt says nothing about can be a version ('4.4'), not a quantity."""
        if self._numeric_dataset is None:
            self._numeric_dataset = 2 * sum(self.numeric(t) for t in self.tasks) > len(self.tasks)
        return self._numeric_dataset

    def options(self, task: Task) -> list[str]:
        return mcq_options(task.prompt) or ["A", "B", "C", "D"]

    def other_tasks(self, rng: random.Random) -> list[Task]:
        """A few tasks at random, for another task's answer: comparing against all of them would make the audit quadratic
        in the sample."""
        return rng.sample(self.tasks, min(OTHER_TASKS, len(self.tasks)))

    def wrong_answer(self, task: Task, rng: random.Random) -> str | None:
        """A plausible answer that is definitely not equivalent to the gold, and not a part of it: next to a list gold,
        a subset adds nothing, so a hedge between the two reads as the gold. A number moves away from zero first (see
        `away`) and is never a hundred times or a hundredth of the gold, so that neither a checker's tolerance nor its
        allowance for percentages makes it a second right answer."""
        g = self.gold(task)
        gt = self.gtype(task)
        cands: list[str] = []
        if self.numeric(task):
            x = to_number(g)
            assert x is not None
            m = re.fullmatch(r"(-?)\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", g) or re.fullmatch(r"(-?)(\d+)\s*/\s*(\d+)", g)
            if m:  # keep the fraction's spelling, change the numerator
                sign, num, den = m.group(1), int(m.group(2)), m.group(3)
                spelled = (lambda a: f"{sign}\\frac{{{a}}}{{{den}}}") if "frac" in g else (lambda a: f"{sign}{a}/{den}")
                cands = [spelled(int(away(num, int(den)))), spelled(num + 1), spelled(num + 2), spelled(num * 2 if num else 1)]
            elif float(x).is_integer() and "." not in g:
                cands = [str(int(away(int(x)))), str(int(x) + 1), str(int(x) - 1), str(int(x) * 2 if x != 0 else 7)]
            else:
                cands = [fmt_num(away(x)), fmt_num(x + 1), fmt_num(x * 2 if x != 0 else 0.5), fmt_num(-x if x != 0 else 1)]
            cands = [c for c in cands if x == 0 or not any(math.isclose(to_number(c), x * k) for k in (100, 0.01))]  # 100/99 for 1/99
        elif gt == "mcq":
            letter = to_letter(g) or "A"
            cands = [o for o in self.options(task) if o != letter] or ["Z"]
        elif gt == "bool":
            cands = [flip_bool(g)]
        else:
            structured = gt in ("json", "list", "grid")
            if structured:
                from rewardaudit.probes.structured import structured_wrong

                cands = structured_wrong(task, self, rng)
            others = [self.gold(t) for t in self.other_tasks(rng) if t.id != task.id and not equiv(t.answer, task.answer, gt)]
            cands += others[:3] + ([] if structured else [g + "x"]) + [g[::-1]]  # a parser would still read g in 'gx'
        for c in cands:
            if (c and not equiv(c, g, gt) and not (gt == "list" and within_list(c, g))
                    and (gt in NOT_LATEX or not malformed_latex(c, g, self.known_commands()))):
                return c
        return None

    def distractor_numbers(self, task: Task, rng: random.Random, k: int = 3) -> list[str]:
        """Numbers that do not equal the gold, to sprinkle into explanations."""
        out: list[str] = []
        pool = [3, 7, 12, 17, 21, 26, 48, 99, 120, 250]
        rng.shuffle(pool)
        for n in pool:
            if not equiv(str(n), task.answer, self.gtype(task)):
                out.append(str(n))
            if len(out) >= k:
                break
        return out


@dataclass
class Probe:
    id: str
    family: str
    name: str
    tier: str  # natural | amplified | contrived
    expect: str  # zero | full
    label: Label
    make: Callable[[Task, random.Random, ProbeContext], list[Variant]]
    applies: Callable[[Task, ProbeContext], bool] = lambda t, c: True
    gold_blind: bool = False  # constructed without reading the gold (a policy could learn it)
    headline: bool = True  # its calls count in the summary's rates (False: an accepted reply is still a finding)
    description: str = ""
    by_variant: bool = False  # distinct replies under one id: its finding reports the reply accepted most often
    sweep: bool = False  # variants grow along one scale: its finding gives the rate of each and where the behaviour ends

    def generate(self, task: Task, rng: random.Random, ctx: ProbeContext) -> list[Variant]:
        """The probe's variants for one task. M variants that are malformed LaTeX are dropped: a grader may
        repair them back to the gold, so they are not definite wrong answers. A variant with a reference answer of its own
        keeps it."""
        if not self.applies(task, ctx):
            return []
        out = [v for v in self.make(task, rng, ctx) if v[1] is not None]
        if self.family == "M":
            out = [v for v in out if not ctx.malformed(v[1], task)]
        return out


def safe_not_solving(comp: str, task: Task, ctx: ProbeContext) -> bool:
    """Guard for NOT_SOLVING probes: drop variants that accidentally contain the gold."""
    return not contains_equiv(comp, ctx.gold(task), ctx.gtype(task))


def prompt_numbers(prompt: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", prompt)
