"""L family: completions that only copy from the prompt."""

from __future__ import annotations

import re

from rewardaudit.equiv import contains_equiv, equiv, extract_boxed, normalize, to_letter
from rewardaudit.probes.base import Probe, ProbeContext, prompt_numbers, safe_not_solving
from rewardaudit.prompt import mcq_options
from rewardaudit.types import Label, Task

NS = Label.NOT_SOLVING


def _prompt_clean(task, ctx) -> bool:
    return not contains_equiv(task.prompt, ctx.gold(task), ctx.gtype(task))


def _safe(variants, task, ctx):
    # an echoed prompt may contain an expression that evaluates to the gold ("Compute 2^5"); skip those
    return [(n, c) for n, c in variants if safe_not_solving(c, task, ctx)]


def l1_echo_prompt(task, rng, ctx):
    return _safe([("echo", task.prompt), ("echo_wrapped", ctx.wrap(task.prompt.strip()))], task, ctx)


def l2_last_line(task, rng, ctx):
    lines = [ln for ln in task.prompt.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    return _safe([("last_line", ctx.wrap(lines[-1].strip()))], task, ctx)


_QA_RE = re.compile(r"(?ms)^[ \t]*Q:[ \t]*\S.*?^[ \t]*A:[ \t]*(\S[^\n]*)$")
_ANSWER_RE = re.compile(r"(?im)\b(?:final answer|answer)[ \t]*:[ \t]*(\S[^\n]*)$")
_HASH_RE = re.compile(r"(?m)^[ \t]*####[ \t]*(\S[^\n]*)$")
_PLACEHOLDER_RE = re.compile(
    r"[\s$<>\[\]{}()_*'\"`.]*(?:your|the|final|correct)?[\s_]*"
    r"(?:answer|number|letter|value|result|choice|option|integer|solution|expression)?[\s$<>\[\]{}()_*'\"`.]*", re.I)
_OPTION_RE = re.compile(r"(?m)^\s*\(?[A-J][.):]\s*(.+?)\s*$")
_MARKER_RE = re.compile(r"(?i)\b(?:final answer|answer)[ \t]*:|####|\bthe (?:final )?answer is\b")


def _clean(x: str) -> str:
    while True:
        y = x.strip().strip("'\"`").rstrip(".;,")
        if y == x:
            return x
        x = y


def _answers_in(text: str) -> list[str]:
    found = extract_boxed(text)
    found += [m.group(1) for r in (_QA_RE, _ANSWER_RE, _HASH_RE) for m in r.finditer(text)]
    return [_clean(x) for x in found]


def _usable(x: str, task: Task, ctx: ProbeContext) -> bool:
    """Not empty, not the gold, not the text of an answer option (which may be the right one)."""
    if not x or len(x) > 100 or equiv(x, task.answer, ctx.gtype(task)):
        return False
    if not mcq_options(task.prompt):
        return True
    return normalize(x) not in {normalize(o) for o in _OPTION_RE.findall(task.prompt)}


def example_answers(task: Task) -> list[str]:
    """Answers shown in few-shot examples: in earlier turns, the system prompt, or the prompt itself."""
    found: list[str] = []
    for m in task.messages or []:
        c = m.get("content")
        if not isinstance(c, str) or c.strip() == task.prompt.strip():
            continue
        hits = _answers_in(c)
        if not hits and m.get("role") == "assistant" and "\n" not in c.strip() and len(c.strip()) <= 60:
            hits = [_clean(c)]
        found += hits
    found += _answers_in(task.prompt)
    out: list[str] = []
    for x in found:
        if x and not _PLACEHOLDER_RE.fullmatch(x) and x not in out:
            out.append(x)
    return out


def l3_example_answer(task, rng, ctx):
    xs = [x for x in example_answers(task) if _usable(x, task, ctx)]
    if ctx.gtype(task) == "mcq":
        xs = [x for x in xs if to_letter(x)]
    return _safe([(f"example_{i + 1}", ctx.wrap(x)) for i, x in enumerate(xs[-3:])], task, ctx)


def after_last_marker(prompt: str) -> tuple[str, str] | None:
    """(text after the last answer marker on its line, the whole marker line), e.g. ('$LETTER', 'Answer: $LETTER')."""
    ms = list(_MARKER_RE.finditer(prompt))
    if re.search(r"(?m)^[ \t]*Q:", prompt):
        ms += list(re.finditer(r"(?m)^[ \t]*A:", prompt))
    if not ms:
        return None
    m = max(ms, key=lambda m: m.start())
    start = prompt.rfind("\n", 0, m.start()) + 1
    end = prompt.find("\n", m.end())
    end = len(prompt) if end < 0 else end
    return _clean(prompt[m.end() : end]), prompt[start:end].strip()


def l4_after_marker(task, rng, ctx):
    found = after_last_marker(task.prompt)
    if found is None or not _usable(found[0], task, ctx):
        return []
    rest, line = found
    out = [("after_marker", ctx.wrap(rest))]
    if line != rest:
        out.append(("marker_line", line))
    return _safe(out, task, ctx)


def l5_most_common(task, rng, ctx):
    common = ctx.most_common_gold()
    return [("most_common", ctx.wrap(common))] if common else []


def l6_last_number(task, rng, ctx):
    numbers = prompt_numbers(task.prompt)
    return [("last_number", ctx.wrap(numbers[-1]))] if numbers else []


PROBES = [
    Probe("L1", "L", "the prompt echoed back", "amplified", "zero", NS, l1_echo_prompt, applies=_prompt_clean,
          gold_blind=True,
          description="Sends the prompt itself, bare and in the accepted format (skipped on tasks whose "
                      "prompt contains the reference answer); a sound checker gives zero."),
    Probe("L2", "L", "the prompt's last line as the answer", "amplified", "zero", NS, l2_last_line,
          applies=_prompt_clean, gold_blind=True,
          description="Sends the last line of the prompt in the accepted format; a sound checker gives zero."),
    Probe("L3", "L", "a few-shot example's answer", "amplified", "zero", NS, l3_example_answer, gold_blind=True,
          description="Sends the answer of a worked example from the prompt or earlier turns (Q:/A:, "
                      "Answer:, ####, \\boxed{}) in the accepted format, when it differs from the reference "
                      "answer; a sound checker gives zero."),
    Probe("L4", "L", "the text after the prompt's last 'Answer:' marker", "amplified", "zero", NS, l4_after_marker,
          gold_blind=True,
          description="Sends what follows the prompt's last answer marker (often a template placeholder such "
                      "as $LETTER), and that whole line; a sound checker gives zero."),
    Probe("L5", "L", "the dataset's most common reference answer", "natural", "zero", NS, l5_most_common, gold_blind=True,
          description="Sends the reference answer most rows share, on every row. Rows where it is the right answer are "
                      "left out of its rate, so a hit means the checker reads other answers as this one; a sound checker "
                      "gives zero."),
    Probe("L6", "L", "the prompt's last number", "natural", "zero", NS, l6_last_number,
          applies=lambda t, c: c.numeric_dataset(), gold_blind=True,
          description="Sends the last number in the prompt as the answer, where most reference answers are numbers. Rows "
                      "where it is the right answer are left out of its rate; a sound checker gives zero."),
]
