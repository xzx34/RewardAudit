"""Dataset checks that need no checker calls. They run on every row, not only on the sampled ones."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from rewardaudit.equiv import contains_equiv, normalize, to_letter
from rewardaudit.goldtype import infer_dataset_type, infer_gold_type
from rewardaudit.prompt import mcq_options
from rewardaudit.types import Task

_SPACE = re.compile(r"\s+")


def run_dataset_checks(tasks: list[Task], left_out: int = 0) -> list[dict[str, Any]]:
    """Each issue: id, name, severity, count, task ids (at most ten) and a note. `left_out`: rows that had an empty
    reference answer and never became tasks (`load_tasks`)."""
    n = len(tasks)
    if n == 0:
        return []
    gold_type, _ = infer_dataset_type(tasks)
    issues: list[dict[str, Any]] = []

    if left_out:
        rows = f"{left_out:,} rows with an empty reference were" if left_out > 1 else "1 row with an empty reference was"
        issues.append({"id": "empty_rows_left_out", "name": f"{rows} left out", "count": left_out, "severity": "low",
                       "note": "they were not sampled; a reply to them could not be scored against anything"})

    # an environment loads its own tasks, and they can still have an empty answer
    empty = [t.id for t in tasks if not t.answer.strip()]
    if empty:
        issues.append({"id": "empty_answers", "name": "empty reference answers", "count": len(empty), "tasks": empty[:10],
                       "severity": "high", "note": "these rows cannot be scored against anything"})

    seen: dict[str, str] = {}
    dups: list[list[str]] = []
    for t in tasks:
        key = _SPACE.sub(" ", t.prompt.strip().lower())
        if key in seen:
            dups.append([seen[key], t.id])
        else:
            seen[key] = t.id
    if dups:
        issues.append({"id": "duplicate_prompts", "name": "duplicate prompts", "count": len(dups), "tasks": dups[:10],
                       "severity": "medium", "note": "the same prompt appears in more than one row"})

    leaks = [t.id for t in tasks if t.answer.strip() and contains_equiv(t.prompt, t.answer, infer_gold_type(t.answer))]
    if gold_type in ("int", "float", "fraction", "latex", "short_str", "long_text", "list") and leaks:
        frac, numeric = len(leaks) / n, gold_type in ("int", "float", "fraction")
        issues.append({"id": "answer_in_prompt", "name": "reference answer appears in the prompt", "count": len(leaks),
                       "tasks": leaks[:10],
                       "severity": ("high" if frac > 0.5 else "low") if numeric else ("high" if frac > 0.2 else "medium"),
                       "note": "the number also appears among the question's own figures" if numeric
                       else "the answer's text occurs inside the prompt"})

    golds = Counter(normalize(t.answer) for t in tasks)
    mode, k = golds.most_common(1)[0]
    if k / n >= 0.2 and len(golds) > 1:
        issues.append({"id": "skewed_answers", "name": "skewed reference answers", "count": k,
                       "severity": "high" if k / n >= 0.4 else "medium",
                       "note": f"always answering '{mode}' would be right on {k / n:.0%} of rows"})

    if gold_type == "mcq":
        bad = [t.id for t in tasks if mcq_options(t.prompt) and (to_letter(t.answer) or "") not in mcq_options(t.prompt)]
        if bad:
            issues.append({"id": "answer_not_an_option", "name": "reference answer is not among the listed options",
                           "count": len(bad), "tasks": bad[:10], "severity": "high", "note": ""})
    return issues
