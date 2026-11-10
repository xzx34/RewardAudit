"""Evidence under each high defect that a reader can check: its example sent again, and cut down to what is still accepted.

None of these calls counts in the rates or the findings.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from rewardaudit.contract import Contract
from rewardaudit.defects import RUN_AS_CODE
from rewardaudit.equiv import states_the_answer
from rewardaudit.goldtype import infer_gold_type
from rewardaudit.metrics import hit
from rewardaudit.probes.registry import all_probes
from rewardaudit.runner import RunnerPool
from rewardaudit.types import Label, Observation, Task, Verdict

RESENDS = 5
MAX_CUT_CALLS = 100  # per reply: once they are spent, the shortest reply so far stands
SENTENCE_END = re.compile(r"[.?!](?=\s)\s*|\n\s*")
WORD_END = re.compile(r"\s+")


def _pieces(text: str, end: re.Pattern[str]) -> list[str]:
    """`text` cut after each match of `end`, so that the pieces join back into it."""
    cuts = [0, *(m.end() for m in end.finditer(text)), len(text)]
    return [text[a:b] for a, b in pairwise(cuts) if b > a]


def _accepted(ex: dict[str, Any], reply: str, v: Verdict, contract: Contract, gold_blind: bool) -> bool:
    """Accepted as the probe statistics count it, for a reply in place of the example's."""
    o = Observation(ex["probe_id"], ex["family"], ex["variant"], ex["task_id"], reply, v, Label(ex["label"]), ex["expect"],
                    ex["tier"])
    return hit(o, contract, gold_blind)[0]


def _shortest_accepted(runner: RunnerPool, task: Task, ex: dict[str, Any], contract: Contract, gold_blind: bool,
                       progress: Callable[[int], None] | None) -> dict[str, Any] | None:
    """The example with sentences, then whitespace-separated words, taken out for as long as the rest is still accepted:
    runs of half the pieces first, then of a quarter, down to one piece at a time, until no single piece can go. A rest
    that states the reference answer alone is not sent, since accepting it is right. None when nothing could be taken out."""
    gold_type, text, calls, best = infer_gold_type(task.answer), ex["completion"], 0, None
    strip = not text[-1:].isspace()  # a space that a cut leaves at the end goes too, unless the reply itself ends in one
    for end in (SENTENCE_END, WORD_END):
        pieces = _pieces(text, end)
        size = max(1, len(pieces) // 2)
        while calls < MAX_CUT_CALLS:
            i, cut = 0, False
            while i < len(pieces) and calls < MAX_CUT_CALLS:
                rest = pieces[:i] + pieces[i + size:]
                reply = "".join(rest).rstrip() if strip else "".join(rest)
                if states_the_answer(reply, task.answer, gold_type, contract.derived):
                    i += size
                    continue
                calls += 1
                v = runner.score(task, reply)
                if progress:
                    progress(1)
                if _accepted(ex, reply, v, contract, gold_blind):
                    pieces, best, cut = rest, {"completion": reply, "score": v.score}, True
                else:
                    i += size
            if size == 1 and not cut:
                break
            size = max(1, size // 2)
        text = "".join(pieces)
    return best


def confirm(runner: RunnerPool, defects: list[dict[str, Any]], tasks: list[Task], contract: Contract,
            progress: Callable[[int], None] | None = None) -> None:
    """Adds `confirmation` to each high defect with an example: `resends`, the scores of RESENDS more calls with the same
    reply (None for a call that returned no score), and, for a wrong answer, hedge or non-answer the checker accepted on
    every one of them, `shortest_accepted`: the shortest reply cut from it that is still accepted, with its score. A right
    answer rejected and a reply that reached an import are only sent again. `progress` is called with the calls just made."""
    by_id = {t.id: t for t in tasks}
    blind = {p.id for p in all_probes() if p.gold_blind}
    for d in defects:
        ex = d["example"]
        if d["severity"] != "high" or not ex:
            continue
        task = by_id[ex["task_id"]]
        verdicts = runner.score_many([(task, ex["completion"])] * RESENDS, progress=progress)
        d["confirmation"] = {"resends": [v.score if v.ok else None for v in verdicts]}
        gold_blind = ex["probe_id"] in blind
        # a reply that should earn nothing is the example of wrong_accepted, hedging_accepted and non_solving_accepted;
        # of the robustness findings, only a reply that reached an import has an example
        if (ex["label"] == Label.NOT_SOLVING.value and d["id"] != RUN_AS_CODE
                and all(_accepted(ex, ex["completion"], v, contract, gold_blind) for v in verdicts)):
            shortest = _shortest_accepted(runner, task, ex, contract, gold_blind, progress)
            if shortest is not None:
                d["confirmation"]["shortest_accepted"] = shortest
