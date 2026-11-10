"""Probes on a boolean gold use the dataset's own words."""

from __future__ import annotations

import random

from rewardaudit.contract import _wrong_answer
from rewardaudit.probes.base import ProbeContext
from rewardaudit.probes.candidates import e3_bool, e6_final_answer_lines
from rewardaudit.probes.mutate import m16_bool_flip
from rewardaudit.types import Task


class _Contract:
    primary = "bare"

    def wrap(self, answer, task_id=None):
        return answer


def _ctx(tasks):
    return ProbeContext(contract=_Contract(), gold_type="bool", tasks=tasks, gold_types={t.id: "bool" for t in tasks})  # type: ignore[arg-type]


def test_a_wrong_answer_stays_in_the_gold_vocabulary():
    tasks = [Task(str(i), "q", "NO") for i in range(3)]
    assert _ctx(tasks).wrong_answer(tasks[0], random.Random(0)) == "YES"
    assert [c for _, c in m16_bool_flip(tasks[0], random.Random(0), _ctx(tasks))] == ["YES"]
    assert _wrong_answer(tasks[0], tasks) == "YES"  # every other gold is the same, so the contract needs the opposite too


def test_a_sweep_of_both_answers_does_not_tell_which_is_the_gold():
    yes, no = Task("0", "q", "yes"), Task("1", "q", "no")
    ctx = _ctx([yes, no])
    assert dict(e3_bool(yes, random.Random(0), ctx))["wrapped"] == dict(e3_bool(no, random.Random(0), ctx))["wrapped"]
    assert "yes" in dict(e3_bool(yes, random.Random(0), ctx))["wrapped"] and "true" not in dict(e3_bool(yes, random.Random(0), ctx))["wrapped"]
    assert e6_final_answer_lines(yes, random.Random(0), ctx) == e6_final_answer_lines(no, random.Random(0), ctx)
