"""Helpers several test files share; they import them with `from conftest import ...`."""

from rewardaudit.contract import Contract
from rewardaudit.goldtype import infer_gold_type
from rewardaudit.metrics import Tally
from rewardaudit.probes.base import ProbeContext
from rewardaudit.types import Kind, Task, Verdict


def probe_context(tasks, primary="boxed", gold_type="int", **kw):
    """A ProbeContext over `tasks` for a checker that accepts the format `primary`, every reference answer passing in it.
    `kw` go to ProbeContext: aliases, reference_is_answer."""
    ids = [t.id for t in tasks]
    contract = Contract(full_score=1.0, accepted=[primary], primary=primary, wrapper_rates={primary: 1.0},
                        self_pass=dict.fromkeys(ids, True), gold_score=dict.fromkeys(ids, 1.0),
                        task_wrappers={i: [primary] for i in ids})
    return ProbeContext(contract=contract, gold_type=gold_type, tasks=tasks,
                        gold_types={t.id: infer_gold_type(t.answer) for t in tasks}, **kw)


class FakeRunner:
    """In-process stand-in for the worker pool around `fn(completion, answer)`, which returns a score, (score,
    sub_scores), or an Exception for a call that raised."""

    max_score = None

    def __init__(self, fn, meta=None):
        self.fn, self.calls, self.meta = fn, 0, meta or {}

    def score(self, task: Task, completion: str) -> Verdict:
        self.calls += 1
        out = self.fn(completion, task.answer)
        if isinstance(out, Exception):
            return Verdict(Kind.ERROR, error=repr(out))
        if isinstance(out, tuple):
            return Verdict(Kind.SCORE, score=float(out[0]), sub_scores=out[1])
        return Verdict(Kind.SCORE, score=float(out))

    def score_many(self, items, progress=None):
        return [self.score(task, completion) for task, completion in items]


def compute(observations, probes, tasks, contract, gold_type, *, controls=None):
    """The audit's statistics over these calls, added in one batch."""
    tally = Tally(probes, tasks, contract, gold_type)
    tally.add(observations, controls)
    return tally.finish()
