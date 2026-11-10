"""Helpers several test files share; they import them with `from conftest import ...`."""

from rewardaudit.metrics import Tally
from rewardaudit.types import Kind, Task, Verdict


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
