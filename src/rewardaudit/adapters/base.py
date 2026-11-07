"""The minimal interface every audited checker exposes, and how tasks are sampled."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rewardaudit.types import Task


@dataclass
class RawScore:
    score: float
    sub_scores: dict[str, float] | None = None
    parsed: str | None = None
    network: bool = False  # the checker asked for a model the adapter stands in for; the worker's guard sees sockets only


def unique_ids(ids: list[str]) -> list[str]:
    """A dataset's own row ids where they tell its rows apart, row numbers otherwise: the audit finds a row by its id."""
    return ids if len(set(ids)) == len(ids) else [str(i) for i in range(len(ids))]


def sample_tasks(pool: list[Task], n: int | None, seed: int) -> list[Task]:
    """`n` tasks drawn uniformly without replacement from the whole dataset; all of them when `n` is None or not below its size."""
    if n is None or n >= len(pool):
        return list(pool)
    return random.Random(seed).sample(pool, n)


class Adapter(ABC):
    """Lives inside the worker process. Only `all_tasks` and `score_raw` are required."""

    name: str = "env"
    meta: dict[str, Any]

    def __init__(self) -> None:
        self.meta = {}

    @abstractmethod
    def all_tasks(self) -> list[Task]:
        """Every row of the dataset."""

    @abstractmethod
    def score_raw(self, task: Task, completion: str) -> RawScore: ...

    def tasks(self, n: int | None, seed: int) -> list[Task]:
        pool = self.all_tasks()
        self.meta["n_rows"] = len(pool)
        return sample_tasks(pool, n, seed)

    def max_score(self) -> float | None:
        """Optional: the nominal full score if the environment declares one."""
        return None
