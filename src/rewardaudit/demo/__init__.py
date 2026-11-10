"""Three toy checkers with planted defects, used by `rewardaudit demo` and the test suite.

  substring : score 1 if the gold appears anywhere in the completion       (accepts a reply listing many candidates)
  mcq_regex : first capital letter A-D in the completion is the answer     (reads 'B' from 'Based on')
  grid_exact: exact match after whitespace normalisation                   (rejects any wrapper or prose)
"""

from __future__ import annotations

import random
import re
from typing import Any

from rewardaudit.adapters.base import Adapter, RawScore
from rewardaudit.types import Task

DEMOS = ("substring", "mcq_regex", "grid_exact")


def _math_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        c = rng.randint(1, 9)
        out.append(Task(id=f"m{i}", prompt=f"Sam has {a} apples and buys {b} more, then gives away {c}. "
                                            f"How many apples does Sam have now? Put the final answer in \\boxed{{}}.",
                        answer=str(a + b - c)))
    return out


def _mcq_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    out = []
    capitals = [("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Norway", "Oslo"),
                ("Egypt", "Cairo"), ("Chile", "Santiago"), ("Canada", "Ottawa")]
    for i in range(n):
        country, city = rng.choice(capitals)
        others = rng.sample([c for _, c in capitals if c != city], 3)
        opts = others + [city]
        rng.shuffle(opts)
        letters = "ABCD"
        gold = letters[opts.index(city)]
        body = "\n".join(f"{letters[j]}. {o}" for j, o in enumerate(opts))
        out.append(Task(id=f"q{i}", prompt=f"What is the capital of {country}?\n{body}\nAnswer with the letter.",
                        answer=gold))
    return out


def _grid_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        rows, cols = rng.randint(2, 3), rng.randint(2, 4)
        grid = [[rng.randint(0, 1) for _ in range(cols)] for _ in range(rows)]
        flipped = [[1 - v for v in row] for row in grid]
        src = "\n".join(" ".join(map(str, r)) for r in grid)
        gold = "\n".join(" ".join(map(str, r)) for r in flipped)
        out.append(Task(id=f"g{i}", prompt=f"Invert every cell of this binary grid (0 becomes 1, 1 becomes 0) "
                                            f"and output the resulting grid, one row per line:\n{src}",
                        answer=gold))
    return out


def score_substring(completion: str, answer: str) -> float:
    return 1.0 if answer in completion else 0.0


def score_mcq_regex(completion: str, answer: str) -> float:
    m = re.search(r"[ABCD]", completion)
    return 1.0 if m and m.group(0) == answer else 0.0


def score_grid_exact(completion: str, answer: str) -> float:
    norm = lambda s: " ".join(s.split())  # noqa: E731
    return 1.0 if norm(completion) == norm(answer) else 0.0


class DemoAdapter(Adapter):
    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        self.which = spec.get("name", "substring")
        if self.which not in DEMOS:
            raise ValueError(f"unknown demo {self.which!r}; choose from {DEMOS}")
        self.name = f"demo/{self.which}"
        self.meta = {"demo": self.which}
        self._sample: list[Task] = []

    def tasks(self, n: int | None, seed: int) -> list[Task]:
        make = {"substring": _math_tasks, "mcq_regex": _mcq_tasks}.get(self.which, _grid_tasks)
        self._sample = make(50 if n is None else n, seed)
        self.meta["n_rows"] = len(self._sample)
        return self._sample

    def all_tasks(self) -> list[Task]:
        return self._sample  # the tasks are made up on request, so the dataset is the sample

    def score_raw(self, task: Task, completion: str) -> RawScore:
        if self.which == "substring":
            return RawScore(score_substring(completion, task.answer))
        if self.which == "mcq_regex":
            return RawScore(score_mcq_regex(completion, task.answer))
        return RawScore(score_grid_exact(completion, task.answer))

    def max_score(self) -> float | None:
        return 1.0
