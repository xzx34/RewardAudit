"""Core data types shared by every layer of the audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Kind(str, Enum):
    """Four-way outcome of one checker call (TRUE/FALSE are both SCORE with a value)."""

    SCORE = "score"
    TIMEOUT = "timeout"
    ERROR = "error"  # checker raised
    CRASH = "crash"  # worker process died


class Label(str, Enum):
    """Whether a completion solves the task, decided at construction time when possible."""

    NOT_SOLVING = "not_solving"
    SOLVING = "solving"


@dataclass
class Task:
    id: str
    prompt: str
    answer: str
    messages: list[dict[str, Any]] | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def chat(self) -> list[dict[str, Any]]:
        return self.messages or [{"role": "user", "content": self.prompt}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            id=str(d["id"]),
            prompt=d.get("prompt", ""),
            answer="" if d.get("answer") is None else str(d["answer"]),
            messages=d.get("messages"),
            info=d.get("info") or {},
        )


@dataclass
class Verdict:
    kind: Kind
    score: float | None = None
    sub_scores: dict[str, float] | None = None
    parsed: str | None = None
    error: str | None = None
    elapsed_ms: float = 0.0
    network: bool = False

    @property
    def ok(self) -> bool:
        return self.kind == Kind.SCORE and self.score is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class Observation:
    """One checker call made by the audit, with everything needed to interpret it."""

    probe_id: str
    family: str
    variant: str
    task_id: str
    completion: str
    verdict: Verdict
    label: Label
    expect: str  # zero | full
    tier: str  # natural | amplified | contrived

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "family": self.family,
            "variant": self.variant,
            "task_id": self.task_id,
            "completion": self.completion,
            "verdict": self.verdict.to_dict(),
            "label": self.label.value,
            "expect": self.expect,
            "tier": self.tier,
        }
