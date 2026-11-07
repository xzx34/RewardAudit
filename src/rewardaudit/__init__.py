"""rewardaudit: black-box audit of checkers in RL environments with verifiable rewards.

    from rewardaudit import audit

    result = audit(score, "data.jsonl", n=200)
    result.print()
"""

from __future__ import annotations

from typing import Any

from rewardaudit.errors import AuditError, NotAuditable

__version__ = "0.1.0"

__all__ = ["AuditError", "NotAuditable", "__version__", "audit"]


def __getattr__(name: str) -> Any:  # workers import this package under an environment's own interpreter, so it stays light
    if name == "audit":
        from rewardaudit.api import audit

        return audit
    raise AttributeError(f"module 'rewardaudit' has no attribute {name!r}")
