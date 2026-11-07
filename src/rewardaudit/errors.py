"""Errors of the audit, and how a long error from a checker or an installer is cut down to one line."""

from __future__ import annotations

import re


class AuditError(Exception):
    """A problem the user can fix. The message says how, so no traceback is shown."""


class NotAuditable(AuditError):
    """The checker cannot be audited: `cls` names the case, `reason` what was seen, `advice` what to do."""

    def __init__(self, cls: str, reason: str, advice: str = "") -> None:
        super().__init__(f"{reason}. {advice}".strip() if advice else reason)
        self.cls, self.reason, self.advice = cls, reason, advice


class WorkerError(RuntimeError):
    """The worker process failed or died."""


class MemoryLimitExceeded(WorkerError):
    """The worker outgrew the memory limit while scoring and was killed."""


def short_error(error: str, k: int = 300) -> str:
    """The place and the message of a traceback ('reward.py, line 3: NameError: x'), or the failure uv reports."""
    lines = [s.strip() for s in error.splitlines() if s.strip()]
    marks = [i for i, s in enumerate(lines) if s.startswith(("×", "error:"))]  # uv: the first failure and its cause
    if marks:
        i = marks[0]
        causes = [s.lstrip("├╰─▶ ").strip() for s in lines[i + 1:i + 4] if s.startswith(("├─▶", "╰─▶"))]
        return (lines[i].lstrip("× ").strip() + (f": {causes[-1]}" if causes else ""))[:k]
    exc = [i for i, s in enumerate(lines) if re.match(r"^[\w.]+(Error|Exception|Exit|Interrupt)\b", s)]
    if not exc:
        return (lines[-1] if lines else "unknown error")[:k]
    last = lines[exc[-1]]
    if "(Request ID:" in last and exc[-1] + 1 < len(lines):  # huggingface_hub: the message is on the next line
        last = f"{last.partition(':')[0]}: {lines[exc[-1] + 1]}"
    frames = [m for s in lines[: exc[-1]] if (m := re.match(r'File "([^"]+)", line (\d+)', s))
              and "/rewardaudit/" not in m.group(1) and "site-packages" not in m.group(1)]
    where = f"{frames[-1].group(1).rsplit('/', 1)[-1]}, line {frames[-1].group(2)}: " if frames else ""
    return (where + last)[:k]
