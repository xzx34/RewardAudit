"""Every probe family. The audit sends the probes tier by tier (natural, amplified, contrived), and within a tier in
this order."""

from __future__ import annotations

from rewardaudit.probes import (
    candidates,
    hedge,
    leak,
    mutate,
    null,
    reference,
    rewrite,
    robustness,
    structured,
)
from rewardaudit.probes.base import Probe

FAMILIES: dict[str, tuple[str, list[Probe]]] = {
    "N": ("replies with no answer in them", null.PROBES),
    "E": ("many candidate answers in one reply", candidates.PROBES),
    "L": ("replies copied from the prompt", leak.PROBES),
    "O": ("a right and a wrong answer in one reply", hedge.PROBES),
    "M": ("the reference answer with a definite error", mutate.PROBES + structured.PROBES_M),
    "Q": ("the reference answer rewritten but still correct", rewrite.PROBES + structured.PROBES_Q),
    "G": ("the row's other accepted answers, and its reference answer written another way", reference.PROBES),
    "P": ("malformed or extreme input that must not make the checker fail or hang", robustness.PROBES),
}


def all_probes(families: list[str] | None = None) -> list[Probe]:
    out: list[Probe] = []
    for fam, (_, probes) in FAMILIES.items():
        if families is None or fam in families:
            out.extend(probes)
    return out
