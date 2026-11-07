"""Every probe family, in the order they run (natural-first within a family is set by tier)."""

from __future__ import annotations

from rewardaudit.probes import enum as enum_family
from rewardaudit.probes import leak, mutate, null, order, parser, rewrite, structured
from rewardaudit.probes.base import Probe

FAMILIES: dict[str, tuple[str, list[Probe]]] = {
    "N": ("replies with no answer in them", null.PROBES),
    "E": ("many candidate answers in one reply", enum_family.PROBES),
    "L": ("replies copied from the prompt", leak.PROBES),
    "O": ("a right and a wrong answer in one reply", order.PROBES),
    "M": ("the reference answer with a definite error", mutate.PROBES + structured.PROBES_M),
    "Q": ("the reference answer rewritten but still correct", rewrite.PROBES + structured.PROBES_Q),
    "P": ("malformed or extreme input that must not make the checker fail or hang", parser.PROBES),
}


def all_probes(families: list[str] | None = None) -> list[Probe]:
    out: list[Probe] = []
    for fam, (_, probes) in FAMILIES.items():
        if families is None or fam in families:
            out.extend(probes)
    return out


def probes_by_family() -> dict[str, list[Probe]]:
    return {fam: probes for fam, (_, probes) in FAMILIES.items()}
