"""The defect catalog: what the findings say about the checker, as opposed to what one probe saw.

Each entry says where the defect is, not how to change the checker. Every probe belongs to exactly one entry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defect:
    id: str
    name: str
    explanation: str
    probes: tuple[str, ...] = ()


FIXED_ANSWER, RUN_AS_CODE, ROBUSTNESS, WRONG_PARTIAL, ABOVE_REFERENCE = "D11", "D12", "D13", "D14", "D18"

DEFECTS = (
    Defect("D1", "Signs and punctuation ignored",
           "A number with its sign flipped or its decimal point moved compares equal to the reference answer.",
           ("M2", "M3", "M22")),
    Defect("D2", "Substring match",
           "A reply counts when the reference answer appears somewhere inside it, or when it is a part of the reference answer.",
           ("M21", "M18", "E1", "E5", "O6")),
    Defect("D3", "Numeric tolerance too wide", "A number that differs from the reference answer still compares equal.", ("M24", "M1")),
    Defect("D4", "One of several answers is read",
           "With several answers marked in one reply, the checker reads one of them, often the first, and ignores the others.",
           ("O5", "O12", "O4", "E6")),
    Defect("D5", "Reads the thinking block", "An answer in the thinking block counts, though the final answer is wrong or missing.",
           ("O10", "O13")),
    Defect("D6", "Hedged answers accepted", "A reply that commits to no single answer earns full score.",
           ("O7", "O11", "O14", "O15", "E2", "E3", "E8")),
    Defect("D7", "Negation ignored", "The right answer earns full score though the reply retracts it, negates it or only supposes it.",
           ("O3",)),
    Defect("D8", "Only a fixed-length window is read", "A right answer loses the score once enough text comes before or after it.",
           ("Q28", "Q29")),
    Defect("D9", "Equivalent rewrites rejected", "A right answer loses the score when written another way.",
           tuple(f"Q{i}" for i in range(1, 32) if i not in (28, 29))),
    Defect("D10", "Copying the prompt pays", "A reply taken from the prompt earns credit without solving the task.",
           ("L1", "L2", "L3", "L4", "L6", "E9")),
    Defect("D11", "A fixed answer pays", "One reply, whatever the task, earns credit more often than a fixed answer would by chance.",
           ("L5", "N1", "N2", "N3", "N5", "N9", "N10")),
    Defect("D12", "Reply text run as code", "The reply's text reached a Python import, so the checker runs what it reads."),
    Defect("D13", "Malformed or extreme replies mishandled",
           "On unusual input the checker raises an error, hangs, crashes or gives credit.", ("P1", "P2", "P5", "P15", "P16", "P17")),
    Defect("D14", "Wrong answers paid in part", "An answer with a definite error earns more than nothing."),
    Defect("D15", "Same input, different scores", "The same reply to the same row gets different scores, so every other number is one draw.",
           ("S1",)),
    Defect("D16", "Reference answer rejected", "The dataset's own reference answer is treated like a wrong one in every format tried.", ("G1",)),
    Defect("D17", "Units ignored", "An answer with a different unit compares equal to the reference answer.", ("M23",)),
    Defect("D18", "Score does not follow the answer", "A reply earns the reference answer's score or more, whatever answer it gives."),
    Defect("D19", "Close answers accepted", "An answer a few characters away from the reference answer compares equal.", ("M5", "M6", "M7")),
    Defect("D20", "Parts of a structured answer ignored",
           "A list, JSON object or grid with a definite error compares equal to the reference answer.",
           ("M8", "M9", "M10", "M11", "M12", "M13", "M17")),
    Defect("D21", "A different answer accepted", "An answer that is plainly another one earns full score.", ("M16", "M19", "M20")),
)

BY_PROBE = {p: d.id for d in DEFECTS for p in d.probes}
