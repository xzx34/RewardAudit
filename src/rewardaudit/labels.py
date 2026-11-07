"""Whether a reply's text states the reference answer. A reply built without reading it can happen to be it (a constant 1
on a reference answer of 1), and that is not counted as accepted."""

from __future__ import annotations

from collections.abc import Iterable

from rewardaudit.equiv import contains_equiv, distinct_candidates, equiv, extract_numbers, to_number


def states_the_answer(text: str, gold: str, gold_type: str | None = None, derived: Iterable[str] = ()) -> bool:
    """Exactly one final-answer candidate, equivalent to the reference answer; hedges and lists of numbers do not count.
    `derived`: wrapper ids of the checker's own answer fields (`Contract.derived`), read as candidates too. A reply that
    is only the answer has no mark for a candidate to be read from, so it counts as it stands."""
    if equiv(text, gold, gold_type):
        return True
    if not contains_equiv(text, gold, gold_type, derived):
        return False
    cands = distinct_candidates(text, gold_type, derived)
    if len(cands) != 1 or not equiv(cands[0], gold, gold_type):
        return False
    if to_number(gold) is None:
        return True
    return len({v for tok in extract_numbers(text) if (v := to_number(tok)) is not None}) < 20
