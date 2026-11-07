"""G family: the dataset's own accepted answers. G1 is no probe: it is the finding of the calls with the reference answers
(see metrics). G2 sends a row's other accepted answers, which a sound checker accepts as it does the reference answer."""

from __future__ import annotations

from rewardaudit.probes.base import Probe
from rewardaudit.types import Label

MAX_ALIASES = 8  # per row; TriviaQA lists about 15 and up to 66


def _has_aliases(task, ctx) -> bool:
    """Other accepted answers beside a reference answer that is one answer. Next to a list, a JSON object or a grid, a
    list may hold its parts, and a part alone is not the answer."""
    return bool(ctx.aliases.get(task.id)) and ctx.gtype(task) not in ("list", "json", "grid")


def g2_other_answers(task, rng, ctx):
    return [(f"alias{i}", ctx.wrap(a)) for i, a in enumerate(ctx.aliases[task.id][:MAX_ALIASES], 1)]


PROBES = [
    Probe("G2", "G", "other accepted answers of the row", "natural", "full", Label.SOLVING, g2_other_answers,
          applies=_has_aliases, headline=False,
          description="Where the checker receives a list of accepted answers, sends each one other than the reference answer "
                      "in the accepted format, one per reply and up to 8 per row; a sound checker gives each full credit. "
                      "Its finding names some that lost the score."),
]
