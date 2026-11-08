"""Two calls files written by `audit --calls`, side by side: where two checkers, or two versions of one, decide differently
on the same rows."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from rewardaudit.errors import AuditError
from rewardaudit.probes.base import Probe
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import ROWS, _grid, _reply, _say, _score
from rewardaudit.types import Label

KEYS = ("task_id", "probe_id", "variant", "completion", "kind", "score", "accepted")
KINDS = ("a_accepts", "b_accepts", "no_score")  # how two aligned calls can differ; no_score: one has none, the other rejected
MAX_EXAMPLES = 3
NO_SCORE = "no_score"
HEADS = (*ROWS, NO_SCORE)  # the rates of the audit's summary, by the same names, and the share of calls with no score
NO_SCORE_LABEL = "Calls with no score"


def _read(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(task_id, probe_id, variant) -> the call. Where a key comes twice, as a wrong answer sent again in another format,
    the later call stands, as it does in the audit."""
    calls: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    call = json.loads(line)
                except ValueError:
                    call = None
                if not isinstance(call, dict) or not set(KEYS) <= set(call):
                    raise AuditError(f"Line {i} of {path} is not a call as audit --calls writes them, with {', '.join(KEYS)}")
                calls[(call["task_id"], call["probe_id"], call["variant"])] = call
    except (OSError, UnicodeDecodeError) as e:  # a file that is not there, or not text
        raise AuditError(f"Cannot read {path}: {getattr(e, 'strerror', None) or e}") from None
    if not calls:
        raise AuditError(f"{path} holds no calls")
    return calls


def _head(p: Probe | None) -> str | None:
    """The rate a probe's calls count toward in the audit's summary; None for a probe outside every rate."""
    if p is None or not p.headline:
        return None
    if p.label == Label.SOLVING:
        return "right_rejected"
    return {"M": "wrong_accepted", "O": "hedging_accepted"}.get(p.family, "non_solving_accepted")


def _heads(calls: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Each rate as the share of the calls behind it, straight from `accepted`; a rate no call measured is left out."""
    probes = {p.id: p for p in all_probes()}
    k, n = dict.fromkeys(HEADS, 0), dict.fromkeys(HEADS, 0)
    for c in calls:
        k[NO_SCORE] += c["score"] is None
        n[NO_SCORE] += 1
        head = _head(probes.get(c["probe_id"]))
        if head is None or c["accepted"] is None:
            continue
        k[head] += (not c["accepted"]) if head == "right_rejected" else c["accepted"]
        n[head] += 1
        if head == "wrong_accepted":  # paid in part: more than nothing, short of an acceptance
            k["wrong_partial"] += not c["accepted"] and c["score"] > 0
            n["wrong_partial"] += 1
    return {h: k[h] / n[h] for h in HEADS if n[h]}


def compare(a_path: str | Path, b_path: str | Path) -> dict[str, Any]:
    """Align the calls of two audits on (task_id, probe_id, variant) and say where their `accepted` differs.

    Returns `aligned` (calls in both files), `agreement` (the share of those with the same `accepted`, two calls with no score
    agreeing), `different_replies` (aligned calls whose replies differ, as where the two audits wrote them in different
    formats), `only_in_a` and `only_in_b` (calls in one file only, counted), `by_probe` (each probe with a difference, the most
    first: the counts `a_accepts`, `b_accepts` and `no_score`, and `examples` of each, up to 3, with task_id, variant,
    completion, score_a, score_b, and completion_b where B's reply differs) and `heads` (each file's rates, as
    `{"a": {...}, "b": {...}}`).
    """
    a, b = _read(a_path), _read(b_path)
    if not {key[0] for key in a} & {key[0] for key in b}:
        raise AuditError(f"{a_path} and {b_path} have no task_id in common; audit both checkers on the same dataset with "
                         "the same --n and --seed")
    aligned = [key for key in a if key in b]
    groups: dict[str, dict[str, Any]] = {}
    differ = different_replies = 0
    for key in aligned:
        x, y = a[key], b[key]
        same = x["completion"] == y["completion"]
        different_replies += not same
        if x["accepted"] == y["accepted"]:
            continue
        differ += 1
        kind = "a_accepts" if x["accepted"] else "b_accepts" if y["accepted"] else "no_score"
        g = groups.setdefault(key[1], {**dict.fromkeys(KINDS, 0), "examples": {k: [] for k in KINDS}})
        g[kind] += 1
        if len(g["examples"][kind]) < MAX_EXAMPLES:
            g["examples"][kind].append({"task_id": key[0], "variant": key[2], "completion": x["completion"],
                                        "score_a": x["score"], "score_b": y["score"],
                                        **({} if same else {"completion_b": y["completion"]})})
    return {
        "a": str(a_path), "b": str(b_path),
        "aligned": len(aligned),
        "agreement": 1 - differ / len(aligned) if aligned else 0.0,
        "different_replies": different_replies,
        "only_in_a": len(a) - len(aligned),
        "only_in_b": len(b) - len(aligned),
        "by_probe": dict(sorted(groups.items(), key=lambda kv: -sum(kv[1][k] for k in KINDS))),
        "heads": {"a": _heads(a.values()), "b": _heads(b.values())},
    }


def _example(ex: dict[str, Any]) -> str:
    """One line: the row and the variant, the reply, and the two scores; both replies where they differ."""
    where, a, b = f"[task {ex['task_id']}, {ex['variant']}]", _score(ex["score_a"]), _score(ex["score_b"])
    if "completion_b" in ex:
        return f"{where} A {a} on {_reply(ex['completion'], 28)}, B {b} on {_reply(ex['completion_b'], 28)}"
    return f"{where} {_reply(ex['completion'], 40)}  A {a}  B {b}"


def render_compare(result: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    _say(console, f"A is {result['a']}, B is {result['b']}.\n", "bold")
    text = (f"Aligned {result['aligned']:,} calls; agree on {result['agreement']:.1%}. "
            f"Only in A: {result['only_in_a']:,}, only in B: {result['only_in_b']:,}.")
    if result["different_replies"]:
        text += (f" On {result['different_replies']:,} of them the two replies differ, as each audit wrote them in a format "
                 "its checker accepts; examples there show both. Both audits with the same --format, one both checkers accept, "
                 "send the same replies.")
    _say(console, text + "\n")
    if result["by_probe"]:
        t = _grid()
        for probe_id, g in result["by_probe"].items():
            counts = f"A accepts {g['a_accepts']:,}, B accepts {g['b_accepts']:,}"
            t.add_row(Text(probe_id), "", Text(counts + (f", no score on one side {g['no_score']:,}" if g["no_score"] else "")))
            for ex in (ex for kind in KINDS for ex in g["examples"][kind]):
                t.add_row("", "", Text(_example(ex), style="dim"))
        console.print(Padding(t, (0, 0, 1, 2), expand=False))
    t = _grid()
    t.add_row("", Text("A", style="bold"), Text("B", style="bold"))
    for head in HEADS:
        rates = [f"{r[head]:.1%}" if head in r else "not measured" for r in (result["heads"]["a"], result["heads"]["b"])]
        t.add_row(Text(ROWS[head][0] if head in ROWS else NO_SCORE_LABEL), *map(Text, rates))
    console.print(Padding(t, (0, 0, 1, 2), expand=False))
    _say(console, "Each rate is the share of the calls behind it in that file, so it can differ from the audit's own.", "dim")
