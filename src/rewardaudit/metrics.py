"""Head metrics, per-probe statistics, severity. Every proportion carries a Wilson 95% interval.

Two granularities:
  * observation level — one checker call; used for pooled rates (wrong accepted, right rejected).
  * task level        — "any variant hit on this task"; used for examples and per-probe findings.
The best gold-blind strategy is selected per (probe, variant) on a discovery split of tasks and
reported on the remaining confirmation split, so the headline number is not inflated by selection.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from rewardaudit.contract import Contract
from rewardaudit.defects import ABOVE_REFERENCE, BY_PROBE, DEFECTS, FIXED_ANSWER, ROBUSTNESS, WRONG_PARTIAL
from rewardaudit.equiv import normalize
from rewardaudit.goldtype import infer_gold_type, mcq_options
from rewardaudit.labels import states_the_answer
from rewardaudit.probes.base import Probe
from rewardaudit.types import Kind, Label, Observation, Task


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class ProbeStats:
    probe: Probe
    n_tasks: int = 0
    task_hits: int = 0  # tasks where any variant hit
    fail_tasks: int = 0  # tasks where any call returned no score
    n_obs: int = 0  # all calls
    n_scored: int = 0  # calls that returned a score; rates use these, failed calls go to robustness
    obs_hits: int = 0
    coincident: int = 0  # gold-blind hits whose completion happens to be the gold (e.g. constant 1 on gold 1)
    partial: int = 0
    errors: int = 0
    timeouts: int = 0
    crashes: int = 0
    n_compared: int = 0  # scored calls of a probe expecting zero, coincidental hits left out
    above_gold: int = 0  # of those, calls scoring strictly above the gold's signal on the same task
    above_tasks: int = 0
    above_example: Observation | None = None
    above_bar: float = 0.0  # what above_example had to beat
    examples: list[Observation] = field(default_factory=list)
    variants: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0, 0]))
    # per variant: [disc_hits, disc_n, conf_hits, conf_n]
    variant_hits: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # [hits, scored]
    variant_examples: dict[str, Observation] = field(default_factory=dict)  # sweeps: the first hit of each variant

    def most_accepted_variant(self) -> tuple[str, int, int]:
        """(variant, hits, scored) of the variant the checker accepts most often."""
        name, (k, n) = max(self.variant_hits.items(), key=lambda kv: (kv[1][0] / kv[1][1] if kv[1][1] else 0.0, kv[1][1]))
        return name, k, n

    @property
    def rate(self) -> float:  # observation-level
        return self.obs_hits / self.n_scored if self.n_scored else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.obs_hits, self.n_scored)

    @property
    def task_rate(self) -> float:
        return self.task_hits / self.n_tasks if self.n_tasks else 0.0

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.ci
        return {
            "id": self.probe.id, "family": self.probe.family, "name": self.probe.name, "tier": self.probe.tier,
            "expect": self.probe.expect, "label": self.probe.label.value, "gold_blind": self.probe.gold_blind,
            "headline": self.probe.headline, "n_tasks": self.n_tasks, "task_hits": self.task_hits,
            "task_rate": self.task_rate, "n_obs": self.n_obs, "n_scored": self.n_scored, "obs_hits": self.obs_hits,
            "rate": self.rate,
            "ci": [lo, hi], "partial": self.partial, "errors": self.errors, "timeouts": self.timeouts,
            "crashes": self.crashes, "coincident": self.coincident, "n_compared": self.n_compared,
            "above_gold": self.above_gold,
            "variants": {v: {"discover": [a, b], "confirm": [c, d]} for v, (a, b, c, d) in self.variants.items()},
            "examples": [o.to_dict() for o in self.examples[:3]],
        }


def _option_chance(tasks: list[Task]) -> tuple[float, str]:
    """The chance that one constant letter is right: one over each task's option count (4 when its prompt lists none)."""
    counts = [len(mcq_options(t.prompt)) or 4 for t in tasks]
    chance = sum(1 / c for c in counts) / len(counts)
    return chance, f"chance 1/{counts[0]}" if len(set(counts)) == 1 else f"chance {chance:.0%} ({min(counts)} to {max(counts)} options)"


def floor_estimate(tasks: list[Task], gold_type: str) -> tuple[float, str]:
    """Best score a constant answer could get: mode frequency, or chance for closed answer spaces."""
    golds = [normalize(t.answer) for t in tasks]
    counts = Counter(golds)
    mode, k = counts.most_common(1)[0]
    mode_rate = k / len(golds)
    common = f"most common reference answer '{mode}' on {k}/{len(golds)} rows"
    if gold_type == "mcq":
        chance, note = _option_chance(tasks)
        return max(mode_rate, chance), f"{common}; {note}"
    if gold_type == "bool":
        return max(mode_rate, 0.5), f"{common}; chance 1/2"
    return mode_rate, common


EPS = 1e-9
MIN_CALLS, MIN_TASKS = 10, 2  # below these a finding is not rated above low: one or two calls say too little
TOO_FEW = "too few calls to rate"


def _require_evidence(f: dict[str, Any], enough: bool) -> dict[str, Any]:
    if f["severity"] != "low" and not enough:
        f["severity"] = "low"
        f["note"] = "; ".join(x for x in (f["note"], TOO_FEW) if x)
    return f


def _hit(o: Observation, contract: Contract, gold_blind: bool = False, eps: float = EPS) -> tuple[bool, bool]:
    """(hit, partial). NOT_SOLVING: hit = accepted at full signal, or for a reply built without the gold, paid by one
    signal key (`Contract.pays_on_one_term`). SOLVING: hit = not accepted."""
    v = o.verdict
    if not v.ok:
        return False, False
    accepted = contract.accepted_verdict(v, o.task_id, eps)
    if o.label == Label.SOLVING:
        return (not accepted), False
    accepted = accepted or (gold_blind and o.label == Label.NOT_SOLVING and contract.pays_on_one_term(v, o.task_id))
    return accepted, (not accepted) and (contract.signal(v) or 0.0) > eps


TIMEOUT_NOTE = ("A timed-out call is counted here as a failed call, never as accepted. The training framework "
                "may treat a timeout differently (zero reward, a retry, or dropping the sample).")
NETWORK_NOTE = ("The reference answers themselves triggered network access, so the checker probably calls a remote "
                "service such as an LLM judge. Network access is blocked during the audit, so these scores may "
                "differ from training, and with the service they may not be deterministic.")


def exception_type(error: str | None) -> str:
    """The exception class named on the last line of a traceback, e.g. 'ValueError'."""
    lines = [s for s in (error or "").splitlines() if s.strip()]
    m = re.match(r"\s*([A-Za-z_][\w.]*)(?::|\s*$)", lines[-1]) if lines else None
    return m.group(1).rsplit(".", 1)[-1] if m else "unknown"


def _failure_example(o: Observation) -> dict[str, Any]:
    return {"probe_id": o.probe_id, "variant": o.variant, "task_id": o.task_id, "completion": o.completion[:400],
            "error": (o.verdict.error or "")[-400:], "elapsed_ms": round(o.verdict.elapsed_ms, 1)}


class _Failures:
    """One kind of failed call: how many, on which probes, one example."""

    def __init__(self) -> None:
        self.count = 0
        self.probes: Counter[str] = Counter()
        self.example: dict[str, Any] | None = None

    def add(self, o: Observation) -> None:
        self.count += 1
        self.probes[o.probe_id] += 1
        self.example = self.example or _failure_example(o)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "probes": dict(self.probes.most_common()), "example": self.example}


class Robustness:
    """S4 timeouts, S5 exceptions, crashes and S7 network attempts over the probe calls, added as they come."""

    def __init__(self) -> None:
        self.total = 0
        self.timeouts, self.errors, self.crashes, self.network = _Failures(), _Failures(), _Failures(), _Failures()
        self.error_types: Counter[str] = Counter()

    def add(self, o: Observation) -> None:
        self.total += 1
        k = o.verdict.kind
        if k == Kind.TIMEOUT:
            self.timeouts.add(o)
        elif k == Kind.CRASH:
            self.crashes.add(o)
        elif k == Kind.ERROR:
            self.errors.add(o)
            self.error_types[exception_type(o.verdict.error)] += 1
        if o.verdict.network:
            self.network.add(o)

    def stats(self, contract: Contract) -> dict[str, Any]:
        """Probe calls and gold calls (the contract pass) are counted separately."""
        total = self.total
        gold = [v for wid, _, _, v in contract.observations if wid != "wrong_answer"]
        gold_kinds = Counter(v.kind for v in gold)
        gold_network = sum(1 for v in gold if v.network)
        return {
            "n_calls": total,
            "error_rate": self.errors.count / total if total else 0.0,
            "timeout_rate": self.timeouts.count / total if total else 0.0,
            "crash_rate": self.crashes.count / total if total else 0.0,
            "network_calls": self.network.count,
            "n_gold_calls": len(gold),
            "timeouts": self.timeouts.to_dict() | {"gold": gold_kinds[Kind.TIMEOUT], "note": TIMEOUT_NOTE},
            "exceptions": self.errors.to_dict()
            | {"by_type": dict(self.error_types.most_common()), "gold": gold_kinds[Kind.ERROR],
               "gold_by_type": dict(Counter(exception_type(v.error) for v in gold if v.kind == Kind.ERROR).most_common())},
            "crashes": self.crashes.to_dict() | {"gold": gold_kinds[Kind.CRASH]},
            "network": self.network.to_dict() | {"gold": gold_network, "note": NETWORK_NOTE if gold_network else ""},
        }


def _metric(key: str, title: str, k: int, n: int, detail: str = "") -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"key": key, "title": title, "value": k / n if n else 0.0, "ci": [lo, hi], "n": n, "k": k, "detail": detail}


NO_DISCRIMINATION = "not measured: reference and wrong answers score the same"
WHOLE_REPLY = "not measured: the checker grades the whole reply by similarity to the reference answer"
OUTRANKS, FORMAT_ONLY, SAME_SCORE = "outranks", "format_only", "same"  # the three no_discrimination cases
REJECTED_GOLD_ID = "G1"
REJECTED_GOLD_NAME = "reference answer rejected in every format tried"


def _not_measured(h: dict[str, Any], why: str) -> dict[str, Any]:
    out = {k: v for k, v in h.items() if k not in ("strategy", "severity", "note", "example")}
    out.update(_metric(h["key"], h["title"], 0, 0, why), not_measured=True)
    if "above_floor" in out:
        out["above_floor"] = False
    return out


def _rejected_severity(probe_id: str, rate: float) -> str:
    return "high" if (probe_id in ("Q5", "Q22") and rate >= 0.2) else ("medium" if rate >= 0.1 else "low")


def _wrong_severity(rate: float) -> str:
    return "high" if rate >= 0.5 else ("medium" if rate >= 0.1 else "low")


def _non_solving_finding(probe: Probe, rate: float, lo: float, floor: float) -> tuple[str, str]:
    """(category, severity) of a probe that expects zero and got credit."""
    if probe.family == "M":
        return "wrong_accepted", _wrong_severity(rate)
    if probe.family == "O":
        return "hedging_accepted", "high" if lo > 0.5 else ("medium" if rate >= 0.2 else "low")
    above = lo > floor
    sev = "high" if (rate >= 0.5 and above) else ("medium" if above else "low")
    return "non_solving_accepted", "medium" if probe.tier == "contrived" and sev == "high" else sev


def _tolerance_severity(variant: str, rate: float) -> str:
    """How wide the accepted error is decides: rounding explains 0.1%, not 5%. Below half the rows it is the rows, not a rule."""
    width = float(variant.rstrip("%"))
    return "low" if rate < 0.5 or width < 1 else "medium" if width < 5 else "high"


def _above_gold_note(ex: Observation, contract: Contract, gold: float) -> str:
    """Against the gold or the gold in the reply's own reasoning, whichever scored higher; both totals on a composite rubric."""
    if not contract.signal_keys or ex.task_id not in contract.gold_total:
        return f"the reply scored above the reference answer: {contract.signal(ex.verdict):.3g} against {gold:.3g} on row {ex.task_id}"
    return (f"the reply scored above the reference answer on {' + '.join(contract.signal_keys)}: {contract.signal(ex.verdict):.3g} against "
            f"{gold:.3g} on row {ex.task_id} (totals {ex.verdict.score:.3g} and {contract.gold_total[ex.task_id]:.3g})")


def _above_gold_findings(stats: dict[str, ProbeStats], floor: float, contract: Contract) -> list[dict[str, Any]]:
    """Replies that should earn nothing and scored strictly above the gold on the same task. This does not rest on the
    gold's score meaning anything: a non-answer outranks the right answer whatever that score is."""
    out = []
    for s in stats.values():
        if not s.above_gold:
            continue
        k, n = (s.above_tasks, s.n_tasks) if s.probe.by_variant else (s.above_gold, s.n_compared)
        rate = k / n
        lo, hi = wilson(k, n)
        category, sev = _non_solving_finding(s.probe, rate, lo, floor)
        ex = s.above_example
        out.append(_require_evidence({
            "probe_id": s.probe.id, "name": s.probe.name, "family": s.probe.family, "tier": s.probe.tier,
            "category": category, "severity": sev, "rate": rate, "ci": [lo, hi], "n": n,
            "task_rate": s.above_tasks / s.n_tasks if s.n_tasks else 0.0, "n_tasks": s.n_tasks,
            "gold_blind": s.probe.gold_blind, "example": ex.to_dict() if ex else None, "above_gold": True,
            "note": _above_gold_note(ex, contract, s.above_bar) if ex else "",
        }, s.n_compared >= MIN_CALLS and s.above_tasks >= MIN_TASKS))
    return out


def _rejected_gold_finding(contract: Contract, tasks: list[Task]) -> dict[str, Any] | None:
    """Most golds pass and some are treated like a wrong answer in every format tried: the dataset's own answer is a
    right answer the checker rejects. Separate from the head metric right_rejected, which counts rewrites of golds
    that pass."""
    rejected = contract.rejected_golds()
    if not rejected or contract.self_pass_rate < 0.5:
        return None
    n, k = len(contract.self_pass), len(rejected)
    tid = rejected[0]
    tried = [w for w, t, _, _ in contract.observations if t == tid and w != "wrong_answer"]
    calls = [(w, comp, v) for w, t, comp, v in contract.observations if t == tid and w != "wrong_answer" and v.ok]
    best = max(calls, key=lambda x: (contract.signal(x[2]) or 0.0, x[0] == contract.primary), default=None)
    gold = next((t.answer for t in tasks if t.id == tid), "")
    partial = sum(not ok and contract.self_fail_reason.get(t) != "error" for t, ok in contract.self_pass.items()) - k
    signal = contract.gold_score[tid]
    best_text = (f"best total {best[2].score:.3g} ({' + '.join(contract.signal_keys)}: {signal:.3g})"
                 if contract.signal_keys and best is not None else f"best score {signal:.3g}")
    note = (f"{k} of {n} rows; row {tid}, reference answer {gold!r}, {best_text} in the {len(tried)} formats tried"
            + (f"; {partial} more reference answers earn partial credit and are not counted" if partial else ""))
    example = None
    if best is not None:
        example = Observation(REJECTED_GOLD_ID, "G", best[0], tid, best[1], best[2], Label.SOLVING, "full", "natural").to_dict()
        example |= {"gold": gold, "best_score": contract.gold_score[tid], "wrappers_tried": tried}
    lo, hi = wilson(k, n)
    return _require_evidence(
        {"probe_id": REJECTED_GOLD_ID, "name": REJECTED_GOLD_NAME, "family": "G", "tier": "natural",
         "category": "right_rejected", "severity": _rejected_severity(REJECTED_GOLD_ID, k / n), "rate": k / n,
         "ci": [lo, hi], "n": n, "unit": "tasks", "task_rate": k / n, "n_tasks": n, "gold_blind": False,
         "example": example, "note": note}, k >= MIN_TASKS)


def nondeterminism_finding(determinism: dict[str, Any]) -> dict[str, Any]:
    """Identical inputs, different scores: every other number in the report is then only one draw."""
    k, n = len(determinism["inconsistent_tasks"]), determinism["checked"]
    lo, hi = wilson(k, n)
    return {"probe_id": "S1", "name": "different scores for the same reply", "family": "S", "tier": "natural",
            "category": "robustness", "severity": "medium", "rate": k / n, "ci": [lo, hi], "n": n, "task_rate": k / n,
            "n_tasks": n, "gold_blind": False, "example": None,
            "note": f"the reference answer, sent {determinism['repeats']} times, got different scores on {k} of {n} rows"}


def _discrimination(contract: Contract, above_findings: list[dict[str, Any]], k: int, n: int) -> dict[str, str]:
    """Which no_discrimination case this is, and the sentence that explains it: replies above the gold first (a low-
    severity rate alone does not decide it), then a reward that follows the format, else the plain same score."""
    flagged = sorted((f for f in above_findings if f["severity"] in ("high", "medium")), key=lambda f: -f["rate"] * f["n"])
    above = [f["probe_id"] for f in flagged]
    wrong = f"wrong answers above it on {contract.wrong_above} of {contract.wrong_checked} rows"
    note = contract.discrimination_note + (f"; {wrong}" if contract.wrong_above else "")
    if above or _wrong_severity(contract.wrong_above / contract.wrong_checked if contract.wrong_checked else 0.0) != "low":
        replies = f"{k} of {n} replies scored above it ({', '.join(above[:6])})" if above else wrong
        return {"case": OUTRANKS, "reason": f"non-answers outrank the reference answer: {replies}; {note}"}
    if contract.format_only:
        return {"case": FORMAT_ONLY, "reason": f"format-only reward: any answer in the right format earns the reference answer's score; {note}"}
    return {"case": SAME_SCORE, "reason": f"{note}; check how the reference answer is delivered"}


class Tally:
    """Adds up the probe calls batch by batch, so that an audit of many tasks need not hold every call in memory. A batch
    must hold all calls of a probe on a task. `controls`: (task id, completion) -> the signal of that reply with the gold as
    its final answer (see pipeline)."""

    def __init__(self, probes: list[Probe], tasks: list[Task], contract: Contract, gold_type: str,
                 discovery_frac: float = 0.4) -> None:
        self.probes, self.tasks, self.contract, self.gold_type = probes, tasks, contract, gold_type
        self.by_id = {p.id: p for p in probes}
        self.passing = set(contract.passing_tasks())
        order = [t.id for t in tasks if t.id in self.passing]
        self.n_disc = max(1, int(len(order) * discovery_frac)) if len(order) >= 5 else len(order)
        self.disc_ids = set(order[: self.n_disc])
        self.golds = {t.id: t.answer for t in tasks}
        self.stats: dict[str, ProbeStats] = {}
        self.robust = Robustness()
        self.null_floor = 0.0
        self.paid_calls, self.paid_share, self.paid_tasks, self.paid_example = 0, 0.0, set(), None  # M calls paid in part

    def add(self, observations: list[Observation], controls: dict[tuple[str, str], float] | None = None) -> None:
        contract, disc_ids, stats = self.contract, self.disc_ids, self.stats
        per_task: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        for o in observations:
            self.robust.add(o)
            if o.task_id in self.passing and o.probe_id in self.by_id:
                per_task[(o.probe_id, o.task_id)].append(o)

        def bar(o: Observation) -> float:
            """What a reply has to beat to be above the gold: the gold's best, and the gold in this reply's own reasoning."""
            gold = contract.gold_score.get(o.task_id, contract.full_score)
            return max(gold, (controls or {}).get((o.task_id, o.completion), gold))

        for (pid, tid), obs in per_task.items():
            probe = self.by_id[pid]
            st = stats.setdefault(pid, ProbeStats(probe))
            st.n_tasks += 1
            any_hit = any_partial = any_above = any_fail = False
            example = None
            for o in obs:
                st.n_obs += 1
                k = o.verdict.kind
                any_fail |= k != Kind.SCORE
                if k == Kind.ERROR:
                    st.errors += 1
                elif k == Kind.TIMEOUT:
                    st.timeouts += 1
                elif k == Kind.CRASH:
                    st.crashes += 1
                h, p = _hit(o, contract, probe.gold_blind)
                if p and probe.family == "M" and probe.headline:
                    self._add_paid(o)
                if probe.family == "N" and o.verdict.ok:
                    self.null_floor = max(self.null_floor, contract.signal(o.verdict) or 0.0)
                slot = st.variants[o.variant]
                if o.verdict.ok:  # the strategy slots keep coincidental hits: a constant policy earns them too
                    if tid in disc_ids:
                        slot[0] += int(h)
                        slot[1] += 1
                    else:
                        slot[2] += int(h)
                        slot[3] += 1
                if h and probe.gold_blind and probe.label == Label.NOT_SOLVING:
                    gold = self.golds[tid]
                    if states_the_answer(o.completion, gold, infer_gold_type(gold), contract.derived):
                        st.coincident += 1
                        continue
                if probe.expect == "zero" and o.verdict.ok:
                    st.n_compared += 1
                    signal = contract.signal(o.verdict)
                    if signal is not None and signal > bar(o) + EPS:
                        st.above_gold += 1
                        if st.above_example is None:
                            st.above_example, st.above_bar = o, bar(o)
                        any_above = True
                st.obs_hits += int(h)
                st.n_scored += int(o.verdict.ok)
                counted = st.variant_hits[o.variant]
                counted[0] += int(h)
                counted[1] += int(o.verdict.ok)
                if h and example is None:
                    example = o
                if h and probe.sweep:
                    st.variant_examples.setdefault(o.variant, o)
                any_hit |= h
                any_partial |= p
            if any_hit:
                st.task_hits += 1
                if example is not None and len(st.examples) < 3:
                    st.examples.append(example)
            if any_partial:
                st.partial += 1
            st.above_tasks += int(any_above)
            st.fail_tasks += int(any_fail)

    def _add_paid(self, o: Observation) -> None:
        signal, gold = self.contract.signal(o.verdict), self.contract.gold_score.get(o.task_id, self.contract.full_score)
        if signal < gold - EPS:
            self.paid_calls += 1
            self.paid_share += signal / gold
            self.paid_tasks.add(o.task_id)
            self.paid_example = self.paid_example or o

    def finish(self) -> dict[str, Any]:
        return _summarize(self)


def compute(
    observations: list[Observation],
    probes: list[Probe],
    tasks: list[Task],
    contract: Contract,
    gold_type: str,
    *,
    discovery_frac: float = 0.4,
    controls: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    tally = Tally(probes, tasks, contract, gold_type, discovery_frac)
    tally.add(observations, controls)
    return tally.finish()


def _sweep(s: ProbeStats) -> tuple[str, int, int, str]:
    """(variant, hits, scored, note) of a probe whose variants grow along one scale. The note gives the acceptance rate of each.
    The variant reported is where the behaviour ends: the widest error accepted at least half the time, or for a right
    answer, the shortest text that loses the score at least half the time."""
    order = sorted(s.variant_hits, key=lambda v: float(re.search(r"\d+(?:\.\d+)?", v).group()))
    solving = s.probe.label == Label.SOLVING
    shown = []
    for v in order:
        k, n = s.variant_hits[v]
        shown.append(f"{v} {(n - k if solving else k) / n:.0%}" if n else f"{v} no score")
    half = [v for v in order if s.variant_hits[v][1] and 2 * s.variant_hits[v][0] >= s.variant_hits[v][1]]
    variant = (half[0] if solving else half[-1]) if half else s.most_accepted_variant()[0]
    return variant, *s.variant_hits[variant], "accepted: " + ", ".join(shown)


def _partial_head(tally: Tally, n: int) -> dict[str, Any]:
    """Wrong answers paid more than nothing and less than the reference answer earns."""
    k = tally.paid_calls
    earned = f", earning {tally.paid_share / k:.0%} of its score on average" if k else ""
    h = _metric("wrong_partial", "Wrong answers paid in part", k, n, f"the reference answer with a definite error{earned}")
    h |= {"severity": "medium" if h["value"] >= 0.1 else "low", "note": "",
          "example": tally.paid_example.to_dict() if tally.paid_example else None}
    return _require_evidence(h, n >= MIN_CALLS and len(tally.paid_tasks) >= MIN_TASKS)


def _summarize(tally: Tally) -> dict[str, Any]:
    stats, probes, tasks, contract, gold_type = tally.stats, tally.probes, tally.tasks, tally.contract, tally.gold_type
    null_floor, passing, n_disc = tally.null_floor, tally.passing, tally.n_disc

    floor, floor_note = floor_estimate(tasks, gold_type)

    # --- head 1: best gold-blind strategy, selected on discovery, reported on confirmation -----
    best: tuple[float, ProbeStats, str, list[int]] | None = None
    for s in stats.values():
        if not (s.probe.gold_blind and s.probe.headline and s.probe.label == Label.NOT_SOLVING):
            continue
        for vname, (dh, dn, ch, cn) in s.variants.items():
            disc_rate = dh / dn if dn else 0.0
            key = (disc_rate, ch / cn if cn else 0.0)
            if best is None or key > (best[0], best[3][2] / best[3][3] if best[3][3] else 0.0):
                best = (disc_rate, s, vname, [dh, dn, ch, cn])
    if best is not None:
        _, s, vname, (dh, dn, ch, cn) = best
        if cn > 0:
            head1 = _metric("non_solving_accepted", "Non-answers accepted", ch, cn,
                            f"best reply {s.probe.id}/{vname} ({s.probe.name}); a fixed answer would be right on {floor:.0%} of rows")
        else:
            head1 = _metric("non_solving_accepted", "Non-answers accepted", dh, dn,
                            f"best reply {s.probe.id}/{vname} ({s.probe.name}); a fixed answer would be right on {floor:.0%} of rows; "
                            "measured on the rows it was chosen on")
        head1["strategy"] = {"probe": s.probe.id, "variant": vname, "discover": [dh, dn], "confirm": [ch, cn]}
    else:
        head1 = _metric("non_solving_accepted", "Non-answers accepted", 0, 0, "no probe that ignores the reference answer ran")
    head1["floor"] = floor
    head1["above_floor"] = head1["ci"][0] > floor
    if head1["n"] and not head1["above_floor"]:
        head1["detail"] += "; not above what a fixed answer gets"

    def pooled(pred) -> tuple[int, int]:
        sel = [s for s in stats.values() if pred(s) and s.probe.headline]
        return sum(s.obs_hits for s in sel), sum(s.n_scored for s in sel)

    hk, hn = pooled(lambda s: s.probe.family == "O" and s.probe.label == Label.NOT_SOLVING)
    wk, wn = pooled(lambda s: s.probe.family == "M")
    rk, rn = pooled(lambda s: s.probe.label == Label.SOLVING)
    head_hedge = _metric("hedging_accepted", "Hedged answers accepted", hk, hn,
                         "one right and one wrong answer in the same reply")
    head2 = _metric("wrong_accepted", "Wrong answers accepted", wk, wn, "the reference answer with a definite error")
    head3 = _metric("right_rejected", "Right answers rejected", rk, rn, "the reference answer rewritten, still correct")

    robustness = tally.robust.stats(contract)
    robustness["null_floor"] = null_floor

    findings: list[dict[str, Any]] = []
    for s in stats.values():
        if s.obs_hits == 0:
            continue
        rate, n, example, note = s.rate, s.n_scored, s.examples[0] if s.examples else None, ""
        lo, hi = s.ci
        if s.probe.sweep:
            variant, k, n, note = _sweep(s)
            rate = k / n
            lo, hi = wilson(k, n)
            example = s.variant_examples.get(variant, example)
        elif s.probe.by_variant:  # distinct replies under one probe: the checker may accept only one of them
            variant, k, n = s.most_accepted_variant()
            rate = k / n
            lo, hi = wilson(k, n)
            example = next((e for e in s.examples if e.variant == variant), example)
        if s.probe.label == Label.SOLVING:
            category, sev = "right_rejected", _rejected_severity(s.probe.id, rate)
        else:
            category, sev = _non_solving_finding(s.probe, rate, lo, floor)
            sev = _tolerance_severity(variant, rate) if s.probe.sweep else sev
        findings.append(_require_evidence({
            "probe_id": s.probe.id, "name": s.probe.name, "family": s.probe.family, "tier": s.probe.tier,
            "category": category, "severity": sev, "rate": rate, "ci": [lo, hi], "n": n,
            "task_rate": s.task_rate, "n_tasks": s.n_tasks, "gold_blind": s.probe.gold_blind, "above_floor": lo > floor,
            "example": example.to_dict() if example else None, "note": note,
        }, s.n_scored >= MIN_CALLS and s.task_hits >= MIN_TASKS))
    for s in stats.values():
        bad = s.errors + s.timeouts + s.crashes
        if bad:
            lo, hi = wilson(bad, s.n_obs)
            findings.append(_require_evidence({
                "probe_id": s.probe.id, "name": s.probe.name, "family": s.probe.family, "tier": s.probe.tier,
                "category": "robustness", "severity": "medium" if (s.crashes or s.timeouts) else "low",
                "rate": bad / s.n_obs, "ci": [lo, hi], "n": s.n_obs, "task_rate": 0.0, "n_tasks": s.n_tasks,
                "example": None, "note": f"{s.errors} errors, {s.timeouts} timeouts, {s.crashes} crashes",
            }, s.fail_tasks >= MIN_TASKS))
    rejected_gold = _rejected_gold_finding(contract, tasks)
    if rejected_gold:
        findings.append(rejected_gold)
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    # within a severity: what a policy could learn without solving first, then wrong answers, hedges, rejections
    def category_rank(f: dict[str, Any]) -> int:
        if f["category"] == "non_solving_accepted":
            return 0 if f.get("gold_blind") else 2
        return {"wrong_accepted": 1, "right_rejected": 3}.get(f["category"], 4)

    heads = [head1, head2, head3, head_hedge, _partial_head(tally, wn)]
    withheld: list[dict[str, Any]] = []
    k_above, n_above = sum(s.above_gold for s in stats.values()), sum(s.n_compared for s in stats.values())
    discrimination = None
    if contract.no_discrimination:
        # reaching the gold's score means nothing here, in either direction; robustness stands, and so does a reply
        # that scored above the gold
        heads = [_not_measured(h, NO_DISCRIMINATION) for h in heads]
        withheld = [f for f in findings if f["category"] != "robustness"]
        above = _above_gold_findings(stats, floor, contract)
        findings = [f for f in findings if f["category"] == "robustness"] + above
        discrimination = _discrimination(contract, above, k_above, n_above)
    elif contract.whole_reply_similarity:
        heads = [_not_measured(h, WHOLE_REPLY) if h["key"] == "right_rejected" else h for h in heads]
        withheld = [f for f in findings if f["category"] == "right_rejected"]
        findings = [f for f in findings if f["category"] != "right_rejected"]
    findings.sort(key=lambda f: (sev_rank[f["severity"]], category_rank(f), -f["rate"]))
    withheld.sort(key=lambda f: (sev_rank[f["severity"]], category_rank(f), -f["rate"]))

    return {
        "head": heads,
        "floor": {"value": floor, "note": floor_note},
        "robustness": robustness,
        "probes": [stats[p.id].to_dict() for p in probes if p.id in stats],
        "findings": findings,
        "withheld_findings": withheld,  # measured, but what they would claim does not hold here (see "head")
        # non-solving replies strictly above the gold's signal on the same task, whatever the class of the checker
        "above_gold": {"k": k_above, "n": n_above, "rate": k_above / n_above if n_above else 0.0,
                       "probes": dict(sorted(((pid, s.above_gold) for pid, s in stats.items() if s.above_gold),
                                             key=lambda x: -x[1])),
                       "wrong_above": contract.wrong_above, "wrong_checked": contract.wrong_checked},
        "discrimination": discrimination,  # {"case", "reason"} when contract.no_discrimination, else None
        "n_tasks": len(tasks),
        "n_passing": len(passing),
        "discovery_n": n_disc,
    }


def _evidence(f: dict[str, Any]) -> dict[str, Any]:
    return {"probe_id": f["probe_id"], "name": f["name"], "severity": f["severity"], "rate": f["rate"], "n": f["n"],
            "unit": f.get("unit", "calls"), "note": f.get("note", "")}


def _wrong_answer_example(contract: Contract) -> dict[str, Any] | None:
    return next((Observation("wrong_answer", "G", "wrong_answer", tid, comp, v, Label.NOT_SOLVING, "zero", "natural").to_dict()
                 for wid, tid, comp, v in contract.observations if wid == "wrong_answer" and contract.accepted_verdict(v, tid)), None)


def _contract_evidence(contract: Contract) -> list[dict[str, Any]]:
    """What the calls with the reference answers and the wrong answers show on their own, with no probe involved."""
    n, out = contract.wrong_checked, []
    if contract.format_only:
        out.append(_require_evidence(
            {"probe_id": "format_only", "name": "any answer in the right format earns the reference answer's score",
             "severity": "high", "rate": contract.wrong_same / n, "n": n, "note": contract.discrimination_note,
             "example": _wrong_answer_example(contract)}, n >= MIN_CALLS and contract.wrong_same >= MIN_TASKS))
    if contract.no_discrimination and contract.wrong_above:
        out.append(_require_evidence(
            {"probe_id": "wrong_above", "name": "wrong answers above the reference answer",
             "severity": _wrong_severity(contract.wrong_above / n), "rate": contract.wrong_above / n, "n": n, "note": "",
             "example": None}, n >= MIN_CALLS and contract.wrong_above >= MIN_TASKS))
    return out


def _defect_of(f: dict[str, Any]) -> str | None:
    """The catalog entry a finding is evidence for. A fixed answer below what chance gives says nothing about the checker."""
    if f.get("above_gold"):
        return ABOVE_REFERENCE
    if f["category"] == "robustness" and f["probe_id"] != "S1":
        return ROBUSTNESS
    defect = BY_PROBE.get(f["probe_id"])
    return None if defect == FIXED_ANSWER and not f.get("above_floor") else defect


def collect_defects(metrics: dict[str, Any], contract: Contract) -> list[dict[str, Any]]:
    """One record per defect with evidence, most severe first: its severity is the highest of its evidence."""
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in metrics["findings"]:
        if (defect := _defect_of(f)) is not None:
            found[defect].append(f)
    found[ABOVE_REFERENCE] += _contract_evidence(contract)
    partial = next(h for h in metrics["head"] if h["key"] == "wrong_partial")
    if partial["k"]:
        found[WRONG_PARTIAL].append({"probe_id": partial["key"], "name": partial["title"], "severity": partial["severity"],
                                     "rate": partial["value"], "n": partial["n"], "note": partial["note"], "example": partial["example"]})
    rank = {"high": 0, "medium": 1, "low": 2}
    out = []
    for d in DEFECTS:
        evidence = sorted(found.get(d.id, []), key=lambda f: (rank[f["severity"]], -f["rate"]))
        if evidence:
            out.append({"id": d.id, "name": d.name, "explanation": d.explanation, "severity": evidence[0]["severity"],
                        "evidence": [_evidence(f) for f in evidence],
                        "example": next((f["example"] for f in evidence if f.get("example")), None)})
    return sorted(out, key=lambda d: rank[d["severity"]])
