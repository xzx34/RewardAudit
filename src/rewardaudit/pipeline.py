"""The audit pipeline: tasks -> gold type -> contract -> determinism -> probes -> metrics -> dataset checks."""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rewardaudit import __version__
from rewardaudit.contract import COMMON_WRAPPERS, Contract, infer_contract
from rewardaudit.gate import check_contract, check_loaded
from rewardaudit.goldtype import infer_dataset_type, infer_task_type
from rewardaudit.metrics import EPS, Tally, collect_defects, nondeterminism_finding
from rewardaudit.probes.base import Probe, ProbeContext
from rewardaudit.probes.registry import all_probes
from rewardaudit.runner import RunnerPool
from rewardaudit.types import Kind, Observation, Task

TIER_RANK = {"natural": 0, "amplified": 1, "contrived": 2}
DETERMINISM_REPEATS = 3
NO_ANSWER, NO_ANSWER_REPEATS = "There is no answer here.", 8  # no lone letter a checker could read as an option
ESTIMATE_TASKS = 20  # tasks whose probe calls are counted ahead, to tell how long the rest will take


@dataclass
class AuditConfig:
    n_tasks: int | None = 100  # None: every row
    seed: int = 0
    timeout_s: float = 10.0
    families: list[str] | None = None
    python: str | None = None
    workers: int = 1
    memory_limit_mb: float | None = 4096  # per worker while scoring; above it the call is a CRASH
    keep_observations: bool = False  # hold every probe call in `AuditResult.observations` (memory grows with the sample)
    formats: list[str] = field(default_factory=list)  # reply formats of the checker's own, such as "<sol>{answer}</sol>"


@dataclass
class AuditResult:
    env_name: str
    meta: dict[str, Any]
    config: AuditConfig
    tasks: list[Task]
    gold_type: str
    gold_type_counts: dict[str, int]
    contract: Contract
    metrics: dict[str, Any]
    dataset_issues: list[dict[str, Any]]
    determinism: dict[str, Any]
    latency_ms: dict[str, float]
    elapsed_s: float
    n_rows: int = 0
    spec: dict[str, Any] = field(default_factory=dict)
    version: str = __version__
    observations: list[Observation] = field(default_factory=list)
    defects: list[dict[str, Any]] = field(default_factory=list)  # most severe first: id, name, severity, evidence, example

    @property
    def findings(self) -> list[dict[str, Any]]:
        """Every finding, most severe first: probe_id, name, category, severity, rate, ci, n and an example."""
        return self.metrics["findings"]

    @property
    def rates(self) -> dict[str, float]:
        """The share of calls on each headline: non_solving_accepted, wrong_accepted, hedging_accepted, right_rejected, wrong_partial.
        A headline no call measured is left out, since 0.0 would read as nothing accepted."""
        return {h["key"]: h["value"] for h in self.metrics["head"] if h["n"] and not h.get("not_measured")}

    def print(self, *, full: bool = False) -> None:
        from rewardaudit.report import render_terminal

        render_terminal(self, full=full)

    def save(self, out_dir: str | Path) -> Path:
        """Write report.md, report.json and repro.py; returns the path of report.md."""
        from rewardaudit.report import write_outputs

        return write_outputs(self, out_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "env": self.env_name,
            "meta": self.meta,
            "spec": self.spec,
            "config": vars(self.config),
            "elapsed_s": self.elapsed_s,
            "n_rows": self.n_rows,
            "n_tasks": len(self.tasks),
            "gold_type": self.gold_type,
            "gold_type_counts": self.gold_type_counts,
            "contract": self.contract.to_dict(),
            "determinism": self.determinism,
            "latency_ms": self.latency_ms,
            "defects": self.defects,
            "metrics": self.metrics,
            "dataset_issues": self.dataset_issues,
        }


Log = Callable[[str], None]


def _check_determinism(runner: RunnerPool, tasks: list[Task], contract: Contract, repeats: int) -> dict[str, Any]:
    """The reference answer, and a reply with no answer in it, each sent several times. A checker may guess at random only
    when it finds no answer."""
    sample = [t for t in tasks if contract.self_pass.get(t.id)][:5]
    inconsistent: list[str] = []
    for t in sample:
        comp = contract.wrap(t.answer, t.id)
        scores = {runner.score(t, comp).score for _ in range(repeats)}
        guesses = {runner.score(t, NO_ANSWER).score for _ in range(NO_ANSWER_REPEATS)}
        if len(scores) > 1 or len(guesses) > 1:
            inconsistent.append(t.id)
    return {"checked": len(sample), "repeats": repeats, "inconsistent_tasks": inconsistent,
            "deterministic": not inconsistent}


def _gold_controls(runner: RunnerPool, observations: list[Observation], tasks_by_id: dict[str, Task], contract: Contract
                   ) -> dict[tuple[str, str], float]:
    """A reply above the gold may owe it to its own reasoning text and not to its answer, where a rubric pays for the text
    in a thinking block. So each such reply goes again with the gold as its final answer, the reasoning unchanged: that
    is the gold it has to beat. (task id, completion) -> the signal of that control; a reply with no final answer to
    replace has none, and a control that is the plain gold in its format is already measured."""
    items: dict[tuple[str, str], tuple[Task, str]] = {}
    for o in observations:
        signal = contract.signal(o.verdict)
        if (o.expect != "zero" or signal is None or not contract.self_pass.get(o.task_id)
                or signal <= contract.gold_score[o.task_id] + EPS):
            continue
        gold = tasks_by_id[o.task_id].answer
        control = contract.with_answer(o.completion, gold, o.task_id)
        if control is not None and control != contract.wrap(gold, o.task_id):
            items[(o.task_id, o.completion)] = (tasks_by_id[o.task_id], control)
    verdicts = runner.score_many(list(items.values())) if items else []
    return {key: s for key, v in zip(items, verdicts, strict=True) if (s := contract.signal(v)) is not None}


def _variants(probe: Probe, tasks: list[Task], ctx: ProbeContext, seed: int) -> list[tuple[Task, str, str]]:
    out: list[tuple[Task, str, str]] = []
    for task in tasks:
        ctx.task = task
        rng = random.Random(f"{seed}:{probe.id}:{task.id}")
        out.extend((task, vname, comp) for vname, comp in probe.generate(task, rng, ctx))
    return out


def _stage(progress: Callable[[str, int, int], None] | None, stage: str, total: int) -> Callable[[int], None] | None:
    """What `RunnerPool.score_many` calls with the calls just finished, as progress(stage, done, total). The total is an estimate."""
    if progress is None:
        return None
    done = [0]

    def step(k: int) -> None:
        done[0] += k
        progress(stage, done[0], max(total, done[0]))

    return step


def run_audit(
    spec: dict[str, Any],
    config: AuditConfig | None = None,
    *,
    log: Log | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> AuditResult:
    """Raises AuditError (NotAuditable among them) with a message for the user when the checker cannot be audited."""
    config = config or AuditConfig()
    log = log or (lambda s: None)
    t_start = time.perf_counter()

    with RunnerPool(spec, workers=config.workers, timeout_s=config.timeout_s, python=config.python,
                    memory_limit_mb=config.memory_limit_mb) as runner:
        log(f"loaded {runner.name}" + (f" ({config.workers} workers)" if config.workers > 1 else ""))
        tasks = runner.tasks(config.n_tasks, config.seed)
        n_rows = int(runner.meta.get("n_rows") or len(tasks))
        log(f"sampled {len(tasks):,} of {n_rows:,} rows (seed {config.seed})")
        gold_type, counts = infer_dataset_type(tasks)
        gold_types = {t.id: infer_task_type(t) for t in tasks}
        check_loaded(runner.meta, tasks)

        answerable = [t for t in tasks if t.answer.strip()] or tasks  # an empty gold is a dataset issue, not a rejection
        t_contract = time.perf_counter()
        contract = infer_contract(runner, answerable, formats=config.formats, progress=_stage(progress, "reference answers", (len(COMMON_WRAPPERS) + 1) * len(answerable)))
        s_per_call = (time.perf_counter() - t_contract) / max(1, contract.calls)
        check_contract(contract, tasks, runner.meta)
        if contract.signal_keys:
            log(f"composite score; correctness judged on {' + '.join(contract.signal_keys)}")
        lat = [v.elapsed_ms for _, _, _, v in contract.observations if v.kind == Kind.SCORE]
        latency = {"median": statistics.median(lat) if lat else 0.0, "max": max(lat) if lat else 0.0,
                   "s_per_call": s_per_call}

        determinism = _check_determinism(runner, tasks, contract, DETERMINISM_REPEATS)
        if not determinism["deterministic"]:
            log("the checker gave different scores to identical inputs")

        ctx = ProbeContext(contract=contract, gold_type=gold_type, tasks=answerable, gold_types=gold_types)
        probes: list[Probe] = sorted(all_probes(config.families), key=lambda p: TIER_RANK.get(p.tier, 9))
        passing = [t for t in tasks if contract.self_pass.get(t.id)]
        head = passing[:ESTIMATE_TASKS]
        estimate = sum(len(_variants(p, head, ctx, config.seed)) for p in probes) * len(passing) // max(1, len(head))
        step = _stage(progress, "probes", estimate)
        tally = Tally(probes, answerable, contract, gold_type)
        by_id = {t.id: t for t in answerable}
        kept: list[Observation] = []
        for probe in probes:
            pending = _variants(probe, passing, ctx, config.seed)
            verdicts = runner.score_many([(t, comp) for t, _, comp in pending], progress=step)
            observations = [Observation(probe.id, probe.family, vname, task.id, comp, v, probe.label, probe.expect, probe.tier)
                            for (task, vname, comp), v in zip(pending, verdicts, strict=True)]
            tally.add(observations, _gold_controls(runner, observations, by_id, contract))
            if config.keep_observations:
                kept.extend(observations)
        metrics = tally.finish()
        if not determinism["deterministic"]:
            metrics["findings"].append(nondeterminism_finding(determinism))
            metrics["findings"].sort(key=lambda f: f["severity"] != "high")  # stable: the mediums stay in order

        # errors while scoring the gold itself are the most important robustness signal of all
        gold_calls = [v for wid, _, _, v in contract.observations if wid != "wrong_answer"]
        gold_kinds = [v.kind for v in gold_calls]
        metrics["robustness"]["gold_calls"] = len(gold_kinds)
        metrics["robustness"]["gold_error_rate"] = (
            sum(1 for k in gold_kinds if k != Kind.SCORE) / len(gold_kinds) if gold_kinds else 0.0
        )
        err_example = next((v.error for v in gold_calls if v.kind == Kind.ERROR and v.error), None)
        metrics["robustness"]["gold_error_example"] = (err_example or "")[-600:]
        issues = runner.dataset_issues()
        env_name, meta = runner.name, dict(runner.meta)
        meta["worker_restarts"] = runner.restarts
        meta["checker_calls"] = runner.calls

    return AuditResult(
        env_name=env_name, meta=meta, config=config, tasks=tasks, gold_type=gold_type,
        gold_type_counts=dict(counts), contract=contract, metrics=metrics,
        dataset_issues=issues, determinism=determinism, latency_ms=latency,
        elapsed_s=time.perf_counter() - t_start, n_rows=n_rows,
        spec={k: v for k, v in spec.items() if k not in ("pickled", "sys_path")},
        observations=kept, defects=collect_defects(metrics, contract),
    )


def exit_code(result: AuditResult, fail_on: dict[str, float] | None = None) -> int:
    """0 clean, 1 above the thresholds (default: any high defect), 2 the checker cannot tell the reference answer from a
    wrong one, so the headline rates were not measured."""
    if result.contract.no_discrimination:
        return 2
    if fail_on:
        return int(any(result.rates.get(key, 0.0) > thr for key, thr in fail_on.items()))
    return int(any(d["severity"] == "high" for d in result.defects))
