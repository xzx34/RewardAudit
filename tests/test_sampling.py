"""Which rows an audit looks at, and how it stays small: a uniform draw from the whole dataset, dataset checks over every
row, metrics added up in batches, no probe that compares a task with all the others."""

import json
import random
import time

import pytest
from rich.console import Console

from rewardaudit.adapters.base import sample_tasks
from rewardaudit.adapters.dataset import load_tasks
from rewardaudit.adapters.function_adapter import FunctionAdapter
from rewardaudit.api import audit
from rewardaudit.contract import Contract
from rewardaudit.gate import label_gold_reason
from rewardaudit.metrics import Tally, compute
from rewardaudit.pipeline import TIER_RANK, AuditConfig, run_audit
from rewardaudit.probes import mutate
from rewardaudit.probes.base import ProbeContext
from rewardaudit.probes.registry import all_probes
from rewardaudit.report import render_terminal
from rewardaudit.types import Kind, Label, Observation, Task, Verdict

ENDS_WITH_GOLD = """
def score(completion, answer):
    return float(completion.strip().endswith(answer))
"""


def _rows(k: int) -> list[dict]:
    return [{"prompt": f"Add the numbers on card {i}.", "answer": str(1000 + 7 * i)} for i in range(k)]


def _pool(k: int) -> list[Task]:
    return [Task(str(i), f"p{i}", str(i)) for i in range(k)]


def _write(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def checker(tmp_path) -> str:
    path = tmp_path / "reward.py"
    path.write_text(ENDS_WITH_GOLD, encoding="utf-8")
    return str(path)


# --- the draw ------------------------------------------------------------------------------------------------------------

def test_a_sample_is_drawn_without_replacement_from_the_whole_dataset():
    pool = _pool(1000)
    ids = [t.id for t in sample_tasks(pool, 100, 0)]
    assert len(ids) == len(set(ids)) == 100 and set(ids) <= {t.id for t in pool}
    assert max(map(int, ids)) >= 500 and ids != [t.id for t in pool[:100]]  # neither the head nor in the dataset's order


def test_every_row_is_as_likely_to_be_drawn():
    pool, counts = _pool(20), [0] * 20
    for seed in range(2000):
        for t in sample_tasks(pool, 5, seed):
            counts[int(t.id)] += 1
    assert all(abs(c - 500) < 100 for c in counts), counts  # five standard deviations around 2000 * 5 / 20


def test_the_same_seed_draws_the_same_rows_and_another_seed_others():
    pool = _pool(200)
    assert sample_tasks(pool, 20, 3) == sample_tasks(pool, 20, 3)
    assert sample_tasks(pool, 20, 3) != sample_tasks(pool, 20, 4)


@pytest.mark.parametrize("n", [None, 30, 31, 10_000])
def test_a_sample_as_large_as_the_dataset_is_the_dataset_in_order(n):
    pool = _pool(30)
    assert sample_tasks(pool, n, 0) == pool


def test_an_adapter_samples_and_says_how_many_rows_there_are(tmp_path, checker):
    ad = FunctionAdapter({"path": checker, "dataset": _write(tmp_path, _rows(30))})
    assert len(ad.tasks(5, 0)) == 5 and ad.meta["n_rows"] == 30
    assert ad.tasks(None, 0) == ad.all_tasks() and len(ad.all_tasks()) == 30


@pytest.mark.parametrize("n, n_tasks", [(5, 5), (None, 12), ("all", 12), (12, 12), (1000, 12)])
def test_an_audit_of_n_rows(tmp_path, checker, n, n_tasks):
    r = audit(checker, _write(tmp_path, _rows(12)), n=n, probes=["N"], workers=1)
    assert len(r.tasks) == n_tasks and r.n_rows == 12
    assert r.to_dict()["n_rows"] == 12 and r.to_dict()["n_tasks"] == n_tasks


def test_an_audits_rows_follow_its_seed_and_not_its_workers(tmp_path, checker):
    data = _write(tmp_path, _rows(40))

    def ids(**kw):
        return [t.id for t in audit(checker, data, n=8, probes=["N"], **{"workers": 1, **kw}).tasks]

    first = ids(seed=1)
    assert ids(seed=1, workers=2) == first and ids(seed=2) != first


# --- row ids -------------------------------------------------------------------------------------------------------------

def test_a_datasets_own_ids_are_used_when_they_tell_the_rows_apart(tmp_path):
    rows = [{**r, "id": f"row-{i}"} for i, r in enumerate(_rows(4))]
    assert [t.id for t in load_tasks(_write(tmp_path, rows))[0]] == ["row-0", "row-1", "row-2", "row-3"]
    rows = [{**r, "id": 100 + i} for i, r in enumerate(_rows(4))]
    assert [t.id for t in load_tasks(_write(tmp_path, rows))[0]] == ["100", "101", "102", "103"]


@pytest.mark.parametrize("ids", [["same"] * 4, ["a", "b", "a", "c"], [1, "1", 2, 3], [None] * 4])
def test_row_ids_stay_unique_when_the_id_column_repeats(tmp_path, ids):
    rows = [{**r, "id": i} for i, r in zip(ids, _rows(4), strict=True)]
    assert [t.id for t in load_tasks(_write(tmp_path, rows))[0]] == ["0", "1", "2", "3"]


def test_an_id_column_that_collides_with_a_row_number_where_it_is_missing(tmp_path):
    rows = _rows(4)
    rows[0]["id"] = 2  # row 2 has no id and takes its number
    assert [t.id for t in load_tasks(_write(tmp_path, rows))[0]] == ["0", "1", "2", "3"]


def test_an_audit_keeps_rows_with_a_repeated_id_apart(tmp_path, checker):
    rows = [{**r, "id": "same"} for r in _rows(10)]
    r = audit(checker, _write(tmp_path, rows), n="all", probes=["N"], workers=1)
    assert len({t.id for t in r.tasks}) == 10 and len(r.contract.self_pass) == 10 and r.contract.self_pass_rate == 1.0


# --- dataset checks see every row ------------------------------------------------------------------------------------------

def _outside_the_sample(k: int, n: int, seed: int) -> list[int]:
    picked = {int(t.id) for t in sample_tasks(_pool(k), n, seed)}
    return [i for i in range(k) if i not in picked]


@pytest.mark.parametrize("workers", [1, 2])
def test_issues_outside_the_sample_are_reported(tmp_path, checker, workers):
    rows = _rows(30)
    a, b, *_ = _outside_the_sample(30, 5, seed=0)
    rows[b]["prompt"] = rows[a]["prompt"]
    r = audit(checker, _write(tmp_path, rows), n=5, seed=0, probes=["N"], workers=workers)
    assert not {str(a), str(b)} & {t.id for t in r.tasks} and r.n_rows == 30
    issues = {i["id"]: i for i in r.dataset_issues}
    assert issues["duplicate_prompts"]["tasks"] == [[str(a), str(b)]] and "empty_rows_left_out" not in issues


@pytest.mark.parametrize("workers", [1, 2])
def test_rows_with_an_empty_reference_are_left_out_before_the_draw(tmp_path, checker, workers):
    rows = _rows(30)
    for i, empty in zip((3, 8, 15, 22), ("", "  \n", None, []), strict=True):
        rows[i]["answer"] = empty
    r = audit(checker, _write(tmp_path, rows), n=20, probes=["N"], workers=workers)
    assert r.n_rows == 26 and len(r.tasks) == 20 and not {"3", "8", "15", "22"} & {t.id for t in r.tasks}
    assert all(t.answer.strip() for t in r.tasks) and r.contract.self_pass_rate == 1.0
    issues = {i["id"]: i for i in r.dataset_issues}
    left = issues["empty_rows_left_out"]
    assert left["name"] == "4 rows with an empty reference were left out" and left["count"] == 4 and left["severity"] == "low"
    assert "empty_answers" not in issues and r.meta["empty_rows_left_out"] == 4


def test_one_row_left_out_is_one_row(tmp_path, checker):
    rows = _rows(12)
    rows[5]["answer"] = ""
    r = audit(checker, _write(tmp_path, rows), n=4, probes=["N"], workers=1)
    console = Console(width=200, record=True, force_terminal=False)
    render_terminal(r, console)
    text = console.export_text()
    assert "Dataset: 1 row with an empty reference was left out (they were not sampled;" in text and "4 of 11 rows" in text


def test_a_skew_of_the_whole_dataset_is_counted_on_all_its_rows(tmp_path, checker):
    rows = _rows(40)
    for row in rows[:12]:
        row["answer"] = "7"
    r = audit(checker, _write(tmp_path, rows), n=5, probes=["N"], workers=1)
    assert next(i for i in r.dataset_issues if i["id"] == "skewed_answers")["count"] == 12


# --- the metrics in batches ------------------------------------------------------------------------------------------------

HEDGES_AND_FAILS = """
def score(completion, answer):
    if "(" in completion and ")" in completion and len(completion) > 400:
        raise ValueError("cannot read this reply")
    return float(answer in completion)
"""


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("measured")
    (tmp / "reward.py").write_text(HEDGES_AND_FAILS, encoding="utf-8")
    golds = [str(100 + 3 * i) for i in range(20)]
    (tmp / "data.jsonl").write_text("\n".join(json.dumps({"prompt": f"Card {i}.", "answer": g}) for i, g in enumerate(golds)))
    spec = {"kind": "function", "path": str(tmp / "reward.py"), "dataset": str(tmp / "data.jsonl")}
    return run_audit(spec, AuditConfig(n_tasks=20, keep_observations=True)), spec


def _in_the_audits_order():
    return sorted(all_probes(), key=lambda p: TIER_RANK.get(p.tier, 9))


def _by_probe(observations):
    groups: dict[str, list] = {}
    for o in observations:
        groups.setdefault(o.probe_id, []).append(o)
    return list(groups.values())


def _by_probe_and_task(observations):
    groups: dict[tuple, list] = {}
    for o in observations:
        groups.setdefault((o.probe_id, o.task_id), []).append(o)
    return list(groups.values())


def _two_parts(observations):
    groups = _by_probe(observations)
    return [[o for g in groups[:7] for o in g], [o for g in groups[7:] for o in g]]


@pytest.mark.parametrize("batches", [_by_probe, _by_probe_and_task, _two_parts])
def test_tally_in_batches_gives_the_metrics_of_compute_on_all_at_once(measured, batches):
    r, _ = measured
    probes = _in_the_audits_order()
    assert r.observations and r.findings
    tally = Tally(probes, r.tasks, r.contract, r.gold_type)
    for batch in batches(r.observations):
        tally.add(batch)
    assert tally.finish() == compute(r.observations, probes, r.tasks, r.contract, r.gold_type)


def test_the_pipeline_reports_what_compute_gives(measured):
    r, _ = measured
    whole = compute(r.observations, _in_the_audits_order(), r.tasks, r.contract, r.gold_type)
    robustness = {k: v for k, v in r.metrics["robustness"].items() if not k.startswith("gold_")}  # the pipeline adds these
    assert {**r.metrics, "robustness": robustness} == whole


def test_a_control_counts_in_the_batch_it_comes_with():
    probe = next(p for p in all_probes() if p.id == "O10")
    tasks = [Task(id=t, prompt="p", answer="7") for t in ("a", "b")]
    reply = "<think>\nnote\n</think>\n\\boxed{9}"
    obs = {t.id: Observation("O10", "O", "think", t.id, reply, Verdict(Kind.SCORE, score=1.5), Label.NOT_SOLVING, "zero", "natural")
           for t in tasks}
    contract = Contract(full_score=1.0, accepted=["think_boxed"], primary="think_boxed", wrapper_rates={"think_boxed": 1.0},
                        self_pass={"a": True, "b": True}, gold_score={"a": 1.0, "b": 1.0})
    controls = {("a", reply): 1.5, ("b", reply): 1.2}  # a reply owes its score to its reasoning on "a", and not on "b"
    tally = Tally([probe], tasks, contract, "int")
    tally.add([obs["a"]], {("a", reply): 1.5})
    tally.add([obs["b"]], {("b", reply): 1.2})
    assert tally.finish() == compute(list(obs.values()), [probe], tasks, contract, "int", controls=controls)
    assert tally.finish()["above_gold"]["k"] == 1


def test_calls_are_kept_only_when_asked_for_and_change_nothing(measured):
    kept, spec = measured
    r = run_audit(spec, AuditConfig(n_tasks=20))
    assert r.observations == [] and kept.observations
    assert [(f["probe_id"], f["rate"], f["severity"]) for f in r.findings] == [(f["probe_id"], f["rate"], f["severity"]) for f in kept.findings]
    assert r.rates == kept.rates and r.meta["checker_calls"] == kept.meta["checker_calls"]


# --- nothing that grows with the square of the dataset --------------------------------------------------------------------

def test_another_tasks_gold_is_looked_for_among_a_few_tasks(monkeypatch):
    compared = []
    real = mutate.equiv
    monkeypatch.setattr(mutate, "equiv", lambda *a, **k: compared.append(a) or real(*a, **k))

    def comparisons(k: int) -> int:
        tasks = [Task(str(i), "p", str(1000 + i)) for i in range(k)]
        contract = Contract(full_score=1.0, accepted=["bare"], primary="bare", wrapper_rates={"bare": 1.0},
                            self_pass={t.id: True for t in tasks}, gold_score={t.id: 1.0 for t in tasks})
        ctx = ProbeContext(contract=contract, gold_type="int", tasks=tasks, task=tasks[0])
        compared.clear()
        assert len(mutate.m20_cross_task(tasks[0], random.Random(0), ctx)) == 2
        return len(compared)

    assert comparisons(50) <= 40 and comparisons(20_000) <= 40


def test_label_golds_are_read_in_time_linear_in_the_rows():
    tasks = [Task(str(i), f"Write it.\ndef fn_{i}(x):", f"fn_{i}") for i in range(20_000)]
    t0 = time.perf_counter()
    assert "identifiers the prompt prints" in label_gold_reason(tasks)
    assert time.perf_counter() - t0 < 5  # comparing each prompt with every gold takes over twenty seconds here
