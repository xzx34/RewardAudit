"""Regression: the three planted defects are found, and nothing else is reported as high severity."""

import pytest

from rewardaudit.pipeline import AuditConfig, run_audit


@pytest.fixture(scope="module")
def results():
    cfg = AuditConfig(n_tasks=16, seed=1)
    return {name: run_audit({"kind": "demo", "name": name}, cfg) for name in ("substring", "mcq_regex", "grid_exact")}


def _findings(result, severity=("high",)):
    return {f["probe_id"]: f for f in result.metrics["findings"] if f["severity"] in severity}


def test_substring_accepts_shotgun(results):
    r = results["substring"]
    heads = {h["key"]: h["value"] for h in r.metrics["head"]}
    assert heads["non_solving_accepted"] >= 0.9
    assert heads["hedging_accepted"] >= 0.8  # below 1: a range such as 'between 74 and 76' does not hold the answer
    assert "E1" in _findings(r)
    assert heads["right_rejected"] == 0.0


def test_mcq_regex_reads_first_letter(results):
    r = results["mcq_regex"]
    high = _findings(r)
    assert "O3" in high or "O7" in high  # first-letter extraction accepts retractions / hedges
    heads = {h["key"]: h["value"] for h in r.metrics["head"]}
    assert heads["wrong_accepted"] == 0.0
    assert "E1" not in high


def test_grid_exact_rejects_prose(results):
    r = results["grid_exact"]
    severities = {f["probe_id"]: f["severity"] for f in r.findings}
    assert r.metrics["compares_whole_reply"] and severities["Q5"] == severities["Q22"] == "low" and severities["Q3"] == "medium"
    heads = {h["key"]: h["value"] for h in r.metrics["head"]}
    assert heads["non_solving_accepted"] == 0.0
    assert heads["wrong_accepted"] == 0.0


def test_contract_and_self_pass(results):
    for r in results.values():
        assert r.contract.self_pass_rate == 1.0
        assert r.determinism["deterministic"]
