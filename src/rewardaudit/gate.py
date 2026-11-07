"""Whether a checker can be audited at all. Probes assume one reply, a deterministic score and a reference answer that
is the right answer to the task; where one of these fails, numbers would mislead, so the audit stops and says why."""

from __future__ import annotations

import json
import re
import statistics
from typing import Any

from rewardaudit.errors import NotAuditable, short_error
from rewardaudit.types import Kind, Task

MULTI_TURN = "rewardaudit audits checkers that score one reply; multi-turn, tool and container environments are out of its scope"
LLM_JUDGE = "rewardaudit audits deterministic checkers; a checker that asks a model or a remote service is out of its scope"
NO_REFERENCE = ("rewardaudit rewrites the right answer and compares scores, so it needs a column holding that answer. "
                "If another column holds it, name it with --answer-col")
NOT_PASSING = ("Check that the answer column holds the final answer, and that the checker gets it in the form it expects. "
               "Pass the reply format with --format, or wrap the checker in a small function (see docs/guide.md)")
ACCEPTED_ANSWERS = ("The answer column holds lists. If each list holds the accepted answers, write replies from one and give "
                    "the checker the whole list: --answer-col COLUMN.0 --reference-col COLUMN")
TIMING_OUT = "Give the checker more time with --timeout"
FAILED = "Fewer --workers may help if the machine ran out of memory"
# 0.1.14 refuses a plain-string `task` column (0.1.13 was never released), and from 0.1.11 reward functions get message
# objects, not dicts.
OLD_VERIFIERS = ("This environment may have been written for an older verifiers: rerun with --verifiers '<0.1.11' "
                 "or --verifiers '<0.1.13'")

PLACEHOLDER_GOLDS = {"", "none", "null", "n/a", "na", "{}", "[]", "-", "?"}
SPEC_KEYS = {"func_name", "instruction_id_list", "instruction_id", "constraint", "constraints", "constraint_type",
             "test_cases", "tests", "rubric", "rubrics", "criteria", "checklist"}
GOLD_FIELDS = ("answer", "target", "solution", "ground_truth", "expected", "label", "final_answer", "reference", "answer_key", "gold")
BASE_RUBRICS = {"Rubric", "RubricGroup", "MultiTurnMonitorRubric"}
# Golds that name the row instead of answering it. An identifier can also be the right answer to a question about the
# prompt, so these count only where the checker does not separate the gold from a wrong answer (`check_contract`).
IDENTIFIER_GOLD = re.compile(r"[A-Za-z][\w-]*[_:][\w:-]*[A-Za-z0-9]")  # two_sum, words:vowel; not T-cell, Paris or 12:30
ROW_ID_GOLD = re.compile(r"[A-Za-z][\w:-]*[\d_:-][\w:-]*")  # plate-01; not B, Paris or 42
PRINTED_TOKEN = re.compile(r"[\w:-]+")  # an identifier counts as printed when it stands alone, not inside a longer one
MIN_PASSING = 5
MIN_GOLD_SHARE = 0.25
# A rollout that never reached a score: the adapter's RolloutError, or a typed error of verifiers.v1 (v1/errors.py).
UNRUN_ERROR = re.compile(r"^(?:[\w.]*\.)?RolloutError\b|^verifiers\.v1\.[\w.]*Error\b", re.M)


def classify_meta(meta: dict[str, Any]) -> tuple[str, str, str] | None:
    """(class, reason, advice) from what a verifiers adapter reports after loading."""
    if meta.get("single_turn") is False:
        return "multi_turn_or_tools", f"The environment class {meta.get('env_class')} is multi-turn", MULTI_TURN
    if meta.get("tools"):
        return "multi_turn_or_tools", f"The environment exposes {meta['tools']} tools", MULTI_TURN
    if meta.get("single_turn") is None and isinstance(meta.get("max_turns"), int) and meta["max_turns"] > 1:
        return "multi_turn_or_tools", f"The environment allows {meta['max_turns']} turns", MULTI_TURN
    if meta.get("needs_container") or meta.get("image"):
        return "multi_turn_or_tools", "The tasks run in a container" + (f" ({meta['image']})" if meta.get("image") else ""), MULTI_TURN
    if meta.get("toolsets"):
        return "multi_turn_or_tools", "The tasks use toolsets: " + ", ".join(meta["toolsets"][:3]), MULTI_TURN
    if meta.get("env_classes") and "framework" in meta and "v0" not in str(meta["framework"]):
        return "multi_turn_or_tools", "The taskset defines its own environment: " + ", ".join(meta["env_classes"][:3]), MULTI_TURN
    if meta.get("judges"):
        return "llm_judge", "The rubric uses judges: " + ", ".join(meta["judges"][:3]), LLM_JUDGE
    if _no_weighted_reward(meta):
        names = ", ".join(meta.get("reward_funcs") or []) or "none"
        return "gold_never_passes", f"The rubric has no reward function with a nonzero weight (functions: {names})", ""
    return None


def _no_weighted_reward(meta: dict[str, Any]) -> bool:
    """A v0 rubric of the stock classes whose functions all weigh 0 (metrics only): every score is 0,
    and recent verifiers refuse to score it at all. A custom rubric class may compute its reward itself."""
    if "v0" not in str(meta.get("framework", "")) or "reward_funcs" not in meta or not meta.get("rubric_classes"):
        return False
    if not set(meta["rubric_classes"]) <= BASE_RUBRICS:
        return False
    weights = meta.get("reward_weights") or []
    return len(weights) == len(meta["reward_funcs"]) and not any(w for w in weights)


def _json_keys(s: str) -> set[str]:
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return set()
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    return set(obj) if isinstance(obj, dict) else set()


def no_reference_reason(tasks: list[Task]) -> str | None:
    """Golds that are not answers: mostly empty or placeholders, or constraint / test specifications."""
    if not tasks:
        return "The dataset has no rows"
    n = len(tasks)
    empty = sum(1 for t in tasks if t.answer.strip().rstrip(".").lower() in PLACEHOLDER_GOLDS)
    if empty >= 0.5 * n:
        return f"{empty} of {n} reference answers are empty or placeholders"
    ids = sum(1 for t in tasks if t.answer.strip() == str(t.id))
    if ids >= 0.5 * n:
        return f"{ids} of {n} reference answers only repeat the row's id"
    spec = [_json_keys(t.answer.strip()) & SPEC_KEYS for t in tasks if t.answer.strip()[:1] in "{["]
    hits = [k for k in spec if k]
    if len(hits) >= 0.5 * n:
        keys = sorted(set().union(*hits))
        return f"{len(hits)} of {n} reference answers are specifications, not answers (keys: {', '.join(keys[:4])})"
    return None


def _holds_lists(tasks: list[Task]) -> bool:
    """Most reference answers are JSON lists of plain values: the column may hold accepted answers, not one answer."""
    def is_list(answer: str) -> bool:
        try:
            value = json.loads(answer)
        except ValueError:
            return False
        return isinstance(value, list) and bool(value) and not any(isinstance(v, (dict, list)) for v in value)

    return 2 * sum(is_list(t.answer) for t in tasks) > len(tasks)


def _row_id_values(info: dict[str, Any]) -> set[str]:
    inner = info.get("info")
    fields = {**info, **inner} if isinstance(inner, dict) else info
    return {v for k, v in fields.items() if (k == "id" or k.endswith("_id")) and isinstance(v, str)}


def _empty_answer_field(gold: str) -> bool:
    try:
        obj = json.loads(gold)
    except (ValueError, TypeError):
        return False
    fields = [obj[k] for k in GOLD_FIELDS if k in obj] if isinstance(obj, dict) else []
    return bool(fields) and all(v in ("", None, [], {}) for v in fields)


def label_gold_reason(tasks: list[Task]) -> str | None:
    """Golds that are labels of the row: an identifier its prompt prints (with no other row's identifier beside it, which
    would make the prompt a list of choices), an id field of the row, or JSON whose answer field is empty."""
    n = len(tasks)
    golds = [t.answer.strip() for t in tasks]
    identifiers = {g for g in golds if IDENTIFIER_GOLD.fullmatch(g)}
    printed = [g for g, t in zip(golds, tasks, strict=True)
               if g in identifiers and set(PRINTED_TOKEN.findall(t.prompt)) & identifiers == {g}]
    if printed and len(printed) >= 0.5 * n:
        return f"{len(printed)} of {n} reference answers are identifiers the prompt prints, not answers (such as {printed[0]})"
    row_ids = [g for g, t in zip(golds, tasks, strict=True) if ROW_ID_GOLD.fullmatch(g) and g in _row_id_values(t.info)]
    if row_ids and len(row_ids) >= 0.5 * n:
        return f"{len(row_ids)} of {n} reference answers only repeat an id field of the row (such as {row_ids[0]})"
    empty = sum(1 for g in golds if _empty_answer_field(g))
    if empty and empty >= 0.5 * n:
        return f"{empty} of {n} reference answers are JSON with an empty answer field"
    return None


def low_gold_share_reason(contract: Any, meta: dict[str, Any]) -> str | None:
    """Passing golds that earn only a small part of what the rubric pays: a label, on a rubric that pays for a written
    report. Reaching such a gold is a mention's credit, not an accepted answer. What the rubric pays is the sum of its
    positive weights, taking each term to reach 1; terms that never apply together overstate it, hence the low bar."""
    weights = [w for w in meta.get("reward_weights") or [] if w > 0]
    totals = [contract.gold_total[t] for t in contract.passing_tasks() if t in contract.gold_total]
    if not weights or not totals:
        return None
    share = statistics.median(totals) / sum(weights)
    if share >= MIN_GOLD_SHARE:
        return None
    return f"Reference answers earn {share:.0%} of what the rubric pays, so they are labels, not full answers"


def classify_contract(contract: Any, tasks: list[Task], min_passing: int = MIN_PASSING) -> tuple[str, str, str] | None:
    """(class, reason, advice) after the gold calls: network use, timeouts, a rollout that cannot run, golds that do not pass."""
    gold = [v for wid, _, _, v in contract.observations if wid != "wrong_answer"]
    if not gold:
        return "gold_never_passes", "No reference answer could be scored", NOT_PASSING
    net = sum(1 for v in gold if v.network)
    if net >= 0.2 * len(gold):
        return "llm_judge", f"{net} of {len(gold)} calls with the reference answer tried to use the network", LLM_JUDGE
    slow = sum(1 for v in gold if v.kind == Kind.TIMEOUT)
    if slow >= 0.5 * len(gold):
        return "timeout", f"{slow} of {len(gold)} calls with the reference answer timed out", TIMING_OUT
    crashed = [v for v in gold if v.kind == Kind.CRASH]
    if len(crashed) >= 0.5 * len(gold):
        return "audit_failed", f"{len(crashed)} of {len(gold)} calls with the reference answer crashed: {(crashed[0].error or '')[:160]}", FAILED
    # no gold was ever scored, and a real run on this verifiers version stops at the same place
    unrun = [v for v in gold if v.kind == Kind.ERROR and UNRUN_ERROR.search(v.error or "")]
    if len(unrun) >= 0.5 * len(gold):
        cause = re.sub(r"(?:^|(?<=: ))[\w.]*(?:RolloutError: |\.(?=\w+Error\b))", "", short_error(unrun[0].error or ""), count=1)
        return "load_failed", f"{len(unrun)} of {len(gold)} rollouts with the reference answer failed on this verifiers: {cause}", OLD_VERIFIERS
    k = len(contract.passing_tasks())
    if k == 0:
        errors = [v for v in gold if v.kind == Kind.ERROR]
        if errors and len(errors) >= 0.5 * len(gold):
            return "gold_never_passes", (f"The checker raised an error on {len(errors)} of {len(gold)} calls with the reference "
                                         f"answer: {short_error(errors[0].error or '')}"), NOT_PASSING
        return "gold_never_passes", f"No reference answer earned full score in any of the {len(contract.wrappers_run)} formats tried", NOT_PASSING
    if k < min(min_passing, len(tasks)):
        same = "; reference and wrong answers score the same on those" if contract.no_discrimination else ""
        return "gold_never_passes", f"Only {k} of {len(tasks)} reference answers earned full score{same}", NOT_PASSING
    return None


def check_loaded(meta: dict[str, Any], tasks: list[Task]) -> None:
    """Before any checker call: raises NotAuditable when the checker's shape or the dataset rules the audit out."""
    verdict = classify_meta(meta)
    if verdict:
        raise NotAuditable(*verdict)
    reason = no_reference_reason(tasks)
    if reason:
        raise NotAuditable("no_reference", reason, NO_REFERENCE)


def check_contract(contract: Any, tasks: list[Task], meta: dict[str, Any]) -> None:
    """After the calls with the reference answer: raises NotAuditable when they show the probes would measure nothing."""
    verdict = classify_contract(contract, tasks)
    if (verdict and verdict[0] == "gold_never_passes") or (not verdict and contract.no_discrimination):
        reason = label_gold_reason(tasks)
        if reason:
            raise NotAuditable("no_reference", reason, NO_REFERENCE)
    if verdict:
        cls, reason, advice = verdict
        if cls == "gold_never_passes" and meta.get("framework"):
            advice = f"{advice}. {OLD_VERIFIERS}"
        if cls == "gold_never_passes" and _holds_lists(tasks):
            advice = f"{ACCEPTED_ANSWERS}. {advice}"
        raise NotAuditable(cls, reason, advice)
    if not contract.no_discrimination:
        reason = low_gold_share_reason(contract, meta)
        if reason:
            raise NotAuditable("no_reference", reason, NO_REFERENCE)
