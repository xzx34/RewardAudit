"""Gold self-pass and format-contract inference.

Send each task's gold through K wrappers. The wrappers that earn full score on most tasks
are the contract the checker actually accepts. Every later probe is wrapped with the
primary (best-accepted) wrapper, so what it tests is content, not format.

Besides the fixed wrappers, each audit derives a few from the environment and the prompts: the
parser's answer tag (`tag:masked_output`), a tag the prompt asks the answer to be in, and a
`Label: <answer>` line the prompt names (`prefix:Final`), each with a reasoning-prefixed variant.
A derived id spells out its own wrapper, so `wrapper_fn` resolves it anywhere (report.json too). So does a format the
user gives with --format, whose id is the format itself (`<sol>{answer}</sol>`).
"""

from __future__ import annotations

import itertools
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from rewardaudit.runner import Runner
from rewardaudit.types import Kind, Task, Verdict

Wrapper = Callable[[str], str]

REASONING = (
    "Let me work through this step by step.\n\n"
    "First I identify what is asked, then I carry out the computation carefully.\n\n"
)

WRAPPERS: list[tuple[str, Wrapper]] = [
    ("bare", lambda a: a),
    ("boxed", lambda a: f"\\boxed{{{a}}}"),
    ("boxed_dollar", lambda a: f"$\\boxed{{{a}}}$"),
    ("answer_tag", lambda a: f"<answer>{a}</answer>"),
    ("final_answer", lambda a: f"Final Answer: {a}"),
    ("answer_colon", lambda a: f"Answer: {a}"),
    ("the_final_answer_is", lambda a: f"The final answer is {a}."),
    ("hash", lambda a: f"#### {a}"),
    ("bold", lambda a: f"**{a}**"),
    ("code_block", lambda a: f"```\n{a}\n```"),
    ("dollar", lambda a: f"${a}$"),
    ("json", lambda a: '{"answer": ' + json.dumps(a, ensure_ascii=False) + "}"),
    ("reasoning_boxed", lambda a: f"{REASONING}Therefore the answer is \\boxed{{{a}}}."),
    ("reasoning_hash", lambda a: f"{REASONING}#### {a}"),
    ("reasoning_final", lambda a: f"{REASONING}Final Answer: {a}"),
    ("reasoning_tag", lambda a: f"{REASONING}<answer>{a}</answer>"),
    ("think_boxed", lambda a: f"<think>\n{REASONING.strip()}\n</think>\n\\boxed{{{a}}}"),
    ("think_tag", lambda a: f"<think>\n{REASONING.strip()}\n</think>\n<answer>{a}</answer>"),
    # a box inside the answer tag (checkers that take the tag's content and then the last box in it)
    ("answer_tag_boxed", lambda a: f"<answer>\\boxed{{{a}}}</answer>"),
    ("think_tag_boxed", lambda a: f"<think>\n{REASONING.strip()}\n</think>\n<answer>\\boxed{{{a}}}</answer>"),
    # for prompts that already open the think block (the completion starts inside it)
    ("open_think_tag", lambda a: f"{REASONING.strip()}\n</think>\n<answer>{a}</answer>"),
    ("open_think_boxed", lambda a: f"{REASONING.strip()}\n</think>\n\\boxed{{{a}}}"),
]

_FIXED: dict[str, Wrapper] = dict(WRAPPERS)

# --- wrappers derived from the environment and the prompts --------------------------------------------

_TAG_NAME = re.compile(r"[A-Za-z_][\w.\-]*")
_LABEL = re.compile(r"[A-Za-z][A-Za-z]*(?: [A-Za-z]+){0,2}")


def wrapper_fn(wid: str) -> Wrapper:
    """The wrapper behind an id: a fixed one ('boxed'), a derived one ('tag:masked_output',
    'reasoning_tag:masked_output', 'prefix:Final', 'reasoning_prefix:Final') or one the user gave ('<sol>{answer}</sol>').
    A derived id names its tag or label, so the id alone rebuilds the wrapper. KeyError for an unknown id."""
    fn = _FIXED.get(wid)
    if fn is not None:
        return fn
    if "{answer}" in wid:
        return lambda a: wid.replace("{answer}", a)
    kind, _, name = wid.partition(":")
    reasoning = REASONING if kind.startswith("reasoning_") else ""
    if kind in ("tag", "reasoning_tag") and _TAG_NAME.fullmatch(name):
        return lambda a: f"{reasoning}<{name}>{a}</{name}>"
    if kind in ("prefix", "reasoning_prefix") and _LABEL.fullmatch(name):
        return lambda a: f"{reasoning}{name}: {a}"
    raise KeyError(wid)


_ANSWER_SLOT = "QQANSWERQQ"

MAX_DERIVED = 4  # tags + prefix labels per audit (two wrappers each), so markup-heavy prompts cost little
# tags that hold reasoning, not the answer; and markup that is never an answer field
_REASONING_TAG = re.compile(r"think|thought|reason|rational|analy|scratch|plan|reflect|explain|explanation|"
                            r"ponder|deliberat|work|step|critique|draft|note", re.I)
_NOT_ANSWER_TAGS = frozenset("answer br div span img li ul ol html body head table tr td th sup sub em strong eos bos "
                             "pad unk".split())
_ANSWER_WORD = re.compile(r"(?<![A-Za-z])(answers?|final|solution|response|respond|output|result|prediction|reply)"
                          r"(?![A-Za-z])", re.I)
_PROMPT_TAG = re.compile(r"<([A-Za-z_][\w.\-]*)>")
_CLAUSE_END = re.compile(r"[.;!?](?:\s|$)|\n\s*\n|>")
# 'Final: <answer>', 'ANSWER: $LETTER', 'Choice: [your choice]' at a line start or after a quote
_PREFIX_LINE = re.compile(r"""(?:^|["'`“])[ \t>*\-]*([A-Za-z][A-Za-z]*(?:[ \t][A-Za-z]+){0,2})[ \t]*:[ \t]*"""
                          r"""(\$[A-Z][A-Z_]{2,}(?![\w$])|<[^<>\n]{1,40}>|\[[^\[\]\n]{1,40}\]|\{[^{}\n]{1,40}\})""", re.M)
_PLACEHOLDER_WORD = re.compile(r"answer|letter|choice|option|number|result|value|solution|final|response|output", re.I)
_FIXED_LABELS = frozenset({"Answer", "Final Answer"})  # answer_colon and final_answer write these
_NOT_LABELS = frozenset("input question example examples note hint problem context task format instructions e q".split())


def _scan_text(t: Task, cap: int = 4000) -> str:
    """The system and user text of a task, each message clipped to its first and last cap/2 characters."""
    parts = []
    for m in t.chat():
        content = m.get("content")
        if m.get("role", "user") in ("system", "user") and isinstance(content, str):
            parts.append(content if len(content) <= cap else content[: cap // 2] + "\n\n" + content[-cap // 2:])
    return "\n\n".join(parts) or t.prompt


def _answer_tags(text: str) -> list[str]:
    """Tags the text asks the answer to be in: a tag closed somewhere in the text or followed by 'tags', named
    as an answer field (`masked_output`) or introduced by a clause about the answer ('put your answer in')."""
    out: list[str] = []
    for m in _PROMPT_TAG.finditer(text):
        name = m.group(1)
        if name in out or len(name) < 2 or name.lower() in _NOT_ANSWER_TAGS or _REASONING_TAG.search(name):
            continue
        if f"</{name}>" not in text and not re.match(r"\s*(?:\.\.\.\s*)?tags?\b", text[m.end():m.end() + 16], re.I):
            continue
        before = text[max(0, m.start() - 200):m.start()]
        clause = _CLAUSE_END.split(before)[-1]
        if _ANSWER_WORD.search(name) or _ANSWER_WORD.search(clause):
            out.append(name)
    return out


def _answer_labels(text: str) -> list[str]:
    """Labels of the answer lines the text spells out with a placeholder ('Final: <answer>', 'ANSWER: $LETTER')."""
    out: list[str] = []
    for m in _PREFIX_LINE.finditer(text):
        label, ph = " ".join(m.group(1).split()), m.group(2)
        if label in out or label in _FIXED_LABELS or label.lower() in _NOT_LABELS:
            continue
        if not ph.startswith("$") and not _PLACEHOLDER_WORD.search(ph):
            continue
        if ph.startswith("<") and f"</{ph[1:-1]}>" in text:  # '<answer>42</answer>' is a tag, not a placeholder
            continue
        out.append(label)
    return out


def derive_wrappers(tasks: list[Task], meta: dict[str, Any] | None = None,
                    limit: int = MAX_DERIVED) -> dict[str, str]:
    """Wrapper ids for formats the fixed wrappers do not produce -> where each came from. The parser's answer tag
    first, then tags and answer-line labels named in at least 30% of the prompts (most frequent first); at most
    `limit` tags and labels, each as the plain form and a reasoning-prefixed one."""
    bases: list[tuple[str, str]] = []
    tag = (meta or {}).get("answer_tag")
    if isinstance(tag, str) and _TAG_NAME.fullmatch(tag) and tag.lower() not in _NOT_ANSWER_TAGS \
            and not _REASONING_TAG.search(tag):
        bases.append((f"tag:{tag}", "the parser's answer tag"))
    texts = [_scan_text(t) for t in tasks]
    need = max(1, 0.3 * len(texts))
    for kind, found, source in (("tag", _answer_tags, "a tag the prompts ask the answer to be in"),
                                ("prefix", _answer_labels, "an answer line the prompts spell out")):
        counts: Counter[str] = Counter(name for text in texts for name in found(text))
        for name, k in counts.most_common():  # ties keep the order of first appearance
            if k >= need and all(b != f"{kind}:{name}" for b, _ in bases):
                bases.append((f"{kind}:{name}", source))
    out: dict[str, str] = {}
    for wid, source in bases[:limit]:
        out[wid] = source
        out[f"reasoning_{wid}"] = source
    return out


# Stage 1 of contract inference: the formats most checkers accept. The rest run only when needed;
# the open-think wrappers join stage 1 when the prompts end inside an open <think>.
COMMON_WRAPPERS = ("bare", "boxed", "answer_tag", "final_answer", "answer_colon", "hash", "reasoning_boxed", "think_tag")
OPEN_THINK = ("open_think_tag", "open_think_boxed")
# Variants of one marker; when any member is accepted the whole family is measured (primary choice needs them).
# A box inside the answer tag carries both markers, so it belongs to both families.
# A checker that takes the bare answer may take any light markup too, so those form the bare family.
FAMILIES = (
    ("bare", "bold", "code_block", "dollar", "json"),
    ("boxed", "boxed_dollar", "reasoning_boxed", "think_boxed", "open_think_boxed", "answer_tag_boxed", "think_tag_boxed"),
    ("answer_tag", "reasoning_tag", "think_tag", "open_think_tag", "answer_tag_boxed", "think_tag_boxed"),
    ("final_answer", "the_final_answer_is", "reasoning_final"),
    ("hash", "reasoning_hash"),
)


@dataclass
class Contract:
    full_score: float  # best signal any gold wrapper achieved
    accepted: list[str]  # wrapper ids accepted on most self-passing tasks
    primary: str  # wrapper used for every later probe
    wrapper_rates: dict[str, float]  # id -> fraction of self-passing tasks where it earned full score
    self_pass: dict[str, bool]  # task id -> any wrapper earned full score
    gold_score: dict[str, float]  # task id -> best gold signal observed
    task_wrappers: dict[str, list[str]] = field(default_factory=dict)  # task id -> wrappers that earned full score
    self_fail_reason: dict[str, str] = field(default_factory=dict)  # task id -> "zero" | "error" | "mixed"
    observations: list[tuple[str, str, str, Verdict]] = field(default_factory=list)
    # For composite rubrics (accuracy + format + length ...): the sub-rewards that read the answer (`_find_signal_keys`).
    # When set, acceptance is judged on their sum against that task's gold, so format-only credit never counts as a hit.
    signal_keys: list[str] | None = None
    signal_note: str = ""
    task_full: dict[str, float] = field(default_factory=dict)  # task id -> the signal a gold that passes earns there
    gold_terms: dict[str, dict[str, float]] = field(default_factory=dict)  # task id -> signal key -> the gold's best, if > 0
    gold_total: dict[str, float] = field(default_factory=dict)  # task id -> the gold's total score (see _gold_total)
    wrappers_run: list[str] = field(default_factory=list)  # wrapper ids actually sent; others were not measured
    wrapper_stage: str = "all"  # all | common | common+families
    wrapper_note: str = ""  # why stage 1 expanded to all wrappers
    calls: int = 0  # checker calls the contract made, wrong-answer pass included
    derived: dict[str, str] = field(default_factory=dict)  # derived wrapper id -> where its format came from
    # The wrong-answer pass on passing tasks: tasks where the wrong answer scored, and those where it reached the
    # gold's own score (see `no_discrimination`), with the mean scores of both.
    wrong_checked: int = 0
    wrong_same: int = 0
    wrong_above: int = 0  # of those, tasks where the wrong answer scored strictly above the gold
    gold_mean: float = 0.0
    wrong_mean: float = 0.0
    wrong_signal: dict[str, float] = field(default_factory=dict)  # task id -> the wrong answer's signal, passing or not
    whole_reply_similarity: bool = False  # see _whole_reply_similarity

    @property
    def self_pass_rate(self) -> float:
        return sum(self.self_pass.values()) / max(1, len(self.self_pass))

    @property
    def no_discrimination(self) -> bool:
        """Wrong answers reach the gold's score on most passing tasks. The "full score" is then only what the gold
        happened to earn, so reaching it says nothing about an answer being right."""
        return 2 * self.wrong_same > self.wrong_checked > 0

    @property
    def discrimination_note(self) -> str:
        return (f"reference and wrong answers score the same on {self.wrong_same} of {self.wrong_checked} rows "
                f"(mean {self.gold_mean:.3g} and {self.wrong_mean:.3g})")

    @property
    def format_only(self) -> bool:
        """No discrimination, and the score follows the format: some wrapper leaves most passing golds below their
        score while a wrong answer reaches it. A sub-reward no gold call ever earned rules this out: that term may read
        the answer, and the gold may simply not reach it."""
        if not self.no_discrimination or min(self.wrapper_rates.values(), default=1.0) >= 0.5:
            return False
        return not _unearned_sub_rewards([v for wid, _, _, v in self.observations if wid != "wrong_answer"])

    def rejected_golds(self, eps: float = 1e-9) -> list[str]:
        """Failing tasks where the checker treats the gold like a wrong answer: its best signal over all wrappers is at
        most 0 or at most the wrong answer's. A gold with partial credit above both is the scale, not a rejection; a
        task whose every gold call raised belongs to robustness."""
        def rejected(tid: str) -> bool:
            best, wrong = self.gold_score[tid], self.wrong_signal.get(tid)
            return best <= eps or (wrong is not None and best <= wrong + eps)

        return [t for t, ok in self.self_pass.items() if not ok and self.self_fail_reason.get(t) != "error" and rejected(t)]

    def signal(self, v: Verdict) -> float | None:
        """The correctness signal of a verdict: the discriminating sub-rewards if known, else the score."""
        if not v.ok:
            return None
        if self.signal_keys and v.sub_scores:
            return float(sum(v.sub_scores.get(k, 0.0) for k in self.signal_keys))
        return v.score

    def accepted_verdict(self, v: Verdict, task_id: str | None = None, eps: float = 1e-9) -> bool:
        """It reached the gold on the terms that decide correctness, and was not marked down to half of the gold's total
        or less: the bar `pays_on_one_term` sets, for a reply the deciding terms alone would call right."""
        full = self.gold_score.get(task_id, self.full_score) if task_id is not None else self.full_score
        if not is_full(self.signal(v), full, eps):
            return False
        total = self.gold_total.get(task_id, 0.0) if self.signal_keys and task_id is not None else 0.0
        return total <= 0 or v.score > 0.5 * total

    def pays_on_one_term(self, v: Verdict, task_id: str, eps: float = 1e-9) -> bool:
        """For a reply built without reading the gold, on a rubric with several signal keys: it reached the gold on one
        of them and earned more than half of the gold's total. The sub-scores alone cannot tell such a reply from a gold
        cut in two (an empty reply and a cut-off gold can earn the same values), so this asks only of the first kind: a
        part of the gold has the right to that term's credit, a reply that read nothing has not."""
        gold, total = self.gold_terms.get(task_id, {}), self.gold_total.get(task_id, 0.0)
        if len(gold) < 2 or total <= 0 or not v.ok or not v.sub_scores or v.score <= 0.5 * total:
            return False
        return any(v.sub_scores.get(k, 0.0) >= x - eps for k, x in gold.items())

    def wrapper_for(self, task_id: str | None) -> str:
        """The primary wrapper, unless it failed on this task; then the first accepted one that passed."""
        if task_id is None:
            return self.primary
        ok = self.task_wrappers.get(task_id)
        if not ok or self.primary in ok:
            return self.primary
        for wid in self.accepted:
            if wid in ok:
                return wid
        return ok[0]

    def wrap(self, answer: str, task_id: str | None = None) -> str:
        return wrapper_fn(self.wrapper_for(task_id))(answer)

    def with_answer(self, completion: str, answer: str, task_id: str | None = None) -> str | None:
        """The same reply with its final answer replaced: its reasoning stays, word for word. None when it does not end
        in an answer in the accepted format, as a sweep of numbers or a refusal does: there is no answer to replace."""
        head, _, tail = self.wrap(_ANSWER_SLOT, task_id).partition(_ANSWER_SLOT)
        mark = head[head.rfind("\n") + 1:]  # what opens the answer on its line; nothing for a plain answer on its own line
        body, tail = completion.rstrip(), tail.rstrip()
        if not body.endswith(tail):
            return None
        before = body[:len(body) - len(tail)]  # the closing mark may be the opening one: **x**, $x$
        start = before.rfind(mark) if mark else before.rfind("\n") + 1
        return None if start < 0 else completion[:start] + self.wrap(answer, task_id)[len(head) - len(mark):]

    def passing_tasks(self) -> list[str]:
        return [t for t, ok in self.self_pass.items() if ok]

    def to_dict(self) -> dict:
        reasons = {}
        for r in self.self_fail_reason.values():
            reasons[r] = reasons.get(r, 0) + 1
        return {
            "full_score": self.full_score,
            "accepted": self.accepted,
            "primary": self.primary,
            "wrapper_rates": self.wrapper_rates,
            "self_pass_rate": self.self_pass_rate,
            "self_fail_reasons": reasons,
            "signal_keys": self.signal_keys,
            "signal_note": self.signal_note,
            "wrappers_run": self.wrappers_run,
            "wrapper_stage": self.wrapper_stage,
            "wrapper_note": self.wrapper_note,
            "calls": self.calls,
            "derived_wrappers": self.derived,
            "no_discrimination": self.no_discrimination,
            "format_only": self.format_only,
            "wrong_answer_pass": {"checked": self.wrong_checked, "same_as_gold": self.wrong_same,
                                  "above_gold": self.wrong_above, "gold_mean": self.gold_mean,
                                  "wrong_mean": self.wrong_mean},
            "rejected_golds": self.rejected_golds()[:50],
            "whole_reply_similarity": self.whole_reply_similarity,
        }


def is_full(score: float | None, full: float, eps: float = 1e-9) -> bool:
    return score is not None and full > 0 and score >= full - eps


MARKED = ("boxed", "answer_tag", "hash", "final_answer", "answer_colon", "boxed_dollar", "the_final_answer_is",
          "bold", "json", "code_block", "dollar")
PUNCTUATION_ONLY = frozenset(("bare", "hash", "bold", "code_block", "dollar"))  # what a checker that strips punctuation cannot tell apart


def ends_open_think(tasks: list[Task]) -> bool:
    """Most prompts end inside an open <think> block, so the completion is expected to start there."""
    def last_text(t: Task) -> str:
        content = t.chat()[-1].get("content") if t.chat() else None
        return content if isinstance(content, str) else t.prompt

    return bool(tasks) and sum(last_text(t).rstrip().endswith("<think>") for t in tasks) > len(tasks) / 2


def _choose_primary(accepted: list[str], rates: dict[str, float], best_rate: float, tasks: list[Task],
                    derived: Iterable[str] = ()) -> str:
    """The wrapper probes will use. A policy under training writes reasoning and then a marked answer, so
    prefer the format the prompt asks for (a derived format first: the parser's answer tag, a tag or answer
    line the prompt names), then any marked format whose reasoning-prefixed variant is also accepted, then the
    plain answer. Prompts that open <think> themselves get the open-think wrappers. A format the user gave comes first."""
    if not accepted:
        return "bare"
    tied = [w for w in accepted if rates[w] >= best_rate - 1e-9]
    for w in derived:
        if "{answer}" in w and w in tied:
            return w
    prompts = " ".join(t.prompt[-600:] for t in tasks[:20]).lower()
    if ends_open_think(tasks):
        order = OPEN_THINK[::-1] if ("\\boxed" in prompts or "boxed{" in prompts) else OPEN_THINK
        for w in order:
            if w in tied:
                return w
    asked = list(derived)
    if "\\boxed" in prompts or "boxed{" in prompts:
        asked += ["boxed", "boxed_dollar"]
    if "<answer>" in prompts:
        asked += ["answer_tag"]
    if "####" in prompts:
        asked += ["hash"]
    if "final answer" in prompts:
        asked += ["final_answer", "the_final_answer_is"]
    if "answer:" in prompts:
        asked += ["answer_colon"]
    for w in asked:
        if w in tied:
            return w
    for w in MARKED:
        if w in tied and (f"reasoning_{w}" in accepted or w in ("boxed_dollar", "the_final_answer_is")):
            return w
    if "bare" in tied:
        return "bare"
    for w in MARKED:
        if w in tied:
            return w
    for w in accepted:
        if rates[w] >= best_rate - 1e-9:
            return w
    return accepted[0]


def _score_all(runner: Runner, items: list[tuple[Task, str]], progress: Callable[[int, int], None] | None) -> list[Verdict]:
    score_many = getattr(runner, "score_many", None)
    if score_many is not None:
        return score_many(items, progress=progress)
    out = []
    for i, (t, comp) in enumerate(items):
        out.append(runner.score(t, comp))
        if progress:
            progress(i + 1, len(items))
    return out


WRONG_SCAN = 100  # tasks to look through for another answer; a dataset with one answer throughout would be quadratic


def _wrong_answer(task: Task, tasks: list[Task]) -> str:
    """Another task's gold if it differs, else a definite error in this gold (a number off by one, another option
    letter, the opposite yes or no): a checker that reads the last number would not notice an appended letter."""
    from rewardaudit.equiv import equiv
    from rewardaudit.goldtype import infer_gold_type
    from rewardaudit.probes.base import ProbeContext

    for other in itertools.islice(tasks, WRONG_SCAN):
        if other.id != task.id and other.answer.strip() and not equiv(other.answer, task.answer):
            return other.answer
    gold_type = infer_gold_type(task.answer)
    ctx = ProbeContext(contract=None, gold_type=gold_type, tasks=tasks, gold_types={task.id: gold_type})  # type: ignore[arg-type]
    return ctx.wrong_answer(task, random.Random(task.id)) or task.answer + "x"


MIN_SERVED = 0.1  # of the tasks: fewer are a chance match of the gold's text, or too few for a term to matter
MIN_NET_SEPARATION = 0.25  # format terms sit at 0; correctness terms start at the chance that two golds differ on them


def _find_signal_keys(gold: list[Verdict], wrong: list[Verdict]) -> tuple[list[str] | None, str]:
    """The sub-rewards that read the answer, from the pairs of a gold and a wrong answer in one format (parallel lists).
    Judged on the tasks where the gold earns the term (at least a tenth of them), so a term that serves one kind of answer
    counts as much as one that serves them all. There the gold must beat the wrong answer, net of the reverse, on at least
    a quarter: a format term pays both alike, and a hedge may lose it while the answer still passes, so it stays out. And
    most of those golds must reach the same maximum: a continuous term (a style classifier, a length bonus) has no such
    ceiling. Zero-weight metrics are not part of the reward and are never chosen."""
    if len({k for v in gold + wrong if v.ok and v.sub_scores for k in v.sub_scores}) < 2:
        return None, ""
    pairs = [(g.sub_scores, w.sub_scores) for g, w in zip(gold, wrong, strict=True)
             if g.ok and w.ok and g.sub_scores and w.sub_scores]
    chosen: list[str] = []
    notes: list[str] = []
    left_out: list[str] = []
    for k in sorted({k for g, w in pairs for k in (*g, *w) if not k.startswith("metric:")}):
        vals = [(g[k], w[k]) for g, w in pairs if k in g and k in w and g[k] == g[k] and w[k] == w[k]]
        if not vals:
            continue
        bottom = min(x for pair in vals for x in pair)
        served = [(a, b) for a, b in vals if a > bottom]
        net = sum((a > b) - (a < b) for a, b in vals)
        if len(served) < max(1.0, MIN_SERVED * len(pairs)) or net < MIN_NET_SEPARATION * len(served):
            continue
        gm, wm = sum(a for a, _ in served) / len(served), sum(b for _, b in served) / len(served)
        where = "" if len(served) == len(pairs) else f" on the {len(served)} of {len(pairs)} rows it serves"
        top = max(a for a, _ in served)
        if 2 * sum(a >= top - 1e-6 * max(1.0, abs(top)) for a, _ in served) <= len(served):
            left_out.append(f"{k} left out: reference {gm:.2f} vs wrong {wm:.2f}{where}, but the reference answers share no maximum")
            continue
        chosen.append(k)
        notes.append(f"{k}: reference {gm:.2f} vs wrong {wm:.2f}{where}")
    if not chosen:
        return None, "; ".join(["no part of the score separates reference from wrong answers; using the total score"] + left_out)
    return chosen, "; ".join(notes + left_out)


def _task_full(keys: list[str], terms: dict[str, dict[str, float]]) -> dict[str, float]:
    """What the signal of a gold that passes comes to on each task, from what the golds earned (task id -> key -> best value
    above 0): the common maximum of every term it earns or should earn. It should earn a term when at least two thirds of
    the golds that earn all this one earns also earn that term: then this gold was marked down. A smaller share is a term
    that serves other kinds of task, and it asks nothing here: some rubrics give one kind of answer a term of its own, or
    leave a third of the tasks without a field a term reads."""
    top = {k: max((t[k] for t in terms.values() if k in t), default=0.0) for k in keys}
    out: dict[str, float] = {}
    for tid, own in terms.items():
        peers = [t for t in terms.values() if own.keys() <= t.keys()]
        out[tid] = sum(top[k] for k in keys if k in own or 3 * sum(k in t for t in peers) >= 2 * len(peers))
    return out


def _gold_total(c: Contract, by_wrapper: dict[str, dict[str, Verdict]], tid: str) -> float:
    """The gold's total score on a task, for the reader to compare a reply with: in the format probes use there when it
    passes, since a reply in that format is what it stands next to; else its best in any format."""
    v = by_wrapper.get(c.wrapper_for(tid), {}).get(tid)
    if c.self_pass.get(tid) and v is not None and v.ok:
        return float(v.score)
    return max((float(d[tid].score) for d in by_wrapper.values() if tid in d and d[tid].ok), default=0.0)


def _gold_wrapper(c: Contract, by_wrapper: dict[str, dict[str, Verdict]], t: Task) -> str | None:
    """The wrapper of a task's best gold call: the one probes use when the gold passes, else the one with the
    highest score (the primary, then accepted wrappers, on ties). None when no gold call scored, and on a checker
    without sub-rewards when the gold earned nothing: a wrong answer could neither name a signal nor change that the
    gold was rejected."""
    if c.self_pass.get(t.id):
        return c.wrapper_for(t.id)
    rank: dict[str, int] = {}
    for w in [c.primary, *c.accepted, *c.wrappers_run]:
        rank.setdefault(w, len(rank))
    scored = [(v.score, -rank[w], w) for w in c.wrappers_run
              if (v := by_wrapper[w][t.id]).ok and v.score == v.score and (v.sub_scores or v.score > 0)]
    return max(scored)[2] if scored else None


def _compare_with_wrong(c: Contract, wrong: dict[str, Verdict], eps: float = 1e-9) -> None:
    """Record how the wrong answers score against the gold: their signal on every task, the rest on the passing ones."""
    c.wrong_signal = {tid: s for tid, v in wrong.items() if (s := c.signal(v)) is not None and s == s}
    scored = [(tid, s) for tid, s in c.wrong_signal.items() if c.self_pass.get(tid)]
    c.wrong_checked = len(scored)
    c.wrong_same = sum(is_full(s, c.gold_score[tid]) for tid, s in scored)
    c.wrong_above = sum(s > c.gold_score[tid] + eps for tid, s in scored)
    if scored:
        c.gold_mean = sum(c.gold_score[tid] for tid, _ in scored) / len(scored)
        c.wrong_mean = sum(s for _, s in scored) / len(scored)


def _whole_reply_similarity(c: Contract, by_wrapper: dict[str, dict[str, Verdict]], eps: float = 1e-9) -> bool:
    """The checker grades the whole reply by closeness to the gold (edit distance, token overlap) and locates no
    answer in it: only the bare gold earns full score, punctuation around it aside, and on most passing tasks the same gold
    in other formats earns varying parts of it. Any text around the answer lowers such a score by definition, so a
    rewrite of the reply measures the metric, not a right answer being rejected."""
    if "bare" not in c.accepted or not PUNCTUATION_ONLY.issuperset(c.accepted):
        return False
    others = [w for w in c.wrappers_run if w not in c.accepted]

    def graded(tid: str) -> bool:
        vals = [c.signal(by_wrapper[w][tid]) for w in others]
        part = [s for s in vals if s is not None and eps < s < c.gold_score[tid] - eps]
        return 2 * len(part) > len(vals) and len({round(s, 6) for s in part}) >= 3

    passing = c.passing_tasks()
    return 2 * sum(graded(tid) for tid in passing) > len(passing) > 0


def _families_of(accepted: list[str]) -> set[str]:
    return {w for fam in FAMILIES if any(a in fam for a in accepted) for w in fam}


def _sub_rewards(verdicts: list[Verdict]) -> list[str]:
    """The terms of the reward these calls report; a rubric with fewer than two has a total and no signal to look for."""
    return sorted({k for v in verdicts if v.ok and v.sub_scores for k in v.sub_scores if not k.startswith("metric:")})


def _unearned_sub_rewards(verdicts: list[Verdict]) -> list[str]:
    """Sub-rewards of a composite rubric that none of these gold calls earned."""
    keys = _sub_rewards(verdicts)
    if len(keys) < 2:
        return []
    subs = [v.sub_scores for v in verdicts if v.ok and v.sub_scores]
    return [k for k in keys if max((s[k] for s in subs if k in s and s[k] == s[k]), default=0.0) <= 0]


def _stage_one_gap(c: Contract, tasks: list[Task], verdicts: list[Verdict], declared: float | None) -> str:
    """Why the common wrappers do not settle the contract ('' when they do), judged on c's signal. A declared full
    score is a total, so it says nothing once the signal is a part of the total."""
    stuck = sum(1 for v in verdicts if v.kind in (Kind.TIMEOUT, Kind.CRASH))
    if verdicts and stuck >= 0.9 * len(verdicts):
        return ""  # the remaining wrappers would only time out as well
    best = max((sum(w in c.task_wrappers.get(t.id, ()) for t in tasks) for w in c.wrappers_run), default=0)
    if best < 0.5 * len(tasks):
        return "no common wrapper earned full score on most tasks"
    if c.signal_keys is None and declared is not None and declared > 0 and c.full_score < declared - 1e-9:
        return f"declared full score {declared:g} not reached"
    never = _unearned_sub_rewards(verdicts)
    return f"sub-reward {never[0]} earned by no common wrapper" if never else ""


def infer_contract(
    runner: Runner,
    tasks: list[Task],
    *,
    wrappers: list[tuple[str, Wrapper]] | None = None,
    formats: Iterable[str] = (),
    progress: Callable[[int, int], None] | None = None,
    two_stage: bool = True,
) -> Contract:
    """Two stages unless `wrappers` is given or `two_stage` is off: the 8 common wrappers, the `formats` the user gave and
    the derived ones (`derive_wrappers`: from the runner's meta and the prompts) first, plus the families of the accepted
    ones; every wrapper only when those settle nothing. On a composite rubric the wrong answers and the signal keys come
    before that decision, since a continuous term in the total leaves most golds below its maximum in any format."""
    derived = {} if wrappers else {**dict.fromkeys(formats, "the format given"),
                                   **derive_wrappers(tasks, getattr(runner, "meta", None))}
    pool = wrappers or WRAPPERS + [(wid, wrapper_fn(wid)) for wid in derived]
    order = [wid for wid, _ in pool]
    fns = dict(pool)
    obs: list[tuple[str, str, str, Verdict]] = []
    by_wrapper: dict[str, dict[str, Verdict]] = {}
    n_calls = 0

    def run(ids: list[str]) -> None:
        nonlocal n_calls
        ids = [w for w in order if w in ids and w not in by_wrapper]
        items = [(wid, t, fns[wid](t.answer)) for wid in ids for t in tasks]
        verdicts = _score_all(runner, [(t, comp) for _, t, comp in items], progress)
        n_calls += len(items)
        for wid in ids:
            by_wrapper[wid] = {}
        for (wid, t, comp), v in zip(items, verdicts, strict=True):
            by_wrapper[wid][t.id] = v
            obs.append((wid, t.id, comp, v))

    def build(signal_keys: list[str] | None, note: str) -> Contract:
        ran = [w for w in order if w in by_wrapper]
        c = Contract(full_score=0.0, accepted=[], primary="bare", wrapper_rates={}, self_pass={}, gold_score={},
                     observations=obs, signal_keys=signal_keys, signal_note=note, wrappers_run=ran, derived=derived)
        scores: dict[str, dict[str, float | None]] = {
            wid: {tid: c.signal(v) for tid, v in by_wrapper[wid].items()} for wid in ran
        }
        observed = [s for d in scores.values() for s in d.values() if s is not None]
        full = max(observed) if observed else 0.0
        declared = runner.max_score
        if signal_keys is None and declared is not None and declared > 0 and any(is_full(s, declared) for s in observed):
            full = float(declared)
        c.full_score = full
        if signal_keys:
            for t in tasks:
                calls = [v.sub_scores for w in ran if (v := by_wrapper[w][t.id]).ok and v.sub_scores]
                best_of = {k: max((s[k] for s in calls if k in s and s[k] == s[k]), default=0.0) for k in signal_keys}
                c.gold_terms[t.id] = {k: x for k, x in best_of.items() if x > 0}
            c.task_full = _task_full(signal_keys, c.gold_terms)
        else:
            c.task_full = {t.id: full for t in tasks}
        for t in tasks:
            vals = [scores[w][t.id] for w in ran if scores[w][t.id] is not None]
            best = max(vals) if vals else 0.0
            c.gold_score[t.id] = best
            c.self_pass[t.id] = is_full(best, c.task_full[t.id])
            c.task_wrappers[t.id] = [w for w in ran if is_full(scores[w][t.id], c.task_full[t.id])]
            if not c.self_pass[t.id]:
                n_err = sum(1 for w in ran if scores[w][t.id] is None)
                c.self_fail_reason[t.id] = "error" if n_err == len(ran) else ("zero" if n_err == 0 else "mixed")
        passing = [t.id for t in tasks if c.self_pass[t.id]]
        for wid in ran:
            c.wrapper_rates[wid] = (sum(is_full(scores[wid][tid], c.task_full[tid]) for tid in passing) / len(passing)) if passing else 0.0
        best_rate = max(c.wrapper_rates.values()) if c.wrapper_rates else 0.0
        c.accepted = [wid for wid in ran if best_rate > 0 and c.wrapper_rates[wid] >= max(0.5, 0.8 * best_rate)]
        c.primary = _choose_primary(c.accepted, c.wrapper_rates, best_rate, tasks, derived)
        c.gold_total = {t.id: _gold_total(c, by_wrapper, t.id) for t in tasks}
        return c

    def all_verdicts() -> list[Verdict]:
        return [v for d in by_wrapper.values() for v in d.values()]

    def add_families(c: Contract) -> bool:
        missing = [w for w in order if w in _families_of(c.accepted) and w not in by_wrapper]
        run(missing)
        return bool(missing)

    wrong: dict[str, tuple[str, Verdict]] = {}  # task id -> the wrapper its wrong answer went in, and the verdict
    keys: list[str] | None = None
    note = ""

    def settle() -> Contract:
        """The contract on the wrappers run so far. Every task whose gold call scored gets a wrong answer (not only the
        self-passing ones: a continuous term in the total can leave almost no gold at the maximum before the signal is
        known). One already sent stays valid while its wrapper still earns that gold's best signal: the pair then
        compares the two in one format at the gold's score. Only the other tasks are sent again."""
        nonlocal keys, note, n_calls
        base = build(None, "")
        known = build(keys, note) if keys else base

        def valid(tid: str) -> bool:
            s = known.signal(by_wrapper[wrong[tid][0]][tid])
            return s is not None and s >= known.gold_score[tid] - 1e-9

        todo = [(t, w) for t in tasks if (w := _gold_wrapper(base, by_wrapper, t)) is not None
                and (t.id not in wrong or (w != wrong[t.id][0] and not valid(t.id)))]
        items = [(t, w, fns[w](_wrong_answer(t, tasks))) for t, w in todo]
        verdicts = _score_all(runner, [(t, comp) for t, _, comp in items], None)
        n_calls += len(items)
        for (t, w, comp), v in zip(items, verdicts, strict=True):
            obs.append(("wrong_answer", t.id, comp, v))
            wrong[t.id] = (w, v)
        pairs = [(t.id, by_wrapper[wrong[t.id][0]][t.id], wrong[t.id][1]) for t in tasks if t.id in wrong]
        keys, note = _find_signal_keys([g for _, g, _ in pairs], [w for _, _, w in pairs])
        if keys:
            return build(keys, note)
        base.signal_note = note
        return base

    staged = two_stage and wrappers is None
    stage, gap = "all", ""
    judged: list[str] | None = None
    if staged:
        run(list(COMMON_WRAPPERS) + (list(OPEN_THINK) if ends_open_think(tasks) else []) + list(derived))
        first = build(None, "")
        # On a composite rubric the signal decides, so the wrong answers come first; every other checker pays for them
        # once, after the wrappers. A sub-reward no common wrapper earned expands whatever the signal says.
        if len(_sub_rewards(all_verdicts())) >= 2 and not _unearned_sub_rewards(all_verdicts()):
            first = settle()
        judged = first.signal_keys
        gap = _stage_one_gap(first, tasks, all_verdicts(), runner.max_score)
        if gap:
            run(order)
        else:
            stage = "common+families" if add_families(first) else "common"
    else:
        run(order)

    contract = settle()
    if staged and stage != "all" and contract.signal_keys != judged and add_families(contract):
        stage = "common+families"  # the signal changed with the wrappers, and it may accept other families
        contract = settle()
    _compare_with_wrong(contract, {tid: v for tid, (_, v) in wrong.items()})
    # "rejected in every format" needs every format, on the few golds it concerns
    rest = [w for w in order if w not in by_wrapper]
    rejected = set(contract.rejected_golds())
    items = [(w, t, fns[w](t.answer)) for t in tasks if t.id in rejected for w in rest]
    verdicts = _score_all(runner, [(t, comp) for _, t, comp in items], None)
    n_calls += len(items)
    for (w, t, comp), v in zip(items, verdicts, strict=True):
        obs.append((w, t.id, comp, v))
        s = contract.signal(v)
        if s is not None and s == s and s > contract.gold_score[t.id]:
            contract.gold_score[t.id] = s
            contract.gold_total[t.id] = float(v.score)
            if is_full(s, contract.task_full[t.id]):
                contract.self_pass[t.id] = True
                contract.self_fail_reason.pop(t.id, None)
                contract.task_wrappers[t.id].append(w)
    contract.whole_reply_similarity = _whole_reply_similarity(contract, by_wrapper)
    contract.wrapper_stage = stage
    contract.wrapper_note = gap
    contract.calls = n_calls
    return contract
