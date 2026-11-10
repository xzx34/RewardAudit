"""The `probes` command: every probe listed by family, or one explained, with examples on built-in sample tasks."""

from __future__ import annotations

import random
from typing import Any

from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from rewardaudit.contract import Contract
from rewardaudit.defects import DEFECTS
from rewardaudit.goldtype import infer_gold_type
from rewardaudit.metrics import REJECTED_GOLD_ID, REJECTED_GOLD_NAME, probe_category
from rewardaudit.probes.base import Probe, ProbeContext
from rewardaudit.probes.registry import FAMILIES
from rewardaudit.report import RELIABLE_ROW, ROWS, _grid, _reply, _say
from rewardaudit.types import Task

SAMPLE_TASKS = [
    Task("int", "Sam has 25 apples and buys 17 more. How many apples does Sam have? Put the final answer in \\boxed{}.", "42"),
    Task("mcq", "What is 6 times 7?\nA. 41\nB. 42\nC. 43\nD. 44\nAnswer with the letter.", "B"),
    Task("expression", "Use 37, 15, 4 and 13 to make 16.", "(37 + 15) * (4 / 13)"),
    Task("fraction", "Simplify 2/4.", "\\frac{1}{2}"),
    Task("decimal", "Solve for x: 2x = -3.", "-1.5"),
    Task("bool", "Is 17 a prime number? Answer yes or no.", "yes"),
    Task("text", "Name the capital of France.", "Paris"),
    Task("latex", "Expand (n+1)^2.", "n^2+2n+1"),
    Task("list", "List the prime factors of 30 in increasing order, separated by commas.", "2, 3, 5"),
    Task("json", "Return the point (3, 4) as a JSON object with keys x and y.", '{"x": 3, "y": 4}'),
    Task("grid", "Invert every cell of this binary grid (0 becomes 1, 1 becomes 0):\n0 1\n1 1", "1 0\n0 0"),
    Task("same", "What is 40 + 2?", "42"),
    Task("unit", "How far is the station?", "5 km"),
    Task("money", "A pen costs 3 dollars. How many dollars do 4 pens cost?", "12"),
    Task("percent", "What percentage of 200 is 50?", "25%"),
    Task("date", "When did the library open?", "2020-03-05"),
    Task("words", "Who wrote Les Miserables?", "Victor Hugo"),
    Task("relation", "Solve x + 2 > 5.", "x > 3"),
    Task("zero", "What is 2 - 2?", "0"),
    Task("set", "List the prime factors of 30 in any order, separated by commas.", "2, 3, 5"),
    Task("accent", "What is the French word for coffee?", "café"),
    Task("example", "Q: What is 1 + 1?\nA: 2\nQ: What is 6 times 7?", "42"),
    Task("candidate", "Which country is in Africa?\nA. Kenya\nB. Samoa\nC. Iceland\nAnswer with the country's name.", "Kenya"),
]
REJECTED_GOLD_HELP = (
    "Most reference answers pass, yet on some rows the checker treats the dataset's own answer like a wrong one in every "
    "format tried: its best score is 0, or no higher than a wrong answer's. A reference answer with partial credit above "
    "that is not counted. No probe sends anything for this: it comes from the calls with the reference answers, and it "
    "stays outside the rate of Rejects right answers. A sound checker accepts the reference answer.")


def _row_label(p: Probe) -> str:
    """The summary row a probe's calls count toward. Any probe's failed calls count toward Runs reliably as well."""
    row = ROWS[probe_category(p)][0]
    if p.family == "P":
        return f"{RELIABLE_ROW}; a reply the checker accepts is a finding under {row}"
    return row if p.headline else f"{row}, as a finding but not in the rate"


def sample_context() -> ProbeContext:
    ids = [t.id for t in SAMPLE_TASKS]
    contract = Contract(full_score=1.0, accepted=["boxed", "answer_tag"], primary="boxed", wrapper_rates={"boxed": 1.0},
                        self_pass={i: True for i in ids}, gold_score={i: 1.0 for i in ids},
                        task_wrappers={i: ["boxed"] for i in ids})
    ctx = ProbeContext(contract=contract, gold_type="int", tasks=SAMPLE_TASKS,
                       gold_types={t.id: infer_gold_type(t.answer) for t in SAMPLE_TASKS},
                       aliases={"words": ["Victor-Marie Hugo", "Hugo"]}, reference_is_answer=True)
    ctx._numeric_dataset = True  # the samples mix answer types on purpose; Q8 still shows its examples
    return ctx


def probe_examples(probe: Probe, k: int = 2) -> list[tuple[Task, str, str | None]]:
    """Up to k (sample task, reply, the reference answer written another way where the checker gets that), of different
    variants of the probe."""
    ctx = sample_context()
    out: dict[str, tuple[Task, str, str | None]] = {}
    for task in SAMPLE_TASKS:
        ctx.task = task
        try:
            for variant, reply, *reference in probe.generate(task, random.Random(f"0:{probe.id}:{task.id}"), ctx):
                out.setdefault(variant, (task, reply, reference[0] if reference else None))
        except Exception:  # noqa: BLE001 - a probe that cannot handle a sample task just shows no example from it
            continue
    return list(out.values())[:k]


def find_explainable(key: str) -> tuple[str, Any] | None:
    """('probe', Probe) | ('family', letter) | ('defect', Defect) | ('finding', id) | None."""
    hit = next((p for _, ps in FAMILIES.values() for p in ps if p.id.lower() == key.lower()), None)
    if hit:
        return "probe", hit
    if key.upper() in FAMILIES:
        return "family", key.upper()
    defect = next((d for d in DEFECTS if d.id.lower() == key.lower()), None)
    if defect:
        return "defect", defect
    if key.upper() in (REJECTED_GOLD_ID, "S1", "S2"):
        return "finding", key.upper()
    if key.lower() in ("format_only", "wrong_above", "wrong_partial"):
        return "finding", key.lower()
    return None


def _list_family(fam: str, console: Console) -> None:
    _say(console, f"{fam}  {FAMILIES[fam][0]}", "bold")
    t = _grid()
    if fam == "G":  # G1 comes from the calls with the reference answers, but readers look for it here
        t.add_row("", Text(REJECTED_GOLD_ID), Text(f"{REJECTED_GOLD_NAME} (a finding, not a probe)"))
    for p in FAMILIES[fam][1]:
        t.add_row("", Text(p.id), Text(p.name))
    console.print(t)


def render_probe_list(console: Console) -> None:
    for fam in FAMILIES:
        _list_family(fam, console)
    _say(console, "\nAlso: D1–D22 (defects), S1, S2, format_only, wrong_above, wrong_partial.")
    _say(console, "\nrewardaudit probes <id> shows what a probe sends and what a sound checker does.", "dim")


def render_explain(key: str, console: Console | None = None) -> bool:
    console = console or Console()
    found = find_explainable(key)
    if found is None:
        _say(console, f"Unknown id {key!r}. rewardaudit probes lists them all.")
        return False
    kind, obj = found
    if kind == "finding":
        if obj == REJECTED_GOLD_ID:
            obj, name, help_text = REJECTED_GOLD_ID, REJECTED_GOLD_NAME, REJECTED_GOLD_HELP
        elif obj == "S1":
            obj, name = "S1", "different scores for the same reply"
            help_text = "The same reply, sent several times on the same row, got different scores. A sound checker returns the same score."
        elif obj == "S2":
            obj, name = "S2", "scores outside the range"
            help_text = "The checker returned nan or inf instead of a finite score. A sound checker returns a finite number."
        elif obj == "format_only":
            name = "any answer in the right format earns the reference answer's score"
            help_text = "Calls with reference and wrong answers show the score follows the reply format. A sound checker separates right answers from wrong ones."
        elif obj == "wrong_above":
            name = "wrong answers above the reference answer"
            help_text = "Calls with wrong answers earned more than the reference answer on the same row. A sound checker scores the reference answer above a wrong one."
        else:
            name, help_text = ROWS["wrong_partial"][1], " ".join(ROWS["wrong_partial"][2:])
        _say(console, f"{obj}  {name}", "bold")
        console.print(Padding(Text(help_text), (0, 0, 0, 2), expand=False))
    elif kind == "defect":
        _say(console, f"{obj.id}  {obj.name}", "bold")
        console.print(Padding(Text(obj.explanation), (0, 0, 0, 2), expand=False))
    elif kind == "family":
        _list_family(obj, console)
        _say(console, f"\nrewardaudit probes <id> shows one of them, such as rewardaudit probes {FAMILIES[obj][1][0].id}.", "dim")
    else:
        examples = probe_examples(obj)
        _say(console, f"{obj.id}  {obj.name}", "bold")
        console.print(Padding(Text(obj.description or ""), (1, 0, 1, 2), expand=False))
        _say(console, f"  Counts toward: {_row_label(obj)}\n")
        _say(console, "  Examples on built-in sample tasks, with \\boxed{} as the accepted format" if examples
             else "  No example: it sends nothing on the built-in sample tasks.", "bold" if examples else "dim")
        for task, reply, reference in examples:
            written = "" if reference is None else f" written as {_reply(reference, 40)}"
            console.print(Padding(Text(f"reference answer {_reply(task.answer, 40)}{written}: {_reply(reply, 200)}"), (0, 0, 0, 4),
                                  expand=False))
    return True
