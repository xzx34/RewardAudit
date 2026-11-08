"""The report: what the terminal, report.md, report.json, repro.py and the probes command say to someone new, at 80 columns."""

import dataclasses
import difflib
import itertools
import json
import platform
import re
import subprocess
import sys
import time

import pytest
from rich.console import Console
from typer.testing import CliRunner

from rewardaudit import audit, cli
from rewardaudit.contract import Contract
from rewardaudit.metrics import Robustness, exception_type
from rewardaudit.pipeline import exit_code
from rewardaudit.probes.base import Probe
from rewardaudit.probes.registry import FAMILIES, all_probes
from rewardaudit.report import (
    _code,
    _md_finding,
    _reply,
    no_discrimination_line,
    render_markdown,
    render_terminal,
    write_outputs,
)
from rewardaudit.types import Kind, Label, Observation, Verdict

runner = CliRunner()
INTERNAL_WORDS = re.compile(r"\b(gold|wrapper|contract|floor|self-pass|primary|tier|signal|threat)s?\b", re.I)
SECTIONS = ["## Summary", "## Defects", "## Findings", "## What to fix", "## Answer formats", "## Reliability", "## Dataset", "## How this was run"]
ROWS_OF_NUMBERS = [{"prompt": f"What is {i} + 10? End with #### and the answer.", "answer": str(i + 10)} for i in range(16)]
ROWS_OF_WORDS = [{"prompt": f"Rewrite item {i} with a dash.", "answer": f"it-em{i} num-ber{i}"} for i in range(16)]


def last_number(completion, answer):
    numbers = re.findall(r"-?\d+", completion)
    return float(bool(numbers) and numbers[-1] == answer)


def any_reply(completion, answer):
    return 0.5 if completion.strip() else 0.0


def similarity(completion, answer):
    return difflib.SequenceMatcher(None, completion, answer, autojunk=False).ratio()


def raises_on_markup(completion, answer):
    if "<answer>" in completion:
        raise ValueError("cannot parse this")
    return float(completion.strip().endswith(answer))


_calls = itertools.count()


def noisy(completion, answer):
    return (1.0 if answer in completion else 0.0) * (0.9 + 0.1 * (next(_calls) % 2))


def several_parts(completion, answer):
    right, formatted = float(completion.strip().endswith(answer)), float("####" in completion)
    return {"score": 0.8 * right + 0.2 * formatted, "correct": right, "format_ok": formatted}


def slow_on_an_empty_reply(completion, answer):
    if not completion:  # of the N probes, only N1 sends one, and it runs first
        time.sleep(0.4)
        return 1.0
    return float(completion.strip().endswith(answer))


def screen(result, *, full=False, out=None):
    console = Console(width=80, record=True, force_terminal=False)
    render_terminal(result, console, full=full, out=out)
    return console.export_text()


def flat(text):
    return " ".join(text.split())


def row_line(text, label):
    return next(line.strip() for line in text.splitlines() if line.strip().startswith(label))


@pytest.fixture(scope="module")
def substring():
    return audit("demo:substring", n=12, workers=1)


@pytest.fixture(scope="module")
def several():
    return audit(several_parts, ROWS_OF_NUMBERS, workers=1)


def test_the_command_line_prints_the_first_screen_at_80_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "console", Console(width=80, force_terminal=False))
    res = runner.invoke(cli.app, ["audit", "demo:substring", "--n", "12", "-q", "--out", str(tmp_path)])
    assert res.exit_code == 1, res.output  # planted defects are findings
    lines = res.output.splitlines()
    assert len(lines) <= 55, res.output  # seven high defects, each with a line or two confirming it
    assert all(len(line.rstrip()) <= 80 for line in lines if not line.startswith("Saved")), res.output
    assert f"Saved report.md, report.json and repro.py in {tmp_path}" in res.output
    assert all((tmp_path / name).exists() for name in ("report.md", "report.json", "repro.py"))


def test_first_screen_says_what_was_found(substring):
    text = screen(substring)
    assert text.startswith(f"rewardaudit {substring.version} · demo/substring · 12 of 12 rows (seed 0) · ")
    high = sum(d["severity"] == "high" for d in substring.defects)
    medium = sum(d["severity"] == "medium" for d in substring.defects)
    assert f"Result: problems found ({high} high, {medium} medium)" in text
    assert "Reference answers accepted on 100% of rows. Replies were sent in the format " in text
    assert "shares of calls, with 95% intervals" in text
    assert f"Defects ({len(substring.defects)})" in text and "Findings (" not in text
    for d in substring.defects:  # nothing is cut off
        assert d["severity"].upper() in text and flat(d["name"]) in flat(text) and d["evidence"][0]["probe_id"] in text
    assert f"--full prints the {len(substring.findings)} findings behind this" in text
    assert "Add --out DIR to save report.md, report.json and repro.py." in text
    assert not INTERNAL_WORDS.search(text), INTERNAL_WORDS.search(text)


@pytest.mark.parametrize("demo", ["substring", "mcq_regex", "grid_exact"])
def test_a_row_says_yes_when_it_has_a_high_or_medium_finding(demo):
    r = audit(f"demo:{demo}", n=12, workers=1)
    serious = [f for f in r.findings if f["severity"] in ("high", "medium")]
    expected = {
        "Accepts wrong answers": any(f["category"] == "wrong_accepted" for f in serious),
        "Rejects right answers": any(f["category"] == "right_rejected" for f in serious),
        "Accepts hedged answers": any(f["category"] == "hedging_accepted" for f in serious),
        "Accepts non-answers": any(f["category"] == "non_solving_accepted" for f in serious),
    }
    heads = {h["key"]: h for h in r.metrics["head"]}
    keys = {"Accepts wrong answers": "wrong_accepted", "Rejects right answers": "right_rejected",
            "Accepts hedged answers": "hedging_accepted", "Accepts non-answers": "non_solving_accepted"}
    text = screen(r)
    for label, yes in expected.items():
        h = heads[keys[label]]
        line = row_line(text, label)
        assert re.fullmatch(rf"{label}\s+{'yes' if yes else 'no'}\s+{h['value']:.0%}\s+\[{h['ci'][0]:.0%}, {h['ci'][1]:.0%}\]", line), line


def test_no_problems_found_and_a_row_nothing_was_sent_for():
    r = audit(last_number, ROWS_OF_NUMBERS, probes=["N"], workers=1)
    assert not r.findings
    text = screen(r)
    assert "Result: no problems found" in text and "No defects." in text
    assert row_line(text, "Accepts wrong answers").split() == ["Accepts", "wrong", "answers", "not", "measured"]
    assert exit_code(r) == 0


def test_full_prints_every_finding_its_interval_and_the_accepted_formats(substring):
    text = screen(substring, full=True)
    assert f"Findings ({len(substring.findings)})" in text and "--full" not in text
    assert "Accepted formats (… stands for reasoning):" in text
    assert all(f["probe_id"] in text for f in substring.findings) and "95% interval [" in text


def test_a_checker_that_does_not_separate_right_from_wrong_makes_no_claim():
    r = audit(any_reply, ROWS_OF_WORDS, workers=1)
    assert r.contract.no_discrimination and exit_code(r) == 2
    text, md = screen(r), render_markdown(r)
    line = no_discrimination_line(r.contract, r.metrics)
    assert "Reference answers and wrong answers score the same on 16 of the 16 rows" in line
    assert "Result: the rates could not be measured" in text and flat(line) in flat(text)
    assert "no problems found" not in text and "No findings." not in text
    for label in ("Accepts non-answers", "Accepts wrong answers", "Accepts hedged answers", "Rejects right answers"):
        assert row_line(text, label).endswith("not measured"), label  # never a misleading "no"
        assert f"| {label} |  | not measured |  | 0 |" in md
    assert "Not counted, for the reason above:" in text and "`withheld_findings`" in md
    assert md.index(line) < md.index("## Summary")  # before any number
    assert "**Reference answers not separated from wrong answers.**" in md
    assert not INTERNAL_WORDS.search(md), INTERNAL_WORDS.search(md)


def test_a_reply_above_the_reference_answer_is_a_yes_without_a_rate():
    def counts_numbers(completion, answer):  # reads every number in the reply: a sweep is many, a reference answer is one
        return 0.955 if len(re.findall(r"-?[0-9]+", completion)) >= 6 else 0.155

    r = audit(counts_numbers, ROWS_OF_NUMBERS, workers=1)
    assert r.contract.no_discrimination and [f for f in r.findings if f.get("above_gold")]
    text = screen(r)
    assert "Result: problems found (" in text and "the rates could not be measured" in text
    assert re.fullmatch(r"Accepts non-answers\s+yes\s+see Defects", row_line(text, "Accepts non-answers"))
    assert "Score does not follow the answer" in flat(text)
    assert "scored above the reference answer" in flat(screen(r, full=True))


def test_a_whole_reply_grader_says_which_row_cannot_be_measured():
    r = audit(similarity, ROWS_OF_WORDS, workers=1)
    assert r.contract.whole_reply_similarity
    text, md = screen(r), render_markdown(r)
    assert row_line(text, "Rejects right answers").endswith("not measured")
    assert re.match(r"Accepts wrong answers\s+no\s+0%", row_line(text, "Accepts wrong answers"))
    assert "grades the whole reply" in flat(text) and md.index("grades the whole reply") < md.index("## Summary")


def test_failed_calls_say_how_many_and_the_exception_type():
    r = audit(raises_on_markup, ROWS_OF_NUMBERS, workers=1)
    rb = r.metrics["robustness"]
    errors = rb["exceptions"]["count"] + rb["exceptions"]["gold"]
    text, md = screen(r), render_markdown(r)
    assert errors and f"{errors} errors (ValueError)" in flat(text) and row_line(text, "Runs reliably").split()[2] == "no"
    assert "of them on reference answers" in flat(text)
    assert "**Errors**:" in md and "Example: probe" in md and "ValueError: cannot parse this" in md and "**Reliability.**" in md


def test_network_attempts_are_counted_and_explained():
    def asks_a_service(completion, answer):
        if "know" in completion:
            import socket

            socket.create_connection(("127.0.0.1", 9), timeout=1)
        return float(completion.strip().endswith(answer))

    r = audit(asks_a_service, ROWS_OF_NUMBERS, probes=["N"], workers=1)
    k = r.metrics["robustness"]["network"]["count"]
    text, md = screen(r), render_markdown(r)
    assert k and f"{k} network attempts" in flat(text) and row_line(text, "Runs reliably").split()[2] == "no"
    assert "**Network attempts**:" in md and "needs no network" in md


def test_a_clean_run_keeps_reliability_short(substring):
    text, md = screen(substring), render_markdown(substring)
    assert re.fullmatch(r"Runs reliably\s+yes\s+no errors, timeouts or crashes in [\d,]+ calls", row_line(text, "Runs reliably"))
    section = md[md.index("## Reliability"):md.index("## Dataset")]
    assert len(section.splitlines()) <= 4 and "Every call returned a score" in section


def test_a_checker_that_is_not_deterministic_says_so():
    r = audit(noisy, ROWS_OF_NUMBERS, workers=1)
    assert not r.determinism["deterministic"]
    text, md = screen(r), render_markdown(r)
    assert "the same reply got different scores on" in flat(text) and row_line(text, "Runs reliably").split()[2] == "no"
    assert "**Different scores for the same reply**" in md


def test_a_score_with_several_parts_names_the_parts_it_judged_on(several):
    keys = " + ".join(several.contract.signal_keys)
    assert f"Whether an answer is right was judged on {keys}" in flat(screen(several))
    md = render_markdown(several)
    assert f"judged on {keys}" in md and "the reference answer's total on this row" in md
    assert "A reply counts as accepted when it reaches" in md


def test_lines_about_how_a_run_was_scored_come_before_the_numbers(substring):
    meta = {**substring.meta, "scoring_path": "direct", "scoring_path_reason": "because", "verifiers_pin": "<0.1.13"}
    r = dataclasses.replace(substring, meta=meta)
    text, md = flat(screen(r)), render_markdown(r)
    for line in ("Measured on <0.1.13, not on the verifiers a fresh install of this package gets.",
                 "direct rubric call, not through the environment's rollout (because)"):
        assert line in text and md.index(line) < md.index("## Summary")
    assert "direct rubric call" not in render_markdown(substring) and "Measured on" not in render_markdown(substring)


def test_a_budget_stops_the_probes_and_the_report_lists_those_not_run(tmp_path):
    r = audit(slow_on_an_empty_reply, ROWS_OF_NUMBERS[:6], probes=["N"], workers=1, budget_s=2.0)
    assert r.meta["probes_not_run"] == ["N2", "N5", "N9", "N3", "N10"]  # N1 alone sleeps 2.4 s, 0.4 s on each of 6 rows
    assert {f["probe_id"] for f in r.findings} == {p["id"] for p in r.metrics["probes"]} == {"N1"}
    assert r.metrics["probes"][0]["n_obs"] == 6  # the probe running when the time ran out is counted in full
    line = "Budget of 2s reached before N2, N5, N9, N3, N10 (5 probes not run); the rows below are measured on fewer probes."
    text, md = flat(screen(r)), render_markdown(r)
    assert line in text and text.index(line) < text.index("Accepts non-answers")
    assert md.index(line) < md.index("## Summary") and "- Timeout: 10 s per call\n- Budget: 2s\n- Workers: 1\n" in md
    write_outputs(r, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["meta"]["probes_not_run"] == r.meta["probes_not_run"] and report["config"]["budget_s"] == 2.0
    r = audit(last_number, ROWS_OF_NUMBERS, probes=["N"], workers=1, budget_s=3600)  # time to spare: every probe runs
    assert "probes_not_run" not in r.meta and len(r.metrics["probes"]) == 6
    assert "Budget of" not in screen(r) and "- Budget: 1h\n" in render_markdown(r)


def test_a_probe_whose_replies_take_long_to_build_is_not_sent_once_the_budget_is_reached(monkeypatch):
    generate = Probe.generate

    def slow(self, task, rng, ctx):
        if self.id == "N2":  # 3 s for its 6 rows, the time ends while they are built
            time.sleep(0.5)
        return generate(self, task, rng, ctx)

    monkeypatch.setattr(Probe, "generate", slow)
    seen = []
    r = audit(last_number, ROWS_OF_NUMBERS[:6], probes=["N"], workers=1, budget_s=2.0, progress=lambda *a: seen.append(a))
    assert r.meta["probes_not_run"] == ["N2", "N5", "N9", "N3", "N10"]
    assert [p["id"] for p in r.metrics["probes"]] == ["N1"]
    # the probes' replies are not all built ahead to count them, so that stage of the progress bar has no total
    assert {total for stage, _, total in seen if stage == "probes"} == {0}
    assert all(total >= done for stage, done, total in seen if stage == "reference answers")


def test_report_md_has_its_sections_in_order_and_nothing_removed(substring):
    md = render_markdown(substring)
    assert md.startswith("# rewardaudit report: demo/substring\n")
    assert [line for line in md.splitlines() if line.startswith("## ")] == SECTIONS
    assert "95% Wilson interval" in md and "shares of calls" in md
    f = substring.findings[0]
    assert f"### {f['severity'].upper()}: {f['probe_id']} {f['name']}" in md and f"Rate {f['rate']:.0%} of " in md
    assert "Example (row `" in md and "reference answer `" in md
    assert "always answering" in md or "No issues." in md
    assert "cover all 12 rows of the dataset" in md
    for gone in ("All probes", "```json", "Calls to first", "adaptive", "budget", "skipped", "Docker", "Withheld"):
        assert gone not in md, gone
    assert not INTERNAL_WORDS.search(md), INTERNAL_WORDS.search(md)


def test_answer_formats_is_a_short_list(substring):
    md = render_markdown(substring)
    section = md[md.index("## Answer formats"):md.index("## Reliability")]
    items = [line for line in section.splitlines() if line.startswith("- ")]
    assert 1 <= len(items) <= 9 and "|" not in section
    assert len(substring.contract.accepted) <= 8 or items[-1].startswith("- and ")


def test_how_this_was_run_names_checker_dataset_and_settings(tmp_path):
    (tmp_path / "reward.py").write_text("def score(completion, answer):\n    return float(answer in completion)\n")
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(row) for row in ROWS_OF_NUMBERS))
    r = audit(str(tmp_path / "reward.py"), str(tmp_path / "data.jsonl"), n=8, seed=3, timeout=7.0, workers=1, probes=["N"])
    md = render_markdown(r)
    section = md[md.index("## How this was run"):]
    for text in ("reward:score", str(tmp_path / "reward.py"), f"`{tmp_path / 'data.jsonl'}`, 16 rows",
                 "8 rows drawn at random, seed 3", "7 s per call", "Workers: 1", "Probe families: N", "`repro.py`"):
        assert text in section, text
    assert "Budget" not in md  # no limit unless one is given


def test_a_checker_on_the_standard_library_imports_none(tmp_path):
    (tmp_path / "reward.py").write_text("import re\n\ndef score(completion, answer):\n    return float(re.sub(r'\\s', '', completion) == answer)\n")
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(row) for row in ROWS_OF_NUMBERS))
    r = audit(str(tmp_path / "reward.py"), str(tmp_path / "data.jsonl"), n=8, workers=1, probes=["N"])
    assert r.environment["packages"] == {} and r.environment["python"] == platform.python_version()
    text, md = screen(r), render_markdown(r)
    assert text.rindex("Dataset: ") < text.index("\nChecker imports: none\n\n")  # the line after the dataset checks
    assert md.index("- Dataset: ") < md.index("- Checker imports: none\n") < md.index("- Sample: ")
    write_outputs(r, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))["environment"] == r.environment


def test_what_reading_a_parquet_dataset_imports_is_not_the_checkers(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "reward.py").write_text("def score(completion, answer):\n    return float(completion.strip() == answer)\n")
    pq.write_table(pa.Table.from_pylist(ROWS_OF_NUMBERS), tmp_path / "data.parquet")
    r = audit(str(tmp_path / "reward.py"), str(tmp_path / "data.parquet"), n=8, workers=1, probes=["N"])
    assert r.environment["packages"] == {} and "Checker imports: none" in screen(r)


def test_cloudpickle_that_carries_a_function_from_python_is_not_its_import():
    def exact(completion, answer):  # defined here, so it travels by value and its module is not imported again
        return float(completion.strip() == answer)

    assert audit(exact, ROWS_OF_NUMBERS, probes=["N"], workers=1).environment["packages"] == {}


def test_checker_imports_name_eight_packages_and_count_the_rest(substring):
    r = dataclasses.replace(substring, environment={**substring.environment, "packages": {f"pkg{i}": f"1.{i}" for i in range(11)}})
    line = "Checker imports: " + ", ".join(f"pkg{i} 1.{i}" for i in range(8)) + " and 3 more"
    assert line in flat(screen(r)) and f"- {line}\n" in render_markdown(r)
    nothing = dataclasses.replace(substring, environment={})  # no worker replied: no line, rather than a wrong "none"
    assert "Checker imports" not in screen(nothing) and "Checker imports" not in render_markdown(nothing)


def test_report_json_is_the_result_without_its_calls(substring, tmp_path):
    path = write_outputs(substring, tmp_path / "out")
    assert path == tmp_path / "out" / "report.md" and path.read_text(encoding="utf-8") == render_markdown(substring)
    data = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "1" and set(data) == {"schema_version", *substring.to_dict()}
    assert "observations" not in data and data["metrics"]["probes"] and data["contract"]["accepted"]


def test_the_rate_line_of_a_finding_says_call_and_row_in_the_singular():
    finding = {"probe_id": "Q1", "name": "extra whitespace", "category": "right_rejected", "severity": "low", "rate": 1.0,
               "ci": [0.21, 1.0], "n": 1, "n_tasks": 1, "note": "", "example": None}
    assert "Rate 100% of 1 call on 1 row, 95% interval" in "\n".join(_md_finding(finding, None))
    assert "Rate 100% of 2 calls on 12 rows," in "\n".join(_md_finding(finding | {"n": 2, "n_tasks": 12}, None))
    assert "Rate 100% of 1 row," in "\n".join(_md_finding(finding | {"unit": "tasks"}, None))


def _run_repro(path, cwd):
    proc = subprocess.run([sys.executable, str(path / "repro.py")], capture_output=True, text=True, timeout=120, cwd=cwd)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_repro_scores_the_example_behind_each_finding_again(substring, tmp_path):
    write_outputs(substring, tmp_path)
    lines = _run_repro(tmp_path, tmp_path)
    examples = [f for f in substring.findings if f.get("example")]
    assert len(lines) == len(examples) > 0
    for line, f in zip(lines, examples, strict=True):
        assert f["probe_id"] in line and f"score={f['example']['verdict']['score']}" in line


@pytest.mark.parametrize("data", ["file", "memory"])
def test_repro_loads_a_function_from_its_file_from_any_directory(tmp_path, monkeypatch, data):
    (tmp_path / "reward.py").write_text("def score(completion, answer):\n    return float(answer in completion)\n")
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(row) for row in ROWS_OF_NUMBERS))
    monkeypatch.chdir(tmp_path)  # relative paths, as on a command line
    r = audit("reward.py", "data.jsonl" if data == "file" else ROWS_OF_NUMBERS, n=8, workers=1)
    write_outputs(r, tmp_path / "out")
    assert len(_run_repro(tmp_path / "out", tmp_path / "out")) == sum(1 for f in r.findings if f.get("example")) > 0


def test_a_function_in_memory_has_no_repro_and_does_not_mention_it(tmp_path):
    r = audit(last_number, ROWS_OF_NUMBERS, workers=1)
    assert "path" not in r.spec
    assert r.save(tmp_path) == tmp_path / "report.md"
    assert not (tmp_path / "repro.py").exists() and (tmp_path / "report.json").exists()
    text, md = screen(r), render_markdown(r)
    assert "repro" not in text and "repro" not in md and "rows passed in memory" in md
    assert "result.save(DIR) writes report.md and report.json." in text and "--out" not in text
    assert "Saved report.md and report.json in" in screen(r, out=tmp_path / "report.md")


@pytest.mark.parametrize("pin, spelled", [(None, "verifiers=None"), ("<0.1.13", "verifiers='<0.1.13'")])
def test_repro_of_a_hub_environment_installs_it_first(substring, tmp_path, pin, spelled):
    meta = {**substring.meta, **({"verifiers_pin": pin} if pin else {})}
    r = dataclasses.replace(substring, spec={"kind": "verifiers", "env": "owner/name"}, meta=meta)
    write_outputs(r, tmp_path)
    code = (tmp_path / "repro.py").read_text(encoding="utf-8")
    compile(code, "repro.py", "exec")
    assert "from rewardaudit.envs import install_env" in code and f"python=str(install_env(SPEC['env'], {spelled})[0])" in code


def test_repro_of_an_environment_in_this_python_installs_nothing(substring, tmp_path):
    write_outputs(dataclasses.replace(substring, spec={"kind": "verifiers", "env": "my-env"}), tmp_path)
    assert "install_env" not in (tmp_path / "repro.py").read_text(encoding="utf-8")


def test_a_reply_is_shown_on_one_line_with_both_ends():
    long = "Let me think. " * 40 + "\n\n\\boxed{42}"
    shown = _reply(long)
    assert "\n" not in shown and shown.endswith("\\boxed{42}") and shown.startswith("Let me think.") and " … " in shown
    assert _reply("") == "(an empty reply)" and _reply(" \n ") == "(spaces and newlines only)"
    assert _reply("a\x00b\n\nc") == "a b ⏎ c"
    assert _code("x") == "`x`" and _code("```\n1") == "```` ```\n1 ````"


# --- the probes command -----------------------------------------------------------------------------------------------


@pytest.fixture
def console_80(monkeypatch):
    monkeypatch.setattr(cli, "console", Console(width=80, force_terminal=False))


def test_probes_lists_every_probe(console_80):
    res = runner.invoke(cli.app, ["probes"])
    assert res.exit_code == 0, res.output
    assert all(p.id in res.output for p in all_probes()) and all(f"{fam}  " in res.output for fam in FAMILIES)
    assert "rewardaudit probes <id>" in res.output
    assert all(len(line.rstrip()) <= 80 for line in res.output.splitlines())
    assert not INTERNAL_WORDS.search(res.output), INTERNAL_WORDS.search(res.output)


@pytest.mark.parametrize("probe", all_probes(), ids=lambda p: p.id)
def test_probes_explains_a_probe(console_80, probe):
    res = runner.invoke(cli.app, ["probes", probe.id])
    assert res.exit_code == 0, res.output
    text = flat(res.output)
    assert text.startswith(f"{probe.id} {probe.name}") and flat(probe.description) in text and "sound" in text
    row = re.search(r"Counts toward: (Accepts non-answers|Accepts wrong answers|Accepts hedged answers|Rejects right answers|Runs reliably)", text)
    assert row, text
    assert not INTERNAL_WORDS.search(text), INTERNAL_WORDS.search(text)


@pytest.mark.parametrize("key, row", [("Q6", "Rejects right answers"), ("O4", "Rejects right answers"), ("M1", "Accepts wrong answers"),
                                      ("O7", "Accepts hedged answers"), ("N1", "Accepts non-answers"), ("P5", "Runs reliably")])
def test_a_probe_counts_toward_the_row_of_its_findings(console_80, key, row):
    assert f"Counts toward: {row}" in runner.invoke(cli.app, ["probes", key]).output


def test_probes_shows_example_replies_and_takes_any_case(console_80):
    res = runner.invoke(cli.app, ["probes", "q6"])
    assert res.exit_code == 0 and "Examples on built-in sample tasks" in res.output
    assert "reference answer 42: " in res.output and "To double-check" in res.output


@pytest.mark.parametrize("fam", list(FAMILIES))
def test_probes_lists_a_family(console_80, fam):
    res = runner.invoke(cli.app, ["probes", fam])
    assert res.exit_code == 0, res.output
    assert FAMILIES[fam][0] in flat(res.output) and all(p.id in res.output for p in FAMILIES[fam][1])


def test_probes_explains_the_finding_no_probe_sends(console_80):
    res = runner.invoke(cli.app, ["probes", "g1"])
    assert res.exit_code == 0 and "reference answer rejected in every format tried" in res.output
    assert "partial credit" in flat(res.output)


def test_an_unknown_probe_fails(console_80):
    res = runner.invoke(cli.app, ["probes", "Z99"])
    assert res.exit_code == 2 and "Unknown id" in res.output


# --- the numbers under Reliability ------------------------------------------------------------------------------------


def test_robustness_breakdown_counts_types_probes_and_gold_network():
    def obs(pid, verdict):
        return Observation(pid, pid[0], "v", "t1", "some input", verdict, Label.NOT_SOLVING, "zero", "natural")

    contract = Contract(full_score=1.0, accepted=["boxed"], primary="boxed", wrapper_rates={}, self_pass={},
                        gold_score={}, observations=[("boxed", "t1", "\\boxed{1}", Verdict(Kind.SCORE, 1.0, network=True)),
                                                     ("wrong_answer", "t1", "\\boxed{2}", Verdict(Kind.SCORE, 0.0))])
    observations = [
        obs("N1", Verdict(Kind.ERROR, error="Traceback (most recent call last):\n  ...\nValueError: bad input")),
        obs("N1", Verdict(Kind.ERROR, error="NetworkBlocked: network access during scoring", network=True)),
        obs("E1", Verdict(Kind.TIMEOUT, error="no result within 10.0s")),
        obs("E1", Verdict(Kind.CRASH, error="worker died")),
        obs("Q1", Verdict(Kind.SCORE, 1.0)),
    ]
    tally = Robustness()
    for o in observations:
        tally.add(o)
    rb = tally.stats(contract)
    assert rb["n_calls"] == 5 and rb["n_gold_calls"] == 1 and rb["network_calls"] == 1
    assert rb["exceptions"]["by_type"] == {"ValueError": 1, "NetworkBlocked": 1}
    assert rb["exceptions"]["probes"] == {"N1": 2} and rb["exceptions"]["example"]["probe_id"] == "N1"
    assert rb["timeouts"]["count"] == 1 and rb["timeouts"]["probes"] == {"E1": 1} and rb["timeouts"]["note"]
    assert rb["crashes"]["count"] == 1
    assert rb["network"]["gold"] == 1 and "remote service" in rb["network"]["note"]
    assert abs(rb["error_rate"] - 0.4) < 1e-9
    assert exception_type("KeyError: 'x'") == "KeyError" and exception_type("") == "unknown"


def test_a_reply_shows_the_spaces_at_its_ends():
    assert _reply("  x  ") == "␣␣x␣␣" and _reply("a\n\nb") == "a ⏎ b" and _reply("\nx\n") == "x"
