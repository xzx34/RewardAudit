"""The Python interface, what stops an audit, and how a problem the user can fix reaches them: an AuditError from Python; from
the command line one line, exit code 2 and no traceback."""

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rewardaudit import AuditError, NotAuditable, audit, cli
from rewardaudit.pipeline import exit_code

ROWS = [{"prompt": f"Add the numbers on card {i}.", "answer": str(1000 + 7 * i)} for i in range(12)]
RATES = {"non_solving_accepted", "wrong_accepted", "hedging_accepted", "right_rejected", "wrong_partial"}


def ends_with_gold(completion, answer):
    return float(completion.strip().endswith(answer))


def _pickled_whole():
    def score(completion, answer):
        return float(completion.strip().endswith(answer))

    return score


EXACT = _pickled_whole()  # unlike a function at the top of a module, it does not make each worker import this file


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _wants_its_tag(completion, answer):
    return float(completion.strip().endswith(f"<sol>{answer}</sol>"))


def test_a_reply_format_of_the_checkers_own_makes_it_auditable():
    with pytest.raises(NotAuditable):
        audit(_wants_its_tag, ROWS, workers=1, probes=["N"])
    r = audit(_wants_its_tag, ROWS, workers=1, probes=["N"], reply_format="<sol>{answer}</sol>")
    assert r.contract.primary == "<sol>{answer}</sol>" and r.contract.self_pass_rate == 1.0
    assert r.to_dict()["contract"]["derived_wrappers"] == {"<sol>{answer}</sol>": "the format given"}
    with pytest.raises(AuditError, match="holds {answer}"):
        audit(_wants_its_tag, ROWS, reply_format="<sol></sol>")


# --- a checker in any form --------------------------------------------------------------------------------------------------

def test_a_plain_function():
    r = audit(ends_with_gold, ROWS, n=8, probes=["n", " O "], timeout=3.0, workers=1)
    assert r.env_name == "ends_with_gold" and r.n_rows == 12 and len(r.tasks) == 8
    assert r.config.families == ["N", "O"] and r.config.timeout_s == 3.0 and r.config.workers == 1
    assert r.contract.self_pass_rate == 1.0
    assert isinstance(r.findings, list) and set(r.rates) == {"non_solving_accepted", "hedging_accepted"}
    assert all(0.0 <= v <= 1.0 for v in r.rates.values())
    assert "pickled" not in r.spec and "sys_path" not in r.spec  # bytes of the function stay out of the report


def test_a_closure_over_a_local_variable():
    weight = 0.25

    def score(completion, answer):
        return weight * float(completion.strip().endswith(answer))

    r = audit(score, ROWS, probes=["N"], workers=1)
    assert r.contract.full_score == 0.25 and r.env_name == "score"


def test_a_lambda_and_what_it_finds():
    r = audit(lambda completion, answer: float(answer in completion), ROWS, n="all", probes=["O"], workers=2)
    assert r.env_name == "<lambda>" and r.rates["hedging_accepted"] > 0.5
    severities = [f["severity"] for f in r.findings]
    assert {"O3", "O7"} <= {f["probe_id"] for f in r.findings} and severities == sorted(severities, key=["high", "medium", "low"].index)
    finding = r.findings[0]
    assert {"probe_id", "name", "category", "severity", "rate", "ci", "n", "example"} <= set(finding) and finding["example"]["completion"]
    assert exit_code(r) == 1


@pytest.mark.parametrize("target", [lambda p: str(p), lambda p: p, lambda p: f"{p}:my_score", lambda p: f"{p.parent / 'other.py'}:my_score"],
                         ids=["path", "Path", "path:function", "other_name"])
def test_a_function_from_a_file(tmp_path, target):
    code = "def my_score(completion, answer):\n    return float(completion.strip().endswith(answer))\n"
    (tmp_path / "reward.py").write_text(code, encoding="utf-8")
    (tmp_path / "other.py").write_text(code + "def helper():\n    pass\n", encoding="utf-8")
    r = audit(target(tmp_path / "reward.py"), ROWS, probes=["N"], workers=1)
    assert r.contract.self_pass_rate == 1.0 and r.meta["function"] == "my_score" and r.spec["path"]


def test_a_function_named_by_its_module(tmp_path, monkeypatch):
    (tmp_path / "scoring_module_by_name.py").write_text("def my_score(completion, answer):\n    return float(completion.strip().endswith(answer))\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert audit("scoring_module_by_name:my_score", ROWS, probes=["N"], workers=1).meta["function"] == "my_score"


def test_a_file_next_to_the_scoring_file_imports(tmp_path):
    (tmp_path / "helper_for_reward.py").write_text("OK = 1.0\n", encoding="utf-8")
    (tmp_path / "reward.py").write_text("from helper_for_reward import OK\n\ndef score(completion, answer):\n"
                                        "    return OK * completion.strip().endswith(answer)\n", encoding="utf-8")
    assert audit(tmp_path / "reward.py", ROWS, probes=["N"], workers=1).contract.self_pass_rate == 1.0


def test_a_function_that_imports_its_own_module_keeps_it(tmp_path, monkeypatch):
    """A function pickled by reference: the worker imports it from where this process found it."""
    (tmp_path / "scoring_module_for_a_test.py").write_text(
        "def score(completion, answer):\n    return float(completion.strip().endswith(answer))\n", encoding="utf-8")
    monkeypatch.syspath_prepend(tmp_path)
    import scoring_module_for_a_test

    monkeypatch.delitem(sys.modules, "scoring_module_for_a_test")
    assert audit(scoring_module_for_a_test.score, ROWS, probes=["N"], workers=1).contract.self_pass_rate == 1.0


# --- rows in any form -------------------------------------------------------------------------------------------------------

class _Frame:
    """What a pandas DataFrame offers the audit."""

    columns = ("prompt", "answer")

    def to_dict(self, orient):
        assert orient == "records"
        return ROWS


@pytest.mark.parametrize("data", [ROWS, tuple(ROWS), iter(ROWS), _Frame()], ids=["list", "tuple", "iterator", "frame"])
def test_rows_in_memory(data):
    r = audit(EXACT, data, n="all", probes=["N"], workers=1)
    assert r.n_rows == 12 and [t.answer for t in r.tasks] == [row["answer"] for row in ROWS]


@pytest.mark.parametrize("as_path", [Path, str], ids=["Path", "str"])
def test_a_dataset_file(tmp_path, as_path):
    path = _write(tmp_path / "rows.jsonl", ROWS)
    r = audit(EXACT, as_path(path), probes=["N"], workers=1)
    assert r.n_rows == 12 and r.spec["dataset"] == str(path)


def test_columns_and_options_reach_the_checker(tmp_path):
    rows = [{"q": row["prompt"], "a": row["answer"], "worked": f"so it is {row['answer']}"} for row in ROWS]
    seen = tmp_path / "seen.txt"

    def score(completion, answer, bonus=0.0):
        seen.write_text(answer, encoding="utf-8")
        return bonus + float(completion.strip().endswith(answer.split()[-1]))

    r = audit(score, _write(tmp_path / "rows.jsonl", rows), prompt_col="q", answer_col="a", reference_col="worked",
              kwargs={"bonus": 1.0}, probes=["N"], workers=1)
    assert r.contract.full_score == 2.0 and seen.read_text(encoding="utf-8").startswith("so it is ")


def test_rows_in_memory_may_hold_values_json_cannot():
    rows = [{**row, "extra": Path("x")} for row in ROWS]
    assert audit(EXACT, rows, probes=["N"], workers=1).tasks[0].info["extra"] == "x"


# --- the result -------------------------------------------------------------------------------------------------------------

def _longest_list(x) -> int:
    if isinstance(x, dict):
        return max(map(_longest_list, x.values()), default=0)
    if isinstance(x, list):
        return max([len(x), *map(_longest_list, x)])
    return 0


def test_save_writes_a_report_without_a_list_of_every_call(tmp_path):
    r = audit(lambda completion, answer: float(answer in completion), ROWS, n="all", workers=2)
    path = r.save(tmp_path / "out")
    assert path == tmp_path / "out" / "report.md" and path.read_text(encoding="utf-8").startswith("# ")
    data = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    calls = r.meta["checker_calls"]
    assert calls > 500 and data["meta"]["checker_calls"] == calls and data["n_rows"] == 12 and r.observations == []
    assert set(r.rates) == RATES
    assert "observations" not in data and "observations" not in data["contract"]
    assert _longest_list(data) < 100  # a list per probe at most, not per call
    assert (tmp_path / "out" / "report.json").read_text(encoding="utf-8").count('"completion"') < calls / 2
    assert data["metrics"]["findings"] == json.loads(json.dumps(r.findings))
    assert "pickled" not in data["spec"]


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_and_log_say_how_far_it_is(workers):
    seen, logged = [], []
    audit(EXACT, ROWS, n=8, probes=["N"], workers=workers, log=logged.append, progress=lambda *a: seen.append(a))
    assert [line for line in logged if line.startswith("sampled")] == ["sampled 8 of 12 rows (seed 0)"]
    assert {stage for stage, _, _ in seen} == {"reference answers", "probes"}
    for stage in ("reference answers", "probes"):
        done = [d for s, d, _ in seen if s == stage]
        assert done == list(range(1, len(done) + 1))
    assert all(done <= total for _, done, total in seen)


def test_print_shows_the_result(capsys):
    audit(lambda completion, answer: float(answer in completion), ROWS, probes=["O"], workers=1).print(full=True)
    assert "O3" in capsys.readouterr().out


# --- what stops an audit -----------------------------------------------------------------------------------------------------

LABEL_ROWS = [{"prompt": f"Write it.\ndef fn_{i}(x):", "answer": f"fn_{i}"} for i in range(8)]


def _stops(checker, rows, **kw) -> NotAuditable:
    with pytest.raises(NotAuditable) as e:
        audit(checker, rows, n="all", workers=1, **kw)
    assert isinstance(e.value, AuditError) and e.value.reason in str(e.value)
    return e.value


def test_reference_answers_that_are_identifiers_the_prompt_prints_are_no_reference():
    any_reply = _stops(lambda completion, answer: 0.5 if completion.strip() else 0.0, LABEL_ROWS)
    assert any_reply.cls == "no_reference" and "8 of 8 reference answers are identifiers the prompt prints" in any_reply.reason
    assert "--answer-col" in any_reply.advice
    never = _stops(lambda completion, answer: 0.0, LABEL_ROWS)
    assert never.cls == "no_reference"


def test_an_identifier_the_checker_separates_is_audited():
    r = audit(lambda completion, answer: float(completion.strip().endswith(answer)), LABEL_ROWS, n="all", probes=["N"], workers=1)
    assert not r.contract.no_discrimination and r.contract.self_pass_rate == 1.0


def test_a_checker_no_reference_answer_passes():
    stop = _stops(lambda completion, answer: 0.0, ROWS)
    assert stop.cls == "gold_never_passes" and "No reference answer earned full score" in stop.reason
    assert "answer column" in stop.advice and "--verifiers" not in stop.advice


def test_a_checker_that_raises_on_every_reference_answer():
    def broken(completion, answer):
        raise KeyError("answer")

    stop = _stops(broken, ROWS)
    assert stop.cls == "gold_never_passes" and "raised an error on" in stop.reason and "KeyError" in stop.reason


def test_too_few_passing_reference_answers():
    stop = _stops(lambda completion, answer: float(answer == "1000" and answer in completion), ROWS)
    assert stop.cls == "gold_never_passes" and "Only 1 of 12" in stop.reason


def _network_checker(how):
    def score(completion, answer):  # the worker's guard refuses these calls: nothing leaves this machine
        import socket
        import urllib.request

        try:
            if how == "http":
                urllib.request.urlopen("http://192.0.2.1/", timeout=1)
            else:
                socket.create_connection(("192.0.2.1", 443), timeout=1)
        except Exception:  # a checker that falls back when its service is away
            if how != "error_swallowed":
                raise
        return float(answer in completion)

    return score


@pytest.mark.parametrize("how", ["socket", "http", "error_swallowed"])
def test_a_checker_that_makes_network_calls_is_an_llm_judge(how):
    stop = _stops(_network_checker(how), ROWS)
    assert stop.cls == "llm_judge" and "tried to use the network" in stop.reason and "remote service" in stop.advice


def test_placeholder_answers_stop_the_audit_before_any_call(tmp_path):
    called = tmp_path / "called"

    def score(completion, answer):
        called.write_text("yes")
        return 1.0

    stop = _stops(score, [{"prompt": f"Card {i}.", "answer": "n/a"} for i in range(8)])
    assert stop.cls == "no_reference" and "empty or placeholders" in stop.reason and not called.exists()


def test_a_checker_that_does_not_separate_is_reported_and_not_stopped():
    r = audit(lambda completion, answer: 0.5 if completion.strip() else 0.0, ROWS, n="all", probes=["N"], workers=1)
    assert r.contract.no_discrimination and exit_code(r) == 2 and all(h.get("not_measured") for h in r.metrics["head"])
    assert r.rates == {}  # not a rate of 0


# --- a checker that does odd things while loading and scoring ----------------------------------------------------------------

NOISY = """
import sys

print("loading the checker")
print("to the error stream", file=sys.stderr)
try:
    ANSWERED = input("Proceed? ")
except EOFError:
    ANSWERED = "end of file"

def score(completion, answer):
    print("scoring", completion)
    return float(ANSWERED == "end of file" and completion.strip().endswith(answer))
"""


def test_a_checker_that_prints_and_asks_does_not_disturb_the_audit(tmp_path):
    (tmp_path / "reward.py").write_text(NOISY, encoding="utf-8")
    r = audit(tmp_path / "reward.py", ROWS, probes=["N"], workers=1)
    assert r.contract.self_pass_rate == 1.0 and r.meta["worker_restarts"] == 0


def test_a_checker_that_ends_its_worker_is_a_failed_call_and_the_audit_goes_on(tmp_path):
    code = ("import os\n\ndef score(completion, answer):\n    if completion == '':\n        os._exit(1)\n"
            "    return float(completion.strip().endswith(answer))\n")
    (tmp_path / "reward.py").write_text(code, encoding="utf-8")
    r = audit(tmp_path / "reward.py", ROWS, n=5, probes=["N"], workers=1)
    assert r.meta["worker_restarts"] > 0 and r.metrics["robustness"]["crashes"]["count"] > 0
    assert r.contract.self_pass_rate == 1.0 and any("crashes" in f["note"] for f in r.findings if f["category"] == "robustness")


# --- problems the user can fix ------------------------------------------------------------------------------------------------

@pytest.fixture
def files(tmp_path) -> dict[str, str]:
    out = {"reward": tmp_path / "reward.py", "data": tmp_path / "data.jsonl", "bad_line": tmp_path / "bad.jsonl",
           "strange_columns": tmp_path / "strange.jsonl", "text": tmp_path / "data.txt", "missing_data": tmp_path / "nope.jsonl",
           "missing_reward": tmp_path / "nope.py", "helpers": tmp_path / "helpers.py", "empty_reward": tmp_path / "empty.py",
           "empty": tmp_path / "empty.jsonl", "scalars": tmp_path / "scalars.jsonl", "mixed": tmp_path / "mixed.jsonl",
           "broken_json": tmp_path / "broken.json", "dict_json": tmp_path / "dict.json", "not_python": tmp_path / "reward.txt",
           "unanswered": tmp_path / "unanswered.jsonl"}
    out["reward"].write_text("def score(completion, answer):\n    return float(answer in completion)\n", encoding="utf-8")
    _write(out["data"], ROWS)
    out["bad_line"].write_text('{"prompt": "p", "answer": "1"}\n{"prompt": "p", \n', encoding="utf-8")
    _write(out["strange_columns"], [{"q": "p", "a": "1"}])
    out["text"].write_text("rows\n", encoding="utf-8")
    out["helpers"].write_text("def one():\n    pass\n\ndef two():\n    pass\n", encoding="utf-8")
    out["empty_reward"].write_text("X = 1\n", encoding="utf-8")
    out["empty"].write_text("\n", encoding="utf-8")
    out["scalars"].write_text("1\n2\n", encoding="utf-8")
    out["mixed"].write_text('{"prompt": "p", "answer": "1"}\n2\n', encoding="utf-8")
    out["broken_json"].write_text("{", encoding="utf-8")
    out["dict_json"].write_text('{"a": 1}', encoding="utf-8")
    out["not_python"].write_text("not python\n", encoding="utf-8")
    _write(out["unanswered"], [{"prompt": "p", "answer": ""}, {"prompt": "q", "answer": None}])
    return {k: str(v) for k, v in out.items()}


# name, target, dataset, keyword arguments for Python, flags for the command line, and what the message must say
PROBLEMS = [
    ("wrong_answer_column", "reward", "data", {"answer_col": "gold"}, ["--answer-col", "gold"], ["no column 'gold'", "prompt, answer"]),
    ("wrong_prompt_column", "reward", "data", {"prompt_col": "text"}, ["--prompt-col", "text"], ["no column 'text'", "prompt, answer"]),
    ("no_known_column", "reward", "strange_columns", {}, [], ["which column holds the prompt", "q, a", "--prompt-col"]),
    ("missing_dataset", "reward", "missing_data", {}, [], ["Dataset not found", "nope.jsonl", "Check the path"]),
    ("unsupported_suffix", "reward", "text", {}, [], ["data.txt", ".jsonl, .json, .csv, .parquet or hf:"]),
    ("bad_json_line", "reward", "bad_line", {}, [], ["line 2 is not valid JSON", "one JSON object per line"]),
    ("json_that_does_not_parse", "reward", "broken_json", {}, [], ["not valid JSON", "name it .jsonl"]),
    ("json_without_rows", "reward", "dict_json", {}, [], ["must hold a list of rows"]),
    ("empty_dataset", "reward", "empty", {}, [], ["is empty"]),
    ("rows_that_are_not_objects", "reward", "scalars", {}, [], ["must be objects with named columns"]),
    ("a_later_row_that_is_not_an_object", "reward", "mixed", {}, [], ["must be objects with named columns"]),
    ("missing_scoring_file", "missing_reward", "data", {}, [], ["Scoring file not found", "nope.py", "Check the path"]),
    ("not_a_python_file", "not_python", "data", {}, [], ["not a Python file", "reward.py:function"]),
    ("no_scoring_function", "helpers", "data", {}, [], ["Cannot tell which function", "Name it `score`", ":function_name"]),
    ("no_function_at_all", "empty_reward", "data", {}, [], ["Cannot tell which function", "Name it `score`"]),
    ("a_function_it_does_not_have", "reward:nope", "data", {}, [], ["has no function 'nope'", ":function_name"]),
    ("invalid_n", "reward", "data", {"n": 0}, ["--n", "0"], ["positive whole number or 'all'"]),
    ("a_word_for_n", "reward", "data", {"n": "many"}, ["--n", "many"], ["positive whole number or 'all'", "many"]),
    ("function_without_dataset", "reward", None, {}, [], ["needs a dataset", "jsonl", "hf:org/name"]),
    ("environment_with_dataset", "owner/name", "data", {}, [], ["owner/name is an environment", "brings its own dataset", "--dataset"]),
    ("unknown_probe_family", "reward", "data", {"probes": ["Z"]}, ["--probes", "Z"], ["Unknown probe family 'Z'", "N, E, L, O, M, Q, G, P"]),
    ("no_reference_answers", "reward", "unanswered", {}, [], ["All 2 rows of", "have an empty reference answer in 'answer'", "--answer-col"]),
]


def _target(name: str, files: dict[str, str]) -> str:
    key, _, function = name.partition(":")
    return files.get(key, key) + (f":{function}" if function else "")


@pytest.mark.parametrize("target, dataset, kw, flags, says", [p[1:] for p in PROBLEMS], ids=[p[0] for p in PROBLEMS])
def test_a_problem_raises_an_error_that_says_what_to_do(files, target, dataset, kw, flags, says):
    with pytest.raises(AuditError) as e:
        audit(_target(target, files), files.get(dataset), workers=1, **kw)
    assert "\n" not in str(e.value) and not isinstance(e.value, NotAuditable)
    for text in says:
        assert text in str(e.value)


def _run(*args: str):
    """COLUMNS: the console wraps a long line at the terminal's width, and here we count lines."""
    return CliRunner().invoke(cli.app, ["audit", *args], env={"COLUMNS": "1000"})


def test_the_command_line_takes_a_reply_format(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text('def score(completion, answer):\n    return float(completion.strip().endswith(f"<sol>{answer}</sol>"))\n')
    data = _write(tmp_path / "data.jsonl", ROWS)
    res = _run(str(reward), "--dataset", str(data), "--format", "<sol>{answer}</sol>", "--probes", "N", "--workers", "1")
    assert res.exit_code != 2 and "in the format <sol>" in res.output, res.output
    _says_one_line(_run(str(reward), "--dataset", str(data), "--format", "<sol></sol>"), "holds {answer}")


def test_the_command_line_takes_answer_field(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text('def score(completion, answer):\n    return float(completion.split("\\n")[0].strip() == answer)\n')
    args = (str(reward), "--dataset", str(_write(tmp_path / "data.jsonl", ROWS)), "--probes", "Q", "--workers", "1")
    inferred, stated = _run(*args), _run(*args, "--answer-field")
    assert inferred.exit_code == 1 and "compares the whole reply" not in inferred.output, inferred.output
    assert stated.exit_code == 0 and "stated with --answer-field, not inferred" in stated.output, stated.output


ALIASES = [{"question": f"Name the thing on card {i}.", "answers": {"text": [f"thing{i}", f"item{i}"]}} for i in range(12)]


def _any_alias(completion, answer):
    return float(completion.strip() in answer)


def test_a_list_of_accepted_answers_goes_to_the_checker_and_one_of_them_into_the_replies():
    stop = _stops(_any_alias, ALIASES, answer_col="answers.text")
    assert "--answer-col COLUMN.0 --reference-col COLUMN" in stop.advice
    r = audit(_any_alias, ALIASES, answer_col="answers.text.0", reference_col="answers.text", workers=1, probes=["N"])
    assert r.contract.self_pass_rate == 1.0 and r.tasks[0].answer.startswith("thing")


def test_a_row_with_no_accepted_answers_is_left_out_and_no_missing_column():
    rows = [*ALIASES, {"question": "Unanswerable?", "answers": {"text": []}}]
    r = audit(_any_alias, rows, n="all", answer_col="answers.text.0", reference_col="answers.text", workers=1, probes=["N"])
    assert len(r.tasks) == r.n_rows == 12 and "12" not in {t.id for t in r.tasks}
    assert [i["count"] for i in r.dataset_issues if i["id"] == "empty_rows_left_out"] == [1]


def test_an_unknown_rate_in_fail_on_is_an_error_and_no_silent_pass():
    res = _run("reward.py", "--dataset", "data.jsonl", "--fail-on", "rejects>0.5")
    _says_one_line(res, "does not know 'rejects'", "right_rejected")


def test_a_class_is_named_as_such(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("class Grader:\n    def __init__(self, strict=True):\n        self.strict = strict\n")
    data = _write(tmp_path / "data.jsonl", ROWS)
    _says_one_line(_run(f"{reward}:Grader", "--dataset", str(data)), "Grader is a class", "small function")


def test_the_command_line_takes_a_reference_column(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("def score(completion, answer):\n    return float(completion.strip() in answer)\n")
    data = _write(tmp_path / "data.jsonl", ALIASES)
    res = _run(str(reward), "--dataset", str(data), "--answer-col", "answers.text.0", "--reference-col", "answers.text",
               "--probes", "N", "--workers", "1")
    assert res.exit_code != 2 and "accepted on 100% of rows" in res.output, res.output


def _says_one_line(res, *says: str) -> None:
    assert res.exit_code == 2, res.output
    assert res.exception is None or isinstance(res.exception, SystemExit)
    assert "Traceback" not in res.output
    lines = [line for line in res.output.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith("Error: "), res.output
    for text in says:
        assert text in lines[0], res.output


@pytest.mark.parametrize("target, dataset, kw, flags, says", [p[1:] for p in PROBLEMS], ids=[p[0] for p in PROBLEMS])
def test_the_same_problem_from_the_command_line(files, target, dataset, kw, flags, says):
    command = [_target(target, files), *(["--dataset", files[dataset]] if dataset else []), *flags]
    _says_one_line(_run(*command, "--quiet", "--workers", "1"), *says)


def test_without_quiet_only_what_was_done_so_far_comes_before_the_error(files):
    res = _run(files["reward"], "--dataset", files["data"], "--answer-col", "gold", "--workers", "1")
    lines = [line for line in res.output.splitlines() if line.strip()]
    assert res.exit_code == 2 and "Traceback" not in res.output and lines[-1].startswith("Error: ") and "no column 'gold'" in lines[-1]
    assert all(line.startswith("loaded") for line in lines[:-1])


@pytest.mark.parametrize("n", [-3, 1.5, True, [5]])
def test_other_values_of_n(n):
    with pytest.raises(AuditError, match="positive whole number"):
        audit(EXACT, ROWS, n=n)


@pytest.mark.parametrize("n", ["-3", "1.5", "all of them"])
def test_other_values_of_n_from_the_command_line(files, n):
    _says_one_line(_run(files["reward"], "--dataset", files["data"], "--n", n), "--n must be a positive whole number or 'all'")


@pytest.mark.parametrize("text, seconds", [("90s", 90), ("10m", 600), ("1h", 3600), ("45", 45), ("1.5m", 90), (" 2m ", 120)])
def test_a_budget_is_a_length_of_time(text, seconds):
    assert cli._budget(text) == seconds


@pytest.mark.parametrize("text", ["10x", "0", "-5s", "m", "1 h", ""])
def test_a_budget_that_is_no_length_of_time_is_one_line(files, text):
    _says_one_line(_run(files["reward"], "--dataset", files["data"], "--budget", text),
                   "--budget must be a length of time such as 90s, 10m or 1h", repr(text))


def test_the_command_line_takes_a_budget(files, tmp_path):
    out = tmp_path / "out"
    res = _run(files["reward"], "--dataset", files["data"], "--budget", "1h", "--probes", "N", "--workers", "1", "--out", str(out))
    assert res.exit_code != 2 and "Budget of" not in res.output, res.output  # an hour is plenty: every probe ran
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["config"]["budget_s"] == 3600
    assert "- Budget: 1h\n" in (out / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("call, says", [
    (lambda: audit(EXACT, ROWS, budget_s=0), "budget_s must be a number of seconds above 0"),
    (lambda: audit(EXACT, ROWS, budget_s="10m"), "budget_s must be a number of seconds above 0, got '10m'"),
    (lambda: audit(EXACT), "needs a dataset"),
    (lambda: audit(EXACT, 7), "Cannot read a dataset from int"),
    (lambda: audit(EXACT, ["a line", "another"]), "Cannot read a dataset from list"),
    (lambda: audit(EXACT, []), "is empty"),
    (lambda: audit(EXACT, ROWS, style="nope", workers=1), "Unknown function style 'nope'"),
    (lambda: audit(EXACT, ROWS, colour="red"), "Unknown option 'colour'"),
    (lambda: audit(7, ROWS), "Cannot audit a int"),
    (lambda: audit(None), "Cannot audit a NoneType"),
])
def test_python_only_problems(call, says):
    with pytest.raises(AuditError, match=says):
        call()


def test_a_function_that_cannot_be_pickled_is_told_to_move_to_a_file():
    class Unpicklable:
        def __reduce__(self):
            raise TypeError("no")

    held = Unpicklable()

    def score(completion, answer):
        return float(bool(held) and answer in completion)

    with pytest.raises(AuditError, match="Put it in a file"):
        audit(score, ROWS)


def test_an_environment_name_without_verifiers_here_says_what_is_wanted(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(AuditError, match="my-env is not a file.*owner/name.*installed in this Python"):
        audit("my-env")


def test_a_missing_library_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    (tmp_path / "rows.parquet").write_bytes(b"")
    from rewardaudit.adapters.dataset import load_tasks

    with pytest.raises(AuditError, match=r"pip install 'rewardaudit\[data\]'"):
        load_tasks("hf:org/name")
    with pytest.raises(AuditError, match=r"pip install 'rewardaudit\[data\]'"):
        load_tasks(str(tmp_path / "rows.parquet"))


@pytest.mark.parametrize("code, says", [
    ("raise ValueError('the setting is missing')\n", "reward.py, line 1: ValueError: the setting is missing"),
    ("import a_module_nobody_has\n", "reward.py, line 1: ModuleNotFoundError: No module named 'a_module_nobody_has'"),
    ("def score(:\n", "SyntaxError"),
], ids=["raises", "import", "syntax"])
def test_a_checker_that_does_not_load_is_one_line_too(tmp_path, code, says):
    (tmp_path / "reward.py").write_text(code, encoding="utf-8")
    data = _write(tmp_path / "data.jsonl", ROWS)
    _says_one_line(_run(str(tmp_path / "reward.py"), "--dataset", str(data), "--quiet", "--workers", "1"),
                   "Could not load the checker", says)


def test_not_auditable_from_the_command_line_is_one_line_too(tmp_path):
    (tmp_path / "reward.py").write_text("def score(completion, answer):\n    return 0.0\n", encoding="utf-8")
    data = _write(tmp_path / "data.jsonl", ROWS)
    _says_one_line(_run(str(tmp_path / "reward.py"), "--dataset", str(data), "--quiet", "--workers", "1"),
                   "No reference answer earned full score", "answer column")


def test_a_bad_fail_on_is_one_line_too(files):
    _says_one_line(_run(files["reward"], "--dataset", files["data"], "--fail-on", "right_rejected"), "--fail-on expects metric>threshold")


# --- exit codes -------------------------------------------------------------------------------------------------------------

def test_exit_codes(tmp_path):
    (tmp_path / "exact.py").write_text("def score(completion, answer):\n    return float(completion.strip().endswith(answer))\n", encoding="utf-8")
    (tmp_path / "any.py").write_text("def score(completion, answer):\n    return float(answer in completion)\n", encoding="utf-8")
    (tmp_path / "same.py").write_text("def score(completion, answer):\n    return 0.5 if completion.strip() else 0.0\n", encoding="utf-8")
    data = str(_write(tmp_path / "data.jsonl", ROWS))

    def code(checker, *flags):
        return _run(str(tmp_path / checker), "--dataset", data, "--workers", "1", "--quiet", *flags).exit_code

    assert code("exact.py", "--probes", "N") == 0
    assert code("any.py", "--probes", "O") == 1
    assert code("any.py", "--probes", "O", "--fail-on", "hedging_accepted>1.0") == 0  # only above these rates
    assert code("any.py", "--probes", "O", "--fail-on", "hedging_accepted>0.1") == 1
    assert code("same.py", "--probes", "N") == 2  # not measured, whatever the thresholds
    assert code("same.py", "--probes", "N", "--fail-on", "hedging_accepted>1.0") == 2


def test_dataset_issues_do_not_set_the_exit_code():
    rows = [dict(row) for row in ROWS] + [{"prompt": "An unanswered card.", "answer": ""}]
    for row in rows[:5]:
        row["answer"] = "1000"  # five rows of twelve: a skew that is high
    rows[2]["prompt"] = rows[1]["prompt"]
    r = audit(EXACT, rows, n="all", probes=["N"], workers=1)
    assert {"skewed_answers", "duplicate_prompts", "empty_rows_left_out"} <= {i["id"] for i in r.dataset_issues}
    assert any(i["severity"] == "high" for i in r.dataset_issues) and exit_code(r) == 0
