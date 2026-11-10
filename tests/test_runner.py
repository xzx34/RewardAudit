"""The worker processes: what a checker can do to them, and what the audit makes of it."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys

import pytest

from rewardaudit.errors import AuditError, WorkerError
from rewardaudit.runner import SRC_ROOT, Runner, RunnerPool, process_memory, worker_env
from rewardaudit.types import Kind, Task

TASK = Task("0", "q", "1")


def _spec(tmp_path, source: str) -> dict:
    (tmp_path / "checker.py").write_text(source, encoding="utf-8")
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps({"prompt": f"q{i}", "answer": "1"}) for i in range(4)), encoding="utf-8")
    return {"kind": "function", "path": str(tmp_path / "checker.py"), "dataset": str(tmp_path / "data.jsonl")}


# --- the memory limit ------------------------------------------------------------------------------------------------------

HUNGRY = """
import time

def score(prompt, completion, answer):
    if completion == "allocate":
        chunks = [b"x" * 10_000_000 for _ in range(100)]  # 1 GB, touched
        time.sleep(5)
        return float(len(chunks))
    return 1.0
"""


def test_memory_limit_turns_into_crash_and_restarts(tmp_path):
    with RunnerPool(_spec(tmp_path, HUNGRY), timeout_s=20.0, memory_limit_mb=200) as r:
        v = r.score(TASK, "allocate")
        assert v.kind == Kind.CRASH and "memory limit" in (v.error or "")
        assert v.elapsed_ms < 10_000 and r.restarts == 1
        ok = r.score(TASK, "1")
        assert ok.kind == Kind.SCORE and ok.score == 1.0


def test_a_worker_killed_for_memory_three_times_in_a_row_is_not_restarted(tmp_path):
    with Runner(_spec(tmp_path, HUNGRY), timeout_s=20.0, memory_limit_mb=200) as r:
        assert r.score(TASK, "allocate").kind == Kind.CRASH and r.score(TASK, "1").score == 1.0  # a success starts the count over
        assert r.score(TASK, "allocate").kind == Kind.CRASH and r.score(TASK, "allocate").kind == Kind.CRASH and not r.retired
        v = r.score(TASK, "allocate")
        assert v.kind == Kind.CRASH and r.retired and r.proc is None and r.restarts == 3
        again = r.score(TASK, "1")
        assert again.kind == Kind.CRASH and "memory limit" in again.error and r.restarts == 3 and r.proc is None


def test_the_other_workers_take_a_retired_workers_queue(tmp_path):
    with RunnerPool(_spec(tmp_path, HUNGRY), workers=2, timeout_s=20.0, memory_limit_mb=200) as pool:
        first, second = pool._runners
        first.memory_deaths, first._memory_error = 3, "worker hit the memory limit"
        verdicts = pool.score_many([(TASK, "1")] * 6)
        assert [v.score for v in verdicts] == [1.0] * 6 and second.calls == 6 and first.calls == 0
        assert pool.score(TASK, "1").score == 1.0 and second.calls == 7
        second.memory_deaths, second._memory_error = 3, "worker hit the memory limit"
        rest = pool.score_many([(TASK, "1")] * 3)  # none left: every call says so at once
        assert [v.kind for v in rest] == [Kind.CRASH] * 3 and all("memory limit" in v.error for v in rest)


def test_a_process_memory_can_be_read():
    assert process_memory(os.getpid()) > 1_000_000


# --- timeouts, errors, a worker that dies -------------------------------------------------------------------------------------

BY_REPLY = """
import os
import time

def score(prompt, completion, answer):
    if completion == "slow":
        time.sleep(60)
    if completion == "raise":
        raise ValueError("cannot read this reply")
    if completion == "die":
        os._exit(1)
    return float(completion)
"""


def test_a_slow_call_is_a_timeout_and_the_next_call_gets_a_new_worker(tmp_path):
    with Runner(_spec(tmp_path, BY_REPLY), timeout_s=0.5) as r:
        v = r.score(TASK, "slow")
        assert v.kind == Kind.TIMEOUT and "no result within 0.5s" in v.error and r.restarts == 1
        assert r.score(TASK, "1").score == 1.0


def test_an_error_keeps_the_worker_and_a_death_replaces_it(tmp_path):
    with Runner(_spec(tmp_path, BY_REPLY), timeout_s=20.0) as r:
        v = r.score(TASK, "raise")
        assert v.kind == Kind.ERROR and "ValueError: cannot read this reply" in v.error and r.restarts == 0 and not v.ok
        v = r.score(TASK, "die")
        assert v.kind == Kind.CRASH and v.error == "worker died" and r.restarts == 1
        assert r.score(TASK, "0.5").score == 0.5


def test_score_many_keeps_the_order_of_the_calls_and_reports_each(tmp_path):
    done = []
    with RunnerPool(_spec(tmp_path, BY_REPLY), workers=3, timeout_s=20.0) as pool:
        replies = [str(i) for i in range(30)]
        verdicts = pool.score_many([(TASK, c) for c in replies], progress=done.append)
        assert [v.score for v in verdicts] == [float(c) for c in replies] and done == [1] * 30
        assert pool.calls == 30 and {r.calls > 0 for r in pool._runners} == {True}


def test_a_checker_that_does_not_load_says_why(tmp_path):
    with pytest.raises(WorkerError, match="ValueError: the setting is missing"):
        RunnerPool(_spec(tmp_path, "raise ValueError('the setting is missing')\n"), workers=2).start()
    with pytest.raises(AuditError, match="Scoring file not found"):
        Runner({"kind": "function", "path": str(tmp_path / "absent.py"), "dataset": "d.jsonl"}).start()


def test_every_worker_loads_an_environments_rows_before_the_first_score():
    """A dataset first loaded by a score call would run under the network guard and its checks count against the checker."""
    class Fake:
        def __init__(self, fails=False):
            self.fails, self.loaded = fails, []

        def tasks(self, n, seed):
            self.loaded.append((n, seed))
            if self.fails:
                raise WorkerError("no memory left")
            return ["task"]

    pool = RunnerPool({"kind": "verifiers", "env": "o/n"}, workers=3)
    pool._runners = [Fake(), Fake(), Fake(fails=True)]
    assert pool.tasks(7, 1) == ["task"]
    assert all(r.loaded == [(7, 1)] for r in pool._runners)
    pool = RunnerPool({"kind": "function", "path": "r.py"}, workers=2)  # a scoring function needs only the row it is handed
    pool._runners = [Fake(), Fake()]
    pool.tasks(7, 1)
    assert pool._runners[1].loaded == []


# --- what a checker sees of the worker ----------------------------------------------------------------------------------------

def test_a_checker_reading_stdin_does_not_take_the_next_request(tmp_path):
    source = "import sys\n\ndef score(prompt, completion, answer):\n    return 1.0 if sys.stdin.read() == '' else 0.0\n"
    with Runner(_spec(tmp_path, source), timeout_s=20.0, startup_timeout_s=20.0) as r:
        assert [r.score(TASK, "1").score for _ in range(2)] == [1.0, 1.0]


def test_what_a_checker_prints_does_not_reach_the_protocol(tmp_path):
    source = ("import os, sys\n\nprint('at load')\n\ndef score(prompt, completion, answer):\n    print('while scoring')\n"
              "    print('to the error stream', file=sys.stderr)\n    return 1.0\n")
    with Runner(_spec(tmp_path, source), timeout_s=20.0) as r:
        assert [r.score(TASK, "1").score for _ in range(2)] == [1.0, 1.0] and r.restarts == 0


MANAGER = """
import multiprocessing

def score(prompt, completion, answer):
    with multiprocessing.get_context("spawn").Manager() as m:  # local IPC over a Unix socket, not network
        d = m.dict()
        d["ok"] = completion.strip() == answer
        return 1.0 if d["ok"] else 0.0
"""


def test_local_ipc_is_not_blocked_as_network(tmp_path):
    with RunnerPool(_spec(tmp_path, MANAGER), workers=1, timeout_s=60) as r:
        t = r.tasks(1, 0)[0]
        v = r.score(t, "1")
    assert v.kind == Kind.SCORE and v.score == 1.0 and not v.network


NETWORK = """
import socket

def score(prompt, completion, answer):
    address = ("192.0.2.1", 443)  # the guard refuses all of these before anything leaves this machine
    if completion == "create_connection":
        socket.create_connection(address, timeout=1)
    if completion == "connect":
        socket.socket().connect(address)
    if completion == "connect_ex":
        socket.socket().connect_ex(address)
    if completion == "swallowed":
        try:
            socket.create_connection(address, timeout=1)
        except Exception:
            return 0.25
    return 1.0
"""


def test_the_network_guard_sees_every_way_to_connect_and_each_call_starts_clean(tmp_path):
    with Runner(_spec(tmp_path, NETWORK), timeout_s=20.0) as r:
        for how in ("create_connection", "connect", "connect_ex"):
            v = r.score(TASK, how)
            assert v.kind == Kind.ERROR and v.network and "NetworkBlocked" in v.error, how
        v = r.score(TASK, "swallowed")
        assert v.kind == Kind.SCORE and v.score == 0.25 and v.network
        v = r.score(TASK, "1")
        assert v.score == 1.0 and not v.network


def test_a_worker_says_as_it_stops_which_packages_the_checker_imported(tmp_path):
    with RunnerPool(_spec(tmp_path, "import pytest\n\ndef score(prompt, completion, answer):\n    return 1.0\n"), workers=2) as pool:
        assert pool.score(TASK, "1").score == 1.0
    env = pool.environment
    assert env["python"] == platform.python_version() and env["platform"] == platform.platform()
    assert env["packages"]["pytest"] == importlib.metadata.version("pytest") and "rewardaudit" not in env["packages"]
    assert list(env["packages"]) == sorted(env["packages"], key=str.lower)


def test_the_standard_library_is_no_package_and_a_dead_worker_says_nothing(tmp_path):
    with Runner(_spec(tmp_path, BY_REPLY), timeout_s=20.0) as r:  # os and time, and the checker's own file
        assert r.score(TASK, "1").score == 1.0
    assert r.environment["packages"] == {}
    gone = Runner(_spec(tmp_path, BY_REPLY))
    gone.start()
    gone.proc.kill()
    gone.proc.wait()
    gone.stop()
    assert gone.environment == {}


@pytest.mark.parametrize("modules", [
    ["rewardaudit"],
    ["rewardaudit.worker", "rewardaudit.dataset_checks", "rewardaudit.adapters.function_adapter", "rewardaudit.adapters.verifiers_adapter",
     "rewardaudit.demo"],
], ids=["package", "worker"])
def test_a_worker_needs_none_of_the_tools_own_dependencies(modules):
    """It runs under an environment's interpreter, where only the standard library and the environment can be counted on."""
    code = ("import importlib, sys\n"
            "sys.modules.update(dict.fromkeys(['rich', 'typer', 'click', 'cloudpickle', 'datasets', 'pyarrow'], None))\n"
            f"for m in {modules!r}:\n    importlib.import_module(m)\n")
    subprocess.run([sys.executable, "-c", code], check=True, env=worker_env({}), capture_output=True, text=True)


def test_a_worker_finds_the_package_and_what_the_user_has_on_their_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/from/the/user")
    monkeypatch.setenv("SOME_TOKEN", "t")
    env = worker_env({"kind": "function", "pickled": "x", "sys_path": ["/from/the/caller"]})
    assert env["PYTHONPATH"].split(os.pathsep) == [SRC_ROOT, "/from/the/caller", "/from/the/user"]
    assert env["SOME_TOKEN"] == "t"  # a user's own checker keeps its environment
