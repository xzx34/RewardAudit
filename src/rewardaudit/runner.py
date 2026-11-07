"""Drive worker subprocesses: tasks, scoring with timeouts, four-way verdicts."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from rewardaudit.errors import AuditError, MemoryLimitExceeded, WorkerError
from rewardaudit.types import Kind, Task, Verdict

SRC_ROOT = str(Path(__file__).resolve().parents[1])
MAX_MEMORY_DEATHS = 3  # in a row; a restart reloads the dataset, so workers would go on being killed


_LIBPROC: list[Any] = []


def process_memory(pid: int) -> int | None:
    """Memory a process holds, in bytes: physical footprint on macOS (counts compressed pages), resident
    set on Linux. None where it cannot be read."""
    if sys.platform == "darwin":
        import ctypes
        import ctypes.util

        try:
            if not _LIBPROC:
                _LIBPROC.append(ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"))
            buf = (ctypes.c_uint64 * 20)()  # rusage_info_v0: uuid (2 words), then resident [8], footprint [9]
            if _LIBPROC[0].proc_pid_rusage(pid, 0, ctypes.byref(buf)) != 0:
                return None
            return int(max(buf[8], buf[9]))
        except (OSError, AttributeError):
            return None
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def worker_env(spec: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    paths = [SRC_ROOT, *(spec.get("sys_path") or [])]  # sys_path: where a function handed to the Python API imports from
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


class RunnerPool:
    """Several workers scoring in parallel. Same interface as Runner plus `score_many`."""

    def __init__(self, spec: dict[str, Any], *, workers: int = 1, timeout_s: float = 10.0, python: str | None = None,
                 startup_timeout_s: float = 3600.0, memory_limit_mb: float | None = 4096) -> None:
        self.spec = spec
        self.workers = max(1, workers)
        self.timeout_s = timeout_s
        self._runners = [Runner(spec, timeout_s=timeout_s, python=python, startup_timeout_s=startup_timeout_s,
                                memory_limit_mb=memory_limit_mb) for _ in range(self.workers)]

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._runners[0].start()  # a checker that does not load says so once
        self._on_others(lambda r: r.start())

    def _on_others(self, call: Any) -> None:
        def one(r: Runner) -> None:
            try:
                call(r)
            except WorkerError:  # its first call tries again
                pass

        threads = [threading.Thread(target=one, args=(r,), daemon=True) for r in self._runners[1:]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def stop(self) -> None:
        for r in self._runners:
            r.stop()

    def __enter__(self) -> RunnerPool:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def name(self) -> str:
        return self._runners[0].name

    @property
    def meta(self) -> dict[str, Any]:
        return self._runners[0].meta

    @property
    def max_score(self) -> float | None:
        return self._runners[0].max_score

    @property
    def calls(self) -> int:
        return sum(r.calls for r in self._runners)

    @property
    def restarts(self) -> int:
        return sum(r.restarts for r in self._runners)

    # --- API -----------------------------------------------------------------

    def tasks(self, n: int | None, seed: int = 0) -> list[Task]:
        tasks = self._runners[0].tasks(n, seed)
        if self.spec.get("kind") != "function":
            # an environment finds a task by its row, so the others load now too: a dataset first loaded by a score
            # call would run under the network guard, and its download checks would count against the checker
            self._on_others(lambda r: r.tasks(n, seed))
        return tasks

    def dataset_issues(self) -> list[dict[str, Any]]:
        return self._runners[0].dataset_issues()

    def _live(self) -> Runner:
        return next((r for r in self._runners if not r.retired), self._runners[0])

    def score(self, task: Task, completion: str, timeout_s: float | None = None) -> Verdict:
        return self._live().score(task, completion, timeout_s)

    def score_many(self, items: list[tuple[Task, str]], progress: Any = None) -> list[Verdict]:
        """`progress` is called with the number of calls just finished."""
        results: list[Verdict | None] = [None] * len(items)
        if self.workers == 1 or len(items) < 2:
            for i, (t, c) in enumerate(items):
                results[i] = self._live().score(t, c)
                if progress:
                    progress(1)
            return results  # type: ignore[return-value]
        q: queue.Queue[int] = queue.Queue()
        for i in range(len(items)):
            q.put(i)
        lock = threading.Lock()

        def work(r: Runner) -> None:
            while not r.retired:  # the queue is left to the workers that still run
                try:
                    i = q.get_nowait()
                except queue.Empty:
                    return
                t, c = items[i]
                results[i] = r.score(t, c)
                if progress:
                    with lock:
                        progress(1)

        threads = [threading.Thread(target=work, args=(r,), daemon=True) for r in self._runners]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for i, v in enumerate(results):  # calls no worker was left for
            if v is None:
                results[i] = self._runners[0].score(*items[i])
        return results  # type: ignore[return-value]


class Runner:
    def __init__(
        self,
        spec: dict[str, Any],
        *,
        timeout_s: float = 10.0,
        python: str | None = None,
        startup_timeout_s: float = 3600.0,
        memory_limit_mb: float | None = 4096,
    ) -> None:
        self.spec = spec
        self.timeout_s = timeout_s
        # watched while scoring, on top of what the worker holds when it starts to score (the dataset, say):
        # a worker above it is killed (CRASH)
        self.memory_limit_mb = memory_limit_mb
        self._memory_base: int | None = None
        self.python = python or sys.executable
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen[str] | None = None
        self._q: queue.Queue[str | None] = queue.Queue()
        self.name = "env"
        self.meta: dict[str, Any] = {}
        self.max_score: float | None = None
        self.restarts = 0
        self.calls = 0
        self.memory_deaths = 0
        self._memory_error = ""

    @property
    def retired(self) -> bool:
        return self.memory_deaths >= MAX_MEMORY_DEATHS

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.proc = subprocess.Popen([self.python, "-m", "rewardaudit.worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True,
                                     encoding="utf-8", env=worker_env(self.spec), cwd=self.spec.get("cwd"))
        self._memory_base = None
        self._q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self._q), daemon=True).start()
        try:
            self._send(self.spec)
            hello = self._recv(self.startup_timeout_s)
        except (WorkerError, BrokenPipeError, OSError) as e:
            self._kill()
            raise WorkerError(str(e)) from None
        if hello is None:
            self.stop()
            raise WorkerError("worker did not start in time")
        if not hello.get("ok"):
            self.stop()
            if hello.get("user_error"):
                raise AuditError(hello["user_error"])
            raise WorkerError("worker failed to load the checker:\n" + str(hello.get("error")))
        self.name = hello.get("name", "env")
        self.meta = hello.get("meta") or {}
        self.max_score = hello.get("max_score")

    @staticmethod
    def _reader(proc: subprocess.Popen[str], q: queue.Queue[str | None]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(json.dumps({"op": "exit"}) + "\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def _kill(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            self.proc = None

    def _restart(self) -> None:
        """A worker that does not come back (no memory left, say) leaves the next call to try again."""
        self._kill()
        self.restarts += 1
        try:
            self.start()
        except WorkerError:
            self._kill()

    def __enter__(self) -> Runner:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # --- protocol ------------------------------------------------------------

    def _recv(self, timeout: float, watch_memory: bool = False) -> dict[str, Any] | None:
        """None on timeout; raises WorkerError if the worker died, MemoryLimitExceeded if it outgrew the limit."""
        deadline = time.monotonic() + timeout
        limit = None
        if watch_memory and self.memory_limit_mb and self.proc is not None:
            if self._memory_base is None:
                self._memory_base = process_memory(self.proc.pid) or 0
            limit = self._memory_base + self.memory_limit_mb * 2**20
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._q.get(timeout=min(remaining, 0.1 if limit else 0.5))
            except queue.Empty:
                used = process_memory(self.proc.pid) if limit and self.proc is not None else None
                if used is not None and limit is not None and used > limit:
                    raise MemoryLimitExceeded(f"worker went over the memory limit ({self.memory_limit_mb:.0f} MB while "
                                              f"scoring; {used / 2**20:.0f} MB in use) and was killed") from None
                continue
            if line is None:
                raise WorkerError("worker exited")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # stray output; keep reading

    def _send(self, req: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _request(self, req: dict[str, Any], timeout: float, watch_memory: bool = False) -> dict[str, Any] | None:
        if self.proc is None:
            self.start()
        try:
            self._send(req)
            return self._recv(timeout, watch_memory)
        except (BrokenPipeError, OSError):
            raise WorkerError("worker pipe closed") from None

    def _loaded(self, req: dict[str, Any], what: str) -> dict[str, Any]:
        resp = self._request(req, self.startup_timeout_s)
        if resp is None:
            raise WorkerError(f"timed out {what}")
        if resp.get("user_error"):
            raise AuditError(resp["user_error"])
        if not resp.get("ok"):
            raise WorkerError(f"{what} failed:\n{resp.get('error')}")
        return resp

    # --- public API ----------------------------------------------------------

    def tasks(self, n: int | None, seed: int = 0) -> list[Task]:
        resp = self._loaded({"op": "tasks", "n": n, "seed": seed}, "loading tasks")
        self.meta = resp.get("meta") or self.meta  # adapters fill in what they learn while loading tasks
        return [Task.from_dict(t) for t in resp["tasks"]]

    def dataset_issues(self) -> list[dict[str, Any]]:
        return self._loaded({"op": "dataset_issues"}, "checking the dataset")["issues"]

    def score(self, task: Task, completion: str, timeout_s: float | None = None) -> Verdict:
        timeout = timeout_s or self.timeout_s
        self.calls += 1
        if self.retired:
            return Verdict(Kind.CRASH, error=self._memory_error)
        t0 = time.perf_counter()
        try:
            resp = self._request({"op": "score", "task": task.to_dict(), "completion": completion}, timeout, True)
        except MemoryLimitExceeded as e:
            self.memory_deaths += 1
            self._memory_error = str(e)
            self._kill() if self.retired else self._restart()
            return Verdict(Kind.CRASH, error=str(e), elapsed_ms=(time.perf_counter() - t0) * 1000)
        except WorkerError:
            self._restart()
            return Verdict(Kind.CRASH, error="worker died", elapsed_ms=(time.perf_counter() - t0) * 1000)
        elapsed = (time.perf_counter() - t0) * 1000
        if resp is None:
            self._restart()
            return Verdict(Kind.TIMEOUT, error=f"no result within {timeout:.1f}s", elapsed_ms=elapsed)
        self.memory_deaths = 0
        if not resp.get("ok"):
            return Verdict(
                Kind.ERROR,
                error=str(resp.get("error", ""))[-2000:],
                elapsed_ms=elapsed,
                network=bool(resp.get("network")),
            )
        score = resp.get("score")
        return Verdict(
            Kind.SCORE,
            score=float(score) if score is not None else None,
            sub_scores=resp.get("sub_scores"),
            parsed=resp.get("parsed"),
            elapsed_ms=float(resp.get("elapsed_ms", elapsed)),
            network=bool(resp.get("network")),
        )
