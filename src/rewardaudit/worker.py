"""Worker process: hosts the adapter and answers JSON-line requests on stdin/stdout.

Runs under whichever interpreter the environment is installed in. The first line on stdin is the spec. Everything the
checker prints goes to stderr so stdout stays a clean protocol channel.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import time
import traceback
from typing import Any

from rewardaudit.errors import AuditError
from rewardaudit.types import MARKER_MODULE


class NetworkBlocked(RuntimeError):
    pass


def _local_ipc(sock: socket.socket) -> bool:
    """Unix-domain sockets are local inter-process communication (multiprocessing managers, pools), not network."""
    return getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX


class _NetworkGuard:
    """Raise on any outbound network connection attempt while active; remember that it happened.
    Unix-domain sockets are allowed: checkers use them for multiprocessing-based timeouts."""

    def __init__(self) -> None:
        self.active = False
        self.tripped = False
        self._orig_connect = socket.socket.connect
        self._orig_connect_ex = socket.socket.connect_ex
        self._orig_create = socket.create_connection

    def install(self) -> None:
        guard = self

        def connect(sock: socket.socket, *a: Any, **k: Any) -> Any:
            if guard.active and not _local_ipc(sock):
                guard.tripped = True
                raise NetworkBlocked("network access during scoring")
            return guard._orig_connect(sock, *a, **k)

        def connect_ex(sock: socket.socket, *a: Any, **k: Any) -> Any:
            if guard.active and not _local_ipc(sock):
                guard.tripped = True
                raise NetworkBlocked("network access during scoring")
            return guard._orig_connect_ex(sock, *a, **k)

        def create_connection(*a: Any, **k: Any) -> Any:
            if guard.active:
                guard.tripped = True
                raise NetworkBlocked("network access during scoring")
            return guard._orig_create(*a, **k)

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
        socket.create_connection = create_connection  # type: ignore[assignment]


class _MarkerWatch:
    """Notices an import of MARKER_MODULE, in this process or a forked child, even if the checker swallows the ImportError
    that follows: a checker that runs a reply's text usually sits inside a try. A file, since a child shares no memory."""

    def __init__(self) -> None:
        self.file = tempfile.TemporaryFile()

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == MARKER_MODULE:
            os.write(self.file.fileno(), b"x")

    def take(self) -> bool:
        seen = os.fstat(self.file.fileno()).st_size > 0
        if seen:
            os.ftruncate(self.file.fileno(), 0)
        return seen


def _top_level(modules: Any) -> set[str]:
    return {name.partition(".")[0] for name in modules}


def _environment(not_checker: set[str]) -> dict[str, Any]:
    """What the checker ran under: this Python, this platform, and {distribution: version} for the third-party modules
    imported in this worker as the checker loaded and scored. That leaves out the standard library and `not_checker`:
    what was there before the checker loaded (this package, `__main__`, what .pth files import at startup), cloudpickle,
    which carries a function over from Python, and what reading the rows imported. A module no installed distribution
    provides, such as the checker's own file, is not listed; one that several provide lists them all."""
    import importlib.metadata
    import platform

    names = _top_level(set(sys.modules)) - not_checker - sys.stdlib_module_names  # a copy made at once: threads import too
    packages: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        # the modules it provides: top_level.txt, or else the first part of each path in RECORD. packages_distributions()
        # does the same through a path object for every file, a tenth of a second per worker in a venv of a hundred
        provides = (dist.read_text("top_level.txt") or "").split() or [
            line.split(",")[0].split("/")[0].partition(".")[0] for line in (dist.read_text("RECORD") or "").splitlines()]
        if names.intersection(provides):
            packages.setdefault(dist.metadata["Name"], dist.version)  # the first on sys.path is the one imported
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def main() -> None:
    not_checker = _top_level(sys.modules) | {"cloudpickle"}  # and reading_rows adds what reading the dataset imports
    out = sys.stdout
    sys.stdout = sys.stderr  # anything the checker prints must not corrupt the protocol
    # and anything it reads must not wait on the protocol: a load that asks a question gets end-of-file, not a hang
    requests = os.fdopen(os.dup(0), encoding="utf-8")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    spec = json.loads(requests.readline())
    guard = _NetworkGuard()
    guard.install()
    marker = _MarkerWatch()
    sys.meta_path.append(marker)

    def send(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    def reading_rows(call: Any, *args: Any) -> Any:
        """What reading the dataset imports (pyarrow, datasets and theirs) is not the checker's."""
        before = set(sys.modules)
        try:
            return call(*args)
        finally:
            not_checker.update(_top_level(set(sys.modules) - before))

    try:
        from rewardaudit.adapters import build_adapter
        from rewardaudit.types import Task

        adapter = build_adapter(spec)
    except AuditError as e:
        send({"ok": False, "user_error": str(e)})
        return
    except Exception:
        send({"ok": False, "error": traceback.format_exc()})
        return
    send({"ok": True, "ready": True, "name": adapter.name, "meta": adapter.meta, "max_score": adapter.max_score()})

    for line in requests:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        op = req.get("op")
        try:
            if op == "tasks":
                tasks = reading_rows(adapter.tasks, req["n"], int(req.get("seed", 0)))
                send({"ok": True, "tasks": [t.to_dict() for t in tasks], "meta": adapter.meta})
            elif op == "dataset_issues":
                from rewardaudit.dataset_checks import run_dataset_checks

                rows = reading_rows(adapter.all_tasks)  # loading tells how many rows were left out
                send({"ok": True, "issues": run_dataset_checks(rows, left_out=adapter.meta.get("empty_rows_left_out", 0))})
            elif op == "score":
                task = Task.from_dict(req["task"])
                guard.tripped = False
                guard.active = True
                buf = io.StringIO()
                t0 = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        raw = adapter.score_raw(task, req["completion"])
                finally:
                    guard.active = False
                elapsed = (time.perf_counter() - t0) * 1000
                send(
                    {
                        "ok": True,
                        "score": raw.score,
                        "sub_scores": raw.sub_scores,
                        "parsed": raw.parsed,
                        "elapsed_ms": elapsed,
                        "network": guard.tripped or raw.network,
                        "reached_import": marker.take(),
                    }
                )
            elif op == "exit":
                send({"ok": True, **_environment(not_checker)})
                break
            else:
                send({"ok": False, "error": f"unknown op {op!r}"})
        except AuditError as e:
            guard.active = False
            send({"ok": False, "user_error": str(e)})
        except NetworkBlocked as e:
            guard.active = False
            send({"ok": False, "error": f"{type(e).__name__}: {e}", "network": True})
        except Exception:
            guard.active = False
            send({"ok": False, "error": traceback.format_exc(limit=6), "network": guard.tripped, "reached_import": marker.take()})


if __name__ == "__main__":
    main()
