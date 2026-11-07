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
import time
import traceback
from typing import Any

from rewardaudit.errors import AuditError


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


def main() -> None:
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

    def send(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    try:
        from rewardaudit.adapters import build_adapter
        from rewardaudit.types import Task

        adapter = build_adapter(spec)
    except AuditError as e:
        send({"ok": False, "fatal": True, "user_error": str(e)})
        return
    except Exception:
        send({"ok": False, "fatal": True, "error": traceback.format_exc()})
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
                tasks = adapter.tasks(req["n"], int(req.get("seed", 0)))
                send({"ok": True, "tasks": [t.to_dict() for t in tasks], "meta": adapter.meta})
            elif op == "dataset_issues":
                from rewardaudit.checks.dataset import run_dataset_checks

                send({"ok": True, "issues": run_dataset_checks(adapter.all_tasks())})
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
                    }
                )
            elif op == "ping":
                send({"ok": True})
            elif op == "exit":
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
            send({"ok": False, "error": traceback.format_exc(limit=6), "network": guard.tripped})


if __name__ == "__main__":
    main()
