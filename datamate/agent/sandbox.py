"""Run model-written code in sandbox sessions.

A session is one worker process (agent/sandbox_child.py) that serves one attempt at one
question, like a notebook kernel:
- the shop's workspace is loaded once per session, not once per code block;
- variables persist between the steps of the attempt;
- the worker starts in the background as soon as the session is opened, i.e. while the model
  is still writing its first code block, so the sandbox adds no waiting time;
- a run that exceeds the timeout kills the worker's whole process group; the session restarts
  transparently on the next step and the result tells the model that its variables are gone;
- file / network / process guards and resource limits: see sandbox_child.py.

    with sandbox.open(ctx) as session:
        r1 = session.run("p = dm.month(7, 2025); S.revenue(p)")
        r2 = session.run("answer('number', S.revenue(p))")   # `p` is still there
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# folder that contains the `datamate` package, so the worker can import it
_PKG_ROOT = str(Path(__file__).resolve().parents[2])
_EOF = object()


@dataclass
class ExecResult:
    stdout: str
    error: str | None
    answer: dict | None  # {"type", "value", "reason"} when the code called answer(...)
    timed_out: bool
    elapsed_ms: int
    restarted: bool = False  # the worker was restarted before this run: earlier variables are gone


@dataclass
class ExecContext:
    workspace: str | None  # pickled Workspace; None when preparing it failed (T empty, S None)
    data_dir: str  # raw files of the business (read-only)
    calendar: str | None
    as_of: str
    question: str


@dataclass
class SandboxConfig:
    timeout_s: float = 30.0
    start_timeout_s: float = 120.0  # loading the workspace in a fresh worker
    guard: bool = True
    mem_mb: int = 8192  # address-space ceiling of one worker (POSIX)
    fsize_mb: int = 256
    deny_read: tuple[str, ...] = ()  # extra folders the code must not read (results, workspace cache…)
    python: str = field(default_factory=lambda: sys.executable)


class _Worker:
    """One child process and the two threads that drain its pipes."""

    def __init__(self, cfg: SandboxConfig, init: dict):
        env = dict(os.environ)
        env["PYTHONPATH"] = _PKG_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env.update(
            PYTHONUTF8="1",
            PYTHONHASHSEED="0",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        self.proc = subprocess.Popen(
            [cfg.python, "-m", "datamate.agent.sandbox_child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=init["scratch"],
            env=env,
            **kwargs,
        )
        self.replies: queue.Queue = queue.Queue()
        self.stderr_tail = b""
        self._readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for t in self._readers:
            t.start()
        self.ready: bool | None = None  # None = still loading
        self.start_error: str | None = None
        self.load_ms = 0
        self.send(init)

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                try:
                    self.replies.put(json.loads(line.decode("utf-8")))
                except ValueError:
                    continue
        except (OSError, ValueError):  # pipe closed under us while the worker is being killed
            pass
        self.replies.put(_EOF)

    def _read_stderr(self) -> None:
        try:
            for chunk in iter(lambda: self.proc.stderr.read(4096), b""):
                self.stderr_tail = (self.stderr_tail + chunk)[-3000:]
        except (OSError, ValueError):
            pass

    def send(self, obj: dict) -> bool:
        try:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=True).encode("ascii") + b"\n")
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def receive(self, timeout: float):
        """A reply dict, _EOF when the worker died, or None on timeout."""
        try:
            return self.replies.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def wait_ready(self, timeout: float) -> bool:
        if self.ready is None:
            msg = self.receive(timeout)
            if isinstance(msg, dict) and msg.get("ready"):
                self.ready, self.load_ms = True, int(msg.get("load_ms", 0))
            else:
                self.ready = False
                why = (
                    "quá thời gian nạp dữ liệu" if msg is None else (msg.get("error") if isinstance(msg, dict) else "")
                )
                self.start_error = (
                    why or self.stderr_tail.decode("utf-8", "replace")[-800:] or "worker dừng khi khởi động"
                )
                self.kill()
        return bool(self.ready)

    def kill(self) -> None:
        """Stop the worker and anything it started."""
        try:
            if os.name == "posix":
                import signal

                os.killpg(self.proc.pid, signal.SIGKILL)
            else:
                self.proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        for t in self._readers:  # they stop at end-of-file once the process is gone
            t.join(timeout=2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except Exception:
                pass


class Session:
    """One attempt's notebook. Not thread-safe: one session belongs to one question thread."""

    def __init__(self, cfg: SandboxConfig, ctx: ExecContext, allow_read: list[str]):
        self.cfg, self.ctx = cfg, ctx
        self._allow_read = allow_read
        self._scratch = tempfile.mkdtemp(prefix="dm_exec_")
        self._worker: _Worker | None = None
        self._starts = 0
        self.runs = 0
        self.exec_ms = 0
        self._start()  # non-blocking: the worker loads while the model writes its first code

    def _start(self) -> None:
        guard = None
        if self.cfg.guard:
            split_root = str(Path(self.ctx.data_dir).resolve().parent.parent)
            guard = {"deny_read": [split_root, *self.cfg.deny_read], "allow_read": self._allow_read}
        init = {
            "op": "init",
            "workspace": self.ctx.workspace,
            "data_dir": self.ctx.data_dir,
            "calendar": self.ctx.calendar,
            "scratch": self._scratch,
            "guard": guard,
            "limits": {
                "mem_mb": self.cfg.mem_mb,
                "fsize_mb": self.cfg.fsize_mb,
                "cpu_s": int(self.cfg.timeout_s * 20),
            },
        }
        self._worker = _Worker(self.cfg, init)
        self._worker.send({"op": "begin", "as_of": self.ctx.as_of, "question": self.ctx.question})
        self._starts += 1

    def run(self, code: str) -> ExecResult:
        t0 = time.monotonic()
        restarted = False
        if self._worker is None:  # killed by a timeout or died: start over, variables are gone
            self._start()
            restarted = True
        w = self._worker
        if not w.wait_ready(self.cfg.start_timeout_s):
            self._worker = None
            return self._result("", f"sandbox không khởi động được: {w.start_error}", None, False, t0, restarted)
        if not w.send({"op": "run", "code": code, "timeout": self.cfg.timeout_s}):
            self._drop()
            return self._result("", "sandbox đã dừng trước khi chạy code", None, False, t0, restarted)
        msg = w.receive(self.cfg.timeout_s)
        if msg is None:
            self._drop()
            return self._result(
                "",
                f"TIMEOUT: code chạy quá {self.cfg.timeout_s:.0f} giây và đã bị dừng. Phiên được khởi động lại: "
                "các biến đã tạo không còn.",
                None,
                True,
                t0,
                restarted,
            )
        if msg is _EOF:
            tail = w.stderr_tail.decode("utf-8", "replace")[-800:]
            self._drop()
            return self._result(
                "",
                "sandbox dừng đột ngột (hết bộ nhớ hoặc code tự thoát tiến trình). Phiên được khởi động lại: "
                f"các biến đã tạo không còn.\n{tail}".rstrip(),
                None,
                False,
                t0,
                restarted,
            )
        return self._result(msg.get("stdout", ""), msg.get("error"), msg.get("answer"), False, t0, restarted)

    def _result(self, stdout, error, answer, timed_out, t0, restarted) -> ExecResult:
        ms = int((time.monotonic() - t0) * 1000)
        self.runs += 1
        self.exec_ms += ms
        return ExecResult(stdout, error, answer, timed_out, ms, restarted)

    def _drop(self) -> None:
        if self._worker is not None:
            self._worker.kill()
            self._worker = None

    def close(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            w.send({"op": "exit"})
            w.kill()
        shutil.rmtree(self._scratch, ignore_errors=True)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Sandbox:
    def __init__(self, timeout_s: float = 30.0, python: str | None = None, config: SandboxConfig | None = None):
        self.cfg = config or SandboxConfig(timeout_s=timeout_s)
        if python:
            self.cfg.python = python
        self.timeout_s = self.cfg.timeout_s

    def open(self, ctx: ExecContext, allow_read: list[str] | None = None) -> Session:
        """allow_read: files of the split that the code may read besides the shop folder."""
        allow = [ctx.data_dir] + ([ctx.calendar] if ctx.calendar else []) + list(allow_read or [])
        return Session(self.cfg, ctx, allow)

    def run(self, code: str, ctx: ExecContext) -> ExecResult:
        """One-shot convenience: a fresh session for a single code block."""
        with self.open(ctx) as session:
            return session.run(code)
