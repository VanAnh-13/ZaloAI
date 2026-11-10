"""Run model-written code in a separate Python process with a wall-clock timeout.

A fresh process per execution keeps runs isolated (no state leaks between
questions, a runaway loop is simply killed). The child loads the workspace
pickle itself, see sandbox_child.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# folder that contains the `datamate` package, so the child can import it
_PKG_ROOT = str(Path(__file__).resolve().parents[2])


@dataclass
class ExecResult:
    stdout: str
    error: str | None
    answer: dict | None  # {"type", "value", "reason"} when the code called answer(...)
    timed_out: bool
    elapsed_ms: int


@dataclass
class ExecContext:
    workspace: str | None  # pickled Workspace; None when preparing it failed (T empty, S None)
    data_dir: str  # raw files of the business (read-only)
    calendar: str | None
    as_of: str
    question: str


class Sandbox:
    def __init__(self, timeout_s: float = 30.0, python: str = sys.executable):
        self.timeout_s = timeout_s
        self.python = python

    def run(self, code: str, ctx: ExecContext) -> ExecResult:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="dm_exec_") as tmp:
            result_path = Path(tmp) / "result.json"
            req = {"code": code, "result": str(result_path), **ctx.__dict__}
            env = dict(os.environ)
            env["PYTHONPATH"] = _PKG_ROOT + os.pathsep + env.get("PYTHONPATH", "")
            env.update(
                PYTHONUTF8="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
            )
            try:
                p = subprocess.run(
                    [self.python, "-m", "datamate.agent.sandbox_child"],
                    input=json.dumps(req, ensure_ascii=False).encode("utf-8"),
                    capture_output=True,
                    timeout=self.timeout_s,
                    cwd=tmp,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return ExecResult(
                    "",
                    f"TIMEOUT: code chạy quá {self.timeout_s:.0f} giây và đã bị dừng.",
                    None,
                    True,
                    int((time.monotonic() - t0) * 1000),
                )
            ms = int((time.monotonic() - t0) * 1000)
            if not result_path.exists():
                err = p.stderr.decode("utf-8", "replace")[-1500:]
                return ExecResult("", f"sandbox lỗi (exit {p.returncode}): {err}", None, False, ms)
            res = json.loads(result_path.read_text(encoding="utf-8"))
        return ExecResult(res.get("stdout", ""), res.get("error"), res.get("answer"), False, ms)
