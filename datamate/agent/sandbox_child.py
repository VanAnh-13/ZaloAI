"""Sandbox worker: one process = one notebook-like session for one attempt at one question.

Started by agent/sandbox.py as `python -m datamate.agent.sandbox_child`. It loads the shop's
workspace once, then runs every code block of the attempt in the same namespace, so variables
survive from one step to the next (no re-typing, no reloading).

Protocol — one JSON object per line on the process's original stdin/stdout:
    parent -> {"op": "init", "workspace", "data_dir", "calendar", "scratch", "guard", "limits"}
    worker -> {"ready": true, "load_ms": 123}
    parent -> {"op": "begin", "as_of": "2026-05-15", "question": "..."}
    parent -> {"op": "run", "code": "...", "timeout": 30}
    worker -> {"stdout": "...", "error": null, "answer": null, "ms": 12}

Namespace given to the code: S, T, UNITS, DATA_END, CAL, AS_OF, QUESTION, DATA_DIR, pd, np, dm
and answer(type, value=None, reason=None).

Protection (the code comes from a model, not from an attacker — the point is that a mistake
cannot damage the run or peek at files it must not see):
- file writes only inside the session's scratch folder; reads of the split folder limited to
  this shop's files (no answers.jsonl, no other shop, no /result);
- no subprocess, no network, no signals to other processes;
- memory, CPU time and file-size limits (POSIX), a watchdog alarm per run;
- the protocol pipes are private file descriptors, so print()/input() cannot corrupt them.
"""

from __future__ import annotations

import ast
import contextlib
import difflib
import io
import json
import os
import sys
import time
import traceback
from datetime import date, datetime
from decimal import Decimal

MAX_STDOUT = 20000
MAX_VALUE_CHARS = 4000


class _Submitted(BaseException):
    """Raised by answer(); BaseException so a bare `except Exception` in model code
    cannot swallow it."""


def to_jsonable(v):
    import numpy as np
    import pandas as pd

    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, np.generic):
        return to_jsonable(v.item())
    if isinstance(v, float):
        return None if v != v else v
    if isinstance(v, int):
        return v
    if v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(v)
        return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, pd.DataFrame):
        df = v if isinstance(v.index, pd.RangeIndex) else v.reset_index()
        return {
            "columns": [str(c) for c in df.columns],
            "rows": [[to_jsonable(x) for x in r] for r in df.itertuples(index=False, name=None)],
        }
    if isinstance(v, pd.Series):
        return [to_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [to_jsonable(x) for x in list(v)]
    if isinstance(v, set):
        return [to_jsonable(x) for x in sorted(v, key=str)]
    return str(v)


# ------------------------------------------------------------------ guard


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _norm_path(p) -> str | None:
    if isinstance(p, int):  # an already open file descriptor
        return None
    try:
        return os.path.normcase(os.path.abspath(os.fsdecode(p)))
    except (TypeError, ValueError):
        return None


class Guard:
    """Audit hook that refuses what model code has no reason to do. Installed once per worker
    (audit hooks cannot be removed), after the workspace is loaded."""

    _WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    _MUTATE = {  # event -> indexes of the path arguments
        "os.remove": (0,),
        "os.rmdir": (0,),
        "os.mkdir": (0,),
        "os.rename": (0, 1),
        "os.truncate": (0,),
        "os.chmod": (0,),
        "os.chown": (0,),
        "os.link": (0, 1),
        "os.symlink": (1,),
        "shutil.rmtree": (0,),
        "shutil.move": (0, 1),
        "shutil.copyfile": (1,),
        "shutil.copytree": (1,),
        "shutil.unpack_archive": (1,),
    }
    _LIST = {"os.listdir", "os.scandir", "os.walk", "glob.glob", "glob.glob/2"}
    _PROCESS = {
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.forkpty",
        "os.startfile",
        "pty.spawn",
        "os.kill",
        "os.killpg",
        "signal.pthread_kill",
    }
    _NETWORK = {
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "urllib.Request",
        "http.client.connect",
        "ftplib.connect",
        "smtplib.connect",
    }

    def __init__(self, scratch: str, deny_read: list[str], allow_read: list[str]):
        def roots(paths):
            out = []
            for p in paths:
                if p:
                    out += [_norm_path(p), os.path.normcase(os.path.realpath(p))]
            return tuple(dict.fromkeys(x for x in out if x))

        self.scratch = roots([scratch])
        self.deny = roots(deny_read)
        # the interpreter, its packages and this package stay readable even under a denied folder
        # (not the whole sys.path: the repo root is on it and may contain the data folders)
        package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.allow = roots(list(allow_read) + [sys.prefix, sys.base_prefix, sys.exec_prefix, package])
        self.devnull = _norm_path(os.devnull)
        self.active = True

    def _check_read(self, raw) -> None:
        p = _norm_path(raw)
        if p and _under(p, self.deny) and not _under(p, self.allow) and not _under(p, self.scratch):
            raise PermissionError(
                f"sandbox: không được đọc {os.fsdecode(raw)!r} — chỉ dùng dữ liệu của shop này (T, S, DATA_DIR)"
            )

    def _check_write(self, raw) -> None:
        p = _norm_path(raw)
        if p and p != self.devnull and not _under(p, self.scratch):
            raise PermissionError(
                f"sandbox: không được ghi/xoá {os.fsdecode(raw)!r} — chỉ ghi được file tạm trong thư mục hiện tại "
                "(đường dẫn tương đối); kết quả thì in ra bằng print()"
            )

    def __call__(self, event: str, args) -> None:
        if not self.active:
            return
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & self._WRITE_FLAGS
            )
            if writing:
                self._check_write(path)
            else:
                self._check_read(path)
        elif event in self._MUTATE:
            for i in self._MUTATE[event]:
                if i < len(args):
                    self._check_write(args[i])
        elif event in self._LIST:
            if args:
                self._check_read(args[0])
        elif event == "sqlite3.connect":
            if args and args[0] not in (":memory:", ""):
                raw = str(args[0])
                self._check_read(raw[5:].split("?")[0] if raw.startswith("file:") else raw)
        elif event in self._PROCESS:
            raise PermissionError("sandbox: không được chạy tiến trình khác (subprocess/os.system/kill)")
        elif event in self._NETWORK:
            raise PermissionError("sandbox: không có mạng — chỉ dùng dữ liệu của shop")


def apply_limits(limits: dict) -> None:
    """Memory / CPU / file-size ceilings. POSIX only; silently skipped elsewhere (Windows dev)."""
    try:
        import resource
    except ImportError:
        return
    mb = 1024 * 1024
    wanted = (
        ("RLIMIT_AS", int(limits.get("mem_mb") or 0) * mb),
        ("RLIMIT_CPU", int(limits.get("cpu_s") or 0)),
        ("RLIMIT_FSIZE", int(limits.get("fsize_mb") or 0) * mb),
        ("RLIMIT_CORE", 0 if limits else None),
    )
    for name, value in wanted:
        if value is None or (value == 0 and name != "RLIMIT_CORE") or not hasattr(resource, name):
            continue
        try:
            _, hard = resource.getrlimit(getattr(resource, name))
            soft = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(getattr(resource, name), (soft, hard))
        except (ValueError, OSError):
            pass


# ------------------------------------------------------------------ kernel


def _show(value) -> str:
    """What a notebook would display for the last expression of a cell."""
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        body = value.head(30).to_string()
        return body + (
            f"\n… ({len(value):,} dòng × {value.shape[1]} cột; chỉ hiện 30 dòng đầu)" if len(value) > 30 else ""
        )
    if isinstance(value, pd.Series):
        body = value.head(40).to_string()
        return body + (f"\n… ({len(value):,} phần tử; chỉ hiện 40 đầu)" if len(value) > 40 else "")
    text = repr(value)
    return (
        text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + f"… (+{len(text) - MAX_VALUE_CHARS} ký tự)"
    )


def _clip_output(out: str) -> str:
    if len(out) <= MAX_STDOUT:
        return out
    half = MAX_STDOUT // 2
    return out[:half] + f"\n… (bỏ {len(out) - MAX_STDOUT} ký tự) …\n" + out[-half:]


class Kernel:
    def __init__(self, init: dict):
        import numpy as np
        import pandas as pd

        from datamate import dm
        from datamate.workspace.cache import load_cached

        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 40)
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.max_rows", 60)

        ws = load_cached(init["workspace"]) if init.get("workspace") else None
        biz = ws.business if ws else None
        cal = None
        if init.get("calendar"):
            cal = pd.read_csv(init["calendar"])
            cal["date"] = pd.to_datetime(cal["date"])
        self.state: dict = {}
        self.cells: dict[str, list[str]] = {}  # cell file name -> source lines, for tracebacks
        self.n = 0

        def answer(type, value=None, reason=None):  # noqa: A002 - name shown to the model
            self.state["answer"] = {
                "type": type,
                "value": to_jsonable(value),
                "reason": None if reason is None else str(reason),
            }
            raise _Submitted

        self.ns = {
            "__name__": "__main__",
            "pd": pd,
            "np": np,
            "dm": dm,
            "S": ws.sales if ws else None,
            "T": biz.frames() if biz else {},
            "UNITS": biz.units if biz else pd.DataFrame(columns=["product", "unit", "factor", "base_unit"]),
            "DATA_END": biz.data_end if biz else None,
            "CAL": cal,
            "AS_OF": None,
            "QUESTION": "",
            "DATA_DIR": init.get("data_dir"),
            "answer": answer,
        }
        self._builtin_names = set(self.ns)

    def begin(self, as_of: str, question: str) -> None:
        self.ns["AS_OF"] = date.fromisoformat(as_of)
        self.ns["QUESTION"] = question

    # -- execution -----------------------------------------------------
    def run(self, code: str) -> dict:
        t0 = time.monotonic()
        self.n += 1
        name = f"<lượt {self.n}>"
        self.cells[name] = code.splitlines()
        self.state.pop("answer", None)
        buf, err = io.StringIO(), None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                self._exec(code, name)
            except _Submitted:
                pass
            except SystemExit:
                pass
            except BaseException as e:  # report every model error back to the model
                err = self._traceback(e)
        return {
            "stdout": _clip_output(buf.getvalue()),
            "error": err,
            "answer": self.state.get("answer"),
            "ms": int((time.monotonic() - t0) * 1000),
        }

    def _exec(self, code: str, name: str) -> None:
        tree = ast.parse(code, name, "exec")
        last = tree.body[-1] if tree.body else None
        if isinstance(last, ast.Expr):  # notebook behaviour: show the value of the last expression
            tree.body = tree.body[:-1]
            exec(compile(tree, name, "exec"), self.ns)
            value = eval(compile(ast.Expression(last.value), name, "eval"), self.ns)
            if value is not None:
                print(_show(value))
        else:
            exec(compile(tree, name, "exec"), self.ns)

    # -- errors --------------------------------------------------------
    def _traceback(self, exc: BaseException) -> str:
        """Only the frames inside the model's code, the exception line, and a hint when one helps."""
        where = []
        if isinstance(exc, SyntaxError) and exc.filename in self.cells:
            where.append(f"  {exc.filename} dòng {exc.lineno}: {(exc.text or '').strip()}")
        for f in traceback.extract_tb(exc.__traceback__):
            if f.filename in self.cells:
                src = self.cells[f.filename]
                line = src[f.lineno - 1].strip() if f.lineno and 0 < f.lineno <= len(src) else ""
                where.append(f"  {f.filename} dòng {f.lineno}: {line}")
        msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        parts = ["Traceback (code của bạn):", *where[-4:], msg[-1500:]]
        hint = self._hint(exc)
        if hint:
            parts.append("Gợi ý: " + hint)
        return "\n".join(parts)

    def _frames(self) -> dict[str, list[str]]:
        """Name -> column names of every DataFrame the code can reach."""
        import pandas as pd

        out = {f"T[{n!r}]": [str(c) for c in df.columns] for n, df in (self.ns.get("T") or {}).items()}
        S = self.ns.get("S")
        if S is not None:
            out["S.orders"] = list(map(str, S.orders.columns))
            out["S.lines"] = list(map(str, S.lines.columns))
        for n, v in self.ns.items():
            if isinstance(v, pd.DataFrame) and n not in self._builtin_names:
                out[n] = [str(c) for c in v.columns]
        return out

    def _hint(self, exc: BaseException) -> str | None:
        try:
            if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], str):
                missing = exc.args[0]
                frames = self._frames()
                if missing in (self.ns.get("T") or {}) or not frames:
                    return None
                tables = list((self.ns.get("T") or {}).keys())
                near_t = difflib.get_close_matches(missing, tables, n=3, cutoff=0.6)
                hits = []
                for name, cols in frames.items():
                    for c in difflib.get_close_matches(missing, cols, n=2, cutoff=0.6):
                        hits.append(f"{name}[{c!r}]")
                if near_t:
                    return f"không có bảng {missing!r}; các bảng: {tables}"
                if hits:
                    return f"không có cột {missing!r}; cột gần giống: {', '.join(hits[:6])}"
                return (
                    f"không có cột/khoá {missing!r}; in df.columns để xem tên cột đúng "
                    "(tên cột giữ nguyên như file gốc)"
                )
            if isinstance(exc, NameError):
                mine = sorted(n for n in self.ns if n not in self._builtin_names and not n.startswith("_"))
                return (
                    "biến chưa được định nghĩa trong phiên này"
                    + (f"; biến bạn đã tạo: {', '.join(mine[:25])}" if mine else "")
                    + "; có sẵn: S, T, UNITS, DATA_END, CAL, AS_OF, QUESTION, DATA_DIR, pd, np, dm, answer"
                )
            if isinstance(exc, AttributeError) and "NoneType" in str(exc) and self.ns.get("S") is None:
                return "`S` là None với shop này: làm trực tiếp trên các bảng `T`"
            if isinstance(exc, MemoryError):
                return "phép tính dùng quá nhiều bộ nhớ (thường do merge nhân chéo): lọc trước rồi mới ghép"
        except Exception:
            return None
        return None


# ------------------------------------------------------------------ protocol


def _watchdog(seconds: float) -> None:
    """Kill this worker if one run overstays (backstop: the parent kills it first). POSIX only."""
    try:
        import signal

        signal.alarm(int(seconds) if seconds > 0 else 0)
    except (ImportError, AttributeError, ValueError, OSError):
        pass


def main() -> None:
    # private copies of the protocol pipes; fds 0/1 become harmless for model code
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    sys.stdin = open(os.devnull, encoding="utf-8")
    sys.dont_write_bytecode = True

    def send(obj: dict) -> None:
        proto_out.write(json.dumps(obj, ensure_ascii=True, default=str).encode("ascii") + b"\n")
        proto_out.flush()

    def recv() -> dict | None:
        line = proto_in.readline()
        return json.loads(line.decode("utf-8")) if line else None

    init = recv()
    if not init or init.get("op") != "init":
        return
    t0 = time.monotonic()
    try:
        kernel = Kernel(init)
    except BaseException as e:
        send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return
    scratch = init.get("scratch")
    if scratch:
        os.makedirs(scratch, exist_ok=True)
        os.chdir(scratch)
        for var in ("TMPDIR", "TEMP", "TMP"):
            os.environ[var] = scratch
        import tempfile

        tempfile.tempdir = scratch
    apply_limits(init.get("limits") or {})
    g = init.get("guard")
    if g:
        sys.addaudithook(Guard(scratch or os.getcwd(), g.get("deny_read") or [], g.get("allow_read") or []))
    send({"ready": True, "load_ms": int((time.monotonic() - t0) * 1000), "pid": os.getpid()})

    while True:
        msg = recv()
        if msg is None or msg.get("op") == "exit":
            return
        if msg.get("op") == "begin":
            kernel.begin(msg["as_of"], msg.get("question", ""))
            continue
        if msg.get("op") == "run":
            _watchdog(float(msg.get("timeout") or 0) + 10)
            res = kernel.run(msg["code"])
            _watchdog(0)
            send(res)


if __name__ == "__main__":
    main()
