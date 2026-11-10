"""Child process of Sandbox.run: load the workspace, execute the code, write a JSON result.

Namespace given to the code: S, T, UNITS, DATA_END, CAL, AS_OF, QUESTION, DATA_DIR,
pd, np, dm and answer(type, value=None, reason=None).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from datetime import date, datetime
from decimal import Decimal

MAX_STDOUT = 20000


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


def _short_traceback(code: str) -> str:
    """Only the frames inside the model's code, plus the exception line."""
    etype, exc, tb = sys.exc_info()
    lines = code.splitlines()
    frames = [f for f in traceback.extract_tb(tb) if f.filename == "<code>"]
    where = [
        f"  dòng {f.lineno}: {lines[f.lineno - 1].strip() if 0 < f.lineno <= len(lines) else ''}" for f in frames[-4:]
    ]
    msg = "".join(traceback.format_exception_only(etype, exc)).strip()
    return "\n".join(["Traceback (code của bạn):", *where, msg[-1500:]])


def main() -> None:
    req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    import numpy as np
    import pandas as pd

    from datamate import dm
    from datamate.workspace.cache import load_cached

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)

    ws = load_cached(req["workspace"]) if req.get("workspace") else None
    biz = ws.business if ws else None
    cal = None
    if req.get("calendar"):
        cal = pd.read_csv(req["calendar"])
        cal["date"] = pd.to_datetime(cal["date"])
    state: dict = {}

    def answer(type, value=None, reason=None):  # noqa: A002 - name shown to the model
        state["answer"] = {"type": type, "value": to_jsonable(value), "reason": None if reason is None else str(reason)}
        raise _Submitted

    ns = {
        "__name__": "__main__",
        "pd": pd,
        "np": np,
        "dm": dm,
        "S": ws.sales if ws else None,
        "T": biz.frames() if biz else {},
        "UNITS": biz.units if biz else pd.DataFrame(columns=["product", "unit", "factor", "base_unit"]),
        "DATA_END": biz.data_end if biz else None,
        "CAL": cal,
        "AS_OF": date.fromisoformat(req["as_of"]),
        "QUESTION": req["question"],
        "DATA_DIR": req["data_dir"],
        "answer": answer,
    }
    buf, err = io.StringIO(), None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(req["code"], "<code>", "exec"), ns)
        except _Submitted:
            pass
        except SystemExit:
            pass
        except BaseException:  # report every model error back to the model
            err = _short_traceback(req["code"])
    out = buf.getvalue()
    if len(out) > MAX_STDOUT:
        out = out[:MAX_STDOUT] + "\n…"
    with open(req["result"], "w", encoding="utf-8") as f:
        json.dump({"stdout": out, "error": err, "answer": state.get("answer")}, f, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
