"""Validate and normalize what the code passed to answer(type, value, reason).

Each type has exactly one accepted format (rulebook §9); a wrong format scores −1,
so instead of guessing, invalid answers raise AnswerError and the message goes
back to the model as feedback.
"""

from __future__ import annotations

import json
import math
import re

from datamate.core.numbers import round_half_up
from datamate.core.text import norm
from datamate.ingest.values import parse_date, parse_number

TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain")


class AnswerError(ValueError):
    pass


def _number(v) -> int | float:
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if isinstance(v, bool) or v is None:
        raise AnswerError(f"cần một số, nhận {v!r}")
    if isinstance(v, str):
        p = parse_number(v)
        if p is None:
            raise AnswerError(f"cần một số (JSON number), nhận chuỗi {v!r}")
        v = p
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise AnswerError(f"cần một số, nhận {v!r}") from None
    if math.isnan(f) or math.isinf(f):
        raise AnswerError(f"số không hợp lệ: {v!r} (NaN/inf — có thể bộ lọc rỗng?)")
    r = round_half_up(f, 2)  # references are integers or rounded to 2 decimals
    return int(r) if float(r).is_integer() else r


def _date(v) -> str:
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    s = str(v).strip()
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", s):
        return s
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[T ]00:00(:00(\.0+)?)?", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(\d{1,2})[/-](\d{4})", s)  # "9/2025" -> month question
    if m:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"
    ts = parse_date(s)
    if ts is not None and ts == ts.normalize():
        return ts.strftime("%Y-%m-%d")
    raise AnswerError(f"date phải là 'YYYY-MM-DD' hoặc 'YYYY-MM', nhận {v!r}")


def _boolean(v) -> bool:
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if isinstance(v, bool):
        return v
    if type(v).__name__ == "bool_":  # numpy bool (already converted by the sandbox, kept for safety)
        return bool(v)
    s = norm(str(v))
    if s in ("true", "co", "có", "đúng", "yes"):
        return True
    if s in ("false", "khong", "không", "sai", "no"):
        return False
    raise AnswerError(f"boolean phải là True/False, nhận {v!r}")


def _string(v) -> str:
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if isinstance(v, (list, tuple, dict)):
        raise AnswerError(f"string phải là một tên duy nhất, nhận {v!r}; nhiều phần tử thì dùng type 'list'")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = re.sub(r"\s+", " ", str(v)).strip()
    if not s:
        raise AnswerError("string rỗng")
    return s


def _cell(v):
    if v is None:
        raise AnswerError("ô/phần tử rỗng (None)")
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return _number(v)
    if isinstance(v, (list, tuple, dict)):
        raise AnswerError(f"phần tử lồng nhau không hợp lệ: {v!r}")
    return _string(v)


def _list(v) -> list:
    if isinstance(v, dict) and "rows" in v:
        raise AnswerError("nhận bảng; với type 'list' hãy truyền một list Python")
    if isinstance(v, (str, int, float)):
        v = [v]
    if not isinstance(v, (list, tuple)):
        raise AnswerError(f"list phải là list Python, nhận {type(v).__name__}")
    if not v:
        raise AnswerError("list rỗng — nếu thật sự không có phần tử nào, kiểm tra lại bộ lọc")
    return [_cell(x) for x in v]


def _table(v) -> dict:
    if not isinstance(v, dict) or "columns" not in v or "rows" not in v:
        raise AnswerError("table phải là {'columns': [...], 'rows': [[...], ...]} hoặc một DataFrame")
    cols = [str(c) for c in v["columns"]]
    rows = v["rows"]
    if not cols or not isinstance(rows, (list, tuple)):
        raise AnswerError("table thiếu cột hoặc dòng")
    out = []
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) != len(cols):
            raise AnswerError(f"mỗi dòng phải có đúng {len(cols)} ô: {r!r}")
        out.append([_cell(x) for x in r])
    return {"columns": cols, "rows": out}


def normalize(payload: dict) -> dict:
    """{'type', 'value', 'reason'} from answer(...) -> submission object without id."""
    t = str(payload.get("type", "")).strip().lower()
    if t not in TYPES:
        raise AnswerError(f"type không hợp lệ {t!r}; chọn một trong {', '.join(TYPES)}")
    if t == "abstain":
        return {"type": "abstain"}
    if t == "unanswerable":
        out = {"type": "unanswerable"}
        if payload.get("reason"):
            out["reason"] = str(payload["reason"])[:300]
        return out
    v = payload.get("value")
    if v is None:
        raise AnswerError(f"thiếu value cho type {t!r}")
    if t in ("number", "percent"):
        return {"type": t, "answer": _number(v)}
    conv = {"string": _string, "date": _date, "boolean": _boolean, "list": _list, "table": _table}[t]
    return {"type": t, "answer": conv(v)}


def canonical(ans: dict) -> str:
    """Key for voting: two answers that would score the same get the same key."""
    t = ans.get("type")
    if t in ("unanswerable", "abstain"):
        return t
    v = ans.get("answer")
    if t == "string":
        v = norm(v)
    elif t == "list":
        v = [norm(x) if isinstance(x, str) else x for x in v]
    elif t == "table":
        v = {
            "columns": [norm(c) for c in v["columns"]],
            "rows": sorted(
                json.dumps([norm(x) if isinstance(x, str) else x for x in r], ensure_ascii=False) for r in v["rows"]
            ),
        }
    return json.dumps([t, v], ensure_ascii=False, sort_keys=True)
