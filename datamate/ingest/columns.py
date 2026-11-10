"""Decide the kind of each raw column (number, date, text, bool, empty) and convert it."""

from __future__ import annotations

import re

import pandas as pd

from datamate.core.text import fold
from datamate.ingest.values import is_blank, parse_date, parse_number, split_number_unit

_DATE_HINT = re.compile(r"\b(ngay|thoi gian|thoi diem|han|gio|date|time|luc|dang ky)\b")
_ID_HINT = re.compile(
    r"\b(ma|so dien thoai|sdt|so dt|dien thoai|phone|imei|bien so|so hd|so phieu|so lo|lo|id|sku|"
    r"code|barcode|so hoa don|dia chi|address)\b"
)
UNIT_SUFFIX = " (đơn vị ghi trong ô)"


def _sample(values: list, n: int = 400) -> list:
    if len(values) <= n:
        return values
    step = len(values) / n
    return [values[int(i * step)] for i in range(n)]


def infer_kind(values: list, name: str, desc: str) -> str:
    nonblank = [v for v in values if not is_blank(v)]
    if not nonblank:
        return "empty"
    hint = fold(f"{name} {desc}")
    s = _sample(nonblank)
    if all(isinstance(v, bool) for v in s) or {fold(str(v)) for v in s} <= {"true", "false"}:
        return "bool"
    date_hint = bool(_DATE_HINT.search(hint))
    n_date = sum(parse_date(v, year_hint=2000, allow_serial=date_hint) is not None for v in s)
    if date_hint and n_date >= 0.9 * len(s):
        return "date"
    id_like = bool(_ID_HINT.search(hint)) and not re.search(r"\b(so luong|so tien|so ngay|so ky)\b", hint)
    strs = [v for v in s if isinstance(v, str)]
    leading_zero = sum(1 for v in strs if re.fullmatch(r"0\d{5,}", v.strip())) > 0.3 * max(len(strs), 1)
    n_num = sum(parse_number(v) is not None for v in s)
    if not id_like and not leading_zero and n_num >= 0.95 * len(s):
        return "number"
    if n_date >= 0.95 * len(s) and strs:
        return "date"
    return "text"


def convert(df: pd.DataFrame, col: str, kind: str) -> pd.Series:
    vals = df[col].tolist()
    if kind == "number":
        return pd.Series([parse_number(v) for v in vals], index=df.index, dtype="float64")
    if kind == "date":
        years = df["_sheet_year"].tolist() if "_sheet_year" in df else [None] * len(vals)
        out = [
            parse_date(v, year_hint=int(y) if y == y and y is not None else None, allow_serial=True)
            for v, y in zip(vals, years, strict=False)
        ]
        return pd.to_datetime(pd.Series(out, index=df.index, dtype=object), errors="coerce")
    if kind == "bool":
        return pd.Series(
            [None if is_blank(v) else fold(str(v)) in ("true", "1", "x", "co", "yes") for v in vals],
            index=df.index,
            dtype=object,
        )
    return pd.Series(
        [None if is_blank(v) else re.sub(r"\s+", " ", str(v)).strip() for v in vals], index=df.index, dtype=object
    )


def units_in_cells(df: pd.DataFrame, col: str) -> pd.Series | None:
    """'2 xe 4m3' -> 'xe 4m3': the unit written next to a number, None if no cell has one."""
    units = [split_number_unit(v)[1] for v in df[col].tolist()]
    return pd.Series(units, index=df.index, dtype=object) if any(units) else None


def summarize(s: pd.Series, kind: str) -> tuple[int, str]:
    """(number of distinct values, one-line summary for the schema card)."""
    nn = s.dropna()
    if nn.empty:
        return 0, "toàn bộ trống"
    nd = int(nn.astype(str).nunique())
    blank = f", {int(s.isna().sum())} ô trống" if s.isna().any() else ""
    if kind == "date":
        has_time = (nn.dt.hour != 0).any() or (nn.dt.minute != 0).any()
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
        return nd, f"{nn.min():{fmt}} → {nn.max():{fmt}}{blank}"
    if kind == "number":
        ints = (nn == nn.round()).all()
        f = (lambda x: f"{int(x):,}") if ints else (lambda x: f"{x:,.2f}")
        return nd, f"min {f(nn.min())}, max {f(nn.max())}{', có số âm' if (nn < 0).any() else ''}{blank}"
    vc = nn.astype(str).value_counts()
    if nd <= 15:
        vals = ", ".join(f"'{v}'({c})" for v, c in vc.items())
        return nd, f"{nd} giá trị: {vals}{blank}"
    ex = ", ".join(f"'{v}'" for v in vc.index[:4])
    return nd, f"{nd} giá trị, ví dụ: {ex}{blank}"
