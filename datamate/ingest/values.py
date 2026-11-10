"""Parse messy Vietnamese spreadsheet values: numbers, money and dates.

Every parser returns None when the value cannot be read, never raises.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta

import pandas as pd

from datamate.core.text import fold

EXCEL_EPOCH = datetime(1899, 12, 30)

_BLANK = {"", "-", "--", "nan", "none", "null", "n/a", "na"}


def is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in _BLANK:
        return True
    return False


# ---------------------------------------------------------------- numbers

_NUM_HEAD = re.compile(r"^(\d[\d.,]*(?:\s\d{3})*(?:[.,]\d+)?)\s*(.*)$")
_MONEY_SUFFIX = {"đ", "₫", "d", "vnd", "vnđ", "đồng", "dong", "vn"}
_MULTIPLIER = {
    "k": 1e3,
    "kđ": 1e3,
    "k đ": 1e3,
    "nghìn": 1e3,
    "ngàn": 1e3,
    "tr": 1e6,
    "triệu": 1e6,
    "trieu": 1e6,
    "tỷ": 1e9,
    "tỉ": 1e9,
    "ty": 1e9,
}
# unit written in the cell; big units may carry digits: "bao 25kg", "xe 4m3", "hộp 30"
_UNIT_WORD = re.compile(r"^[^\W\d_][\w.]*(?:\s[\w.]+){0,3}$")
# spoken-style money: "1tr900" = 1.900.000, "2 triệu 45 nghìn", "1k5" = 1.500
_TR_REST = re.compile(r"^(?:tr|triệu|trieu)\s*(\d{1,3})\s*(k|nghìn|ngàn|nghin|ngan)?\s*(?:đ|d|đồng)?$")
_K_REST = re.compile(r"^k(\d{1,3})$")


def _parse_digits(p: str, decimal_first: bool = False) -> float | None:
    """decimal_first: a single separator is a decimal mark ('1824,235' tr = 1824.235 triệu)."""
    p = p.replace(" ", "")
    if "," in p and "." in p:
        dec = "," if p.rfind(",") > p.rfind(".") else "."
        thou = "." if dec == "," else ","
        p = p.replace(thou, "").replace(dec, ".")
    elif "," in p or "." in p:
        sep = "," if "," in p else "."
        parts = p.split(sep)
        if not decimal_first and (len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3)):
            # thousands separators: 1.234.567 / 38,000
            if not (1 <= len(parts[0]) <= 3 and all(len(x) == 3 for x in parts[1:])):
                return None
            p = "".join(parts)
        elif len(parts) == 2:
            p = parts[0] + "." + parts[1]
        else:
            return None
    try:
        return float(p)
    except ValueError:
        return None


def split_number_unit(v) -> tuple[float | None, str | None]:
    """Like parse_number, and also returns a unit written in the cell:
    '2 xe 4m3' -> (2.0, 'xe 4m3'), '55.000đ' -> (55000.0, None)."""
    if isinstance(v, bool) or is_blank(v):
        return None, None
    if isinstance(v, (int, float)):
        return float(v), None
    if not isinstance(v, str):
        return None, None
    raw = v.strip().replace(" ", " ")
    s = raw.lower()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s, raw = True, s[1:-1].strip(), raw[1:-1].strip()
    if s[:1] in "-−+":
        neg = neg or s[0] != "+"
        s, raw = s[1:].strip(), raw[1:].strip()
    m = _NUM_HEAD.match(s)
    if not m:
        return None, None
    suffix = m.group(2).strip().rstrip(".")
    unit = None
    m_tr, m_k = _TR_REST.match(suffix), _K_REST.match(suffix)
    if m_tr or m_k:
        head = _parse_digits(m.group(1))
        if head is None:
            return None, None
        if m_tr and m_tr.group(2):  # "1 triệu 45 nghìn" -> 1 045 000
            num = head * 1e6 + int(m_tr.group(1)) * 1e3
        elif m_tr:  # "1tr900" -> 1 900 000; "1tr5" -> 1 500 000 (like 1.5 triệu)
            num = head * 1e6 + int(m_tr.group(1).ljust(3, "0")) * 1e3
        else:  # "1k5" -> 1 500
            num = head * 1e3 + int(m_k.group(1).ljust(3, "0"))
    elif suffix in _MULTIPLIER:
        num = _parse_digits(m.group(1), decimal_first=True)
        if num is not None:
            num *= _MULTIPLIER[suffix]
    else:
        num = _parse_digits(m.group(1))
        if suffix in ("", "%") or suffix in _MONEY_SUFFIX:
            pass
        elif _UNIT_WORD.match(suffix) and len(suffix) <= 20:
            unit = raw[m.start(2) :].strip().rstrip(".")  # keep original casing
        else:
            return None, None
    if num is None:
        return None, None
    return (-num if neg else num), unit


def parse_number(v) -> float | None:
    """'55.000đ' -> 55000, '49k' -> 49000, '38,000' -> 38000, '1.234,5' -> 1234.5,
    '5 kg' -> 5, '2 xe 4m3' -> 2, '1824,235tr' -> 1824235000, '(1.000)' -> -1000,
    12 -> 12.0. Booleans and blanks -> None."""
    return split_number_unit(v)[0]


# ------------------------------------------------------------------ dates

_ISO = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ t](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?)?(?:z|[+-]\d{2}:?\d{2})?$"
)
_DMY = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})(?:[ ,]+(\d{1,2})[:h](\d{2})(?::(\d{2}))?)?$")
_THG = re.compile(r"^(\d{1,2})\s*thg\s*(\d{1,2}),?\s*(\d{4})(?:[ ,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_VN_LONG = re.compile(r"^(?:ngay\s*)?(\d{1,2})\s*thang\s*(\d{1,2})\s*(?:nam\s*)?(\d{4})$")
_DM = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})$")
_TIME_FIRST = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s+(.+)$")


def _mk(y, mo, d, hh=None, mm=None, ss=None) -> pd.Timestamp | None:
    try:
        y = int(y)
        if y < 100:
            y += 2000
        return pd.Timestamp(datetime(y, int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0)))
    except (ValueError, TypeError):
        return None


def parse_date(v, year_hint: int | None = None, allow_serial: bool = False) -> pd.Timestamp | None:
    """Day-first parsing. Handles datetime objects, Excel serials (only when
    allow_serial), 'dd/mm/yyyy [hh:mm[:ss]]', 'yyyy-mm-dd ...', '1 thg 6 2025',
    'ngày 5 tháng 3 năm 2025' and 'dd/mm' (year from year_hint)."""
    if is_blank(v) or isinstance(v, bool):
        return None
    if isinstance(v, pd.Timestamp):
        return v
    if isinstance(v, datetime):
        return pd.Timestamp(v)
    if isinstance(v, date):
        return pd.Timestamp(datetime(v.year, v.month, v.day))
    if isinstance(v, (int, float)):
        if allow_serial and 20000 < v < 80000:
            return pd.Timestamp(EXCEL_EPOCH + timedelta(days=float(v))).round("s")
        return None
    if not isinstance(v, str):
        return None
    s = fold(v.strip())
    m = _TIME_FIRST.match(s)
    if m and not _ISO.match(s):
        d = parse_date(m.group(4), year_hint, allow_serial)
        return None if d is None else _mk(d.year, d.month, d.day, m.group(1), m.group(2), m.group(3))
    for rx, order in ((_ISO, "ymd"), (_DMY, "dmy"), (_THG, "dmy"), (_VN_LONG, "dmy")):
        m = rx.match(s)
        if m:
            g = m.groups()
            if order == "ymd":
                return _mk(g[0], g[1], g[2], *g[3:6])
            return _mk(g[2], g[1], g[0], *(g[3:6] if len(g) > 3 else ()))
    m = _DM.match(s)
    if m and year_hint:
        return _mk(year_hint, m.group(2), m.group(1))
    if allow_serial and re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_date(float(s), allow_serial=True)
    return None


def parse_time(v) -> time | None:
    """'06:05', '14:30:00', datetime -> time-of-day."""
    if isinstance(v, datetime):
        return v.time()
    if isinstance(v, time):
        return v
    if isinstance(v, str):
        m = re.search(r"(\d{1,2})[:h](\d{2})(?::(\d{2}))?", v)
        if m:
            try:
                return time(int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
            except ValueError:
                return None
    return None


def month_year_from_name(name: str) -> tuple[int | None, int | None]:
    """'T01_2025', '04.2025', 'Thang 02 2025', 'BanHang_T1_2026', 'X_2025' -> (month, year)."""
    s = fold(name)
    m = re.search(r"(?<!\d)(\d{1,2})\D{0,3}(20\d{2})(?!\d)", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(?<!\d)(20\d{2})\D{0,3}(\d{1,2})(?!\d)", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"(?<!\d)(20\d{2})(?!\d)", s)
    if m:
        return None, int(m.group(1))
    return None, None
