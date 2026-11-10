"""Find the periods a (noisy) Vietnamese question talks about, for prompt hints.

    extract("dthu th7/2025 so với tháng trước", as_of)  ->  [Period(tháng 7/2025), ...]

Ambiguous spellings ("8/26", "9/6") yield several candidates; the agent decides.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

from datamate.core.periods import (
    Period,
    as_date,
    before_tet,
    day,
    last_month,
    last_n_days,
    last_week,
    month,
    quarter,
    same_days_prev_month,
    tet,
    this_month,
    this_week,
    today,
    until,
    year,
    yesterday,
    ytd,
)
from datamate.core.text import fold

_MY = re.compile(
    r"(?:\b(?:thang|th|t)\s*\.?\s*)?(?<![\d/.])(\d{1,2})\s*(?:/|-|\.|\s+nam\s+)\s*(20\d{2}|\d{2})(?![\d/])"
)
_Q = re.compile(r"\b(?:quy|q)\s*(\d)(?:\s*(?:/|-|nam)\s*(20\d{2}|\d{2}))?\b")
_DMY = re.compile(r"(?<![\d/])(\d{1,2})[/.\-](\d{1,2})[/.\-](20\d{2})(?![\d/])")
_M_ONLY = re.compile(r"\b(?:thang|th)\s*(\d{1,2})\b(?!\s*(?:/|-|\.|nam)\s*\d)")
_N_DAYS = re.compile(r"\b(\d{1,3})\s*ngay\s*(gan nhat|gan day|qua|vua qua|cuoi|gan)\b")
_SNAPSHOT = re.compile(r"(den het|toi het|tinh den|tinh toi|den cuoi|cuoi)\s*(ngay\s*)?$")


_RELATIVE = (
    (r"\b(hom nay|hnay|bua nay|ngay nay)\b", today),
    (r"\b(den het|toi het|tinh den|tinh toi|cuoi ngay|het)\s*(ngay\s*)?(hom nay|hnay)\b", until),
    (r"\b(hom qua|hqua|bua qua)\b", yesterday),
    (r"\btuan (nay|ni)\b", this_week),
    (r"\btuan (truoc|roi|qua)\b", last_week),
    (r"\b(thang nay|thang ni)\b", this_month),
    (r"\bthang (truoc|roi)\b", last_month),
    (r"\b(tu dau nam|nam nay)\b", ytd),
    (r"\bnam (ngoai|truoc|roi)\b", lambda t: year(t.year - 1)),
)


def extract(text: str, as_of, cal: pd.DataFrame | None = None) -> list[Period]:
    """Candidate periods mentioned in the question: explicit dates/months/quarters first,
    in reading order, then relative expressions ("tuần rồi", "tháng trước"…)."""
    s = fold(text)
    t = as_date(as_of)
    out: list[Period] = []

    def add(p: Period | None):
        if p and p not in out:
            out.append(p)

    def inside(pos: int, spans: list[tuple[int, int]]) -> bool:
        return any(a <= pos < b for a, b in spans)

    explicit: list[tuple[int, Period]] = []  # (position in the text, period)
    dmy_spans = []
    for m in _DMY.finditer(s):
        try:
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            continue
        dmy_spans.append(m.span())
        explicit.append((m.start(), until(d) if _SNAPSHOT.search(s[: m.start()]) else day(d)))
    q_spans = []
    for m in _Q.finditer(s):
        q = int(m.group(1))
        if 1 <= q <= 4:
            q_spans.append(m.span())
            explicit.append((m.start(), quarter(q, int(m.group(2)) if m.group(2) else None, t)))
    for m in _MY.finditer(s):
        if inside(m.start(1), dmy_spans) or inside(m.start(1), q_spans):  # "quý 2/2026" is not "tháng 2/2026"
            continue
        mm, yy = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12:
            explicit.append((m.start(), month(mm, yy)))
        if len(m.group(2)) == 2 and m.group(0)[:1].isdigit():
            try:  # bare "d/m" could also be a day of this year: "9/6"
                explicit.append((m.start() + 1, day(date(t.year, yy, mm))))
            except ValueError:
                pass
    for m in _M_ONLY.finditer(s):
        mm = int(m.group(1))
        if 1 <= mm <= 12 and not inside(m.start(1), q_spans):
            explicit.append((m.start(), month(mm, as_of=t)))
    for m in _N_DAYS.finditer(s):
        explicit.append((m.start(), last_n_days(int(m.group(1)), t)))
    for _, p in sorted(explicit, key=lambda x: x[0]):
        add(p)
    for pat, fn in _RELATIVE:
        if re.search(pat, s):
            add(fn(t))
    if re.search(r"\b(thang nay|thang ni)\b", s) and re.search(r"thang truoc|so voi", s):
        add(same_days_prev_month(t))
    if cal is not None:
        for m in re.finditer(r"\btet\b(?:\s*(?:nam\s*)?(20\d{2}))?", s):
            ys = [int(m.group(1))] if m.group(1) else [t.year, t.year - 1]
            for yy in ys:
                try:
                    p = tet(yy, cal)
                except ValueError:
                    continue
                if m.group(1) or p.start <= t:
                    add(p)
                    if "truoc tet" in s:
                        add(before_tet(yy, cal))
                    break
    return out
