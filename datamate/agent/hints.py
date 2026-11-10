"""Deterministic analysis of a question, turned into prompt hints (all marked "may be wrong")
and into the context the verifier uses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from datamate.core.periods import Period, prev_period
from datamate.core.text import key
from datamate.nlu.answer_type import expected_type
from datamate.nlu.entities import EntityIndex
from datamate.nlu.normalize import rewrite
from datamate.nlu.periods import extract

_COMPARE = re.compile(r"\b(so voi|tang|giam|thay doi|chenh lech|len hay xuong|hon hay kem)\b")


@dataclass
class Analysis:
    rewritten: str | None
    periods: list[Period]
    expected: tuple[str, str] | None  # (type, why)
    names: list[tuple[str, str, str]]  # (value, table, column)
    coverage: tuple[date, date] | None  # first order date .. last data date
    outside: list[Period] = field(default_factory=list)  # periods entirely outside the data
    partial: list[Period] = field(default_factory=list)  # periods only partly inside

    @property
    def expected_type(self) -> str | None:
        return self.expected[0] if self.expected else None

    @property
    def all_outside(self) -> bool:
        checked = [p for p in self.periods if not p.label.startswith("đến hết")]
        return bool(checked) and all(p in self.outside for p in checked)

    @property
    def all_inside(self) -> bool:
        checked = [p for p in self.periods if not p.label.startswith("đến hết")]
        return bool(checked) and self.coverage is not None and not self.outside and not self.partial

    def lines(self) -> list[str]:
        out = []
        if self.rewritten:
            out.append(f'Câu hỏi viết lại: "{self.rewritten}"')
        if self.periods:
            out.append("Kỳ thời gian nhận ra: " + "; ".join(str(p) for p in self.periods[:6]))
        if self.expected:
            out.append(f"Kiểu đáp án dự đoán: {self.expected[0]} ({self.expected[1]})")
        if self.names:
            out.append(
                "Tên trong dữ liệu giống chữ trong câu hỏi: " + "; ".join(f"'{v}' ({t}.{c})" for v, t, c in self.names)
            )
        if self.coverage:
            out.append(f"Dữ liệu bán hàng có từ {self.coverage[0]} đến {self.coverage[1]}.")
        for p in self.outside:
            out.append(f"CẢNH BÁO: kỳ {p} nằm HOÀN TOÀN ngoài khoảng dữ liệu → nhiều khả năng unanswerable (§7).")
        for p in self.partial:
            out.append(f"Kỳ {p} chỉ có một phần dữ liệu → tính trên phần có dữ liệu (§7).")
        return out


def _coverage(sales_coverage: tuple[str, str] | None, data_end: str | None) -> tuple[date, date] | None:
    if not sales_coverage:
        return None
    start, end = date.fromisoformat(sales_coverage[0]), date.fromisoformat(sales_coverage[1])
    if data_end:
        end = max(end, date.fromisoformat(data_end))
    return start, end


def analyze(
    question: str,
    as_of: str,
    cal: pd.DataFrame | None,
    index: EntityIndex | None,
    sales_coverage: tuple[str, str] | None = None,
    data_end: str | None = None,
) -> Analysis:
    rewritten = rewrite(question)
    if key(rewritten) == key(question):
        rewritten = None
    text = rewritten or question
    try:
        periods = extract(text, as_of, cal)
    except Exception:  # hints must never break a question
        periods = []
    a = Analysis(
        rewritten,
        periods,
        expected_type(text),
        index.mentioned(text) if index else [],
        _coverage(sales_coverage, data_end),
    )
    if a.coverage:
        start, end = a.coverage
        checked = [p for p in periods if not p.label.startswith("đến hết")]
        if checked and _COMPARE.search(key(text)):
            checked.append(prev_period(checked[0]))  # "so với tháng trước" needs the previous period too
        for p in checked:
            if p.end < start or p.start > end:
                a.outside.append(p)
            elif p.start < start or p.end > end:
                a.partial.append(p)
    return a
