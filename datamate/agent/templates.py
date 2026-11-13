"""Stage 1 of the cascade: answer plain sales questions without any LLM call.

A question is "plain" when, once the period words and filler words are removed, nothing is
left: no product, customer or group filter, no extra condition. Its wording then maps to one
of a few intents (revenue, order count, cancelled orders, cancel rate, average order value,
best day, % change of revenue) and the period parser gives the period.

`python -m eval.oracle --train <train>` checks exactly this code against the train answers
(it is how the sales model and the period rules are validated), so keep the two in sync:
any wording added here must keep that report at 100 %.

Nothing here depends on a business id or a question id.
"""

from __future__ import annotations

import re

from datamate.agent.query import Query
from datamate.core.periods import Period, prev_period, same_days_prev_month, same_period_last_year, this_month
from datamate.core.text import key

_REVENUE = r"doanh thu|dthu|bao nhieu tien|may tien|nhieu tien|thu ve|thu duoc"
_FILLER = set(
    """
cua hang shop quan tiem doanh thu dthu tong cong ban duoc dc ra bao nhieu nhiu bnhiu nhieu may tien la trong cho
toi minh biet giup xem voi hoi nhe nha a the vay z ha khong ko k oi tinh so don bi huy ti ty le phan tram gia tri
trung binh quan mot moi tren mang ve ngay nao cao nhat dat xet theo thay doi tang giam hay lien truoc thang nam quy
hom nay qua tuan roi dem co da dang tat ca cac lai duoc luong hang ngay hnay hqua bua dat gi sao duoc nhe dum giup
jum cko mk mik e anh chi em ban ah u uh chua hien thi lam on bn nhiu nhieu phan tram kia vua bang tron
""".split()
)
_PERIOD_WORDS = re.compile(
    r"\b(thang|th|t|quy|q)\s*\d{1,2}(\s*(/|-|nam)\s*\d{2,4})?|\b\d{1,2}/\d{1,2}/\d{4}|\b\d{1,3}\s*ngay\b|\d+"
)


def is_plain(s: str) -> bool:
    """s: folded question without punctuation (core.text.key)."""
    rest = _PERIOD_WORDS.sub(" ", s)
    return all(tok in _FILLER for tok in re.findall(r"[a-z]+", rest))


def intent(s: str, raw: str) -> str | None:
    """s: folded question without punctuation (core.text.key); raw: original text (for '%')."""
    percent = "%" in raw or re.search(r"ti le|ty le|phan tram", s)
    if re.search(r"\bhuy\b", s):
        if percent:
            return "cancel_rate"
        if re.search(r"\b(bao nhieu|may|so|dem)\b", s):
            return "cancel_count"
        return None
    if re.search(r"trung binh|binh quan", s) and re.search(r"\bdon\b", s):
        return "avg_order"
    if re.search(r"\bngay nao\b", s) and re.search(r"nhat", s):
        return "best_day"
    if re.search(r"truoc|so voi", s) and percent and re.search(_REVENUE, s):
        return "pct_change"
    if re.search(_REVENUE, s):
        return "revenue"
    if re.search(r"\bdon\b", s) and re.search(r"bao nhieu|may", s):
        return "order_count"
    return None


def base_period(s: str, periods: list[Period]) -> Period:
    """The period a percentage change is measured against."""
    p = periods[0]
    if re.search(r"cung ky nam (truoc|ngoai)", s):
        return same_period_last_year(p)
    if re.search(r"(lien|ngay) truoc|truoc do|thang truoc|ky truoc", s) or len(periods) < 2:
        return same_days_prev_month(p.end) if p == this_month(p.end) else prev_period(p)
    return periods[1]


_QUERY = {
    "revenue": ("value", "revenue"),
    "order_count": ("value", "orders"),
    "cancel_count": ("value", "cancelled_orders"),
    "cancel_rate": ("value", "cancel_rate"),
    "avg_order": ("value", "avg_order_value"),
    "best_day": ("best_day", "revenue"),
    "pct_change": ("change", "revenue"),
}


def resolve(question: str, periods: list[Period]) -> tuple[str, Query] | None:
    """(intent, query) for a plain question, None when the question needs a model.
    periods: nlu.periods.extract(question, as_of, calendar), in reading order."""
    s = key(question)
    kind = intent(s, question)
    if kind is None or not is_plain(s) or not periods:
        return None
    op, metric = _QUERY[kind]
    base = base_period(s, periods) if kind == "pct_change" else None
    return kind, Query(op, metric, periods[0], base)
