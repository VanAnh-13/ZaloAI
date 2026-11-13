"""Deterministic query engine on the sales model.

A `Query` says WHAT to compute (metric, period, filters, ranking…); `execute` computes it with
the rulebook mechanics that never change: valid orders, period by order date, half-up rounding,
ranking tie-break, names as written in the data, the answer format of each type.

Two cheap stages of the cascade produce queries (the expensive code agent is the third):
- agent/templates.py — regular expressions, no LLM, for plain questions;
- agent/planner.py   — the LLM fills a small JSON spec, it never computes anything itself.

Whatever the engine cannot express raises QueryError and the question goes on to the code agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from datamate.core.numbers import pct_change, round2
from datamate.core.periods import DATA_START, Period
from datamate.semantics.sales import LINE_DIMS, ORDER_DIMS, QueryError, SalesView

OPS = ("value", "change", "top", "best_day")
METRICS = ("revenue", "orders", "cancelled_orders", "cancel_rate", "avg_order_value", "qty", "base_qty")
RANK_MEASURES = {"revenue": "revenue", "orders": "orders", "qty": "qty", "base_qty": "base_qty"}
DIMS = LINE_DIMS + ORDER_DIMS

__all__ = ["DIMS", "METRICS", "OPS", "Query", "QueryError", "Result", "execute"]


@dataclass
class Query:
    op: str  # value | change | top | best_day
    metric: str
    period: Period | None = None  # None = all the data
    base: Period | None = None  # op "change": the period compared against
    filters: dict = field(default_factory=dict)  # dimension -> name (or list of names)
    by: str | None = None  # op "top": dimension to rank
    k: int | None = None
    ascending: bool = False

    def describe(self) -> dict:
        return {
            "op": self.op,
            "metric": self.metric,
            "period": str(self.period) if self.period else None,
            "base": str(self.base) if self.base else None,
            "filters": self.filters or None,
            "by": self.by,
            "k": self.k,
            "order": "asc" if self.ascending else "desc",
        }


@dataclass
class Result:
    answer: dict  # submission object without id: {"type", "answer"} or {"type": "unanswerable", "reason"}
    evidence: dict  # how it was computed (logged; also used by the confidence features)


def _outside(p: Period | None, coverage: tuple[date, date] | None) -> bool:
    """True when the period lies entirely outside the dates the shop has sales data for.
    Snapshot periods ("đến hết ngày X", from the start of the data) are never outside."""
    if p is None or coverage is None or p.start <= DATA_START:
        return False
    return p.end < coverage[0] or p.start > coverage[1]


def _number(x) -> int | float:
    r = round2(x)
    return int(r) if float(r).is_integer() else r


def _metric(S: SalesView, metric: str, p: Period | None):
    if metric == "revenue":
        return S.revenue(p)
    if metric == "orders":
        return S.order_count(p)
    if metric == "cancelled_orders":
        return S.cancelled_count(p)
    if metric == "cancel_rate":
        return S.cancel_rate(p)
    if metric == "avg_order_value":
        return S.avg_order_value(p)
    if metric in ("qty", "base_qty"):
        return _number(S.qty(p, base=metric == "base_qty"))
    raise QueryError(f"metric không hỗ trợ: {metric!r}")


def _daily(S: SalesView, metric: str, p: Period | None):
    if metric == "revenue":
        return S.daily_revenue(p)
    if metric == "orders":
        v = S.valid_orders(p)
        return v.groupby(v["date"].dt.normalize())["order_id"].nunique()
    if metric in ("qty", "base_qty"):
        L = S.valid_lines(p)
        if metric not in L:
            raise QueryError(f"S.lines không có cột `{metric}`")
        return L.groupby(L["order_date"].dt.normalize())[metric].sum()
    raise QueryError(f"best_day không hỗ trợ metric {metric!r}")


def execute(q: Query, S: SalesView | None, coverage: tuple[date, date] | None = None) -> Result:
    """Compute a query. Raises QueryError when the engine must not answer."""
    if S is None:
        raise QueryError("shop này không có mô hình bán hàng S")
    if q.op not in OPS:
        raise QueryError(f"op không hỗ trợ: {q.op!r}")
    if q.metric not in METRICS:
        raise QueryError(f"metric không hỗ trợ: {q.metric!r}")
    ev = {"query": q.describe()}
    # rulebook §7: a period entirely outside the data cannot be answered (a part inside is computed)
    for name, p in (("kỳ hỏi", q.period), ("kỳ so sánh", q.base if q.op == "change" else None)):
        if _outside(p, coverage):
            ev["outside"] = name
            reason = f"{name} ({p}) nằm ngoài khoảng dữ liệu {coverage[0]} → {coverage[1]}"
            return Result({"type": "unanswerable", "reason": reason}, ev)
    view = S.where(**q.filters) if q.filters else S
    ev["orders_in_period"] = int(len(view.orders_in(q.period)))

    if q.op == "value":
        v = _metric(view, q.metric, q.period)
        if v is None:
            raise QueryError("không có đơn nào trong kỳ nên không tính được tỉ lệ / trung bình")
        return Result({"type": "percent" if q.metric == "cancel_rate" else "number", "answer": v}, ev)

    if q.op == "change":
        if q.metric == "cancel_rate":
            raise QueryError("không tính % thay đổi của một tỉ lệ")
        if q.period is None or q.base is None:
            raise QueryError("op 'change' cần cả period và base")
        new, old = _metric(view, q.metric, q.period), _metric(view, q.metric, q.base)
        ev["values"] = {"period": new, "base": old}
        change = None if new is None or old is None else pct_change(new, old)
        if change is None:
            raise QueryError("kỳ so sánh bằng 0 hoặc không có đơn nên không tính được % thay đổi")
        return Result({"type": "percent", "answer": change}, ev)

    if q.op == "top":
        if q.metric not in RANK_MEASURES:
            raise QueryError(f"không xếp hạng theo metric {q.metric!r}")
        if not q.by:
            raise QueryError("op 'top' cần `by`")
        if q.by in ("hour", "weekday"):  # how such an answer is written depends on the rulebook: leave it to the model
            raise QueryError(f"xếp hạng theo {q.by!r} để agent viết code xử lý")
        k = int(q.k or 1)
        if k < 1 or k > 50:
            raise QueryError(f"k không hợp lệ: {q.k!r}")
        ranked = view.rank(q.by, RANK_MEASURES[q.metric], q.period, k=k, ascending=q.ascending)
        ev["ranking"] = ranked.to_dict("records")
        if ranked.attrs.get("tie_at_boundary"):
            raise QueryError("hạng k và k+1 bằng nhau — câu hỏi kiểu này đã bị loại nên nhiều khả năng đang hiểu sai")
        if len(ranked) < k:
            raise QueryError(f"chỉ có {len(ranked)} giá trị, không đủ top {k}")
        names = [str(x) for x in ranked[q.by].tolist()]
        if k > 1:
            return Result({"type": "list", "answer": names}, ev)
        # "ngày nào / tháng nào … nhất" is a date answer: YYYY-MM-DD, or YYYY-MM for a month (§9)
        return Result({"type": "date" if q.by in ("day", "month") else "string", "answer": names[0]}, ev)

    # best_day
    if q.ascending:
        raise QueryError("ngày thấp nhất phụ thuộc cách tính ngày không có đơn")
    daily = _daily(view, q.metric, q.period)
    if not len(daily):
        raise QueryError("không có đơn hợp lệ nào trong kỳ")
    best = daily[daily == daily.max()].index.min()  # equal days: the earliest
    ev["days_tied"] = int((daily == daily.max()).sum())
    return Result({"type": "date", "answer": best.strftime("%Y-%m-%d")}, ev)
