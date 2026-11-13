"""Stage 2 of the cascade: the LLM fills a small JSON spec (QUERY_SPEC), the query engine computes.

One short model call instead of a multi-step coding session. The model only chooses the
meaning — which metric, which period, which filter, which ranking — and may always answer
`"route": "code"` to hand the question to the code agent. Everything it writes is validated:
unknown fields, names that are not in the data, impossible periods → no vote from this stage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from datamate.agent.query import DIMS, METRICS, OPS, Query, QueryError, Result, execute
from datamate.core.periods import Period, prev_period, same_period_last_year
from datamate.semantics.sales import TIME_DIMS, SalesView

# The marker QUERY_SPEC in the first line lets the offline mock client recognize this prompt.
SYSTEM = """\
Bạn là bộ phân tích câu hỏi của DataMate và chỉ làm một việc: đọc MỘT câu hỏi về dữ liệu bán hàng của MỘT shop rồi điền một QUERY_SPEC (JSON). Một máy tính toán tất định sẽ thực hiện spec đó. Bạn KHÔNG tính toán và KHÔNG viết code.

## Máy tính toán làm được gì
Nó chạy trên mô hình bán hàng đã áp luật của RULEBOOK: chỉ tính đơn hợp lệ (không huỷ, không trả hàng — kể cả trả một phần), kỳ tính theo ngày đặt đơn, voucher/phí giao không phải doanh thu, làm tròn và xếp hạng đúng luật.

metric:
- "revenue": doanh thu = tổng thành tiền dòng hàng của các đơn hợp lệ đặt trong kỳ
- "orders": số đơn hợp lệ đặt trong kỳ
- "cancelled_orders": số đơn bị huỷ trong các đơn đặt trong kỳ
- "cancel_rate": tỉ lệ % đơn huỷ trên MỌI đơn đặt trong kỳ
- "avg_order_value": giá trị trung bình một đơn hợp lệ (doanh thu / số đơn hợp lệ)
- "qty": tổng số lượng bán, theo đơn vị ghi ở dòng hàng
- "base_qty": tổng số lượng bán đã quy về đơn vị cơ bản (chỉ khi shop có quy đổi đơn vị)

op:
- "value": một con số — metric trong `period`
- "change": % thay đổi của metric ở `period` so với `base` (giảm ra số âm). `base` là "prev" (kỳ liền trước tương ứng: tháng trước, quý trước, tuần trước; "tháng này" thì so cùng số ngày của tháng trước), "last_year" (cùng kỳ năm trước) hoặc một kỳ cụ thể {"start": ..., "end": ...}
- "top": xếp hạng các giá trị của `by` theo metric (revenue | orders | qty | base_qty) và lấy `k` cái đầu; "order" là "desc" (nhiều nhất/cao nhất) hoặc "asc" (ít nhất/thấp nhất). k = 1 khi hỏi "… nào … nhất"
- "best_day": ngày có metric (revenue | orders | qty) cao nhất trong `period`

by (chỉ cho "top"): product, product_group, customer, customer_group, channel, payment, staff — chỉ những chiều có trong mục "Các chiều của shop" — hoặc day, month ("tháng nào doanh thu cao nhất" → by "month", k 1).
filters (tuỳ chọn): lọc theo tên, ví dụ {"product": "Sting dâu 330ml", "channel": "Shopee"}. Chép tên ĐÚNG NGUYÊN VĂN từ mục "Các chiều của shop" hoặc từ gợi ý; tên không có trong dữ liệu thì spec bị loại.
period: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}, gồm cả hai đầu. Ưu tiên chép từ "Kỳ thời gian nhận ra" trong gợi ý; tự tính theo RULEBOOK nếu gợi ý thiếu hoặc sai. null = toàn bộ dữ liệu.

## route
- "sales": câu hỏi diễn đạt được ĐẦY ĐỦ và CHÍNH XÁC bằng các trường ở trên.
- "code": mọi trường hợp khác — công nợ, tồn kho, nhập hàng, lãi và giá vốn, lô và hạn dùng, giao hàng, sổ đặc thù; định nghĩa riêng trong RULEBOOK (khách trung thành, khách mới, giờ cao điểm…); điều kiện lọc mà filters không tả được (theo ngày trong tuần, theo ca, theo ngưỡng số tiền…); câu hỏi cần nhiều số hoặc một bảng; câu hỏi có/không; hoặc bạn không chắc. Một agent viết code sẽ xử lý. Chọn "code" luôn an toàn; điền spec sai thì bị trừ điểm.
- "unanswerable": theo RULEBOOK dữ liệu không đủ để trả lời — kỳ hỏi nằm HOÀN TOÀN ngoài khoảng ngày có dữ liệu, hoặc shop không ghi loại thông tin cần. Kỳ chỉ có một phần dữ liệu thì vẫn tính được.

## Trả lời
Đúng MỘT đối tượng JSON, không viết gì trước hay sau. "analysis" viết trước tiên, 1–2 câu: câu hỏi cần gì, kỳ nào, có điều kiện gì.
{"analysis": "...", "route": "sales", "op": "value", "metric": "revenue", "filters": {}, "period": {"start": "2025-07-01", "end": "2025-07-31"}, "base": null, "by": null, "k": null, "order": "desc"}
{"analysis": "...", "route": "sales", "op": "top", "metric": "qty", "filters": {}, "period": {"start": "2025-10-01", "end": "2025-10-31"}, "base": null, "by": "product", "k": 3, "order": "desc"}
{"analysis": "...", "route": "sales", "op": "change", "metric": "revenue", "filters": {"channel": "Shopee"}, "period": {"start": "2026-03-01", "end": "2026-03-31"}, "base": "prev", "by": null, "k": null, "order": "desc"}
{"analysis": "...", "route": "code"}
{"analysis": "...", "route": "unanswerable"}
"""

MAX_VALUES_SHOWN = 40


def shop_dimensions(S: SalesView | None, coverage: tuple[date, date] | None) -> str:
    """The part of the shop the planner needs: which dimensions exist and how their values are written."""
    if S is None:
        return '## Các chiều của shop\nShop này KHÔNG có mô hình bán hàng: luôn trả lời route "code".'
    lines = ["## Các chiều của shop (tên viết đúng như trong dữ liệu)"]
    if coverage:
        lines.append(f"- Dữ liệu bán hàng có từ {coverage[0]} đến {coverage[1]}.")
    for dim, values in S.dims().items():
        shown = ", ".join(f'"{v}"' for v in values[:MAX_VALUES_SHOWN])
        more = f", … (+{len(values) - MAX_VALUES_SHOWN} tên khác — xem gợi ý)" if len(values) > MAX_VALUES_SHOWN else ""
        lines.append(f"- {dim} ({len(values)}): {shown}{more}")
    extras = [m for m in ("qty", "base_qty") if m in S.lines]
    lines.append("- metric theo số lượng dùng được: " + (", ".join(extras) if extras else "(không có cột số lượng)"))
    if not S.per_line:
        lines.append("- Dữ liệu chỉ có hoá đơn, không có dòng hàng: không lọc/xếp hạng theo product, không có qty.")
    return "\n".join(lines)


def build_messages(q: dict, rulebook: str, dimensions: str, hint_lines: list[str], weekday: str) -> list[dict]:
    parts = []
    if rulebook.strip():
        parts.append("# RULEBOOK (luật tính đáp án của bộ dữ liệu này)\n" + rulebook.strip())
    parts.append(dimensions)
    parts.append(f'# CÂU HỎI\n- Hôm nay (AS_OF): {q["as_of_date"]} ({weekday})\n- Câu hỏi: "{q["question"]}"')
    if hint_lines:
        parts.append("## Gợi ý tự động (có thể sai — tự kiểm tra lại)\n" + "\n".join(f"- {h}" for h in hint_lines))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# ------------------------------------------------------------------ parsing


class SpecError(ValueError):
    """The model's JSON is not a usable spec."""


def extract_json(text: str) -> dict:
    """The first complete JSON object in the reply (models sometimes wrap it in ``` or prose)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except ValueError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    raise SpecError("không thấy đối tượng JSON trong câu trả lời")


def _period(v, known: list[Period], what: str) -> Period | None:
    if v is None:
        return None
    if not isinstance(v, dict) or "start" not in v or "end" not in v:
        raise SpecError(f'{what} phải là {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} hoặc null')
    try:
        start, end = date.fromisoformat(str(v["start"])[:10]), date.fromisoformat(str(v["end"])[:10])
    except ValueError:
        raise SpecError(f"{what}: ngày không hợp lệ {v!r}") from None
    if start > end or not (2000 <= start.year <= 2100 and 2000 <= end.year <= 2100):
        raise SpecError(f"{what}: khoảng ngày không hợp lệ {start} → {end}")
    for p in known:  # the same dates as a recognized period: keep its label ("tháng này", "7 ngày gần nhất"…)
        if (p.start, p.end) == (start, end):
            return p
    return Period(start, end)


def to_query(spec: dict, known_periods: list[Period]) -> Query:
    """Validated Query from a route == "sales" spec."""
    op, metric = spec.get("op"), spec.get("metric")
    if op not in OPS:
        raise SpecError(f"op phải là một trong {OPS}, nhận {op!r}")
    if metric not in METRICS:
        raise SpecError(f"metric phải là một trong {METRICS}, nhận {metric!r}")
    period = _period(spec.get("period"), known_periods, "period")
    filters = spec.get("filters") or {}
    if not isinstance(filters, dict):
        raise SpecError("filters phải là một đối tượng {chiều: tên}")
    clean = {}
    for dim, value in filters.items():
        if value in (None, "", []):
            continue
        if dim not in DIMS:
            raise SpecError(f"không lọc được theo {dim!r}")
        if isinstance(value, list):
            if not all(isinstance(x, (str, int)) for x in value):
                raise SpecError(f"filters[{dim!r}] phải là tên hoặc danh sách tên")
            clean[dim] = [str(x) for x in value]
        elif isinstance(value, (str, int)) and not isinstance(value, bool):
            clean[dim] = str(value)
        else:
            raise SpecError(f"filters[{dim!r}] phải là tên hoặc danh sách tên")
    base = None
    if op == "change":
        raw = spec.get("base")
        if period is None:
            raise SpecError("op 'change' cần period")
        if raw in (None, "prev"):
            base = prev_period(period)
        elif raw == "last_year":
            base = same_period_last_year(period)
        else:
            base = _period(raw, known_periods, "base")
    by, k = spec.get("by"), spec.get("k")
    if op == "top":
        if by not in DIMS + TIME_DIMS:
            raise SpecError(f"by phải là một trong {DIMS + TIME_DIMS}, nhận {by!r}")
        if k is None:
            k = 1
        if isinstance(k, bool) or not isinstance(k, int):
            raise SpecError(f"k phải là số nguyên, nhận {k!r}")
    order = spec.get("order") or "desc"
    if order not in ("desc", "asc"):
        raise SpecError(f'order phải là "desc" hoặc "asc", nhận {order!r}')
    return Query(
        op, metric, period, base, clean, by if op == "top" else None, k if op == "top" else None, order == "asc"
    )


# ------------------------------------------------------------------ stage


@dataclass
class Plan:
    route: str | None = None  # sales | code | unanswerable | None (the stage failed)
    spec: dict | None = None
    answer: dict | None = None  # the stage's vote; None = no vote
    evidence: dict | None = None
    error: str | None = None
    thinking: str | None = None
    raw: str | None = None
    llm_ms: int = 0
    cached: bool = False
    usage: dict | None = None


_TYPE_OF_OP = {
    "change": ("percent",),
    "best_day": ("date",),
    "top": ("string", "list", "date"),
    "value": ("number", "percent"),
}


def run_plan(
    client,
    q: dict,
    S: SalesView | None,
    rulebook: str,
    dimensions: str,
    hint_lines: list[str],
    weekday: str,
    known_periods: list[Period],
    coverage: tuple[date, date] | None,
    expected_type: str | None,
    thinking: bool = False,
    max_tokens: int = 1024,
) -> Plan:
    plan = Plan()
    try:
        reply = client.chat(
            build_messages(q, rulebook, dimensions, hint_lines, weekday),
            enable_thinking=thinking,
            max_tokens=max_tokens,
        )
    except Exception as e:  # server down, timeout…: this stage gives no vote
        plan.error = f"llm: {e}"
        return plan
    plan.thinking, plan.raw, plan.llm_ms, plan.cached = reply.reasoning, reply.content, reply.latency_ms, reply.cached
    plan.usage = {k: v for k, v in (reply.usage or {}).items() if isinstance(v, int)}
    try:
        spec = extract_json(re.sub(r"<think>.*?</think>", "", reply.content, flags=re.S))
        plan.spec = spec
        route = spec.get("route")
        if route not in ("sales", "code", "unanswerable"):
            raise SpecError(f"route không hợp lệ: {route!r}")
        plan.route = route
        if route == "code":
            return plan
        if route == "unanswerable":
            plan.answer = {"type": "unanswerable", "reason": str(spec.get("analysis") or "")[:300]}
            return plan
        query = to_query(spec, known_periods)
        result: Result = execute(query, S, coverage)
        plan.evidence = result.evidence
        got = result.answer["type"]
        if got != "unanswerable":
            # the question's wording says which type is expected; a plan of another shape misread it
            if expected_type and got != expected_type:
                raise SpecError(f"spec cho đáp án kiểu {got!r} nhưng câu hỏi cần {expected_type!r}")
            if got not in _TYPE_OF_OP[query.op]:
                raise SpecError(f"kiểu đáp án {got!r} không khớp op {query.op!r}")
        plan.answer = result.answer
    except (SpecError, QueryError) as e:
        plan.error = str(e)
    except Exception as e:  # a bug in the engine must never take the question down
        plan.error = f"{type(e).__name__}: {e}"
    return plan
