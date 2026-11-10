"""`dm` — the helper library available to model-written code (façade only: the
implementations live in datamate.core and datamate.semantics)."""

from datamate.core.numbers import pct, pct_change, round2, round_half_up, round_money
from datamate.core.periods import (
    Period,
    as_date,
    before_tet,
    day,
    last_month,
    last_n_days,
    last_week,
    month,
    months_ending,
    prev_period,
    quarter,
    same_days_prev_month,
    same_period_last_year,
    shift_of,
    tet,
    this_month,
    this_week,
    today,
    until,
    vn_weekday,
    year,
    yesterday,
    ytd,
)
from datamate.core.text import find, fold, key, norm
from datamate.semantics.ranking import top_k

__all__ = [
    "DOC",
    "Period",
    "as_date",
    "before_tet",
    "day",
    "find",
    "fold",
    "key",
    "last_month",
    "last_n_days",
    "last_week",
    "month",
    "months_ending",
    "norm",
    "pct",
    "pct_change",
    "prev_period",
    "quarter",
    "round2",
    "round_half_up",
    "round_money",
    "same_days_prev_month",
    "same_period_last_year",
    "shift_of",
    "tet",
    "this_month",
    "this_week",
    "today",
    "top_k",
    "until",
    "vn_weekday",
    "year",
    "yesterday",
    "ytd",
]

# Reference shown to the model in the system prompt. tests/test_dm.py checks that every
# dm.<name> mentioned here exists.
DOC = """\
Kỳ thời gian — mọi kỳ là `Period(start, end)` gồm cả 2 đầu; `p.mask(df[col])` → mask các dòng có ngày trong kỳ:
- dm.today(AS_OF), dm.yesterday(AS_OF), dm.day(d)
- dm.this_week(AS_OF) (thứ Hai → hôm nay), dm.last_week(AS_OF) (trọn tuần trước, thứ Hai → Chủ nhật)
- dm.this_month(AS_OF) (ngày 1 → hôm nay), dm.last_month(AS_OF), dm.same_days_prev_month(AS_OF) (so "tháng này" với tháng trước)
- dm.month(m, y) — cả tháng; dm.month(m, as_of=AS_OF) khi câu hỏi không ghi năm (luật năm nay / năm trước)
- dm.quarter(q, y) hoặc dm.quarter(q, as_of=AS_OF); dm.year(y); dm.ytd(AS_OF) (từ đầu năm → hôm nay)
- dm.last_n_days(n, AS_OF) (n ngày kể cả hôm nay); dm.until(d) (từ đầu dữ liệu → hết ngày d: còn nợ, còn tồn…)
- dm.months_ending(m, y, n) → list n tháng liền nhau kết thúc ở m/y, cũ → mới
- dm.prev_period(p) (kỳ trước tương ứng: tháng/quý/năm trước; "tháng này" → cùng số ngày tháng trước), dm.same_period_last_year(p)
- dm.tet(y, CAL), dm.before_tet(y, CAL) (Tết = 23 tháng Chạp → mùng 10 tháng Giêng, tra CAL)
- dm.shift_of(ts) → "sáng" (06:00–13:59) | "chiều" (14:00–21:59) | None; dm.vn_weekday(d) → "Thứ Hai"…
Số (làm tròn nửa lên — KHÔNG dùng round() của Python):
- dm.round_money(x) → int đồng; dm.round2(x) → 2 chữ số
- dm.pct(part, whole) → phần trăm 2 chữ số (12.4 nghĩa là 12,4%); dm.pct_change(new, old) → % thay đổi, giảm ra số âm
Tên / thực thể:
- dm.find(candidates, query, k=5) → [(tên, điểm 0–100)]: so khớp không dấu, chịu lỗi gõ; dùng để tìm đúng tên trong dữ liệu
- dm.key(s) → khoá so sánh không dấu, bỏ dấu câu; dm.norm(s) → chuẩn hoá như bộ chấm (giữ dấu)
Xếp hạng:
- dm.top_k(df, by, k=None, code=None, name=None, ascending=False) → (rows, tie_at_boundary): xếp `by` giảm dần,
  bằng nhau thì theo cột mã `code` (không có mã thì `name`) tăng dần. tie_at_boundary=True thường nghĩa là đang tính sai.

Mô hình bán hàng `S` (có thể là None — xem schema card; đã áp luật §1 của rulebook):
- S.orders: mỗi dòng một đơn — order_id, date (ngày đặt), datetime (nếu có giờ), status ∈ {hoàn thành, huỷ, trả hàng,
  đang giao}, status_raw, customer, customer_code, customer_key (mã khách, hoặc tên chuẩn hoá; NaN = khách lẻ), walk_in,
  payment, channel, staff, customer_group (nếu có), revenue (doanh thu đơn = tổng thành tiền dòng), n_lines,
  valid (= không huỷ, không trả hàng — kể cả trả một phần)
- S.lines: mỗi dòng một dòng hàng — order_id, order_date, product, product_code, unit, qty, base_qty, unit_price,
  discount, amount (thành tiền dòng), unit_cost, product_group, status, is_return (dòng trả hàng), customer_key, walk_in
  (chỉ có những cột tìm được trong dữ liệu)
- S.orders_in(p) — mọi đơn đặt trong kỳ (mọi trạng thái); S.valid_orders(p); S.valid_lines(p) (dòng của đơn hợp lệ)
- S.revenue(p) → int; S.order_count(p); S.cancelled_count(p); S.cancel_rate(p) → %; S.avg_order_value(p) → int;
  S.daily_revenue(p) → Series doanh thu theo ngày. p=None nghĩa là toàn bộ dữ liệu.
"""
