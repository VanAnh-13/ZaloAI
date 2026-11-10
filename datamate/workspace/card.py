"""Schema card: the markdown description of one business shown to the model."""

from __future__ import annotations

from datamate.ingest import Business
from datamate.semantics import SalesView

CARD_MAX_CHARS = 16000
TRUNCATED = "\n… (cắt bớt; dùng code để xem thêm)"


def render_card(
    biz: Business, sales: SalesView | None, sales_error: str | None = None, max_chars: int = CARD_MAX_CHARS
) -> str:
    """Full card if it fits; otherwise shorter and shorter column summaries; cut only as a last resort."""
    card = _render(biz, sales, sales_error, summary_chars=None)
    for chars in (80, 40):
        if len(card) <= max_chars:
            break
        card = _render(biz, sales, sales_error, summary_chars=chars)
    if len(card) > max_chars:
        card = card[:max_chars] + TRUNCATED
    return card


def _sales_section(sales: SalesView | None, sales_error: str | None) -> list[str]:
    lines = ["\n## Mô hình bán hàng chuẩn hoá `S` (tự nhận diện — kiểm tra lại trước khi tin)"]
    if sales is None:
        reason = "lỗi khi dựng" if sales_error else "không thấy bảng bán hàng rõ ràng"
        return lines + [f"`S` = None ({reason}): hãy làm trực tiếp trên các bảng `T`."]
    lines.append("Cột chuẩn ← nguồn: " + "; ".join(f"{role} ← `{src}`" for role, src in sales.sources.items()))
    lines += [f"- {n}" for n in sales.notes]
    lines.append(f"- `S.lines` có cột: {', '.join(sales.lines.columns)}")
    lines.append(f"- `S.orders` có cột: {', '.join(sales.orders.columns)}")
    return lines


def _render(biz: Business, sales: SalesView | None, sales_error: str | None, summary_chars: int | None) -> str:
    r = biz.readme
    lines = [f"# Shop `{biz.biz_id}`"]
    if r.data_end:
        lines.append(f"Dữ liệu có đến hết **{r.data_end:%d/%m/%Y}** (biến `DATA_END`).")
    lines.append("")
    lines.append("## README (đã lược phần mô tả cột, xem các bảng bên dưới)")
    lines.append(r.text_without_columns())
    if len(biz.units):
        ex = "; ".join(f"{u.product}: 1 {u.unit} = {u.factor:g} {u.base_unit}" for u in biz.units.head(4).itertuples())
        lines.append(
            f"\nQuy đổi đơn vị từ README: {len(biz.units)} dòng trong DataFrame `UNITS` "
            f"(cột product, unit, factor, base_unit). Ví dụ: {ex}"
        )
    lines += _sales_section(sales, sales_error)
    lines.append("\n## Bảng dữ liệu gốc (dict `T`, mỗi bảng là pandas DataFrame đã làm sạch)")
    for t in biz.tables.values():
        lines.append(f"\n### T[{t.name!r}] — {len(t.df):,} dòng")
        src = ", ".join(t.sources[:3]) + (f" … (+{len(t.sources) - 3} nguồn, đã gộp)" if len(t.sources) > 3 else "")
        lines.append(f"Nguồn: {src}")
        for n in t.notes[:3]:
            lines.append(f'Ghi chú khi đọc: "{n[:200]}"')
        for a in t.actions:
            lines.append(f"Đã xử lý: {a}")
        compact = summary_chars is not None
        empty = [c.name for c in t.columns if c.kind == "empty"] if compact else []
        lines.append("| cột | README mô tả | kiểu | tóm tắt |")
        lines.append("|---|---|---|---|")
        for c in t.columns:
            if c.name in empty:
                continue
            summary = c.summary
            if compact and len(summary) > summary_chars:
                summary = summary[:summary_chars] + "…"
            lines.append(f"| `{c.name}` | {c.desc} | {c.kind} | {summary} |")
        if empty:
            lines.append("Cột trống hoàn toàn: " + ", ".join(f"`{n}`" for n in empty))
    lines.append(
        "\nCột `_file`, `_sheet`, `_row` (và `_sheet_month`, `_sheet_year` nếu suy được từ tên sheet/file) "
        "cho biết nguồn của từng dòng."
    )
    return "\n".join(lines)
