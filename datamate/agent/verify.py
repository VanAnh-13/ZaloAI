"""Sanity checks on an answer before it is accepted. A warning sends the answer back to
the model once ("kiểm tra lại"); the warnings also feed the confidence estimate."""

from __future__ import annotations

from datamate.agent.hints import Analysis
from datamate.core.text import key
from datamate.nlu.entities import EntityIndex


def check(ans: dict, analysis: Analysis, index: EntityIndex | None) -> list[str]:
    t = ans.get("type")
    if t == "abstain":
        return []
    out = []
    exp = analysis.expected_type
    if exp and t not in (exp, "unanswerable"):
        out.append(f"câu hỏi có vẻ cần type '{exp}' nhưng bạn nộp '{t}'")
    if t == "unanswerable":
        if analysis.all_inside:
            out.append(
                "kỳ hỏi nằm trong khoảng có dữ liệu — chỉ trả lời unanswerable nếu shop thật sự không ghi "
                "loại thông tin cần"
            )
        return out
    if analysis.all_outside:
        out.append("kỳ hỏi nằm hoàn toàn ngoài khoảng dữ liệu — có thể phải là unanswerable (§7)")
    v = ans.get("answer")
    if t in ("number", "percent") and v == 0 and analysis.all_inside:
        out.append("kết quả bằng 0 dù kỳ hỏi có dữ liệu — kiểm tra lại bộ lọc (tên, trạng thái, kỳ)")
    if index is not None and index.spellings:
        names = [v] if t == "string" else [x for x in v if isinstance(x, str)] if t == "list" else []
        unknown = [n for n in names if len(key(n)) >= 2 and n not in index]
        if unknown:
            out.append(
                f"tên {unknown[:3]} không có nguyên văn trong dữ liệu (sai dấu cũng bị chấm sai) — "
                "chép đúng tên như trong bảng"
            )
    return out
