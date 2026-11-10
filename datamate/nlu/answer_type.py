"""Guess the expected answer type from the wording of the question (rulebook §9).

Heuristic and shown to the model as a hint only; the verifier also uses it to ask the
model to double-check an answer whose type looks wrong.
"""

from __future__ import annotations

import re

from datamate.core.text import fold, key

# dates and period words are removed first, so "tháng 3/2025 món nào" is not read as "2025 món"
_DATES = re.compile(
    r"\b(thang|th|quy|q|nam|ngay|tuan)\s*\d+([/.-]\d{1,4}){0,2}(\s*nam\s*\d+)?"
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}[/.-]\d{2,4}"
)

_THING = (
    r"(mat hang|san pham|hang|mon|khach|khach hang|nhan vien|nv|nha cung cap|ncc|lo|thuoc|tho|xe|dong may|hang"
    r"|chi nhanh|ban|ca|kenh|nhom|size|so)"
)
_TAIL = r"(\s+(a|ha|nhi|vay|z|v|the|nhe|nha|ban|shop|oi))*\s*$"


def expected_type(question: str) -> tuple[str, str] | None:
    """(type, why) or None when the wording gives no clear signal."""
    s = key(_DATES.sub(" ", fold(question)))
    raw = question.lower()
    if re.search(r"\bngay nao\b", s):
        return "date", "hỏi 'ngày nào' → \"YYYY-MM-DD\""
    if re.search(r"\bthang nao\b", s):
        return "date", "hỏi 'tháng nào' → \"YYYY-MM\""
    if re.search(
        rf"\b(nhung|cac|\d+) {_THING}\b.*\b(moi|tung) (nguoi|khach|mat hang|san pham|mon|loai|nhan vien)\b", s
    ) or re.search(rf"\b(nhung|cac|\d+) {_THING}\b.*\b(kem|va|cung)\b (so|tien|gia tri|so luong|so tien)\b", s):
        return "table", "hỏi nhiều đối tượng kèm số đo của từng đối tượng → table"
    if re.search(
        r"\b(moi|tung) (thang|loai|ky|ngay|nam|quy)\b.*\b(bao nhieu|bnhiu|may)\b|\blan luot\b|"
        r"\b(bao nhieu|bnhiu|may)\b.*\b(moi|tung) (thang|loai)\b|\b(chan|nguyen)\b.*\ble\b",
        s,
    ):
        return "list", "hỏi nhiều số → list theo đúng thứ tự câu hỏi"
    if "%" in raw or re.search(r"\b(ti le|ty le|phan tram)\b", s):
        return "percent", "hỏi phần trăm → số phần trăm (12.4 nghĩa là 12,4%)"
    if re.search(rf"\b(co|da)\b.*\b(khong|ko|k|chua|hong)\b{_TAIL}|\b(phai|dung) khong\b", s):
        return "boolean", "câu hỏi có/không → True/False"
    if re.search(rf"\btop\s*\d+|\b\d+ {_THING}\b|\bliet ke\b|\bnhung {_THING}\b", s):
        return "list", "hỏi nhiều đối tượng → list có thứ tự"
    if re.search(rf"\b{_THING} nao\b|\bai\b|\bla gi\b|\bgi\b|\bj\b", s):
        return "string", "hỏi một đối tượng → string (tên nguyên văn trong dữ liệu)"
    if re.search(r"\b(bao nhieu|bnhiu|bn|may|nhieu|nhiu)\b", s):
        return "number", "hỏi số lượng / số tiền → number"
    return None
