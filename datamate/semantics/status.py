"""Map the many ways shops write an order status to the four rulebook statuses (§8),
and recognize walk-in customers (khách lẻ)."""

from __future__ import annotations

import re

from datamate.core.text import key

COMPLETED, CANCELLED, RETURNED, SHIPPING = "hoàn thành", "huỷ", "trả hàng", "đang giao"
STATUSES = (COMPLETED, CANCELLED, RETURNED, SHIPPING)

# first match wins; patterns run on folded text ("Đã hủy" -> "da huy")
_RULES = (  # word boundaries matter: "van chuyen" contains "huy"
    (CANCELLED, re.compile(r"\bhuy\b|cancel|khong giao duoc|giao that bai|that bai|failed|\bvoid\b|bo don|khach bo")),
    (
        RETURNED,
        re.compile(r"tra hang|tra lai|doi tra|hang tra ve|hoan hang|hoan tra|hoan tien|return|refund|khach tra"),
    ),
    (
        SHIPPING,
        re.compile(
            r"dang giao|dang van chuyen|dang ship|dang chuyen|cho giao|chua giao|shipping|delivering|"
            r"in transit|dang xu ly|cho xu ly|dang gui|cho lay hang|dang dong goi|processing"
        ),
    ),
    (
        COMPLETED,
        re.compile(
            r"hoan thanh|xong|thanh cong|da lay|da giao|giao thanh cong|complete|done|success|"
            r"da thanh toan|finish|\bok\b"
        ),
    ),
)

_WALK_IN = {"khach le", "le", "kl", "khach vang lai", "vang lai", "khach", "khach hang le", "guest", "walk in", "khl"}


def canonical_status(raw) -> str | None:
    """'xong' -> 'hoàn thành', 'hủy đơn' -> 'huỷ', 'hàng trả về' -> 'trả hàng'; None if unrecognized."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    s = key(raw)
    for status, rx in _RULES:
        if rx.search(s):
            return status
    return None


def status_from_note(raw) -> str:
    """Shops without a status column write it in the note; a note that does not mention a
    status means a completed order (rulebook §8)."""
    st = canonical_status(raw)
    return st if st in (CANCELLED, RETURNED, SHIPPING) else COMPLETED


def is_walk_in_name(raw) -> bool:
    if raw is None or (isinstance(raw, float) and raw != raw):
        return True
    s = key(raw)
    return not s or s in _WALK_IN
