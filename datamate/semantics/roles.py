"""Which column of a table plays which role in the sales model.

Evidence, strongest first: the README description of the column (shops share a small
vocabulary: "mã đơn", "thành tiền sau giảm", "trạng thái đơn"…), then the column name.
Patterns run on folded names ("Mã hóa đơn" -> "ma hoa don", "NgayBan" -> "ngay ban").
Nothing here depends on a business id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from datamate.core.text import key
from datamate.ingest.model import Table


@dataclass(frozen=True)
class Rule:
    kinds: tuple[str, ...]
    desc: str = ""  # pattern on the folded README description (worth 3 points)
    name: str = ""  # pattern on the folded column name (worth 2 points)
    exclude: str = ""  # pattern on description or name that rules the column out


_NUM, _TXT, _DATE, _ID = ("number",), ("text",), ("date",), ("text", "number")

# Order matters: on equal score the earlier role wins the column.
RULES: dict[str, Rule] = {
    "order_id": Rule(
        _ID,  # "số thứ tự đơn" is an order key, "số thứ tự dòng trong đơn" is not
        desc=r"^(ma|so) (don|hoa don|bill|giao dich|chung tu)\b|^so thu tu (don|hoa don)\b",
        name=r"^(ma|so|id) ?(don|don hang|hoa don|hd|bill|chung tu|ct|giao dich|gd)$"
        r"|^(don|don hang|hoa don|hd|bill) (so|no|id)$|^(hd|bill|hoa don)$"
        r"|^(order|invoice)( id| no| code)?$",
        exclude=r"\bnhap\b|phieu thu|phieu chi|tra hang|tra lai",
    ),
    "product_code": Rule(
        _ID,
        desc=r"^ma (hang|san pham|mat hang|sp|hh)\b",
        name=r"^(ma|id) ?(hang|hang hoa|hh|sp|san pham|mat hang|hang ban|san pham ban)$"
        r"|^(ma )?sku$|^(product|item) (id|code)$",
        exclude=r"barcode|vach|lien quan|\blo\b",
    ),
    "product": Rule(
        _TXT,
        desc=r"^ten (hang|mon|san pham|mat hang|sp)\b|^san pham$|^mat hang$",
        name=r"^(ten )?(hang|hang hoa|hang ban|mat hang|san pham|sp|mon|mon an|chi tiet hang|hh)$"
        r"|^noi dung$|^(product|item)( name)?$",
        exclude=r"khach|nhom|\bma\b|loai|\bgia\b|dvt|don vi|so luong",
    ),
    "customer_code": Rule(
        _ID,
        desc=r"^ma khach",
        name=r"^(ma kh|ma khach|ma khach hang|so kh|ma khach mua|ma kh mua|id kh|ma the)$" r"|^customer (id|code)$",
    ),
    "customer": Rule(
        _TXT,
        desc=r"^ten (khach|kh)\b",
        name=r"^(khach|khach hang|ten khach|ten khach hang|ten kh|kh|doi tac|ben mua|nguoi mua"
        r"|khach hang ten|customer|customer name)$",
        exclude=r"\bma\b|\bso\b|nhom|loai|sdt|dien thoai|dia chi|\btra\b|\bno\b",
    ),
    "base_qty": Rule(
        _NUM,
        desc=r"don vi co ban",
        name=r"^(sl|so luong|slg) (quy doi|co ban|le|quy ra le|quy ra dv le|dvt co ban)$" r"|^quy ra dv le$",
        exclude=r"\btra\b|nhap|giao|\bton\b",
    ),
    "qty": Rule(
        _NUM,
        desc=r"^so luong( ban| san pham| hang)?$",
        name=r"^(so luong|sl|slg|sluong|soluong|qty|s l)( ban| hang| san pham| mua)?$",
        exclude=r"quy doi|co ban|quy ra|\ble\b|\btra\b|nhap|giao|\bton\b|\blo\b",
    ),
    "amount": Rule(
        _NUM,
        desc=r"^thanh tien( sau giam)?$|^tien ban$|thanh tien cua dong|^tri gia$",
        name=r"^(thanh tien|t tien|tt|tri gia|tien|tien ban|amount|thanh tien d|line total|tong tien"
        r"|thanh tien dong|so tien)$",
        exclude=r"truoc giam|\bvon\b|nhap|\bno\b|\btra\b|\bthu\b|^giam|tien giam|tien hang|phai tra",
    ),
    "goods_total": Rule(
        _NUM,
        desc=r"^(tong )?tien hang\b|cong thanh tien",
        name=r"^(tien hang|tong tien hang|tien hang truoc giam|subtotal)$",
    ),
    "unit_price": Rule(
        _NUM,
        desc=r"^don gia$|^gia ban$",
        name=r"^(don gia|gia|gia ban|gia le|gia ban le|gia san pham|gia don vi|gia 1 dv|price"
        r"|unit price|gia niem yet|don gia d)$",
        exclude=r"\bvon\b|nhap|\bcu\b|\bmoi\b|quy doi|hien tai",
    ),
    "discount": Rule(
        _NUM,
        desc=r"giam gia|tien giam|chiet khau|khuyen mai",
        name=r"^(giam gia|giam|chiet khau|ck|khuyen mai|tien giam|km|giam d|giam gia dong)$",
        exclude=r"truoc|sau|phai tra|da tinh|\btong\b",
    ),
    "unit_cost": Rule(_NUM, desc=r"^gia von", name=r"^gia von( luc ban)?$|^(unit )?cost$"),
    "date": Rule(
        _DATE,
        desc=r"^(ngay|thoi diem|thoi gian|gio) (ban|dat|lap|mua|giao dich|tao don|dat don|dat hang)\b",
        name=r"^(ngay|thoi gian|thoi diem|ngay gio|gio)( (ban|dat|dat hang|lap|lap hd|mua|giao dich|gd"
        r"|tao|tao don|hd|hoa don|chung tu|ct|tao hd))?$|^ngay thang$|^(order )?date$|^created( at)?$",
        exclude=r"giao hang|ngay giao\b(?! dich)|ngay cho|\bthu\b|nhap|sinh|\bhan\b|ap dung|tao khach"
        r"|tao mat hang|dang ky|\btra\b|bao hanh|\bgop\b|\bnhan\b",
    ),
    "status": Rule(
        _TXT,
        desc=r"^trang thai (don|cua don)|^tinh trang don",
        name=r"^(trang thai|tinh trang)( (giao|giao hang|don|don hang|xu ly|hd|hoa don))?$"
        r"|^(status|ket qua|xu ly|ghi chu tt|tt don)$",
        exclude=r"nhap|hddt|hoat dong|khach|phieu",
    ),
    "payment": Rule(
        _TXT,
        desc=r"hinh thuc thanh toan|phuong thuc thanh toan",
        name=r"^(phuong thuc|phuong thuc tt|phuong thuc thanh toan|httt|pttt|hinh thuc thanh toan"
        r"|hinh thuc tt|thanh toan bang|tra bang|payment|payment method)$",
        exclude=r"\bthu\b",
    ),
    "channel": Rule(
        _TXT, desc=r"^kenh", name=r"^(kenh|kenh ban|ban qua|nguon|nguon don|loai ban|kenh dat|channel|source)$"
    ),
    "staff": Rule(
        _TXT,
        desc=r"nhan vien|thu ngan",
        name=r"^(nhan vien|nv|ma nv|nguoi ban|thu ngan|ma nv ban|nv ban|nhan vien ban|staff)$",
    ),
    "unit": Rule(
        _TXT,
        desc=r"^don vi tinh$|^don vi$",
        name=r"^(dvt|don vi|don vi tinh|dv tinh|d vi|unit|dv)$",
        exclude=r"co ban|goc|nho|lon|quy doi|nhap",
    ),
    "return_qty": Rule(
        _NUM, desc=r"(so luong|sl) (khach )?tra", name=r"^(sl|so luong) tra( dvt co ban)?$|^so luong tra lai$"
    ),
    "note": Rule(_TXT, desc=r"^ghi chu$", name=r"^(ghi chu|note|ghi chu don|luu y)$", exclude=r"ghi chu tt"),
    "customer_group": Rule(
        _TXT,
        desc=r"^nhom khach|^hang khach|^loai khach",
        name=r"^(nhom khach|nhom khach hang|hang|hang khach|hang thanh vien|loai khach|nhom" r"|doi tuong kh)$",
        exclude=r"loai hinh",
    ),
    "product_group": Rule(
        _TXT,
        desc=r"^nhom hang|^danh muc|^loai (san pham|hang)",
        name=r"^(nhom hang|nhom hang hoa|danh muc|loai san pham|phan loai|nhom|loai hang"
        r"|nhom hang 3 cap|nhom sp|category)$",
        exclude=r"loai hang hoa",
    ),
}
_PRIORITY = {role: i for i, role in enumerate(RULES)}


def fold_name(s: str) -> str:
    """'NgayBan' -> 'ngay ban', 'Mã hóa đơn' -> 'ma hoa don' (split camelCase, then fold)."""
    s = str(s)
    return key("".join(" " + ch if i and ch.isupper() and s[i - 1].islower() else ch for i, ch in enumerate(s)))


def _main_clause(desc: str) -> str:
    """'thành tiền của dòng; Voucher … ghi số âm' -> 'thanh tien cua dong';
    'giờ đặt đơn (dòng trả hàng: lúc trả)' -> 'gio dat don'."""
    return key(re.sub(r"\([^)]*\)", " ", desc).split(";")[0])


def score_columns(table: Table, order_keys: tuple[str, ...] = ()) -> list[tuple[int, str, str]]:
    """Every (score, role, column) with a positive score, best first.
    order_keys: column names the README declares as the order key ("các dòng cùng STT là một đơn")."""
    keys = {fold_name(k) for k in order_keys}
    out = []
    for c in table.columns:
        if c.name.startswith("_") or c.kind == "empty":
            continue
        d, n = _main_clause(c.desc), fold_name(c.name)
        if n in keys:
            out.append((6, "order_id", c.name))
        for role, rule in RULES.items():
            if c.kind not in rule.kinds:
                continue
            if rule.exclude and (re.search(rule.exclude, d) or re.search(rule.exclude, n)):
                continue
            score = (3 if rule.desc and d and re.search(rule.desc, d) else 0) + (
                2 if rule.name and re.search(rule.name, n) else 0
            )
            if score:
                out.append((score, role, c.name))
    out.sort(key=lambda x: (-x[0], _PRIORITY[x[1]], x[2]))
    return out


def candidates(table: Table, role: str) -> list[str]:
    """Columns that could play `role`, best first (e.g. a date column and a date-time column)."""
    return [col for _, r, col in score_columns(table) if r == role]


def detect_roles(table: Table, order_keys: tuple[str, ...] = ()) -> dict[str, str]:
    """role -> column name; each column gets at most one role."""
    scored = score_columns(table, order_keys)
    # in a customer list (codes/names of customers, no orders), "Hạng"/"Nhóm" is the customer group
    is_customer_list = any(r in ("customer_code", "customer") for _, r, _ in scored) and not any(
        r in ("order_id", "amount", "qty") for _, r, _ in scored
    )
    roles: dict[str, str] = {}
    used: set[str] = set()
    for _, role, col in scored:
        if is_customer_list and role in ("product", "product_group"):
            continue
        if role not in roles and col not in used:
            roles[role] = col
            used.add(col)
    return roles
