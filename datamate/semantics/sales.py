"""Canonical sales model of one business, with the rulebook §1 definitions on top.

    S = build_sales(biz)
    S.lines    # one row per order line: order_id, order_date, product, qty, amount, is_return, …
    S.orders   # one row per order: order_id, date, status, customer_key, revenue, valid, …
    S.revenue(month(7, 2025))

Built automatically from column roles (semantics.roles); every choice is recorded in
`sources` / `notes` so the agent can check it against the raw tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from datamate.core.numbers import pct, round_money
from datamate.core.periods import Period
from datamate.core.text import key
from datamate.ingest.columns import UNIT_SUFFIX
from datamate.ingest.model import Business, Table
from datamate.semantics.roles import candidates, detect_roles
from datamate.semantics.status import (
    CANCELLED,
    COMPLETED,
    RETURNED,
    SHIPPING,
    canonical_status,
    is_walk_in_name,
    status_from_note,
)

LINE_ROLES = (
    "product",
    "product_code",
    "unit",
    "qty",
    "base_qty",
    "unit_price",
    "discount",
    "amount",
    "unit_cost",
    "return_qty",
)
ORDER_ROLES = ("date", "status", "note", "customer", "customer_code", "payment", "channel", "staff", "goods_total")
_RANK = {COMPLETED: 1, SHIPPING: 2, CANCELLED: 3, RETURNED: 4}
_BY_RANK = {v: k for k, v in _RANK.items()}
_RETURN_TABLE = re.compile(r"tra hang|tra lai|doi tra|hoan hang|return")
_NOT_SALES_TABLE = re.compile(r"nhap|purchase|giao hang|\blo\b|ton kho|thanh toan|phieu thu|bao hanh|\bgop\b|so quy")
# lines that are not goods: order-level discounts and shipping are not revenue (§1) …
_ADJUSTMENT = re.compile(
    r"^(voucher|ma giam gia|giam gia|dung \d+ diem|dung diem|tru diem|diem|phi ship|phi giao"
    r"|phi giao hang|phi van chuyen|ship|van chuyen|khuyen mai|chiet khau)\b"
)
# … while a service fee is revenue but not an item to rank or count (§3)
_SERVICE_FEE = re.compile(r"phi phuc vu|phi dich vu|service")
ITEM, SERVICE_FEE, ADJUSTMENT = "item", "service_fee", "adjustment"


@dataclass
class SalesView:
    lines: pd.DataFrame
    orders: pd.DataFrame
    sources: dict[str, str] = field(default_factory=dict)  # role -> "table.column"
    notes: list[str] = field(default_factory=list)

    # ---- rulebook §1 ---------------------------------------------------
    def orders_in(self, period: Period | None = None) -> pd.DataFrame:
        """Every order placed in the period (by order date), whatever its status."""
        o = self.orders
        return o if period is None else o[period.mask(o["date"])]

    def valid_orders(self, period: Period | None = None) -> pd.DataFrame:
        """Orders that count for revenue: not cancelled, not returned (even partly)."""
        o = self.orders_in(period)
        return o[o["valid"]]

    def valid_lines(self, period: Period | None = None, items_only: bool = True) -> pd.DataFrame:
        """Lines of the valid orders placed in the period, return lines excluded. items_only drops
        service fees, vouchers, points and shipping lines (not items to rank or count)."""
        ids = self.valid_orders(period)["order_id"]
        L = self.lines
        keep = L["order_id"].isin(ids) & ~L["is_return"]
        return L[keep & (L["kind"] == ITEM)] if items_only else L[keep]

    def revenue(self, period: Period | None = None) -> int:
        return round_money(self.valid_orders(period)["revenue"].sum())

    def order_count(self, period: Period | None = None) -> int:
        return int(len(self.valid_orders(period)))

    def cancelled_count(self, period: Period | None = None) -> int:
        return int((self.orders_in(period)["status"] == CANCELLED).sum())

    def cancel_rate(self, period: Period | None = None) -> float | None:
        """Cancelled orders / every order placed in the period, in percent (2 decimals)."""
        o = self.orders_in(period)
        return pct(int((o["status"] == CANCELLED).sum()), len(o))

    def avg_order_value(self, period: Period | None = None) -> int | None:
        v = self.valid_orders(period)
        return round_money(v["revenue"].sum() / len(v)) if len(v) else None

    def daily_revenue(self, period: Period | None = None) -> pd.Series:
        v = self.valid_orders(period)
        return v.groupby(v["date"].dt.normalize())["revenue"].sum()

    def coverage(self) -> tuple[date, date] | None:
        d = self.orders["date"].dropna()
        return (d.min().date(), d.max().date()) if len(d) else None


# ---------------------------------------------------------------- building


def _ids(s: pd.Series) -> pd.Series:
    """Order/customer ids as clean strings: 1001.0 -> '1001', ' ORD1 ' -> 'ORD1'."""

    def one(v):
        if v is None or (isinstance(v, float) and v != v):
            return np.nan
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip() or np.nan

    return s.map(one)


def _name_key(v):
    return key(v) if isinstance(v, str) and v.strip() else np.nan


def _has_time(s: pd.Series) -> bool:
    d = s.dropna()
    return bool(len(d)) and bool(((d.dt.hour != 0) | (d.dt.minute != 0)).any())


def _is_line_table(name: str, r: dict) -> bool:
    return (
        "order_id" in r
        and bool({"product", "product_code"} & r.keys())
        and bool({"amount", "unit_price"} & r.keys())
        and not _NOT_SALES_TABLE.search(key(name))
    )


def _is_invoice_table(name: str, r: dict) -> bool:
    return (
        "order_id" in r
        and bool({"goods_total", "amount"} & r.keys())
        and "date" in r
        and not _NOT_SALES_TABLE.search(key(name))
    )


def _is_row_sale_table(name: str, r: dict) -> bool:
    """A sales book without order ids: every row is one sale."""
    return (
        "order_id" not in r
        and bool({"product", "product_code"} & r.keys())
        and bool({"amount", "unit_price"} & r.keys())
        and "date" in r
        and not _NOT_SALES_TABLE.search(key(name))
    )


class _Builder:
    def __init__(self, biz: Business, roles: dict[str, dict[str, str]], line_table: str, per_line: bool):
        self.biz, self.roles, self.line_table, self.per_line = biz, roles, line_table, per_line
        self.sources: dict[str, str] = {}
        self.notes: list[str] = []

    def build(self) -> SalesView:
        t, r = self.biz.tables[self.line_table], self.roles[self.line_table]
        L = pd.DataFrame(index=t.df.index)
        if "order_id" in r:
            L["order_id"] = _ids(t.df[r["order_id"]])
            self.sources["order_id"] = f"{self.line_table}.{r['order_id']}"
        else:
            L["order_id"] = [f"dong_{i + 1}" for i in range(len(L))]
            self.sources["order_id"] = "(không có cột mã đơn: mỗi dòng là một đơn)"
        for role in LINE_ROLES + ORDER_ROLES:
            if role in r:
                L[role] = t.df[r[role]]
                self.sources[role] = f"{self.line_table}.{r[role]}"
        kind = "dòng hàng" if self.per_line else "hoá đơn (không có chi tiết dòng hàng)"
        self.notes.append(f"Bảng {kind}: T[{self.line_table!r}] ({len(L):,} dòng).")
        self._datetime(t, L)
        self._unit_fallback(t, r, L)
        header = self._pick_header(L["order_id"])
        if header:
            self._join_header(header, L)
        self._join_groups(L)
        self._amount(L)
        self._line_kinds(L)
        self._statuses(L)
        orders = self._orders(L)
        L["order_date"] = L["order_id"].map(orders.set_index("order_id")["date"])
        for c in ("customer_key", "walk_in"):
            L[c] = L["order_id"].map(orders.set_index("order_id")[c])
        self._summary(orders)
        return SalesView(L.reset_index(drop=True), orders, self.sources, self.notes)

    # -- columns -------------------------------------------------------
    def _datetime(self, t: Table, L: pd.DataFrame) -> None:
        """'date' = order date; 'datetime' = order date-time when some date column has times."""
        cands = [c for c in candidates(t, "date") if c in t.df]
        timed = [c for c in cands if _has_time(t.df[c])]
        if timed:
            L["datetime"] = t.df[timed[0]]
            self.sources["datetime"] = f"{self.line_table}.{timed[0]}"
            untimed = [c for c in cands if c not in timed]
            if untimed:
                L["date"] = t.df[untimed[0]]
                self.sources["date"] = f"{self.line_table}.{untimed[0]}"

    def _unit_fallback(self, t: Table, r: dict, L: pd.DataFrame) -> None:
        if "unit" not in L and "qty" in r and r["qty"] + UNIT_SUFFIX in t.df:
            L["unit"] = t.df[r["qty"] + UNIT_SUFFIX]
            self.sources["unit"] = f"{self.line_table}.{r['qty'] + UNIT_SUFFIX}"

    def _pick_header(self, line_ids: pd.Series) -> str | None:
        """Table with one row per order that brings order-level roles (date, status, customer…)."""
        ids = set(line_ids.dropna())
        best = None
        for name, r in self.roles.items():
            if name == self.line_table or "order_id" not in r or _NOT_SALES_TABLE.search(key(name)):
                continue
            hid = _ids(self.biz.tables[name].df[r["order_id"]]).dropna()
            if hid.empty or hid.nunique() < 0.99 * len(hid):
                continue
            overlap = len(ids & set(hid)) / max(1, min(len(ids), hid.nunique()))
            useful = [x for x in ORDER_ROLES + ("datetime",) if x in r]
            if overlap < 0.5 or not useful:
                continue
            score = (len(useful), overlap)
            if best is None or score > best[0]:
                best = (score, name)
        return best[1] if best else None

    def _join_header(self, name: str, L: pd.DataFrame) -> None:
        t, r = self.biz.tables[name], self.roles[name]
        hid = _ids(t.df[r["order_id"]])
        keep = ~hid.duplicated() & hid.notna()

        def mapper(col: str) -> pd.Series:
            return L["order_id"].map(pd.Series(t.df.loc[keep, col].values, index=hid[keep].values))

        joined = []
        for role in ORDER_ROLES:
            if role in r and (role not in L or L[role].isna().all()):
                L[role] = mapper(r[role])
                self.sources[role] = f"{name}.{r[role]}"
                joined.append(role)
        timed = [c for c in candidates(t, "date") if _has_time(t.df[c])]
        if "datetime" not in L and timed:
            L["datetime"] = mapper(timed[0])
            self.sources["datetime"] = f"{name}.{timed[0]}"
            joined.append("datetime")
        if joined:
            self.notes.append(f"Ghép bảng đơn T[{name!r}] theo mã đơn để lấy: {', '.join(joined)}.")

    def _join_groups(self, L: pd.DataFrame) -> None:
        """customer_group from a customer list, product_group from a product list."""
        for group, keys in (
            ("customer_group", ("customer_code", "customer")),
            ("product_group", ("product_code", "product")),
        ):
            if group in L:
                continue
            for name, r in self.roles.items():
                if name == self.line_table or group not in r:
                    continue
                k = next((k for k in keys if k in r and k in L), None)
                if k is None:
                    continue
                df = self.biz.tables[name].df
                src = _ids(df[r[k]]) if k.endswith("code") else df[r[k]].map(_name_key)
                dst = _ids(L[k]) if k.endswith("code") else L[k].map(_name_key)
                keep = src.notna() & ~src.duplicated()
                mapped = dst.map(pd.Series(df.loc[keep, r[group]].values, index=src[keep].values))
                if mapped.notna().mean() >= 0.5:
                    L[group] = mapped
                    self.sources[group] = f"{name}.{r[group]} (nối theo {k})"
                    break

    def _amount(self, L: pd.DataFrame) -> None:
        if not self.per_line:
            col = "goods_total" if "goods_total" in L else "amount"
            L["amount"] = pd.to_numeric(L[col], errors="coerce")
            self.sources["amount"] = self.sources[col]
            self.notes.append(f"Doanh thu mỗi đơn = `{self.sources[col]}` (bảng không có dòng hàng).")
            return
        if "amount" not in L and {"qty", "unit_price"} <= set(L):
            disc = L["discount"].fillna(0) if "discount" in L else 0
            L["amount"] = L["qty"] * L["unit_price"] - disc
            self.notes.append("Không thấy cột thành tiền: amount = qty × unit_price − discount.")
        elif "amount" not in L:
            L["amount"] = np.nan
            self.notes.append("CẢNH BÁO: không tính được thành tiền dòng; S.revenue() không dùng được.")

    def _statuses(self, L: pd.DataFrame) -> None:
        if "status" in L:
            raw = L["status"]
            L["status_raw"] = raw
            L["status"] = raw.map(canonical_status)
            vc = raw.value_counts()
            mapping = [
                f"'{v}'→{canonical_status(v) or 'KHÔNG NHẬN RA (coi như hoàn thành)'} ({n:,})" for v, n in vc.items()
            ]
            self.notes.append("Trạng thái: " + "; ".join(mapping[:12]) + ("; …" if len(mapping) > 12 else ""))
        elif "note" in L:
            L["status"] = L["note"].map(status_from_note)
            self.notes.append(f"Không có cột trạng thái: đọc từ ghi chú `{self.sources['note']}` (§8).")
        else:
            L["status"] = COMPLETED
            self.notes.append("Không có cột trạng thái: mọi đơn coi là hoàn thành.")
        neg = pd.Series(False, index=L.index)
        for c in ("amount", "qty"):
            if c in L:
                neg |= pd.to_numeric(L[c], errors="coerce") < 0
        # a negative voucher / points line is not a return; a negative item line is
        L["is_return"] = (neg & (L["kind"] == ITEM)) | (L["status"] == RETURNED)

    def _line_kinds(self, L: pd.DataFrame) -> None:
        """item / service_fee (revenue, not an item) / adjustment (voucher, points, shipping: not revenue)."""
        L["kind"] = ITEM
        if not self.per_line or "product" not in L:
            return
        name = L["product"].map(lambda v: key(v) if isinstance(v, str) else "")
        no_qty = L["qty"].isna() if "qty" in L else pd.Series(False, index=L.index)
        amount = pd.to_numeric(L["amount"], errors="coerce")
        adjustment = name.str.match(_ADJUSTMENT) | (no_qty & (amount <= 0)) | (no_qty & (name == ""))
        service = name.str.contains(_SERVICE_FEE) & ~adjustment
        L.loc[adjustment, "kind"] = ADJUSTMENT
        L.loc[service, "kind"] = SERVICE_FEE
        n_adj, n_fee = int(adjustment.sum()), int(service.sum())
        if n_adj or n_fee:
            ex = ", ".join(f"'{v}'" for v in L.loc[adjustment, "product"].dropna().unique()[:4])
            self.notes.append(
                f"Dòng không phải hàng: {n_adj:,} dòng voucher/điểm/phí giao (không tính doanh thu, "
                f"vd {ex or '—'}), {n_fee:,} dòng phí phục vụ (tính doanh thu, không phải mặt hàng); "
                "cột `kind` của S.lines."
            )

    def _returned_ids(self, L: pd.DataFrame) -> set:
        """Orders with a return anywhere: return lines, 'số lượng trả' > 0, or a returns table."""
        ids = set(L.loc[L["is_return"], "order_id"].dropna())
        if "return_qty" in L:
            ids |= set(L.loc[pd.to_numeric(L["return_qty"], errors="coerce") > 0, "order_id"].dropna())
        for name, r in self.roles.items():
            if name == self.line_table or "order_id" not in r:
                continue
            df, extra = self.biz.tables[name].df, set()
            if "return_qty" in r:
                extra = set(
                    _ids(df.loc[pd.to_numeric(df[r["return_qty"]], errors="coerce") > 0, r["order_id"]]).dropna()
                )
            elif _RETURN_TABLE.search(key(name)):
                extra = set(_ids(df[r["order_id"]]).dropna())
            if extra:
                self.notes.append(f"Đơn có trả hàng lấy thêm từ T[{name!r}]: {len(extra):,} đơn.")
                ids |= extra
        return ids

    # -- orders --------------------------------------------------------
    def _orders(self, L: pd.DataFrame) -> pd.DataFrame:
        L = L[L["order_id"].notna()]
        g_all = L.groupby("order_id", sort=False)
        kept = L[~L["is_return"]]
        o = pd.DataFrame(index=g_all.size().index)
        o.index.name = "order_id"
        for c in ("date", "datetime"):
            if c in L:
                d = pd.to_datetime(L[c], errors="coerce")
                o[c] = d[~L["is_return"]].groupby(kept["order_id"]).min().reindex(o.index)
                o[c] = o[c].fillna(d.groupby(L["order_id"]).min())
        if "date" not in o:
            o["date"] = o["datetime"] if "datetime" in o else pd.NaT
        rank = L["status"].map(_RANK).fillna(_RANK[COMPLETED])
        o["status"] = rank.groupby(L["order_id"]).max().map(_BY_RANK)
        o.loc[o.index.isin(self._returned_ids(L)), "status"] = RETURNED
        first = g_all.first()
        for c in ("status_raw", "customer", "customer_code", "customer_group", "payment", "channel", "staff", "note"):
            if c in L:
                o[c] = first[c]
        if self.per_line:  # §1: goods lines (and service fees); vouchers, points and shipping excluded
            o["revenue"] = (
                kept[kept["kind"] != ADJUSTMENT]
                .groupby("order_id", sort=False)["amount"]
                .sum()
                .reindex(o.index)
                .fillna(0.0)
            )
        else:  # one row per order (duplicated rows of an order count once)
            o["revenue"] = first["amount"].fillna(0.0)
        o["n_lines"] = g_all.size()
        self._customers(o)
        o["valid"] = ~o["status"].isin([CANCELLED, RETURNED])
        o = o.reset_index().sort_values(["date", "order_id"], kind="mergesort").reset_index(drop=True)
        return o

    def _customers(self, o: pd.DataFrame) -> None:
        """Rulebook §3: with customer codes, one code = one customer and an order without a
        code is a walk-in; otherwise names that differ only in spelling are one customer."""
        has_codes = "customer_code" in o and o["customer_code"].notna().any()
        if has_codes:
            code = _ids(o["customer_code"])
            o["walk_in"] = code.isna()
            o["customer_key"] = code.where(~o["walk_in"])
            self.notes.append(f"Khách xác định theo mã `{self.sources['customer_code']}`; đơn không có mã là khách lẻ.")
        elif "customer" in o:
            o["walk_in"] = o["customer"].map(is_walk_in_name)
            o["customer_key"] = o["customer"].map(_name_key).where(~o["walk_in"])
            self.notes.append(
                f"Khách xác định theo tên `{self.sources['customer']}` (bỏ dấu, hoa thường); "
                "'Khách lẻ'/'kl'/'lẻ'/trống là khách lẻ."
            )
        else:
            o["walk_in"] = True
            o["customer_key"] = np.nan

    def _summary(self, o: pd.DataFrame) -> None:
        vc = o["status"].value_counts()
        span = ""
        d = o["date"].dropna()
        if len(d):
            span = f", ngày đặt {d.min():%Y-%m-%d} → {d.max():%Y-%m-%d}"
        self.notes.append(
            f"S.orders: {len(o):,} đơn{span}; hợp lệ {int(o['valid'].sum()):,}; "
            + ", ".join(f"{s} {int(vc.get(s, 0)):,}" for s in (COMPLETED, SHIPPING, CANCELLED, RETURNED))
        )


def build_sales(biz: Business) -> SalesView | None:
    """None when no table looks like sales (the agent then works on the raw tables only)."""
    keys = tuple(biz.readme.order_keys)
    roles = {n: detect_roles(t, keys) for n, t in biz.tables.items() if not t.df.empty}
    for is_kind, per_line in ((_is_line_table, True), (_is_invoice_table, False), (_is_row_sale_table, True)):
        cands = [(len(biz.tables[n].df), n) for n, r in roles.items() if is_kind(n, r)]
        if cands:
            return _Builder(biz, roles, max(cands)[1], per_line).build()
    return None
