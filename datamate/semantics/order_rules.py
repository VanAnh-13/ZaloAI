"""Rulebook §8 for tables written line by line per order:
- blank cells on the later lines of an order repeat the first line;
- two identical lines of the same order are an export error and count once.

Applies to the tables whose roles say "order lines" (order id + product + quantity or
amount); tables not written per order keep every row (§8: each row is one record).
"""

from __future__ import annotations

from datamate.ingest.columns import summarize
from datamate.ingest.loaders import PROVENANCE
from datamate.ingest.model import Business, Table
from datamate.semantics.roles import detect_roles


def is_order_lines(roles: dict[str, str]) -> bool:
    return (
        "order_id" in roles
        and bool({"product", "product_code"} & roles.keys())
        and bool({"qty", "amount", "base_qty"} & roles.keys())
    )


def apply_order_rules(t: Table, oid: str) -> bool:
    """Returns True when the table changed."""
    df, cols, changed = t.df, list(t.df.columns), False
    blank_id = df[oid].isna()
    if blank_id.any() and blank_id.mean() < 0.9:
        df[oid] = df[oid].ffill()
        t.actions.append(f"điền mã đơn trống ở dòng tiếp theo của đơn (cột `{oid}`, {int(blank_id.sum())} dòng)")
        changed = True
    g = df.groupby(oid, sort=False)
    first = g.cumcount() == 0
    multi = g[oid].transform("size") > 1
    later = multi & ~first
    if later.any():
        filled = []
        for c in cols:
            if c in PROVENANCE or c == oid:
                continue
            na = df[c].isna()
            # blank on (almost) every later line but filled on the first line: an order-level column
            if na[later].mean() >= 0.9 and na[first & multi].mean() <= 0.2:
                df[c] = df.groupby(oid, sort=False)[c].transform(lambda s: s.ffill())
                filled.append(c)
        if filled:
            t.actions.append(
                "điền ô trống ở các dòng sau của đơn theo dòng đầu: " + ", ".join(f"`{c}`" for c in filled)
            )
            changed = True
    dup = df.duplicated(subset=[c for c in cols if c not in PROVENANCE], keep="first") & df[oid].notna()
    if dup.any():
        df = df[~dup].reset_index(drop=True)
        t.actions.append(f"bỏ {int(dup.sum())} dòng trùng y hệt trong cùng một đơn (lỗi xuất file, §8)")
        changed = True
    t.df = df
    return changed


def clean_business(biz: Business) -> None:
    """Apply §8 to every order-lines table of the business, in place."""
    keys = tuple(biz.readme.order_keys)
    for t in biz.tables.values():
        if t.df.empty:
            continue
        roles = detect_roles(t, keys)
        if is_order_lines(roles) and apply_order_rules(t, roles["order_id"]):
            for ci in t.columns:  # refresh the card summaries
                ci.n_distinct, ci.summary = summarize(t.df[ci.name], ci.kind)
                if ci.n_bad:
                    ci.summary += f"; {ci.n_bad} ô không đọc được thành {ci.kind}"
