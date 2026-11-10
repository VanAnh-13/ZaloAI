"""Workspace = everything the agent needs about one business, built once and cached:
typed tables (ingest), the canonical sales model (semantics) and the schema card."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

from datamate.ingest import Business, build_business
from datamate.semantics import SalesView, build_sales
from datamate.semantics.order_rules import clean_business
from datamate.workspace.card import render_card


@dataclass
class Workspace:
    business: Business
    sales: SalesView | None
    card: str
    entities: list[tuple[str, str, str]] = field(default_factory=list)  # (value, table, column)
    sales_error: str | None = None

    @property
    def biz_id(self) -> str:
        return self.business.biz_id


def collect_entities(biz: Business, max_distinct: int = 5000) -> list[tuple[str, str, str]]:
    """Names that questions may mention (products, customers, groups…): text columns with
    a manageable number of distinct values."""
    out = []
    for t in biz.tables.values():
        for c in t.columns:
            if c.kind == "text" and 2 <= c.n_distinct <= max_distinct and not c.name.startswith("_"):
                out.extend((v, t.name, c.name) for v in t.df[c.name].dropna().astype(str).unique())
    return out


def build_workspace(biz_dir: Path) -> Workspace:
    biz = build_business(Path(biz_dir))
    clean_business(biz)  # rulebook §8 on order-lines tables
    sales, sales_error = None, None
    try:
        sales = build_sales(biz)
    except Exception:  # the raw tables stay usable even if the sales model cannot be built
        sales_error = traceback.format_exc(limit=4)
    return Workspace(biz, sales, render_card(biz, sales, sales_error), collect_entities(biz), sales_error)
