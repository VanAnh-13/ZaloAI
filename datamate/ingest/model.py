"""Typed tables of one business, as produced by ingest."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from datamate.ingest.readme import Readme


@dataclass
class ColumnInfo:
    name: str
    kind: str  # number | date | text | empty | bool
    desc: str = ""  # README description, "" when the README does not describe the column
    n_distinct: int = 0
    summary: str = ""
    n_bad: int = 0  # non-blank raw cells that could not be converted to `kind`


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    sources: list[str]
    columns: list[ColumnInfo]
    notes: list[str] = field(default_factory=list)  # title/filter/footer rows found in the sheets
    actions: list[str] = field(default_factory=list)  # cleaning steps applied

    def column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)


@dataclass
class Business:
    biz_id: str
    readme: Readme
    tables: dict[str, Table]
    units: pd.DataFrame  # unit conversions listed in the README: product, unit, factor, base_unit

    @property
    def data_end(self) -> date | None:
        return self.readme.data_end

    def frames(self) -> dict[str, pd.DataFrame]:
        return {n: t.df for n, t in self.tables.items()}
