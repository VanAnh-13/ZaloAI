"""Business folder -> typed tables.

biz = build_business(Path("/data/data/biz_101"))
biz.tables["ban_hang"].df      # typed DataFrame, original column names
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datamate.ingest.columns import UNIT_SUFFIX, convert, infer_kind, summarize, units_in_cells
from datamate.ingest.loaders import PROVENANCE, RawTable, load_tables
from datamate.ingest.model import Business, ColumnInfo, Table
from datamate.ingest.readme import parse_readme
from datamate.ingest.values import is_blank


def typed_table(rt: RawTable) -> Table:
    df = rt.df
    infos, new = [], {}
    for col in df.columns:
        if col in PROVENANCE:
            new[col] = df[col]
            continue
        desc = rt.descriptions.get(col, "")
        kind = infer_kind(df[col].tolist(), col, desc)
        new[col] = convert(df, col, kind)
        n_raw = sum(not is_blank(v) for v in df[col].tolist())
        infos.append(ColumnInfo(col, kind, desc, n_bad=max(0, n_raw - int(new[col].notna().sum()))))
        if kind == "number":
            units = units_in_cells(df, col)  # "2 xe 4m3": the number alone is ambiguous
            if units is not None:
                new[col + UNIT_SUFFIX] = units
                infos.append(ColumnInfo(col + UNIT_SUFFIX, "text", f"đơn vị viết kèm số trong ô gốc của cột `{col}`"))
    t = Table(rt.name, pd.DataFrame(new), rt.sources, infos, list(rt.notes))
    for ci in t.columns:
        ci.n_distinct, ci.summary = summarize(t.df[ci.name], ci.kind)
        if ci.n_bad:
            ci.summary += f"; {ci.n_bad} ô không đọc được thành {ci.kind}"
    return t


def build_business(biz_dir: Path) -> Business:
    biz_dir = Path(biz_dir)
    readme_path = biz_dir / "README.md"
    readme = parse_readme(readme_path.read_text(encoding="utf-8") if readme_path.exists() else "")
    tables = {}
    for rt in load_tables(biz_dir, readme):
        t = typed_table(rt) if not rt.df.empty else Table(rt.name, rt.df, rt.sources, [], rt.notes)
        tables[t.name] = t
    units = pd.DataFrame([u.__dict__ for u in readme.units], columns=["product", "unit", "factor", "base_unit"])
    return Business(biz_dir.name, readme, tables, units)
