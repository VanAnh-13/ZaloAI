"""Read every data file of a business into raw tables.

Generic on purpose: nothing here may depend on a business id. Header rows are
detected (title/filter rows above the header, two-level headers), sheets and
split files with the same header are concatenated, provenance is kept in
_file / _sheet / _row columns.
"""

from __future__ import annotations

import csv
import io
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from datamate.core.text import fold, key
from datamate.ingest.readme import Readme
from datamate.ingest.values import is_blank, month_year_from_name, parse_date, parse_number

HEADER_SCAN_ROWS = 15
# total / note rows that hand-kept books put inside the data ("Tổng tháng", "Ghi chú: …")
_FOOTER = re.compile(r"^(tổng|cộng|ghi chú|lưu ý|total|subtotal|grand total)\b")
_FOOTER_FOLDED = re.compile(r"^(tong cong|tong thang|tong so|tong tien|ghi chu|luu y)\b")
PROVENANCE = ("_file", "_sheet", "_row", "_sheet_month", "_sheet_year")


@dataclass
class RawSheet:
    file: str
    sheet: str | None
    rows: list[list]


@dataclass
class RawTable:
    name: str
    df: pd.DataFrame  # object dtype, original column names + provenance columns
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # title/filter rows found above headers
    descriptions: dict[str, str] = field(default_factory=dict)  # column -> README description
    from_sqlite: bool = False


# ----------------------------------------------------------------- readers


def read_xlsx(path: Path) -> list[RawSheet]:
    from python_calamine import CalamineWorkbook

    wb = CalamineWorkbook.from_path(str(path))
    out = []
    for name in wb.sheet_names:
        rows = wb.get_sheet_by_name(name).to_python(skip_empty_area=False)
        out.append(RawSheet(path.name, name, [list(r) for r in rows]))
    return out


def read_csv(path: Path) -> list[RawSheet]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0] if text else ""
    delim = max([",", ";", "\t", "|"], key=first.count) if first else ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)]
    return [RawSheet(path.name, None, rows)]


def read_sqlite(path: Path) -> list[tuple[str, pd.DataFrame]]:
    # /data is mounted read-only: immutable=1 stops sqlite from creating journal files
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        names = [
            r[0]
            for r in con.execute(
                "select name from sqlite_master "
                "where type in ('table','view') and name not like 'sqlite_%' order by name"
            )
        ]
        return [(n, pd.read_sql_query(f'select * from "{n}"', con)) for n in names]
    finally:
        con.close()


# ---------------------------------------------------------- header finding


def _cell_kind(v) -> str:
    if is_blank(v):
        return "blank"
    if isinstance(v, (datetime, date, time)):
        return "date"
    if isinstance(v, bool):
        return "text"
    if isinstance(v, (int, float)):
        return "num"
    s = str(v)
    if parse_number(s) is not None:
        return "num"
    if parse_date(s) is not None:
        return "date"
    return "text"


def detect_header(rows: list[list], readme_keys: set[str]) -> int:
    """Index of the header row: most README column names matched, otherwise the
    widest all-text row (earliest on ties)."""
    width = max((sum(not is_blank(c) for c in r) for r in rows[:200]), default=0)
    if width == 0:
        return 0
    best, best_score = 0, float("-inf")
    for i, r in enumerate(rows[:HEADER_SCAN_ROWS]):
        cells = [c for c in r if not is_blank(c)]
        if not cells:
            continue
        kinds = [_cell_kind(c) for c in cells]
        text_ratio = kinds.count("text") / len(kinds)
        fill = len(cells) / width
        matched = sum(1 for c in cells if isinstance(c, str) and key(c) in readme_keys)
        score = matched * 10 + fill * text_ratio * 5
        if len({str(c).strip() for c in cells}) < len(cells):
            score -= 2  # header names are normally unique
        if score > best_score:
            best, best_score = i, score
    return best


def _header_names(cells: list) -> list[str]:
    names, seen = [], {}
    for j, c in enumerate(cells):
        n = "" if is_blank(c) else re.sub(r"\s+", " ", str(c)).strip()
        n = n or f"col_{j + 1}"
        if n in seen:
            seen[n] += 1
            n = f"{n} ({seen[n]})"
        else:
            seen[n] = 1
        names.append(n)
    return names


def is_footer_row(cells: list, width: int) -> bool:
    """A total/note row inside the data: first text cell starts with Tổng/Cộng/Ghi chú…
    and the row is mostly empty. Keeping it would double-count sums."""
    filled = [c for c in cells if not is_blank(c)]
    texts = [c for c in filled if isinstance(c, str)]
    if not texts or len(filled) > max(3, width // 2):
        return False
    first = texts[0].strip().lower()
    return bool(_FOOTER.match(first) or _FOOTER_FOLDED.match(fold(first)))


def sheet_to_frame(sh: RawSheet, readme: Readme) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    desc = readme.columns_for(sh.file, sh.sheet)
    rows = sh.rows
    if not rows:
        return pd.DataFrame(), [], desc
    h = detect_header(rows, {key(c) for c in desc})
    header = _header_names(rows[h])
    notes = [" | ".join(str(c).strip() for c in r if not is_blank(c)) for r in rows[:h]]
    notes = [n for n in notes if n]
    width = len(header)
    body = []
    header_keys = [key(x) for x in header]
    footers: dict[str, int] = {}
    for i, r in enumerate(rows[h + 1 :], start=h + 2):
        r = list(r) + [None] * (width - len(r))
        extra = [c for c in r[width:] if not is_blank(c)]
        r = r[:width]
        if all(is_blank(c) for c in r) and not extra:
            continue
        if [key(c) if isinstance(c, str) else None for c in r] == header_keys:
            continue  # header repeated inside the data
        if is_footer_row(r, width):
            label = next(str(c).strip() for c in r if isinstance(c, str) and not is_blank(c))
            footers[label[:120]] = footers.get(label[:120], 0) + 1
            continue
        body.append(r + [i])
    if footers:
        n = sum(footers.values())
        notes.append(
            f"đã bỏ {n} dòng tổng/ghi chú nằm trong dữ liệu: " + "; ".join(f"'{k}'" for k in list(footers)[:6])
        )
    df = pd.DataFrame(body, columns=header + ["_row"], dtype=object)
    m, y = month_year_from_name(sh.sheet or "")
    if y is None:
        m, y = month_year_from_name(Path(sh.file).stem)
    df.insert(0, "_file", sh.file)
    df.insert(1, "_sheet", sh.sheet)
    if y is not None:
        df["_sheet_year"] = y
        df["_sheet_month"] = m
    return df, notes, desc


# ------------------------------------------------------------- assembling

_SPLIT_SUFFIX = [
    re.compile(r"[_\- ]?(?:t|thang|th)?\d{1,2}[_\- ]\d{4}$", re.I),
    re.compile(r"[_\- ]\d{4}$"),
    re.compile(r"[_\- ]\d{1,2}$"),
]


def file_base(stem: str) -> str:
    """'SoBanHang_1' / 'SoBanHang_2026' / 'BanHang_T10_2025' -> 'SoBanHang' / 'BanHang'."""
    prev = None
    while prev != stem:
        prev = stem
        for rx in _SPLIT_SUFFIX:
            new = rx.sub("", stem)
            if new:
                stem = new
    return stem


def load_tables(biz_dir: Path, readme: Readme) -> list[RawTable]:
    sheets: list[tuple[RawSheet, pd.DataFrame, list[str], dict[str, str]]] = []
    tables: list[RawTable] = []
    for path in sorted(biz_dir.iterdir()):
        suf = path.suffix.lower()
        try:
            if suf in (".xlsx", ".xlsm", ".xls"):
                raw = read_xlsx(path)
            elif suf in (".csv", ".txt", ".tsv"):
                raw = read_csv(path)
            elif suf in (".sqlite", ".db", ".sqlite3"):
                for name, df in read_sqlite(path):
                    df = df.astype(object).where(df.notna(), None)
                    df.insert(0, "_file", path.name)
                    df.insert(1, "_sheet", name)
                    tables.append(
                        RawTable(
                            name,
                            df,
                            [f"{path.name}:{name}"],
                            descriptions=readme.columns_for(path.name, name),
                            from_sqlite=True,
                        )
                    )
                continue
            else:
                continue
        except Exception as e:  # never let one bad file kill the business
            tables.append(RawTable(f"LOI_{path.stem}", pd.DataFrame(), [path.name], notes=[f"không đọc được: {e}"]))
            continue
        for sh in raw:
            df, notes, desc = sheet_to_frame(sh, readme)
            if not df.empty:
                sheets.append((sh, df, notes, desc))

    # group sheets with the same header (and file base) into one table
    groups: dict[tuple, list] = {}
    for item in sheets:
        sh, df = item[0], item[1]
        cols = tuple(key(c) for c in df.columns if c not in PROVENANCE)
        groups.setdefault((file_base(Path(sh.file).stem), cols), []).append(item)
    headers_per_file: dict[str, set] = {}
    for (_, cols), items in groups.items():
        for it in items:
            headers_per_file.setdefault(it[0].file, set()).add(cols)

    used: set[str] = set()
    for (base, _), items in groups.items():
        sh0 = items[0][0]
        multi = len(headers_per_file[sh0.file]) > 1
        generic_sheet = (
            sh0.sheet is None
            or re.fullmatch(r"(?i)sheet\s*\d*", sh0.sheet or "")
            or month_year_from_name(sh0.sheet) != (None, None)
        )  # "T7-2025" names a period
        name = sh0.sheet if multi and not generic_sheet else base
        n, i = name, 2
        while n in used:
            n, i = f"{name}_{i}", i + 1
        used.add(n)
        df = pd.concat([it[1] for it in items], ignore_index=True)
        notes = list(dict.fromkeys(nt for it in items for nt in it[2]))
        desc: dict[str, str] = {}
        for it in items:
            desc.update(it[3])
        srcs = [f"{it[0].file}" + (f"[{it[0].sheet}]" if it[0].sheet else "") for it in items]
        tables.append(RawTable(n, df, srcs, notes, desc))
    return tables
