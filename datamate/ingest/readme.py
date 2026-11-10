"""Parse a business README.md: files, per-file column descriptions, conventions
(Quy ước), the last data date and the unit-conversion list."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from datamate.core.text import key


@dataclass
class ColumnSection:
    file: str  # file name, may be a pattern such as BanHang_T<tháng>_<năm>.xlsx
    sheet: str | None
    columns: dict[str, str]  # raw column name -> description

    def matches(self, file: str, sheet: str | None) -> bool:
        rx = "^" + re.sub(r"<[^>]+>", ".+?", re.escape(self.file).replace(r"\<", "<").replace(r"\>", ">")) + "$"
        if not re.match(rx, file, flags=re.I):
            return False
        return self.sheet is None or sheet is None or key(self.sheet) == key(sheet)


@dataclass
class UnitRule:
    product: str
    unit: str
    factor: float
    base_unit: str


@dataclass
class Readme:
    text: str
    intro: str = ""
    files: list[str] = field(default_factory=list)
    file_notes: dict[str, str] = field(default_factory=dict)
    sections: list[ColumnSection] = field(default_factory=list)
    conventions: list[str] = field(default_factory=list)
    data_end: date | None = None
    units: list[UnitRule] = field(default_factory=list)
    order_keys: list[str] = field(default_factory=list)  # "các dòng cùng **STT** là một đơn" -> ["STT"]

    def columns_for(self, file: str, sheet: str | None = None) -> dict[str, str]:
        """Column descriptions for one file/sheet; exact sheet sections win."""
        out: dict[str, str] = {}
        for sec in self.sections:
            if sec.matches(file, sheet):
                if sec.sheet is not None and sheet is not None:
                    return dict(sec.columns)
                out.update(sec.columns)
        return out

    def text_without_columns(self) -> str:
        """README minus the per-column sections and the long unit list (both are
        shown elsewhere in the schema card)."""
        out, skip = [], False
        for line in self.text.splitlines():
            if line.startswith("## "):
                skip = line.startswith("## Cột trong")
            if skip:
                continue
            if self.units and re.match(r"^\s+-\s+.+:\s*1\s+[^=]+=\s*[\d.,]+", line):
                continue
            out.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


_SEC = re.compile(r"^## Cột trong `([^`]+)`(?:,\s*sheet `([^`]+)`)?")
_COL = re.compile(r"^- \*\*(.+?)\*\*:\s*(.+)$")
_FILE = re.compile(r"^- `([^`]+)`(?:\s*\(([^)]*)\))?(?::\s*(.+))?$")
_UNIT = re.compile(r"^\s+-\s+(.+):\s*1\s+([^=]+?)\s*=\s*([\d.,]+)\s*(.+?)\s*$")
_END = re.compile(r"đến hết (?:ngày\s*)?(\d{1,2})/(\d{1,2})/(\d{4})")
_ORDER_KEY = re.compile(r"dòng (?:có )?cùng \*{0,2}`?([^*`,;.]{1,40}?)`?\*{0,2} (?:là|thuộc) (?:một|1|cùng một) đơn")


def parse_readme(text: str) -> Readme:
    r = Readme(text=text)
    section, current = None, None
    intro_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            m = _SEC.match(line)
            if m:
                current = ColumnSection(m.group(1), m.group(2), {})
                r.sections.append(current)
                section = "columns"
            else:
                current = None
                section = line[3:].strip().lower()
            continue
        if line.startswith("# "):
            continue
        if section is None:
            if line.strip():
                intro_lines.append(line.strip())
        elif section == "columns" and current is not None:
            m = _COL.match(line)
            if m:
                current.columns[m.group(1).strip()] = m.group(2).strip()
        elif section == "file":
            m = _FILE.match(line.strip())
            if m:
                r.files.append(m.group(1))
                if m.group(3):
                    r.file_notes[m.group(1)] = m.group(3).strip()
        elif section.startswith("quy ước"):
            mu = _UNIT.match(line)
            if mu:
                try:
                    r.units.append(
                        UnitRule(
                            mu.group(1).strip(),
                            mu.group(2).strip(),
                            float(mu.group(3).replace(",", ".")),
                            mu.group(4).strip(),
                        )
                    )
                except ValueError:
                    pass
            elif line.startswith("- "):
                r.conventions.append(line[2:].strip())
    r.intro = " ".join(intro_lines)
    r.order_keys = list(dict.fromkeys(k.strip() for k in _ORDER_KEY.findall(text)))
    m = _END.search(text)
    if m:
        try:
            r.data_end = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return r
