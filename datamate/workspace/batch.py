"""Prepare the workspaces of many businesses in parallel processes.

The orchestrator keeps only a light `WorkspaceRef` per business; tables and the sales
model stay in the pickle that the sandbox loads.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from datamate.workspace.build import Workspace
from datamate.workspace.cache import cache_path, load_or_build
from datamate.workspace.card import TRUNCATED


@dataclass
class WorkspaceRef:
    biz_id: str
    data_dir: str
    pickle: str | None  # None when preparing failed: the agent must read DATA_DIR itself
    card: str
    data_end: str | None = None
    coverage: tuple[str, str] | None = None  # first/last order date of the sales model
    entities: list[tuple[str, str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


def _warnings(ws: Workspace) -> list[str]:
    out = []
    for t in ws.business.tables.values():
        if t.df.empty:
            out.append(f"bảng rỗng {t.name}: {t.notes}")
        for c in t.columns:
            if c.name.startswith("col_"):
                out.append(f"{t.name}.{c.name}: cột không có tên")
            if c.n_bad:
                out.append(f"{t.name}.{c.name}: {c.n_bad} ô không đọc được ({c.kind})")
    if ws.business.data_end is None:
        out.append("README không có ngày hết dữ liệu")
    if ws.sales is None:
        out.append(
            "không dựng được mô hình bán hàng S"
            + (f": {ws.sales_error.strip().splitlines()[-1]}" if ws.sales_error else "")
        )
    if ws.card.endswith(TRUNCATED):
        out.append(f"card bị cắt ({len(ws.card)} ký tự)")
    return out


def _fallback_card(d: Path, error: str) -> str:
    readme = (d / "README.md").read_text(encoding="utf-8") if (d / "README.md").exists() else ""
    files = ", ".join(sorted(p.name for p in d.iterdir() if p.name != "README.md"))
    return (
        f"# Shop `{d.name}`\n**Nạp dữ liệu tự động bị lỗi** ({error.strip().splitlines()[-1][:200]}). "
        f"`T` rỗng và `S` = None: hãy tự đọc file gốc trong thư mục `DATA_DIR` bằng pandas.\nFile: {files}\n\n"
        f"## README\n{readme}"
    )


def prepare_one(biz_dir: str, cache_dir: str) -> WorkspaceRef:
    t0 = time.time()
    d = Path(biz_dir)
    try:
        ws = load_or_build(d, Path(cache_dir))
        cov = ws.sales.coverage() if ws.sales is not None else None
        return WorkspaceRef(
            d.name,
            str(d),
            str(cache_path(d, cache_dir)),
            ws.card,
            str(ws.business.data_end) if ws.business.data_end else None,
            (str(cov[0]), str(cov[1])) if cov else None,
            ws.entities,
            _warnings(ws),
            seconds=time.time() - t0,
        )
    except Exception:
        err = traceback.format_exc(limit=6)
        return WorkspaceRef(d.name, str(d), None, _fallback_card(d, err), error=err, seconds=time.time() - t0)


def prepare_many(data_dir: Path, biz_ids: list[str], cache_dir: Path, workers: int = 4) -> dict[str, WorkspaceRef]:
    dirs = [str(Path(data_dir) / "data" / b) for b in biz_ids]
    if workers <= 1 or len(dirs) <= 1:
        refs = [prepare_one(d, str(cache_dir)) for d in dirs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(dirs))) as ex:
            refs = list(ex.map(prepare_one, dirs, [str(cache_dir)] * len(dirs)))
    return {r.biz_id: r for r in refs}
