"""Prepare (and cache) the workspace of every business of a split, and report problems.

    python -m datamate.workspace --data C:/Users/Admin/Downloads/train/train --cards runs/cards

predict.sh runs it in the background while vLLM loads, so the run starts with a warm cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from datamate.config import load_settings
from datamate.workspace.batch import prepare_many


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    ap = argparse.ArgumentParser(prog="python -m datamate.workspace")
    ap.add_argument("--data", required=True, help="split folder that contains data/<biz>/")
    ap.add_argument("--ws", default=settings.ws_dir, help="cache folder for the pickled workspaces")
    ap.add_argument("--workers", type=int, default=settings.ingest_workers)
    ap.add_argument("--only", nargs="*", help="business ids to prepare")
    ap.add_argument("--cards", help="also write each schema card to this folder")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = Path(args.data)
    ids = sorted(p.name for p in (data / "data").iterdir() if p.is_dir())
    if args.only:
        ids = [b for b in ids if b in set(args.only)]
    t0 = time.time()
    refs = prepare_many(data, ids, Path(args.ws), args.workers)
    cards = Path(args.cards) if args.cards else None
    if cards:
        cards.mkdir(parents=True, exist_ok=True)
    n_err = n_warn = 0
    for b in ids:
        r = refs[b]
        status = "LỖI" if r.error else "ok"
        print(f"{b}: {status} {r.seconds:5.1f}s card={len(r.card):5d} ký tự, {len(r.warnings)} cảnh báo", flush=True)
        for w in r.warnings:
            print(f"     - {w}")
        if r.error:
            print("     " + r.error.strip().replace("\n", "\n     "))
        n_err += bool(r.error)
        n_warn += len(r.warnings)
        if cards:
            (cards / f"{b}.md").write_text(r.card, encoding="utf-8")
    print(f"\n{len(ids)} shop, {n_err} lỗi, {n_warn} cảnh báo, {time.time() - t0:.1f}s")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
