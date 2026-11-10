"""Pickle cache of workspaces. The sandbox loads the same pickle for every code run.

The fingerprint covers the business files and the source code that builds a
workspace, so editing a parser or a role rule invalidates stale caches.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

from datamate.workspace.build import Workspace, build_workspace

CACHE_VERSION = 3
_PKG = Path(__file__).resolve().parents[1]
_SOURCES = ("core", "ingest", "semantics", "workspace")


def _code_hash() -> bytes:
    h = hashlib.sha1(str(CACHE_VERSION).encode())
    for sub in _SOURCES:
        for p in sorted((_PKG / sub).glob("*.py")):
            h.update(p.read_bytes())
    return h.digest()


def fingerprint(biz_dir: Path) -> str:
    h = hashlib.sha1(_code_hash())
    for p in sorted(Path(biz_dir).iterdir()):
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def cache_path(biz_dir: Path, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{Path(biz_dir).name}.pkl"


def load_cached(path: Path) -> Workspace:
    with open(path, "rb") as f:
        _, _, ws = pickle.load(f)
    return ws


def load_or_build(biz_dir: Path, cache_dir: Path) -> Workspace:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_path(biz_dir, cache_dir)
    fp = fingerprint(biz_dir)
    if p.exists():
        try:
            with open(p, "rb") as f:
                ver, fp_old, ws = pickle.load(f)
            if ver == CACHE_VERSION and fp_old == fp:
                return ws
        except Exception:
            pass
    ws = build_workspace(biz_dir)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump((CACHE_VERSION, fp, ws), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)  # atomic: concurrent readers never see a half-written file
    return ws
