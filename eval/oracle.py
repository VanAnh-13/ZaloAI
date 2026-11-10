"""Check the sales model (SalesView) and period parsing against train answers.

    python -m eval.oracle --train C:/Users/Admin/Downloads/train/train [--ws runs/ws] [--show 20]

Only "plain" questions are used: once period words and filler words are removed, nothing
may remain (no product, customer or group filter). For those, the answer is computed
deterministically from S and compared with answers.jsonl. A mismatch points at a bug in
ingest, the sales model or the period helpers — no LLM involved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datamate.config import load_settings
from datamate.core.numbers import pct_change
from datamate.core.periods import prev_period, same_days_prev_month, same_period_last_year, this_month
from datamate.core.text import key
from datamate.nlu.periods import extract
from datamate.workspace import load_cached, prepare_many
from eval.scorer import score_one

_REVENUE = r"doanh thu|dthu|bao nhieu tien|may tien|nhieu tien|thu ve|thu duoc"
_FILLER = set(
    """
cua hang shop quan tiem doanh thu dthu tong cong ban duoc dc ra bao nhieu nhiu bnhiu nhieu may tien la trong cho
toi minh biet giup xem voi hoi nhe nha a the vay z ha khong ko k oi tinh so don bi huy ti ty le phan tram gia tri
trung binh quan mot moi tren mang ve ngay nao cao nhat dat xet theo thay doi tang giam hay lien truoc thang nam quy
hom nay qua tuan roi dem co da dang tat ca cac lai duoc luong hang ngay hnay hqua bua dat gi sao duoc nhe dum giup
jum cko mk mik e anh chi em ban ah u uh chua hien thi lam on bn nhiu nhieu phan tram kia vua bang tron
""".split()
)
_PERIOD_WORDS = re.compile(
    r"\b(thang|th|t|quy|q)\s*\d{1,2}(\s*(/|-|nam)\s*\d{2,4})?|\b\d{1,2}/\d{1,2}/\d{4}|\b\d{1,3}\s*ngay\b|\d+"
)


def is_plain(s: str) -> bool:
    rest = _PERIOD_WORDS.sub(" ", s)
    return all(tok in _FILLER for tok in re.findall(r"[a-z]+", rest))


def intent(s: str, raw: str) -> str | None:
    """s: folded question without punctuation (core.text.key); raw: original text (for '%')."""
    percent = "%" in raw or re.search(r"ti le|ty le|phan tram", s)
    if re.search(r"\bhuy\b", s):
        if percent:
            return "cancel_rate"
        if re.search(r"\b(bao nhieu|may|so|dem)\b", s):
            return "cancel_count"
        return None
    if re.search(r"trung binh|binh quan", s) and re.search(r"\bdon\b", s):
        return "avg_order"
    if re.search(r"\bngay nao\b", s) and re.search(r"nhat", s):
        return "best_day"
    if re.search(r"truoc|so voi", s) and percent and re.search(_REVENUE, s):
        return "pct_change"
    if re.search(_REVENUE, s):
        return "revenue"
    if re.search(r"\bdon\b", s) and re.search(r"bao nhieu|may", s):
        return "order_count"
    return None


def _base_period(s: str, periods: list):
    """The period a percentage change is measured against."""
    p = periods[0]
    if re.search(r"cung ky nam (truoc|ngoai)", s):
        return same_period_last_year(p)
    if re.search(r"(lien|ngay) truoc|truoc do|thang truoc|ky truoc", s) or len(periods) < 2:
        return same_days_prev_month(p.end) if p == this_month(p.end) else prev_period(p)
    return periods[1]


def compute(kind: str, S, periods: list, s: str) -> tuple[str, object] | None:
    """(answer type, value) from the sales model, None if the question cannot be mapped."""
    if not periods:
        return None
    p = periods[0]
    if kind == "revenue":
        return "number", S.revenue(p)
    if kind == "order_count":
        return "number", S.order_count(p)
    if kind == "cancel_count":
        return "number", S.cancelled_count(p)
    if kind == "cancel_rate":
        return "percent", S.cancel_rate(p)
    if kind == "avg_order":
        return "number", S.avg_order_value(p)
    if kind == "best_day":
        daily = S.daily_revenue(p)
        return ("date", daily.idxmax().strftime("%Y-%m-%d")) if len(daily) else None
    if kind == "pct_change":
        return "percent", pct_change(S.revenue(p), S.revenue(_base_period(s, periods)))
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.oracle")
    ap.add_argument("--train", required=True)
    ap.add_argument("--ws", default=load_settings().ws_dir)
    ap.add_argument("--workers", type=int, default=load_settings().ingest_workers)
    ap.add_argument("--show", type=int, default=15, help="mismatches to print")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = Path(args.train)
    qs = [json.loads(x) for x in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    gold = {a["id"]: a for a in map(json.loads, (root / "answers.jsonl").read_text(encoding="utf-8").splitlines())}
    import pandas as pd

    cal = pd.read_csv(root / "calendar.csv")
    refs = prepare_many(root, sorted({q["business_id"] for q in qs}), Path(args.ws), args.workers)
    sales = {b: (load_cached(r.pickle).sales if r.pickle else None) for b, r in refs.items()}

    stats: dict[str, Counter] = defaultdict(Counter)
    by_biz: dict[str, Counter] = defaultdict(Counter)
    misses = []
    for q in qs:
        s = key(q["question"])
        kind = intent(s, q["question"])
        if kind is None or not is_plain(s):
            continue
        g = gold[q["id"]]
        S = sales.get(q["business_id"])
        if S is None or g["type"] == "unanswerable":
            stats[kind]["skip"] += 1
            continue
        res = compute(kind, S, extract(q["question"], q["as_of_date"], cal), s)
        if res is None or res[1] is None:
            stats[kind]["skip"] += 1
            continue
        ok = score_one({"type": res[0], "answer": res[1]}, g) == 1
        stats[kind]["ok" if ok else "wrong"] += 1
        by_biz[q["business_id"]]["ok" if ok else "wrong"] += 1
        if not ok:
            misses.append((q["business_id"], kind, q["question"], q["as_of_date"], res[1], g["answer"]))

    print(f"{'mẫu câu':<14} {'đúng':>6} {'sai':>6} {'bỏ qua':>7} {'tỉ lệ đúng':>11}")
    tot = Counter()
    for k in sorted(stats):
        c = stats[k]
        tot.update(c)
        n = c["ok"] + c["wrong"]
        print(f"{k:<14} {c['ok']:>6} {c['wrong']:>6} {c['skip']:>7} {c['ok'] / n if n else 0:>11.3f}")
    n = tot["ok"] + tot["wrong"]
    print(f"{'TỔNG':<14} {tot['ok']:>6} {tot['wrong']:>6} {tot['skip']:>7} {tot['ok'] / n if n else 0:>11.3f}")
    bad = sorted(((c["wrong"], b) for b, c in by_biz.items() if c["wrong"]), reverse=True)
    if bad:
        print("\nshop có câu sai:", ", ".join(f"{b}({w}/{by_biz[b]['ok'] + w})" for w, b in bad))
    for m in misses[: args.show]:
        print(f"  [{m[0]} {m[1]}] {m[2]!r} (as_of {m[3]}): tính={m[4]!r} đáp án={m[5]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
