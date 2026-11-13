"""Check stage 1 of the cascade (templates + query engine + sales model + period rules) against
the train answers. No LLM involved.

    python -m eval.oracle --train C:/Users/Admin/Downloads/train/train [--ws runs/ws] [--show 20]

This runs exactly what the pipeline runs for a plain question (agent/templates.py →
agent/query.py), so the table below is the measured precision of the answers that cost no LLM
call. A mismatch points at a bug in ingest, the sales model, the period helpers or a template.
Keep every row at 100 % before trusting effort "fast".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from datamate.agent.query import QueryError, execute
from datamate.agent.templates import resolve
from datamate.config import load_settings
from datamate.nlu.periods import extract
from datamate.workspace import load_cached, prepare_many
from eval.scorer import score_one


def coverage_of(ref) -> tuple[date, date] | None:
    """Same coverage the pipeline gives the engine: first order date .. last data date."""
    if not ref.coverage:
        return None
    start, end = date.fromisoformat(ref.coverage[0]), date.fromisoformat(ref.coverage[1])
    if ref.data_end:
        end = max(end, date.fromisoformat(ref.data_end))
    return start, end


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
        hit = resolve(q["question"], extract(q["question"], q["as_of_date"], cal))
        if hit is None:
            continue
        kind, query = hit
        g = gold[q["id"]]
        S = sales.get(q["business_id"])
        if S is None:
            stats[kind]["skip"] += 1
            continue
        try:
            answer = execute(query, S, coverage_of(refs[q["business_id"]])).answer
        except QueryError:
            stats[kind]["skip"] += 1  # the engine declines: the question goes on to the model
            continue
        # periods outside the data are a rule of their own (§7): reported on a separate row
        row = "ngoài dữ liệu" if answer["type"] == "unanswerable" else kind
        ok = score_one(answer, g) == 1
        stats[row]["ok" if ok else "wrong"] += 1
        by_biz[q["business_id"]]["ok" if ok else "wrong"] += 1
        if not ok:
            misses.append(
                (
                    q["business_id"],
                    row,
                    q["question"],
                    q["as_of_date"],
                    answer.get("answer", "unanswerable"),
                    g.get("answer", g["type"]),
                )
            )

    covered = sum(c["ok"] + c["wrong"] for c in stats.values())
    print(f"{covered}/{len(qs)} câu ({covered / max(1, len(qs)):.1%}) được tầng 1 trả lời, không cần LLM\n")
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
