"""Local re-implementation of the official DataMate scoring (Evaluation tab).

    python -m eval.scorer --pred runs/x/submission.jsonl --gold <train>/answers.jsonl \
        --questions <train>/questions.jsonl [--ids-file eval/splits/regression_300.txt]

Per question: +1 correct, 0 abstain (or missing), −1 wrong.  PA = mean.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from datamate.core.text import norm

VALUE_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table"}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_eq(pred, gold) -> bool:
    """Exact match; only floating-point noise is ignored."""
    return _is_num(pred) and math.isclose(float(pred), float(gold), rel_tol=1e-9, abs_tol=1e-6)


def _pct_eq(pred, gold) -> bool:
    if not _is_num(pred):
        return False
    from datamate.core.numbers import round2

    return math.isclose(round2(pred), round2(gold), abs_tol=1e-9)


def _str_eq(pred, gold, alias) -> bool:
    if not isinstance(pred, str):
        return False
    accepted = {norm(gold)} | {norm(a) for a in (alias or []) if isinstance(a, str)}
    return norm(pred) in accepted


def _cell_eq(pred, gold, alias=None) -> bool:
    if _is_num(gold):
        return _num_eq(pred, gold)
    return _str_eq(pred, gold, alias)


def _list_eq(pred, gold, alias) -> bool:
    if not isinstance(pred, list) or len(pred) != len(gold):
        return False
    alias = alias or []
    return all(
        _cell_eq(p, g, alias[i] if i < len(alias) else None) for i, (p, g) in enumerate(zip(pred, gold, strict=True))
    )


def _table_eq(pred, gold, alias) -> bool:
    if (
        not isinstance(pred, dict)
        or not isinstance(pred.get("columns"), list)
        or not isinstance(pred.get("rows"), list)
    ):
        return False
    gcols, pcols = [norm(c) for c in gold["columns"]], [norm(c) for c in pred["columns"]]
    if len(gcols) != len(pcols):
        return False
    # columns by normalized name; if the names do not match, by position
    order = (
        [pcols.index(c) for c in gcols]
        if sorted(gcols) == sorted(pcols) and len(set(pcols)) == len(pcols)
        else list(range(len(gcols)))
    )
    prows = [r for r in pred["rows"] if isinstance(r, list) and len(r) == len(gcols)]
    if len(prows) != len(pred["rows"]) or len(prows) != len(gold["rows"]):
        return False
    alias = alias or []
    used = [False] * len(prows)
    for i, grow in enumerate(gold["rows"]):  # rows are matched as a set
        row_alias = alias[i] if i < len(alias) else None
        for j, prow in enumerate(prows):
            if used[j]:
                continue
            if all(_cell_eq(prow[order[k]], g, row_alias if k == 0 else None) for k, g in enumerate(grow)):
                used[j] = True
                break
        else:
            return False
    return True


def score_one(pred: dict | None, gold: dict) -> int:
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    pt, gt = pred.get("type"), gold["type"]
    if gt == "unanswerable":
        return 1 if pt == "unanswerable" else -1
    if pt != gt or pt not in VALUE_TYPES or "answer" not in pred:
        return -1
    p, g, alias = pred["answer"], gold["answer"], gold.get("alias")
    ok = {
        "number": lambda: _num_eq(p, g),
        "percent": lambda: _pct_eq(p, g),
        "string": lambda: _str_eq(p, g, alias),
        "date": lambda: isinstance(p, str) and p.strip() == g,
        "boolean": lambda: isinstance(p, bool) and p == g,
        "list": lambda: _list_eq(p, g, alias),
        "table": lambda: _table_eq(p, g, alias),
    }[gt]()
    return 1 if ok else -1


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(preds: dict[str, dict], golds: dict[str, dict], universe: list[str]) -> dict:
    scores = {i: score_one(preds.get(i), golds[i]) for i in universe}
    n = len(scores)
    c = sum(1 for s in scores.values() if s == 1)
    w = sum(1 for s in scores.values() if s == -1)
    return {
        "n": n,
        "correct": c,
        "wrong": w,
        "abstain": n - c - w,
        "pa": (c - w) / n if n else 0.0,
        "acc_answered": c / (c + w) if c + w else 0.0,
        "scores": scores,
    }


def _table(title: str, groups: dict[str, list[int]]) -> None:
    print(f"\n{title:<16} {'n':>5} {'đúng':>5} {'sai':>5} {'bỏ':>5} {'PA':>7}")
    for k in sorted(groups):
        s = groups[k]
        c, w = s.count(1), s.count(-1)
        print(f"{k:<16} {len(s):>5} {c:>5} {w:>5} {len(s) - c - w:>5} {(c - w) / len(s):>7.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.scorer")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--questions", help="questions.jsonl, for the per-business breakdown")
    ap.add_argument("--ids-file", help="score only these ids (default: every gold id)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    golds = {g["id"]: g for g in read_jsonl(args.gold)}
    preds = {p["id"]: p for p in read_jsonl(args.pred)}
    universe = sorted(golds)
    if args.ids_file:
        keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
        universe = [i for i in universe if i in keep]
    r = evaluate(preds, golds, universe)
    print(
        f"N={r['n']}  đúng={r['correct']}  sai={r['wrong']}  bỏ={r['abstain']}  "
        f"PA={r['pa']:.4f}  acc-khi-trả-lời={r['acc_answered']:.3f}  (có dự đoán: {sum(i in preds for i in universe)})"
    )
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, s in r["scores"].items():
        by_type[golds[i]["type"]].append(s)
    _table("theo type", by_type)
    if args.questions:
        biz = {q["id"]: q["business_id"] for q in read_jsonl(args.questions)}
        by_biz: dict[str, list[int]] = defaultdict(list)
        for i, s in r["scores"].items():
            by_biz[biz.get(i, "?")].append(s)
        _table("theo shop", by_biz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
