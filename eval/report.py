"""Where a run wins, loses and spends its time — per stage of the cascade.

    python -m eval.report --logs runs/x/logs/logs.jsonl --gold <train>/answers.jsonl \
        [--ids-file eval/splits/regression_300.txt] [--show 15] [--failures runs/x/failures.jsonl]

Reads the per-question logs (votes of every stage, timing, code) and the reference answers:
- PA, and what the abstention rule gained or cost;
- precision of each voter (template / planner / code attempt) — which stage can be trusted;
- precision by agreement pattern — the evidence behind "two votes or abstain";
- breakdown by answer type and by shop, worst first;
- LLM calls, tokens and seconds per stage — where a slow run loses its time;
- the wrong answers, grouped by kind of mistake, with the question, the votes and the last code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datamate.agent.answer import canonical
from eval.scorer import read_jsonl, score_one


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def mistake(pred: dict, gold: dict) -> str:
    """Rough family of a wrong answer, to see what to fix first."""
    pt, gt = pred.get("type"), gold["type"]
    if gt == "unanswerable":
        return "đáng ra unanswerable"
    if pt == "unanswerable":
        return "unanswerable sai"
    if pt != gt:
        return f"sai kiểu ({pt} thay vì {gt})"
    p, g = pred.get("answer"), gold.get("answer")
    if gt in ("number", "percent") and _is_num(p) and _is_num(g):
        if g and abs(p - g) / abs(g) < 0.02:
            return "số lệch < 2% (làm tròn, một vài đơn/dòng)"
        if g and p and (abs(p / g - 1) > 0.5):
            return "số lệch lớn (sai kỳ, sai bộ lọc hoặc sai công thức)"
        return "số lệch vừa (đơn huỷ/trả, bộ lọc)"
    if gt == "list" and isinstance(p, list) and isinstance(g, list):
        if len(p) != len(g):
            return "list sai số phần tử"
        return (
            "list đúng phần tử nhưng sai thứ tự" if sorted(map(str, p)) == sorted(map(str, g)) else "list sai phần tử"
        )
    if gt == "string":
        return "sai tên"
    if gt == "date":
        return "sai ngày/tháng"
    return f"sai giá trị ({gt})"


def _line(label: str, scores: list[int], width: int = 34) -> str:
    n, c, w = len(scores), scores.count(1), scores.count(-1)
    prec = f"{c / (c + w):.3f}" if c + w else "  —  "
    return f"{label:<{width}} {n:>5} {c:>5} {w:>5} {n - c - w:>5} {prec:>7} {(c - w) / n if n else 0:>7.3f}"


def _table(title: str, groups: dict[str, list[int]], order: list[str] | None = None, width: int = 34) -> None:
    print(f"\n{title:<{width}} {'n':>5} {'đúng':>5} {'sai':>5} {'bỏ':>5} {'chính xác':>7} {'PA':>7}")
    for k in order or sorted(groups):
        print(_line(k, groups[k], width))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.report")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ids-file", help="only these ids")
    ap.add_argument("--show", type=int, default=12, help="wrong answers to print")
    ap.add_argument("--failures", help="write every wrong or abstained question to this jsonl file")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gold = {g["id"]: g for g in read_jsonl(args.gold)}
    logs = [r for r in read_jsonl(args.logs) if r["id"] in gold]
    if args.ids_file:
        keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
        logs = [r for r in logs if r["id"] in keep]
    if not logs:
        print("không có câu nào khớp giữa log và đáp án")
        return 1

    final = [score_one(r.get("answer"), gold[r["id"]]) for r in logs]
    n, c, w = len(final), final.count(1), final.count(-1)
    print(
        f"N={n}  đúng={c}  sai={w}  bỏ={n - c - w}  PA={(c - w) / n:.4f}  chính xác khi trả lời={c / max(1, c + w):.3f}"
    )
    # what if every question with at least one vote had been answered with the leading vote?
    voted = [score_one(r.get("voted"), gold[r["id"]]) for r in logs]
    vc, vw = voted.count(1), voted.count(-1)
    print(
        f"Nếu nộp mọi đáp án dẫn đầu (không abstain): đúng={vc} sai={vw} PA={(vc - vw) / n:.4f} "
        f"→ luật abstain đang {'thêm' if (c - w) >= (vc - vw) else 'làm mất'} {abs((c - w) - (vc - vw)) / n:.4f} PA"
    )

    # ---- precision of each voter --------------------------------------------------------
    by_voter: dict[str, list[int]] = defaultdict(list)
    for r in logs:
        k = 0
        for v in r.get("votes") or []:
            name = v["source"]
            if name == "code":
                k += 1
                name = f"code (lượt {k})"
            by_voter[name].append(score_one(v["answer"], gold[r["id"]]))
    order = [k for k in ("template", "plan") if k in by_voter] + sorted(k for k in by_voter if k.startswith("code"))
    _table("từng người bỏ phiếu (mọi phiếu)", by_voter, order)

    # ---- by agreement pattern -----------------------------------------------------------
    by_pattern: dict[str, list[int]] = defaultdict(list)
    for r, s_voted in zip(logs, voted, strict=True):
        votes = r.get("votes") or []
        if not votes:
            by_pattern["không có phiếu nào"].append(0)
            continue
        lead = canonical(r["voted"])
        same = sorted({v["source"] for v in votes if canonical(v["answer"]) == lead})
        n_same = int(r.get("n_votes") or len(same))
        distinct = int((r.get("confidence_features") or {}).get("distinct_answers", 1))
        label = f"{n_same} phiếu [{'+'.join(same)}]" + (" · có phiếu khác" if distinct > 1 else "")
        by_pattern[label].append(s_voted)
    _table("đáp án dẫn đầu theo kiểu đồng thuận", by_pattern, sorted(by_pattern, key=lambda k: -len(by_pattern[k])), 44)

    # ---- by type / shop -----------------------------------------------------------------
    by_type: dict[str, list[int]] = defaultdict(list)
    by_biz: dict[str, list[int]] = defaultdict(list)
    for r, s in zip(logs, final, strict=True):
        by_type[gold[r["id"]]["type"]].append(s)
        by_biz[r.get("business_id", "?")].append(s)
    _table("theo type", by_type)
    worst = sorted(by_biz, key=lambda b: (sum(by_biz[b]) / len(by_biz[b]), b))
    _table("theo shop (kém nhất trước)", by_biz, worst[:12])

    # ---- time ---------------------------------------------------------------------------
    t = Counter()
    per_stage = defaultdict(Counter)
    for r in logs:
        t.update({k: int(v) for k, v in (r.get("timing") or {}).items()})
        t.update({k: int(v) for k, v in (r.get("usage") or {}).items() if isinstance(v, int)})
        t["elapsed_ms"] += int(r.get("elapsed_ms", 0))
        for st in r.get("stages") or []:
            per_stage[st["stage"]]["n"] += 1
            per_stage[st["stage"]]["llm_ms"] += int(st.get("llm_ms") or 0)
            per_stage[st["stage"]]["llm_calls"] += int(st.get("llm_calls") or (1 if st["stage"] == "plan" else 0))
            per_stage[st["stage"]]["exec_ms"] += int(st.get("exec_ms") or 0)
    print(
        f"\nThời gian: tổng thời gian các câu {t['elapsed_ms'] / 1000:.0f}s (chạy song song nên lớn hơn đồng hồ); "
        f"LLM {t['llm_ms'] / 1000:.0f}s qua {t['llm_calls']} lần gọi ({t['cached_calls']} từ cache); "
        f"sandbox {t['exec_ms'] / 1000:.1f}s; token sinh ra {t['completion_tokens']:,}, "
        f"token prompt {t['prompt_tokens']:,}"
    )
    print(f"{'tầng':<10} {'số lần':>7} {'gọi LLM':>8} {'LLM (s)':>9} {'s/lần':>7} {'sandbox (s)':>12}")
    for name in ("template", "plan", "code"):
        if name in per_stage:
            ps = per_stage[name]
            print(
                f"{name:<10} {ps['n']:>7} {ps['llm_calls']:>8} {ps['llm_ms'] / 1000:>9.0f} "
                f"{ps['llm_ms'] / 1000 / max(1, ps['n']):>7.1f} {ps['exec_ms'] / 1000:>12.1f}"
            )
    no_llm = sum(1 for r in logs if not (r.get("timing") or {}).get("llm_calls"))
    print(f"{no_llm}/{n} câu ({no_llm / n:.1%}) xong mà không gọi LLM lần nào")

    # ---- mistakes -----------------------------------------------------------------------
    kinds = Counter()
    abst = Counter()
    failures = []
    for r, s in zip(logs, final, strict=True):
        g = gold[r["id"]]
        if s == -1:
            kinds[mistake(r["answer"], g)] += 1
        elif s == 0:
            votes = r.get("votes") or []
            lead_ok = score_one(r.get("voted"), g) == 1
            why = (
                "không có phiếu nào"
                if not votes
                else ("có phiếu ĐÚNG nhưng không đủ đồng thuận" if lead_ok else "phiếu dẫn đầu sai → bỏ là đúng")
            )
            abst[why] += 1
        if s != 1:
            failures.append(
                {
                    "id": r["id"],
                    "business_id": r.get("business_id"),
                    "score": s,
                    "question": r.get("question"),
                    "as_of_date": r.get("as_of_date"),
                    "gold": {k: g[k] for k in ("type", "answer") if k in g},
                    "answer": r.get("answer"),
                    "votes": r.get("votes"),
                    "stages": [
                        {k: st.get(k) for k in ("stage", "route", "intent", "spec", "answer", "error", "warnings")}
                        for st in r.get("stages") or []
                    ],
                    "last_code": (r.get("code") or [None])[-1],
                }
            )
    if kinds:
        print("\nCâu SAI theo loại lỗi:")
        for k, v in kinds.most_common():
            print(f"  {v:>4}  {k}")
    if abst:
        print("\nCâu BỎ (abstain):")
        for k, v in abst.most_common():
            print(f"  {v:>4}  {k}")
    shown = [f for f in failures if f["score"] == -1][: args.show]
    if shown:
        print(f"\n{len(shown)} câu sai đầu tiên:")
    for f in shown:
        print(f"\n[{f['business_id']} {f['id']}] {f['question']!r} (as_of {f['as_of_date']})")
        print(f"   đáp án đúng: {json.dumps(f['gold'], ensure_ascii=False)[:300]}")
        print(f"   đã nộp     : {json.dumps(f['answer'], ensure_ascii=False)[:300]}")
        for v in f["votes"] or []:
            print(f"   phiếu {v['source']:<8}: {json.dumps(v['answer'], ensure_ascii=False)[:200]}")
        if f["last_code"]:
            print("   code cuối:\n      " + f["last_code"].strip().replace("\n", "\n      ")[:900])
    if args.failures:
        Path(args.failures).write_text(
            "\n".join(json.dumps(f, ensure_ascii=False, default=str) for f in failures) + "\n", encoding="utf-8"
        )
        print(f"\nđã ghi {len(failures)} câu sai/bỏ vào {args.failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
