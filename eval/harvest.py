"""Build the worked examples of the code agent from a run on the TRAIN split.

    python -m eval.harvest --train <train> --logs runs/x/logs/logs.jsonl \
        [--out datamate/agent/examples.jsonl] [--max 600] [--per-group 6]

For every question whose code attempt gave the reference answer, the program of that attempt
(its code blocks, in order, without the ones that failed) is executed again in a fresh sandbox
session. Only programs that reproduce the reference answer on their own are kept — an example
is never something the model merely claimed. The selection is spread over shops and answer
types, shortest program first.

The file holds questions and programs of the train split only (never public/private test) and
ships inside the image; record it in training_code/data_manifest.json (path, SHA-256, count —
printed at the end) because it is data produced with the model.

To measure honestly on train afterwards, run with DATAMATE_EXAMPLES_HOLDOUT=1: a question then
never sees examples from its own shop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from datamate.agent.answer import AnswerError, normalize
from datamate.agent.examples import DEFAULT_PATH, signature
from datamate.agent.sandbox import ExecContext, Sandbox, SandboxConfig
from datamate.config import load_settings
from datamate.workspace import prepare_many
from eval.scorer import read_jsonl, score_one


def candidate_programs(rec: dict, gold: dict) -> list[list[str]]:
    """Code blocks of each attempt that reached the reference answer, shortest attempt first."""
    out = []
    for st in rec.get("stages") or []:
        if st.get("stage") != "code" or not st.get("answer") or score_one(st["answer"], gold) != 1:
            continue
        blocks = [s["code"] for s in st.get("steps") or [] if s.get("code") and not s.get("error")]
        if blocks:
            out.append(blocks)
    return sorted(out, key=lambda b: (len(b), sum(map(len, b))))


def reproduces(sandbox: Sandbox, ctx: ExecContext, blocks: list[str], gold: dict) -> bool:
    with sandbox.open(ctx) as session:
        for code in blocks:
            res = session.run(code)
            if res.error:
                return False
            if res.answer is not None:
                try:
                    return score_one(normalize(res.answer), gold) == 1
                except AnswerError:
                    return False
    return False


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    ap = argparse.ArgumentParser(prog="python -m eval.harvest")
    ap.add_argument("--train", required=True, help="train split folder (questions.jsonl, answers.jsonl, data/)")
    ap.add_argument("--logs", required=True, help="logs.jsonl of a run on train")
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--ws", default=settings.ws_dir)
    ap.add_argument("--max", type=int, default=600, help="examples to keep in total")
    ap.add_argument("--per-group", type=int, default=6, help="examples per (shop, answer type, wording)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = Path(args.train)
    questions = {q["id"]: q for q in read_jsonl(root / "questions.jsonl")}
    gold = {g["id"]: g for g in read_jsonl(root / "answers.jsonl")}
    logs = [r for r in read_jsonl(args.logs) if r["id"] in gold and r["id"] in questions]
    refs = prepare_many(root, sorted({questions[r["id"]]["business_id"] for r in logs}), Path(args.ws), 4)
    cal = root / "calendar.csv"
    sandbox = Sandbox(config=SandboxConfig(timeout_s=settings.exec_timeout_s))

    kept: dict[tuple, list[dict]] = defaultdict(list)
    tried = ok = 0
    for rec in logs:
        q, g = questions[rec["id"]], gold[rec["id"]]
        ref = refs[q["business_id"]]
        if ref.pickle is None:
            continue
        ctx = ExecContext(ref.pickle, ref.data_dir, str(cal) if cal.exists() else None, q["as_of_date"], q["question"])
        for blocks in candidate_programs(rec, g)[:2]:
            tried += 1
            if reproduces(sandbox, ctx, blocks, g):
                ok += 1
                kept[(q["business_id"], g["type"])].append(
                    {
                        "id": q["id"],
                        "business_id": q["business_id"],
                        "question": q["question"],
                        "as_of": q["as_of_date"],
                        "type": g["type"],
                        "code": "\n\n".join(b.strip() for b in blocks),
                    }
                )
                break
    print(f"{len(logs)} câu trong log; {tried} chương trình thử lại, {ok} tái lập đúng đáp án")

    # spread over shops and types; inside a group: one per wording first, shortest program first
    chosen: list[dict] = []
    for group in kept.values():
        group.sort(key=lambda e: (len(e["code"]), e["id"]))
        seen: dict[str, int] = defaultdict(int)
        picked = []
        for e in group:
            sig = signature(e["question"])
            if seen[sig] == 0 and len(picked) < args.per_group:
                picked.append(e)
            seen[sig] += 1
        group[:] = picked
    groups = sorted(kept.items())
    while len(chosen) < args.max and any(g for _, g in groups):
        for _, g in groups:
            if g and len(chosen) < args.max:
                chosen.append(g.pop(0))
    chosen.sort(key=lambda e: e["id"])

    out = Path(args.out)
    text = "\n".join(json.dumps(e, ensure_ascii=False) for e in chosen) + ("\n" if chosen else "")
    out.write_text(text, encoding="utf-8")
    by_type = defaultdict(int)
    for e in chosen:
        by_type[e["type"]] += 1
    print(f"đã ghi {len(chosen)} ví dụ vào {out}: {dict(sorted(by_type.items()))}")
    print(
        f"data_manifest.json → path: {out.as_posix()}, sha256: {hashlib.sha256(text.encode('utf-8')).hexdigest()}, "
        f"examples: {len(chosen)}, source: train split (questions + model-written programs verified against answers)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
