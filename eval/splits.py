"""Deterministic dev splits of the train set.

    python -m eval.splits --train C:/Users/Admin/Downloads/train/train

Writes eval/splits/:
- holdout_biz.txt / holdout_ids.txt — whole businesses never used for prompt work (generalization check)
- dev_ids.txt                       — everything else
- regression_300.txt                — stratified (business × answer type) sample for quick iterations
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "splits"


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.splits")
    ap.add_argument("--train", required=True, help="train split folder (questions.jsonl, answers.jsonl)")
    ap.add_argument("--holdout", type=int, default=8, help="number of holdout businesses")
    ap.add_argument("--regression", type=int, default=300)
    args = ap.parse_args(argv)

    root = Path(args.train)
    qs = [json.loads(x) for x in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ans = {
        a["id"]: a
        for a in (json.loads(x) for x in (root / "answers.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())
    }
    bizs = sorted({q["business_id"] for q in qs}, key=_h)
    holdout = set(bizs[: args.holdout])

    strata: dict[tuple, list[str]] = defaultdict(list)
    for q in sorted(qs, key=lambda q: _h(q["id"])):
        strata[(q["business_id"], ans[q["id"]]["type"])].append(q["id"])
    # round-robin over strata so rare types and every business are represented
    rng = random.Random(0)
    keys = sorted(strata)
    rng.shuffle(keys)
    regression: list[str] = []
    while len(regression) < args.regression and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(regression) < args.regression:
                regression.append(strata[k].pop(0))

    OUT.mkdir(exist_ok=True)
    (OUT / "holdout_biz.txt").write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")
    (OUT / "holdout_ids.txt").write_text(
        "\n".join(sorted(q["id"] for q in qs if q["business_id"] in holdout)) + "\n", encoding="utf-8"
    )
    (OUT / "dev_ids.txt").write_text(
        "\n".join(sorted(q["id"] for q in qs if q["business_id"] not in holdout)) + "\n", encoding="utf-8"
    )
    (OUT / "regression_300.txt").write_text("\n".join(sorted(regression)) + "\n", encoding="utf-8")
    print(f"holdout: {sorted(holdout)}; regression: {len(regression)} câu; ghi vào {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
