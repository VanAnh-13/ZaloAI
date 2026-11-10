"""CLI: /data -> /result/submission.jsonl + /result/logs/logs.jsonl

python -m datamate.run --data /data --out /result                    # container (predict.sh)
python -m datamate.run --data <split> --out runs/x --limit 20 --mock # offline smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from datamate.agent.llm import ChatClient, MockClient
from datamate.config import load_settings
from datamate.pipeline import Pipeline


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args(argv: list[str] | None):
    ap = argparse.ArgumentParser(prog="python -m datamate.run")
    ap.add_argument("--data", default="/data", help="split folder with questions.jsonl, rulebook.md, data/")
    ap.add_argument("--out", default="/result")
    ap.add_argument("--ws", help="workspace cache folder (default: settings.ws_dir)")
    ap.add_argument("--limit", type=int, help="only the first N questions (dev)")
    ap.add_argument("--ids-file", help="only question ids listed in this file, one per line (dev)")
    ap.add_argument("--business", nargs="*", help="only these business ids (dev)")
    ap.add_argument("--resume", action="store_true", help="keep existing outputs, skip answered ids (dev)")
    ap.add_argument("--mock", action="store_true", help="offline mock LLM, no GPU needed (dev)")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--attempts", type=int)
    return ap.parse_args(argv)


def select(questions: list[dict], args) -> list[dict]:
    if args.ids_file:
        keep = {x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()}
        questions = [q for q in questions if q["id"] in keep]
    if args.business:
        questions = [q for q in questions if q["business_id"] in set(args.business)]
    questions = sorted(questions, key=lambda q: (q["business_id"], q["id"]))
    return questions[: args.limit] if args.limit else questions


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = load_settings(concurrency=args.concurrency, attempts=args.attempts, ws_dir=args.ws)
    client = MockClient() if args.mock else ChatClient(settings)
    questions = select(read_jsonl(Path(args.data) / "questions.jsonl"), args)
    t0 = float(os.environ.get("DATAMATE_T0") or time.time())
    Pipeline(settings, Path(args.data), Path(args.out), client, t0=t0).run(questions, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
