"""Write /result/submission.jsonl and /result/logs/logs.jsonl, one line per question.

Each answer is flushed and fsynced immediately: if the run is stopped at the time
limit, everything written so far still counts (Rule 8).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class ResultWriter:
    def __init__(self, out_dir: Path, resume: bool = False):
        self.sub_path = Path(out_dir) / "submission.jsonl"
        self.log_path = Path(out_dir) / "logs" / "logs.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if resume and self.sub_path.exists():
            for line in self.sub_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.done.add(json.loads(line)["id"])
        else:
            for p in (self.sub_path, self.log_path):
                p.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict, log: dict) -> None:
        sub_line = json.dumps(record, ensure_ascii=False)
        log_line = json.dumps(log, ensure_ascii=False, default=str)
        with self._lock:
            if record["id"] in self.done:
                return
            # submission first: it is what gets scored
            for path, line in ((self.sub_path, sub_line), (self.log_path, log_line)):
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            self.done.add(record["id"])
