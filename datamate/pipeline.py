"""Composition root: prepare the workspaces, then answer every question concurrently and
write each result as soon as it is known."""

from __future__ import annotations

import json
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

from datamate.agent.confidence import Calibrator
from datamate.agent.examples import ExampleBank
from datamate.agent.planner import shop_dimensions
from datamate.agent.sandbox import Sandbox, SandboxConfig
from datamate.agent.solver import Resources, Shop, solve
from datamate.agent.templates import Names
from datamate.config import Settings
from datamate.nlu.entities import EntityIndex
from datamate.results import ResultWriter
from datamate.workspace import load_cached, prepare_many


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TimeBudget:
    """Drops to effort "fast" (first vote wins, one code attempt) when the run falls behind the deadline."""

    def __init__(self, deadline: float, total: int, margin_s: float = 300.0):
        self.deadline, self.total, self.margin = deadline, total, margin_s
        self.start = time.time()
        self.done = 0
        self._lock = threading.Lock()

    def record(self) -> None:
        with self._lock:
            self.done += 1

    def behind(self) -> bool:
        with self._lock:
            done = self.done
        if done < 5:
            return False
        projected = (time.time() - self.start) / done * (self.total - done)
        return projected >= self.deadline - time.time() - self.margin

    def effort(self, configured: str) -> str:
        return "fast" if configured != "fast" and self.behind() else configured


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        data_dir: Path,
        out_dir: Path,
        client,
        calibrator: Calibrator | None = None,
        t0: float | None = None,
    ):
        self.s = settings
        self.data_dir, self.out_dir = Path(data_dir), Path(out_dir)
        self.client = client
        self.calibrator = calibrator or Calibrator.load()
        self.t0 = t0 or time.time()
        self.deadline = self.t0 + settings.deadline_s  # the 6 h budget starts with predict.sh

    def _resources(self) -> Resources:
        rb = self.data_dir / "rulebook.md"
        cal_path = self.data_dir / "calendar.csv"
        cal = None
        if cal_path.exists():
            cal = pd.read_csv(cal_path)
            cal["date"] = pd.to_datetime(cal["date"])
        sandbox = Sandbox(
            config=SandboxConfig(
                timeout_s=self.s.exec_timeout_s,
                guard=self.s.sandbox_guard,
                mem_mb=self.s.sandbox_mem_mb,
                # model code reads this shop's files only: not the answers, not the results, not other shops' caches
                deny_read=(str(self.out_dir.resolve()), str(Path(self.s.ws_dir).resolve())),
            )
        )
        examples = ExampleBank.load()
        if len(examples):
            log(f"{len(examples)} ví dụ mẫu (few-shot {self.s.few_shot}, holdout theo shop: {self.s.examples_holdout})")
        return Resources(
            self.client,
            sandbox,
            self.s,
            rb.read_text(encoding="utf-8") if rb.exists() else "",
            str(cal_path) if cal is not None else None,
            cal,
            self.calibrator,
            examples,
            str(rb) if rb.exists() else None,
        )

    def _shops(self, biz_ids: list[str]) -> dict[str, Shop]:
        refs = prepare_many(self.data_dir, biz_ids, Path(self.s.ws_dir), self.s.ingest_workers)
        shops = {}
        for b, r in refs.items():
            if r.error:
                log(f"workspace lỗi {b}: {r.error.strip().splitlines()[-1]}")
            sales = None
            if r.pickle and (self.s.use_templates or self.s.use_planner):
                try:  # the query engine of stages 1–2 runs in this process; the tables stay in the pickle
                    sales = load_cached(Path(r.pickle)).sales
                except Exception as e:
                    log(f"không nạp được mô hình bán hàng {b}: {e}")
            cov = (date.fromisoformat(r.coverage[0]), date.fromisoformat(r.coverage[1])) if r.coverage else None
            if cov and r.data_end:
                cov = (cov[0], max(cov[1], date.fromisoformat(r.data_end)))
            names = Names(sales.dims()) if sales is not None else None
            shops[b] = Shop(r, EntityIndex(r.entities), sales, shop_dimensions(sales, cov), names)
        return shops

    def run(self, questions: list[dict], resume: bool = False) -> dict:
        questions = sorted(questions, key=lambda q: (q["business_id"], q["id"]))  # fixed order, shops grouped
        writer = ResultWriter(self.out_dir, resume=resume)
        todo = [q for q in questions if q["id"] not in writer.done]
        res = self._resources()
        log(f"{len(todo)} câu cần trả lời; chuẩn bị {len({q['business_id'] for q in todo})} shop…")
        shops = self._shops(sorted({q["business_id"] for q in todo}))
        log(f"workspace xong sau {time.time() - self.t0:.0f}s kể từ lúc bắt đầu")
        llm_up = self.client.wait_ready(max(0.0, min(900.0, self.deadline - time.time() - 600)))
        if not llm_up:
            # without a model only stage 1 can vote: take its answers rather than abstain on everything
            log("LLM chưa sẵn sàng — chỉ tầng 1 (không cần LLM) trả lời được; các câu còn lại sẽ abstain")
        budget = TimeBudget(self.deadline, len(todo))

        def work(q: dict) -> dict:
            t_q = time.monotonic()
            try:
                if time.time() > self.deadline - 60:
                    raise TimeoutError("hết thời gian trước khi bắt đầu câu này")
                effort = budget.effort(self.s.effort) if llm_up else "fast"
                rec, lg = solve(q, shops[q["business_id"]], res, self.deadline, effort)
            except Exception as e:  # one question never kills the run
                rec = {"id": q["id"], "type": "abstain"}
                lg = {
                    "id": q["id"],
                    "answer": rec,
                    "elapsed_ms": int((time.monotonic() - t_q) * 1000),
                    "thinking": None,
                    "code": None,
                    "output": None,
                    "business_id": q["business_id"],
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=5),
                }
            writer.write(rec, lg)
            budget.record()
            return rec

        types: Counter = Counter()
        with ThreadPoolExecutor(max_workers=self.s.concurrency) as ex:
            futures = [ex.submit(work, q) for q in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                types[fut.result()["type"]] += 1
                if i % 10 == 0 or i == len(futures):
                    log(f"{i}/{len(futures)} câu; {dict(types)}; {time.time() - self.t0:.0f}s")
        totals = writer.totals

        for q in questions:  # anything not written is an explicit abstain
            if q["id"] not in writer.done:
                writer.write(
                    {"id": q["id"], "type": "abstain"},
                    {
                        "id": q["id"],
                        "answer": {"type": "abstain"},
                        "elapsed_ms": 0,
                        "thinking": None,
                        "error": "không được xử lý",
                    },
                )
        summary = {
            "questions": len(questions),
            "types": dict(types),
            "seconds": round(time.time() - self.t0, 1),
            # where the time went: answered by which stage, LLM calls and seconds, sandbox seconds
            "answered_by": dict(totals["answered_by"]),
            "llm_calls": totals["llm_calls"],
            "llm_cached_calls": totals["cached_calls"],
            "llm_seconds": round(totals["llm_ms"] / 1000, 1),
            "sandbox_seconds": round(totals["exec_ms"] / 1000, 1),
            "completion_tokens": totals["completion_tokens"],
            "prompt_tokens": totals["prompt_tokens"],
            "settings": {k: v for k, v in self.s.__dict__.items() if "key" not in k},
        }
        (self.out_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(
            f"xong: {summary['types']} trong {summary['seconds']}s; quyết định bởi {summary['answered_by']}; "
            f"{summary['llm_calls']} lần gọi LLM ({summary['llm_cached_calls']} từ cache, {summary['llm_seconds']}s), "
            f"sandbox {summary['sandbox_seconds']}s"
        )
        return summary
