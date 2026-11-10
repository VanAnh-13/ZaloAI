"""Solve one question: K attempts with different prompt variants, vote, then decide whether
the answer is reliable enough to submit (+1 right / −1 wrong / 0 abstain)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from datamate.agent.answer import canonical
from datamate.agent.confidence import Calibrator, features
from datamate.agent.hints import analyze
from datamate.agent.loop import Attempt, run_attempt
from datamate.agent.prompts import build_messages
from datamate.agent.sandbox import ExecContext, Sandbox
from datamate.agent.verify import check
from datamate.config import Settings
from datamate.nlu.entities import EntityIndex
from datamate.workspace import WorkspaceRef

ABSTAIN = {"type": "abstain"}


@dataclass
class Resources:
    """Run-level dependencies shared by every question."""

    client: object  # ChatClient or MockClient
    sandbox: Sandbox
    settings: Settings
    rulebook: str
    calendar_path: str | None
    cal: pd.DataFrame | None
    calibrator: Calibrator


@dataclass
class Shop:
    ref: WorkspaceRef
    index: EntityIndex


def vote(answers: list[dict | None], min_agreement: float) -> tuple[dict, float]:
    """Unique plurality with share >= min_agreement wins; ties or weak majorities abstain.
    Failed attempts (None) count in the denominator."""
    keys = [canonical(a) for a in answers if a is not None]
    if not keys:
        return dict(ABSTAIN), 0.0
    counts = Counter(keys).most_common()
    top, n = counts[0]
    share = n / len(answers)
    if (len(counts) > 1 and counts[1][1] == n) or share < min_agreement:
        return dict(ABSTAIN), share
    return next(a for a in answers if a is not None and canonical(a) == top), share


def _decided(answers: list[dict | None], remaining: int) -> bool:
    """True when the remaining attempts cannot change the vote winner."""
    counts = Counter(canonical(a) for a in answers if a is not None).most_common(2)
    if not counts:
        return False
    second = counts[1][1] if len(counts) > 1 else 0
    return counts[0][1] > second + remaining


def solve(q: dict, shop: Shop, res: Resources, deadline: float, attempts: int | None = None) -> tuple[dict, dict]:
    """Returns (submission record, log record)."""
    t0 = time.monotonic()
    k = attempts or res.settings.attempts
    analysis = analyze(q["question"], q["as_of_date"], res.cal, shop.index, shop.ref.coverage, shop.ref.data_end)
    hint_lines = analysis.lines()
    ctx = ExecContext(shop.ref.pickle, shop.ref.data_dir, res.calendar_path, q["as_of_date"], q["question"])

    def verify(ans: dict) -> list[str]:
        return check(ans, analysis, shop.index)

    done: list[Attempt] = []
    for v in range(k):
        msgs = build_messages(q, shop.ref.card, res.rulebook, hint_lines, variant=v)
        done.append(run_attempt(res.client, res.sandbox, msgs, ctx, res.settings, deadline, variant=v, verify=verify))
        if _decided([a.answer for a in done], k - len(done)):
            break
    voted, share = vote([a.answer for a in done], res.settings.min_agreement)
    feats = features(done, voted, share, analysis.expected_type, analysis.all_outside)
    keep = voted["type"] == "abstain" or res.calibrator.keep(feats)
    final = voted if keep else dict(ABSTAIN)

    steps = [s for a in done for s in a.steps]
    thinking = [s["thinking"] for s in steps]
    executed = [s for s in steps if s["code"]]
    log = {
        "id": q["id"],
        "answer": final,
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
        # Docker guide §3.3: raw reasoning of every model call, in call order; null if none
        "thinking": thinking if any(thinking) else None,
        "code": [s["code"] for s in executed] or None,
        "output": [s["output"] for s in executed] or None,
        "business_id": q["business_id"],
        "as_of_date": q["as_of_date"],
        "question": q["question"],
        "hints": hint_lines,
        "voted": voted,
        "vote_share": round(share, 3),
        "confidence_features": feats,
        "p_correct": res.calibrator.probability(feats),
        "attempts": [
            {
                "variant": a.variant,
                "answer": a.answer,
                "llm_calls": len(a.steps),
                "error": a.error,
                "warnings": a.warnings,
            }
            for a in done
        ],
        "usage": dict(sum((a.usage for a in done), Counter())),
    }
    return {"id": q["id"], **final}, log
