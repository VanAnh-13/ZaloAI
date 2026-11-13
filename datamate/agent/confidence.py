"""Decide whether to submit an answer or abstain (score: +1 right, −1 wrong, 0 abstain).

Features describe how the answer was reached — above all which voters of the cascade agree on
it. A logistic model fitted on dev runs (eval/calibrate.py) turns them into P(correct) and we
answer when P >= threshold. Without a calibration file the solver uses its plain rule: two
agreeing votes (effort "balanced") or one (effort "fast").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from datamate.agent.answer import canonical

DEFAULT_PATH = Path(__file__).with_name("calibration.json")
FEATURES = (
    "votes",  # voters that gave the submitted answer
    "voters",  # voters that gave any answer
    "vote_share",  # votes / stages that were asked (failed attempts count)
    "distinct_answers",
    "template_agrees",  # +1 the template stage gave this answer, −1 it gave another one, 0 it did not vote
    "plan_agrees",  # same for the planner stage
    "code_votes",  # code attempts that gave this answer
    "code_attempts",
    "warnings",  # verifier warnings on the submitted answer
    "challenged",
    "exec_errors",
    "llm_calls",
    "type_matches",
    "is_unanswerable",
    "periods_outside",
)


def _agrees(votes: list, source: str, key: str) -> float:
    mine = [canonical(v.answer) for v in votes if v.source == source]
    if not mine:
        return 0.0
    return 1.0 if key in mine else -1.0


def features(votes: list, final: dict, attempts: list, analysis, warnings: list[str], stages: list[dict]) -> dict:
    """votes: solver.Vote list · attempts: loop.Attempt list · analysis: hints.Analysis."""
    key = canonical(final) if votes else ""
    keys = [canonical(v.answer) for v in votes]
    asked = sum(1 for st in stages if st["stage"] != "plan" or st.get("route") != "code")
    expected = analysis.expected_type
    return {
        "votes": float(keys.count(key)),
        "voters": float(len(votes)),
        "vote_share": keys.count(key) / asked if asked else 0.0,
        "distinct_answers": float(len(set(keys))),
        "template_agrees": _agrees(votes, "template", key),
        "plan_agrees": _agrees(votes, "plan", key),
        "code_votes": float(sum(1 for v in votes if v.source == "code" and canonical(v.answer) == key)),
        "code_attempts": float(len(attempts)),
        "warnings": float(len(warnings)),
        "challenged": float(sum(1 for a in attempts if a.challenged)),
        "exec_errors": float(sum(a.exec_errors for a in attempts)),
        "llm_calls": float(sum(len(a.steps) for a in attempts)),
        "type_matches": float(expected is None or final.get("type") == expected),
        "is_unanswerable": float(final.get("type") == "unanswerable"),
        "periods_outside": float(analysis.all_outside),
    }


@dataclass
class Calibrator:
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    threshold: float = 0.0  # 0 keeps every answer

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Calibrator":
        if not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d.get("weights", {}), d.get("bias", 0.0), d.get("threshold", 0.0))

    @property
    def fitted(self) -> bool:
        return bool(self.weights)

    def probability(self, feats: dict) -> float | None:
        if not self.weights:
            return None
        z = self.bias + sum(w * float(feats.get(k, 0.0)) for k, w in self.weights.items())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def keep(self, feats: dict) -> bool:
        p = self.probability(feats)
        return p is None or p >= self.threshold
