"""Decide whether to submit an answer or abstain (score: +1 right, −1 wrong, 0 abstain).

Features describe how the answer was reached; a logistic model fitted on dev runs
(eval/calibrate.py) turns them into P(correct). We answer when P >= threshold.
Without a calibration file every voted answer is kept.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from datamate.agent.answer import canonical

DEFAULT_PATH = Path(__file__).with_name("calibration.json")
FEATURES = (
    "vote_share",
    "attempts",
    "distinct_answers",
    "warnings",
    "challenged",
    "exec_errors",
    "llm_calls",
    "type_matches",
    "is_unanswerable",
    "periods_outside",
)


def features(attempts: list, final: dict, share: float, expected_type: str | None, all_outside: bool) -> dict:
    """attempts: list of loop.Attempt."""
    answers = [a.answer for a in attempts if a.answer is not None]
    return {
        "vote_share": share,
        "attempts": len(attempts),
        "distinct_answers": len({canonical(a) for a in answers}),
        "warnings": sum(len(a.warnings) for a in attempts),
        "challenged": sum(1 for a in attempts if a.challenged),
        "exec_errors": sum(a.exec_errors for a in attempts),
        "llm_calls": sum(len(a.steps) for a in attempts),
        "type_matches": float(expected_type is None or final.get("type") == expected_type),
        "is_unanswerable": float(final.get("type") == "unanswerable"),
        "periods_outside": float(all_outside),
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

    def probability(self, feats: dict) -> float | None:
        if not self.weights:
            return None
        z = self.bias + sum(w * float(feats.get(k, 0.0)) for k, w in self.weights.items())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def keep(self, feats: dict) -> bool:
        p = self.probability(feats)
        return p is None or p >= self.threshold
