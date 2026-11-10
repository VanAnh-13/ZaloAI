"""Fit the abstention model on a dev run (needs real LLM runs; see README).

    python -m eval.calibrate --logs runs/dev/logs/logs.jsonl --gold <train>/answers.jsonl \
        [--out datamate/agent/calibration.json]

Every question whose voted answer is not "abstain" becomes one sample: x = the logged
confidence features, y = voted answer correct. An L2-regularized logistic regression gives
P(correct); the threshold that maximizes PA on the run is stored with the weights.
Fit on held-out businesses (eval/splits/holdout_ids.txt) to avoid an optimistic threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from datamate.agent.confidence import DEFAULT_PATH, FEATURES
from eval.scorer import read_jsonl, score_one


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> tuple[np.ndarray, float]:
    """Newton / IRLS on standardized features; returns weights and bias in the original scale."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = l2 * np.eye(Z.shape[1])
    reg[0, 0] = 0.0  # no penalty on the bias
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) + reg @ w
        hess = Z.T @ (Z * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    coef = w[1:] / sd
    return coef, float(w[0] - (coef * mu).sum())


def best_threshold(p: np.ndarray, y: np.ndarray, n_total: int) -> tuple[float, float]:
    """Threshold maximizing PA = (correct kept − wrong kept) / all questions."""
    best_t, best_pa = 0.0, (2 * y.sum() - len(y)) / n_total
    for t in np.unique(p):
        keep = p >= t
        pa = (2 * y[keep].sum() - keep.sum()) / n_total
        if pa > best_pa:
            best_t, best_pa = float(t), float(pa)
    return best_t, best_pa


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.calibrate")
    ap.add_argument("--logs", required=True, help="logs.jsonl of a run on questions with known answers")
    ap.add_argument("--gold", required=True, help="answers.jsonl")
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gold = {g["id"]: g for g in read_jsonl(args.gold)}
    logs = [r for r in read_jsonl(args.logs) if r["id"] in gold and "confidence_features" in r]
    samples = [r for r in logs if r.get("voted", {}).get("type") != "abstain"]
    if len(samples) < 20:
        print(f"chỉ có {len(samples)} mẫu — cần ít nhất 20 câu đã trả lời để fit")
        return 1
    X = np.array([[float(r["confidence_features"].get(f, 0.0)) for f in FEATURES] for r in samples])
    y = np.array([1.0 if score_one(r["voted"], gold[r["id"]]) == 1 else 0.0 for r in samples])
    coef, bias = fit_logistic(X, y, args.l2)
    p = 1.0 / (1.0 + np.exp(-(X @ coef + bias)))
    t, pa = best_threshold(p, y, len(logs))
    pa_all = (2 * y.sum() - len(y)) / len(logs)
    out = {
        "weights": dict(zip(FEATURES, map(float, coef), strict=False)),
        "bias": bias,
        "threshold": t,
        "fitted_on": {
            "logs": args.logs,
            "questions": len(logs),
            "answered": len(samples),
            "pa_answer_all": round(pa_all, 4),
            "pa_with_threshold": round(pa, 4),
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(logs)} câu, {len(samples)} đã trả lời, accuracy {y.mean():.3f}")
    print(f"PA trả lời hết = {pa_all:.4f};  PA với ngưỡng {t:.3f} = {pa:.4f}")
    print("trọng số:", ", ".join(f"{k}={v:+.3f}" for k, v in out["weights"].items()), f"bias={bias:+.3f}")
    print(f"đã ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
