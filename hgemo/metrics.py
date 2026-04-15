"""Multi-label evaluation metrics.

Computes the standard metrics reported in the paper:
  - Hamming Loss
  - Ranking Loss
  - Label-Ranking Average Precision (LRAP)
  - Micro-F1 and Macro-F1
  - One-Error
  - Subset 0/1 Loss

Also includes Pareto-front utilities (non-dominated sorting, hypervolume).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
    zero_one_loss,
)


@dataclass(frozen=True)
class MLResult:
    """Container for one multi-label evaluation."""
    hamming: float
    ranking: float
    avg_precision: float
    f1_micro: float
    f1_macro: float
    one_error: float
    zero_one_loss: float


def one_error(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fraction of instances whose top-ranked label is irrelevant."""
    if y_true.size == 0:
        return 0.0
    top = np.argmax(y_score, axis=1)
    errors = sum(1 for i, t in enumerate(top) if y_true[i, t] == 0)
    return float(errors) / y_true.shape[0]


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> MLResult:
    """Compute the full metric suite from predictions and scores."""
    return MLResult(
        hamming=float(hamming_loss(y_true, y_pred)),
        ranking=float(label_ranking_loss(y_true, y_score)),
        avg_precision=float(
            label_ranking_average_precision_score(y_true, y_score)
        ),
        f1_micro=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        one_error=one_error(y_true, y_score),
        zero_one_loss=float(zero_one_loss(y_true, y_pred)),
    )


# ── Pareto-front utilities ────────────────────────────────────────────

def pareto_nondominated(points: np.ndarray) -> np.ndarray:
    """Return the subset of non-dominated rows (minimisation)."""
    if points.size == 0:
        return points
    n = points.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        dominates = np.all(points[i] <= points, axis=1) & np.any(
            points[i] < points, axis=1
        )
        is_nd[dominates] = False
    return points[is_nd]


def pareto_nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask selecting non-dominated rows (keeps alignment)."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    n = pts.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        dominates = np.all(pts[i] <= pts, axis=1) & np.any(
            pts[i] < pts, axis=1
        )
        is_nd[dominates] = False
    return is_nd


def hypervolume_3d(
    points: np.ndarray,
    ref: tuple[float, float, float],
) -> float:
    """Approximate 3-D hypervolume (sum of dominated boxes).

    Exact for well-separated fronts; used for early-stopping and
    convergence logging rather than for definitive Pareto comparisons.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    pts = pareto_nondominated(pts)
    r0, r1, r2 = float(ref[0]), float(ref[1]), float(ref[2])
    pts = pts[(pts[:, 0] < r0) & (pts[:, 1] < r1) & (pts[:, 2] < r2)]
    if pts.size == 0:
        return 0.0
    hv = 0.0
    for p in pts:
        hv += (
            max(0.0, r0 - p[0])
            * max(0.0, r1 - p[1])
            * max(0.0, r2 - p[2])
        )
    return float(hv)
