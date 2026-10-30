"""Evaluation metrics, all from scikit-learn.

``roc_auc_score``, ``average_precision_score`` and ``balanced_accuracy_score``
are used directly. Nothing is reimplemented here; this module only assembles
them into one record per state and guards the summary statistics against
non-finite values.

PR-AUC is reported as average precision, and its chance level is the positive
class prevalence, not 0.5 -- on this corpus 89/267 = 0.3333.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from .validation import ValidationError, safe_argmax

METRIC_NAMES = ("roc_auc", "pr_auc", "balanced_accuracy", "accuracy")


@dataclass
class StateMetrics:
    """Out-of-fold performance at one state."""

    state: int
    n: int
    n_E: int
    n_B: int
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    accuracy: float
    degenerate: bool
    all_folds_converged: bool
    max_solver_iterations: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_out_of_fold(y: np.ndarray, scores: np.ndarray,
                         predictions: Optional[np.ndarray] = None,
                         state: int = -1,
                         degenerate: bool = False,
                         all_folds_converged: bool = True,
                         max_solver_iterations: int = 0,
                         threshold: float = 0.5) -> StateMetrics:
    """Score one state's out-of-fold predictions."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValidationError(f"state {state}: non-finite out-of-fold scores")
    if predictions is None:
        predictions = (scores >= threshold).astype(int)
    predictions = np.asarray(predictions).astype(int)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValidationError(
            f"state {state}: both classes must be present (E={n_pos}, B={n_neg})")

    metrics = StateMetrics(
        state=int(state),
        n=int(y.shape[0]),
        n_E=n_pos,
        n_B=n_neg,
        roc_auc=float(roc_auc_score(y, scores)),
        pr_auc=float(average_precision_score(y, scores)),
        balanced_accuracy=float(balanced_accuracy_score(y, predictions)),
        accuracy=float(accuracy_score(y, predictions)),
        degenerate=bool(degenerate),
        all_folds_converged=bool(all_folds_converged),
        max_solver_iterations=int(max_solver_iterations),
    )
    for name in METRIC_NAMES:
        value = getattr(metrics, name)
        if not np.isfinite(value):
            raise ValidationError(f"state {state}: {name} is not finite")
        if not 0.0 <= value <= 1.0:
            raise ValidationError(f"state {state}: {name} = {value} is out of [0, 1]")
    return metrics


def chance_levels(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    prevalence = float((y == 1).mean())
    return {"roc_auc": 0.5, "pr_auc": prevalence,
            "balanced_accuracy": 0.5, "prevalence": prevalence}


def best_state(records: Sequence[StateMetrics], metric: str = "roc_auc",
               exclude_degenerate: bool = True) -> StateMetrics:
    """Highest-scoring state, refusing to let a non-finite value win.

    Degenerate states are excluded by default: a constant design matrix cannot
    win a maximum on merit, and including it only invites confusion.
    """
    if metric not in METRIC_NAMES:
        raise ValidationError(f"unknown metric {metric!r}")
    candidates = [r for r in records
                  if not (exclude_degenerate and r.degenerate)]
    if not candidates:
        raise ValidationError("no non-degenerate state to choose from")
    values = [getattr(r, metric) for r in candidates]
    return candidates[safe_argmax(values, f"{metric} over states")]


def metrics_table(records: Sequence[StateMetrics]) -> List[Dict[str, object]]:
    return [r.as_dict() for r in records]


def compare_to_reference(records: Sequence[StateMetrics],
                         reference: Dict[int, Dict[str, float]],
                         metrics: Sequence[str] = METRIC_NAMES,
                         tolerance: float = 1e-6) -> List[Dict[str, object]]:
    """Row-per-(state, metric) comparison against frozen reference values."""
    rows: List[Dict[str, object]] = []
    for record in records:
        expected = reference.get(record.state)
        for name in metrics:
            reproduced = float(getattr(record, name))
            if expected is None or name not in expected:
                rows.append({"state": record.state, "metric": name,
                             "reference": "", "reproduced": reproduced,
                             "absolute_difference": "",
                             "within_tolerance": "no_reference"})
                continue
            ref = float(expected[name])
            difference = abs(reproduced - ref)
            rows.append({"state": record.state, "metric": name,
                         "reference": ref, "reproduced": reproduced,
                         "absolute_difference": difference,
                         "within_tolerance": bool(difference <= tolerance)})
    return rows


def max_absolute_difference(comparison: Sequence[Dict[str, object]]) -> float:
    """Largest absolute metric difference over every compared cell."""
    values = [float(row["absolute_difference"]) for row in comparison
              if row["absolute_difference"] != ""]
    if not values:
        raise ValidationError("no comparable cells")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValidationError("a metric difference is not finite")
    return float(array.max())
