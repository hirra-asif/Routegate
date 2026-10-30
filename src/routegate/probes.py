"""The linear probe, and out-of-fold scoring with it.

The probe is a two-step scikit-learn ``Pipeline``. Putting the scaler inside the
pipeline is the whole point: ``fit`` on the training rows estimates the mean and
standard deviation from those rows only, and the held-out rows are transformed
with the training statistics and never contribute to them. There is no separate
scaling step that could accidentally see the whole design matrix.

    StandardScaler()
    LogisticRegression(penalty="l2", C=0.1, class_weight="balanced",
                       max_iter=5000, solver="lbfgs", random_state=20260805,
                       tol=1e-6)

Labels: E = 1 (over-refusal), B = 0 (benign compliance). ``predict_proba`` is
read at the column for class 1, located by ``classes_`` rather than assumed.

On ``tol``. Every argument above is the frozen specification; ``tol`` is the one
that had to be stated rather than left at its default, and it is a convergence
setting, not part of what is being estimated. The reference pilot minimised this
same objective to a gradient tolerance of 1e-6, explicitly noting that this is
tighter than scikit-learn's default ``tol=1e-4``. Left at the default, lbfgs
stops short of the optimum and state 45 reproduces to 6.3e-05 rather than 1e-6 --
a disagreement that is entirely the stopping rule and not the model. At 1e-6 the
metrics are converged: 1e-5, 1e-6, 1e-8 and 1e-12 all return the same values to
every digit, and what remains is the reference file's own six-decimal rounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from .splits import FoldAssignment
from .validation import ValidationError

PROBE_PENALTY = "l2"
PROBE_C = 0.1
PROBE_CLASS_WEIGHT = "balanced"
PROBE_MAX_ITER = 5000
PROBE_SOLVER = "lbfgs"
PROBE_RANDOM_STATE = 20260805
PROBE_TOL = 1e-6              # the pilot's gradient tolerance; sklearn defaults to 1e-4
DECISION_THRESHOLD = 0.5

POSITIVE_CLASS = 1


def make_probe(C: float = PROBE_C,
               max_iter: int = PROBE_MAX_ITER,
               class_weight: Optional[str] = PROBE_CLASS_WEIGHT,
               solver: str = PROBE_SOLVER,
               random_state: int = PROBE_RANDOM_STATE,
               tol: float = PROBE_TOL) -> Pipeline:
    """The frozen probe: standardise on training rows, then L2 logistic."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty=PROBE_PENALTY,
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            tol=tol,
        )),
    ])


def make_categorical_probe(C: float = PROBE_C,
                           max_iter: int = PROBE_MAX_ITER,
                           class_weight: Optional[str] = PROBE_CLASS_WEIGHT,
                           solver: str = PROBE_SOLVER,
                           random_state: int = PROBE_RANDOM_STATE,
                           tol: float = PROBE_TOL) -> Pipeline:
    """The same probe, over raw category strings one-hot encoded per fold.

    ``OneHotEncoder`` is the first pipeline step, so its categories are learned
    from the training rows of each fold and nothing else. ``handle_unknown=
    "ignore"`` is what makes that safe: a category that appears only in the
    held-out rows encodes as all-zero instead of raising or, worse, forcing the
    encoder to be fitted on the whole column.
    """
    return Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  dtype=np.float64)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty=PROBE_PENALTY,
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            tol=tol,
        )),
    ])


def raw_space_direction(coef: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The probe direction in raw activation units.

    A fitted ``Pipeline(StandardScaler, LogisticRegression)`` scores a row as

        w . (x - mean) / scale + b

    so the weight on the *raw* feature is ``w / scale``, not ``w``. Reading the
    standardised coefficients as a direction in activation space silently
    reweights every dimension by its own standard deviation.

    Fold-specific by construction: each fold has its own ``scale``, so each fold
    has its own direction. Averaging coefficients across folds before dividing
    would mix weights that were never on the same footing.
    """
    coef = np.asarray(coef, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if coef.shape != scale.shape:
        raise ValidationError(
            f"coefficients {coef.shape} and scaler scale {scale.shape} disagree")
    if not np.all(scale > 0):
        raise ValidationError("a StandardScaler scale of zero cannot be divided by")
    return coef / scale


def probe_specification() -> Dict[str, object]:
    """The probe configuration, for the run record."""
    return {
        "pipeline": ["sklearn.preprocessing.StandardScaler",
                     "sklearn.linear_model.LogisticRegression"],
        "penalty": PROBE_PENALTY,
        "C": PROBE_C,
        "class_weight": PROBE_CLASS_WEIGHT,
        "max_iter": PROBE_MAX_ITER,
        "solver": PROBE_SOLVER,
        "random_state": PROBE_RANDOM_STATE,
        "tol": PROBE_TOL,
        "tol_note": "the reference pilot's gradient tolerance; scikit-learn's "
                    "default of 1e-4 stops lbfgs short of the optimum and "
                    "reproduces state 45 only to 6.3e-05",
        "decision_threshold": DECISION_THRESHOLD,
        "label_encoding": {"E": 1, "B": 0},
        "scaler_scope": "fitted inside each training fold only, via Pipeline",
    }


@dataclass
class FoldOutcome:
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    scores: np.ndarray          # P(E) on the held-out rows
    decisions: np.ndarray       # signed distance to the boundary
    n_iter: int
    converged: bool
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float


@dataclass
class OutOfFoldResult:
    """Every analysed row scored exactly once, by a probe that never saw it."""

    scores: np.ndarray
    decisions: np.ndarray
    fold_of_row: np.ndarray
    predictions: np.ndarray
    folds: List[FoldOutcome] = field(default_factory=list)
    degenerate: bool = False

    @property
    def all_converged(self) -> bool:
        return all(f.converged for f in self.folds)

    @property
    def max_iterations(self) -> int:
        return max((f.n_iter for f in self.folds), default=0)


def fit_probe_on_fold(X: np.ndarray, y: np.ndarray,
                      train_index: np.ndarray, test_index: np.ndarray,
                      fold: int = 0,
                      probe_factory=make_probe,
                      keep_fold_detail: bool = True,
                      transform_factory=None) -> FoldOutcome:
    """Fit one probe on ``train_index`` and score ``test_index``.

    The single fold is the unit of work everything else is built from:
    out-of-fold scoring loops over it, and nested state selection uses it for
    both its inner search and its outer refit, so there is one place where
    "fit on training rows, score held-out rows" is implemented.
    """
    import warnings

    pipeline = probe_factory()
    transform = (transform_factory(X, train_index)
                 if transform_factory is not None else None)
    train_rows = X[train_index] if transform is None else transform(X[train_index])
    test_rows = X[test_index] if transform is None else transform(X[test_index])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(train_rows, y[train_index])
        converged = not any(issubclass(w.category, ConvergenceWarning)
                            for w in caught)

    classifier = pipeline.named_steps["classifier"]
    column = int(np.flatnonzero(classifier.classes_ == POSITIVE_CLASS)[0])
    scaler = pipeline.named_steps.get("scaler")
    return FoldOutcome(
        fold=int(fold),
        train_index=train_index,
        test_index=test_index,
        scores=pipeline.predict_proba(test_rows)[:, column],
        decisions=pipeline.decision_function(test_rows),
        n_iter=int(np.max(classifier.n_iter_)),
        converged=converged,
        scaler_mean=(np.array(scaler.mean_)
                     if keep_fold_detail and scaler is not None else np.empty(0)),
        scaler_scale=(np.array(scaler.scale_)
                      if keep_fold_detail and scaler is not None else np.empty(0)),
        coef=np.array(classifier.coef_[0]) if keep_fold_detail else np.empty(0),
        intercept=float(classifier.intercept_[0]),
    )


def cross_val_out_of_fold(X: np.ndarray, y: np.ndarray,
                          folds: FoldAssignment,
                          probe_factory=make_probe,
                          keep_fold_detail: bool = True,
                          transform_factory=None) -> OutOfFoldResult:
    """Fit one probe per fold and score the rows that fold held out.

    A constant design matrix (state 0 on this corpus) is handled explicitly
    rather than left to chance: ``StandardScaler`` maps a zero-variance column
    to a constant zero, the classifier then fits an intercept only, and the
    scores come out identical for every row. That is the correct answer -- a
    feature set with no variation carries no information -- and it is flagged
    on the result so a caller can report it as a degenerate control instead of
    a finding.

    ``transform_factory``, when given, is called once per fold as
    ``factory(X, train_index)`` and must return a callable that transforms a
    block of rows. It exists for feature transformations that have to be
    *fitted* -- norm-residualisation is one -- so that the fit sees training
    rows only. Sample-local transformations such as unit-L2 normalisation need
    no factory: applying them to the whole matrix beforehand is arithmetically
    identical and cannot leak.

    ``X`` may be non-numeric (raw category strings) when the pipeline's first
    step encodes it; in that case the encoder is fitted per fold like every
    other step.
    """
    import warnings

    X = np.asarray(X)
    numeric = X.dtype.kind in "fiub"
    if numeric:
        X = X.astype(np.float64)
    y = np.asarray(y).astype(int)
    if X.shape[0] != y.shape[0]:
        raise ValidationError(
            f"design matrix has {X.shape[0]} rows, labels have {y.shape[0]}")
    if folds.n != y.shape[0]:
        raise ValidationError(
            f"fold assignment covers {folds.n} rows, labels have {y.shape[0]}")

    degenerate = bool(X.shape[1] > 0 and (
        float(X.std(axis=0).max()) == 0.0 if numeric
        else all(len(set(X[:, j].tolist())) == 1 for j in range(X.shape[1]))))

    scores = np.full(y.shape[0], np.nan, dtype=np.float64)
    decisions = np.full(y.shape[0], np.nan, dtype=np.float64)
    fold_of_row = np.full(y.shape[0], -1, dtype=int)
    outcomes: List[FoldOutcome] = []

    for k, (train_index, test_index) in enumerate(folds.splits()):
        outcome = fit_probe_on_fold(X, y, train_index, test_index, fold=k,
                                    probe_factory=probe_factory,
                                    keep_fold_detail=keep_fold_detail,
                                    transform_factory=transform_factory)
        scores[test_index] = outcome.scores
        decisions[test_index] = outcome.decisions
        fold_of_row[test_index] = k
        outcomes.append(outcome)

    if np.isnan(scores).any():
        raise ValidationError("a row was never held out and so was never scored")
    if not np.isfinite(scores).all() or not np.isfinite(decisions).all():
        raise ValidationError("the probe produced a non-finite out-of-fold score")

    return OutOfFoldResult(
        scores=scores,
        decisions=decisions,
        fold_of_row=fold_of_row,
        predictions=(scores >= DECISION_THRESHOLD).astype(int),
        folds=outcomes if keep_fold_detail else [],
        degenerate=degenerate,
    )
