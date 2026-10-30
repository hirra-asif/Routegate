"""The probe and the metrics.

Covers requirements 11, 12 and 13 of the test plan: logistic-regression
determinism, state-0 zero-variance handling, and finite metrics.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from conftest import needs_real_inputs
from routegate.metrics import (
    METRIC_NAMES,
    best_state,
    chance_levels,
    compare_to_reference,
    evaluate_out_of_fold,
    max_absolute_difference,
)
from routegate.probes import (
    DECISION_THRESHOLD,
    PROBE_C,
    PROBE_CLASS_WEIGHT,
    PROBE_MAX_ITER,
    PROBE_RANDOM_STATE,
    PROBE_SOLVER,
    PROBE_TOL,
    cross_val_out_of_fold,
    make_probe,
    probe_specification,
)
from routegate.splits import build_stratified_folds
from routegate.validation import ValidationError


def _folds(y, seed=20260805):
    return build_stratified_folds(y, [f"r{i}" for i in range(len(y))], seed=seed)


# ------------------------------------------------- the frozen specification
def test_the_probe_is_the_specified_scikit_learn_estimator():
    pipeline = make_probe()
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.penalty == "l2"
    assert classifier.C == PROBE_C == 0.1
    assert classifier.class_weight == PROBE_CLASS_WEIGHT == "balanced"
    assert classifier.max_iter == PROBE_MAX_ITER == 5000
    assert classifier.solver == PROBE_SOLVER == "lbfgs"
    assert classifier.random_state == PROBE_RANDOM_STATE == 20260805


def test_the_convergence_tolerance_is_the_reference_pilots_not_the_default():
    """``tol`` is stated, not inherited.

    scikit-learn defaults to 1e-4; the reference pilot converged to 1e-6 and
    said so. At the default, lbfgs stops short of the optimum and state 45
    reproduces only to 6.3e-05, which would fail the 1e-6 gate for a reason
    that has nothing to do with the model.
    """
    assert PROBE_TOL == 1e-6
    assert make_probe().named_steps["classifier"].tol == 1e-6
    assert LogisticRegression().tol == 1e-4, (
        "if scikit-learn's default ever changes, the note above needs revisiting")


def test_the_specification_record_matches_the_estimator():
    spec = probe_specification()
    assert spec["C"] == 0.1
    assert spec["class_weight"] == "balanced"
    assert spec["max_iter"] == 5000
    assert spec["solver"] == "lbfgs"
    assert spec["random_state"] == 20260805
    assert spec["tol"] == 1e-6
    assert spec["label_encoding"] == {"E": 1, "B": 0}


# ------------------------------------------------------- 11. determinism
def test_the_probe_reproduces_bit_for_bit(synthetic_problem):
    X, y = synthetic_problem
    folds = _folds(y)
    first = cross_val_out_of_fold(X, y, folds)
    second = cross_val_out_of_fold(X, y, folds)
    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.decisions, second.decisions)
    for a, b in zip(first.folds, second.folds):
        np.testing.assert_array_equal(a.coef, b.coef)
        assert a.intercept == b.intercept


def test_two_fresh_pipelines_agree_exactly(synthetic_problem):
    X, y = synthetic_problem
    a = make_probe().fit(X, y)
    b = make_probe().fit(X, y)
    np.testing.assert_array_equal(a.named_steps["classifier"].coef_,
                                  b.named_steps["classifier"].coef_)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_the_positive_class_column_is_located_not_assumed(synthetic_problem):
    X, y = synthetic_problem
    pipeline = make_probe().fit(X, y)
    classifier = pipeline.named_steps["classifier"]
    assert list(classifier.classes_) == [0, 1]
    column = int(np.flatnonzero(classifier.classes_ == 1)[0])
    folds = _folds(y)
    result = cross_val_out_of_fold(X, y, folds)
    train, test = folds.splits()[0]
    refit = make_probe().fit(X[train], y[train])
    np.testing.assert_allclose(result.folds[0].scores,
                               refit.predict_proba(X[test])[:, column],
                               rtol=0, atol=0)


def test_the_probe_recovers_a_planted_direction(synthetic_problem):
    X, y = synthetic_problem
    result = cross_val_out_of_fold(X, y, _folds(y))
    metrics = evaluate_out_of_fold(y, result.scores, result.predictions, state=0)
    assert metrics.roc_auc > 0.85


# ------------------------------------------ 12. zero-variance / state zero
def test_a_constant_design_matrix_is_flagged_and_scored_at_chance():
    y = np.array([1] * 40 + [0] * 80)
    X = np.full((len(y), 16), 7.5)
    result = cross_val_out_of_fold(X, y, _folds(y))
    assert result.degenerate is True
    assert np.isfinite(result.scores).all()
    np.testing.assert_allclose(result.scores, result.scores[0])
    metrics = evaluate_out_of_fold(y, result.scores, result.predictions, state=0,
                                   degenerate=result.degenerate)
    assert metrics.roc_auc == pytest.approx(0.5)
    assert metrics.pr_auc == pytest.approx(len(y[y == 1]) / len(y))
    assert metrics.balanced_accuracy == pytest.approx(0.5)
    assert metrics.degenerate is True


def test_a_degenerate_state_cannot_win_the_maximum():
    records = [
        evaluate_out_of_fold(np.array([1] * 10 + [0] * 20),
                             np.r_[np.full(10, 0.9), np.full(20, 0.1)],
                             state=1, degenerate=False),
        evaluate_out_of_fold(np.array([1] * 10 + [0] * 20),
                             np.full(30, 0.5), state=0, degenerate=True),
    ]
    assert best_state(records, "roc_auc").state == 1


def test_a_non_finite_metric_never_wins_a_maximum():
    values = [0.9, float("nan")]
    from routegate.validation import safe_argmax
    with pytest.raises(ValidationError, match="non-finite"):
        safe_argmax(values, "roc_auc over states")


@needs_real_inputs
def test_state_zero_is_constant_on_the_real_corpus(inputs):
    matrix = inputs.state_matrix(0)
    assert float(matrix.std(axis=0).max()) == 0.0
    assert len(np.unique(matrix, axis=0)) == 1


@needs_real_inputs
def test_a_transformer_block_state_is_not_constant(inputs):
    assert float(inputs.state_matrix(1).std(axis=0).max()) > 0.0
    assert float(inputs.state_matrix(45).std(axis=0).max()) > 0.0


# --------------------------------------------------------- 13. finite metrics
@pytest.mark.parametrize("case", ["separable", "noise", "near_constant"])
def test_metrics_are_finite_and_in_range(case):
    rng = np.random.default_rng(11)
    y = np.array([1] * 40 + [0] * 80)
    if case == "separable":
        X = np.zeros((120, 12))
        X[y == 1] += 6.0
        X += rng.normal(scale=0.1, size=X.shape)
    elif case == "noise":
        X = rng.normal(size=(120, 12))
    else:
        X = np.ones((120, 12)) + rng.normal(scale=1e-9, size=(120, 12))
    result = cross_val_out_of_fold(X, y, _folds(y))
    metrics = evaluate_out_of_fold(y, result.scores, result.predictions, state=3,
                                   degenerate=result.degenerate)
    for name in METRIC_NAMES:
        value = getattr(metrics, name)
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_metrics_come_from_scikit_learn(synthetic_problem):
    X, y = synthetic_problem
    result = cross_val_out_of_fold(X, y, _folds(y))
    metrics = evaluate_out_of_fold(y, result.scores, result.predictions, state=9)
    assert metrics.roc_auc == roc_auc_score(y, result.scores)
    assert metrics.pr_auc == average_precision_score(y, result.scores)
    assert metrics.balanced_accuracy == balanced_accuracy_score(y, result.predictions)


def test_a_non_finite_score_is_refused():
    y = np.array([1, 1, 0, 0])
    with pytest.raises(ValidationError, match="non-finite"):
        evaluate_out_of_fold(y, np.array([0.1, np.nan, 0.2, 0.3]), state=5)


def test_a_single_class_is_refused():
    with pytest.raises(ValidationError, match="both classes"):
        evaluate_out_of_fold(np.ones(6, dtype=int), np.linspace(0, 1, 6), state=5)


def test_chance_levels_use_prevalence_for_pr_auc():
    y = np.array([1] * 89 + [0] * 178)
    levels = chance_levels(y)
    assert levels["roc_auc"] == 0.5
    assert levels["pr_auc"] == pytest.approx(89 / 267)
    assert levels["prevalence"] == pytest.approx(0.3333333, abs=1e-6)


def test_the_decision_threshold_is_a_half():
    assert DECISION_THRESHOLD == 0.5


# ------------------------------------------------------------- comparison
def test_comparison_reports_absolute_differences():
    y = np.array([1] * 10 + [0] * 20)
    record = evaluate_out_of_fold(y, np.r_[np.full(10, 0.9), np.full(20, 0.1)],
                                  state=7)
    reference = {7: {"roc_auc": record.roc_auc,
                     "pr_auc": record.pr_auc + 1e-3,
                     "balanced_accuracy": record.balanced_accuracy,
                     "accuracy": record.accuracy}}
    rows = compare_to_reference([record], reference, tolerance=1e-6)
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["roc_auc"]["within_tolerance"] is True
    assert by_metric["pr_auc"]["within_tolerance"] is False
    assert max_absolute_difference(rows) == pytest.approx(1e-3)


def test_a_state_without_a_reference_is_marked_not_compared():
    y = np.array([1] * 10 + [0] * 20)
    record = evaluate_out_of_fold(y, np.r_[np.full(10, 0.9), np.full(20, 0.1)],
                                  state=7)
    rows = compare_to_reference([record], {}, tolerance=1e-6)
    assert all(r["within_tolerance"] == "no_reference" for r in rows)
    with pytest.raises(ValidationError, match="no comparable cells"):
        max_absolute_difference(rows)
