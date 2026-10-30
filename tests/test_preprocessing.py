"""Preprocessing discipline.

Covers requirement 10 of the test plan: the scaler is fitted on training rows
only. The decisive test corrupts the held-out rows and checks that nothing the
pipeline learned moves.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from routegate.probes import cross_val_out_of_fold, make_probe
from routegate.splits import build_stratified_folds


def _folds(y, n_splits=5, seed=20260805):
    return build_stratified_folds(y, [f"r{i}" for i in range(len(y))],
                                  n_splits=n_splits, seed=seed)


def test_the_scaler_lives_inside_the_pipeline():
    pipeline = make_probe()
    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_fold_scaler_statistics_are_the_training_rows_own(synthetic_problem):
    X, y = synthetic_problem
    folds = _folds(y)
    result = cross_val_out_of_fold(X, y, folds)
    for fold in result.folds:
        expected = StandardScaler().fit(X[fold.train_index])
        np.testing.assert_allclose(fold.scaler_mean, expected.mean_, rtol=0, atol=0)
        np.testing.assert_allclose(fold.scaler_scale, expected.scale_, rtol=0, atol=0)
        whole = StandardScaler().fit(X)
        assert not np.allclose(fold.scaler_mean, whole.mean_), (
            "a fold scaler must not equal the whole-data scaler")


def test_corrupting_the_held_out_rows_moves_nothing_that_was_fitted(synthetic_problem):
    """The decisive leakage test.

    If any fitted quantity had seen a held-out row, replacing those rows with
    garbage would move it. The held-out scores must move -- otherwise the
    corruption was inert and the test proves nothing.
    """
    X, y = synthetic_problem
    folds = _folds(y)
    clean = cross_val_out_of_fold(X, y, folds)
    rng = np.random.default_rng(99)

    for k, reference in enumerate(clean.folds):
        corrupted = X.copy()
        corrupted[reference.test_index] = rng.normal(
            loc=500.0, scale=250.0,
            size=(reference.test_index.size, X.shape[1]))
        dirty = cross_val_out_of_fold(corrupted, y, folds).folds[k]

        np.testing.assert_allclose(dirty.scaler_mean, reference.scaler_mean,
                                   rtol=0, atol=0)
        np.testing.assert_allclose(dirty.scaler_scale, reference.scaler_scale,
                                   rtol=0, atol=0)
        np.testing.assert_allclose(dirty.coef, reference.coef, rtol=0, atol=0)
        assert dirty.intercept == reference.intercept
        assert not np.allclose(dirty.scores, reference.scores), (
            "the corruption did not reach the held-out scores")


def test_held_out_rows_are_transformed_with_training_statistics(synthetic_problem):
    X, y = synthetic_problem
    folds = _folds(y)
    result = cross_val_out_of_fold(X, y, folds)
    for fold in result.folds:
        scaler = StandardScaler().fit(X[fold.train_index])
        transformed = scaler.transform(X[fold.test_index])
        recomputed = 1.0 / (1.0 + np.exp(-(transformed @ fold.coef + fold.intercept)))
        np.testing.assert_allclose(recomputed, fold.scores, rtol=1e-9, atol=1e-12)


def test_a_zero_variance_column_does_not_produce_a_non_finite_value(synthetic_problem):
    X, y = synthetic_problem
    X = X.copy()
    X[:, 0] = 3.0
    scaler = StandardScaler().fit(X)
    assert scaler.scale_[0] == 1.0
    transformed = scaler.transform(X)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[:, 0], 0.0)


def test_every_row_is_scored_exactly_once(synthetic_problem):
    X, y = synthetic_problem
    folds = _folds(y)
    result = cross_val_out_of_fold(X, y, folds)
    assert result.scores.shape[0] == y.shape[0]
    assert np.isfinite(result.scores).all()
    assert set(result.fold_of_row.tolist()) == set(range(folds.n_splits))
    for k, (_, test) in enumerate(folds.splits()):
        np.testing.assert_array_equal(np.flatnonzero(result.fold_of_row == k),
                                      np.sort(test))


def test_a_fold_assignment_of_the_wrong_length_is_refused(synthetic_problem):
    from routegate.validation import ValidationError
    X, y = synthetic_problem
    folds = _folds(y[:40])
    with pytest.raises(ValidationError):
        cross_val_out_of_fold(X, y, folds)
