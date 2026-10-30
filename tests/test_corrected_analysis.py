"""The corrected analysis: variants, repeated splits, selection and the null.

Every requirement the corrected protocol adds over the reproduction is pinned
here — that the split assignment is built once and reused, that a fitted
transformation never sees a held-out row, that the decision threshold is fixed
at 0.5, that a permutation searches every included state, that a probe
direction is reconstructed through the scaler it was fitted with, and that a
subgroup too small to cross-validate is refused rather than reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from conftest import needs_real_inputs
from routegate.controls import (
    ACTIVATION_VARIANTS,
    PROMPT_LENGTH_NOTE,
    apply_norm_residualiser,
    categorical_column,
    fit_norm_residualiser,
    permuted_labels,
    prompt_length_features,
    unit_l2_normalised_states,
    variant_matrix,
    variant_transform_factory,
)
from routegate.data import COVARIATE_COLUMNS, token_count_columns
from routegate.probes import (
    DECISION_THRESHOLD,
    cross_val_out_of_fold,
    make_categorical_probe,
    make_probe,
    raw_space_direction,
)
from routegate.selection import (
    mean_cross_validated_roc_auc,
    nested_state_selection,
    permutation_null,
    search_maximum,
)
from routegate.splits import (
    build_repeated_stratified_folds,
    build_stratified_folds,
    folds_for_subset,
    load_repeated_fold_assignments,
    write_repeated_fold_assignments,
)
from routegate.validation import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ANALYSIS = REPO_ROOT / "experiments" / "residual_geometry" / "run_analysis.py"
CONFIG = REPO_ROOT / "configs" / "residual_geometry.json"


def _states(y, n_states=4, dim=12, seed=20260805):
    """A small cube [state, row, dim] with one informative state and a constant one."""
    rng = np.random.default_rng(seed)
    cube = rng.normal(size=(n_states, y.shape[0], dim))
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    cube[2][y == 1] += 3.0 * direction
    cube[0] = 1.0                                   # the degenerate state
    return cube


@pytest.fixture
def problem():
    y = np.array([1] * 30 + [0] * 60, dtype=int)
    return _states(y), y, [f"r{i}" for i in range(y.shape[0])]


# ---------------------------------------------- one split assignment, reused
def test_repeated_folds_are_distinct_draws_from_their_seeds(problem):
    _, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=4)
    assert repeated.n_repeats == 4
    assignments = {tuple(f.fold_of_row.tolist()) for f in repeated.assignments}
    assert len(assignments) == 4
    again = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=4)
    for first, second in zip(repeated.assignments, again.assignments):
        assert np.array_equal(first.fold_of_row, second.fold_of_row)


def test_the_written_assignment_round_trips(problem, tmp_path):
    _, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=3)
    path = tmp_path / "cv_assignments.csv"
    write_repeated_fold_assignments(path, repeated, y)
    reloaded = load_repeated_fold_assignments(path, row_ids)

    assert reloaded.n_repeats == repeated.n_repeats
    assert reloaded.base_seed == repeated.base_seed
    for first, second in zip(repeated.assignments, reloaded.assignments):
        assert np.array_equal(first.fold_of_row, second.fold_of_row)


def test_the_written_assignment_names_every_row_repeat_and_fold(problem, tmp_path):
    import csv

    _, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=3)
    path = tmp_path / "cv_assignments.csv"
    write_repeated_fold_assignments(path, repeated, y)
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))

    assert len(rows) == 3 * len(row_ids)
    for column in ("row_id", "repeat", "fold", "split", "y_true"):
        assert column in rows[0]
    assert {int(r["repeat"]) for r in rows} == {0, 1, 2}


def test_a_reloaded_assignment_is_rebound_to_the_row_order(problem, tmp_path):
    _, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=2)
    path = tmp_path / "cv_assignments.csv"
    write_repeated_fold_assignments(path, repeated, y)

    shuffled = list(reversed(row_ids))
    reloaded = load_repeated_fold_assignments(path, shuffled)
    assert reloaded.assignments[0].row_ids == shuffled
    assert np.array_equal(reloaded.assignments[0].fold_of_row,
                          repeated.assignments[0].fold_of_row[::-1])


# ------------------------------------------------------- activation variants
def test_unit_l2_is_sample_local(problem):
    cube, y, _ = problem
    X = cube[2]
    whole = unit_l2_normalised_states(X)
    for i in range(X.shape[0]):
        row = unit_l2_normalised_states(X[i:i + 1])
        assert np.allclose(row[0], whole[i])
    assert np.allclose(np.linalg.norm(whole, axis=1), 1.0)


def test_unit_l2_needs_no_fold_fitted_transform():
    assert variant_transform_factory("unit_l2") is None
    assert variant_transform_factory("raw") is None
    assert variant_transform_factory("activation_norm") is None
    assert variant_transform_factory("norm_residualised") is not None


def test_the_activation_norm_variant_is_one_scalar_per_example(problem):
    cube, _, _ = problem
    features = variant_matrix("activation_norm", cube[2])
    assert features.shape == (cube.shape[1], 1)
    assert np.allclose(features[:, 0], np.linalg.norm(cube[2], axis=1))


def test_norm_residualisation_is_fitted_on_training_rows_only(problem):
    cube, y, row_ids = problem
    X = cube[2]
    folds = build_stratified_folds(y, row_ids, n_splits=5)
    train_index, test_index = folds.splits()[0]

    factory = variant_transform_factory("norm_residualised")
    transform = factory(X, train_index)
    before = transform(X[test_index]).copy()

    corrupted = X.copy()
    corrupted[test_index] = 1e6                       # garbage, held out only
    coefficients_after = fit_norm_residualiser(corrupted[train_index])
    assert np.allclose(fit_norm_residualiser(X[train_index]), coefficients_after)
    assert np.allclose(before, apply_norm_residualiser(X[test_index],
                                                       coefficients_after))


def test_every_declared_variant_is_known():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for variant in config["activation_variants"]["included"]:
        assert variant in ACTIVATION_VARIANTS


# ----------------------------------------------------------------- threshold
def test_balanced_accuracy_uses_a_fixed_half_threshold(problem):
    from routegate.metrics import evaluate_out_of_fold
    from sklearn.metrics import balanced_accuracy_score

    cube, y, row_ids = problem
    folds = build_stratified_folds(y, row_ids, n_splits=5)
    result = cross_val_out_of_fold(cube[2], y, folds)
    metrics = evaluate_out_of_fold(y, result.scores, result.predictions)

    assert DECISION_THRESHOLD == 0.5
    assert metrics.balanced_accuracy == pytest.approx(
        balanced_accuracy_score(y, (result.scores >= 0.5).astype(int)))
    assert np.array_equal(result.predictions, (result.scores >= 0.5).astype(int))


# ------------------------------------------------- categorical controls (OHE)
def test_the_categorical_probe_fits_its_encoder_on_training_rows():
    y = np.array([1] * 20 + [0] * 40, dtype=int)
    categories = ["a"] * 30 + ["b"] * 29 + ["rare"]
    X = categorical_column(categories)
    folds = build_stratified_folds(y, [f"r{i}" for i in range(y.shape[0])],
                                   n_splits=5)
    result = cross_val_out_of_fold(X, y, folds,
                                   probe_factory=make_categorical_probe)
    assert np.isfinite(result.scores).all()

    encoder = make_categorical_probe().named_steps["encoder"]
    assert encoder.handle_unknown == "ignore"

    # a category held out of training encodes as all-zero rather than raising
    train = np.array([i for i, c in enumerate(categories) if c != "rare"])
    fitted = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    fitted.fit(X[train])
    assert fitted.transform([["rare"]]).sum() == 0.0


def test_the_categorical_probe_is_the_same_estimator_with_an_encoder_in_front():
    pipeline = make_categorical_probe()
    assert [name for name, _ in pipeline.steps] == ["encoder", "scaler",
                                                    "classifier"]
    assert isinstance(pipeline.named_steps["encoder"], OneHotEncoder)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    reference = make_probe().named_steps["classifier"]
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    for parameter in ("C", "penalty", "class_weight", "solver", "tol",
                      "max_iter", "random_state"):
        assert getattr(classifier, parameter) == getattr(reference, parameter)


# ------------------------------------------------------- prompt-length control
def test_no_token_count_is_invented():
    manifest_columns = ["row_index", "row_id", "final_ebh", "source_family",
                        "prompt_text", "prompt_sha256"]
    assert token_count_columns(manifest_columns) == []
    assert token_count_columns(["row_id", "n_tokens"]) == ["n_tokens"]

    features = prompt_length_features(["a b c", "dd"])
    assert features.shape == (2, 2)
    assert features[0].tolist() == [5.0, 3.0]         # characters, whitespace
    assert "token count" in PROMPT_LENGTH_NOTE.lower()


def test_the_config_records_that_no_token_count_exists():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "not available" in config["controls"]["prompt_token_count"]
    assert config["controls"]["prompt_length_features"] == [
        "prompt_characters", "prompt_whitespace_tokens"]


def test_the_declared_categorical_columns_are_the_ones_the_loader_uses():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    declared = config["controls"]["categorical_columns"]
    for role, column in COVARIATE_COLUMNS.items():
        assert declared[role] == column


# ----------------------------------------------------------- direction recovery
def test_the_raw_space_direction_divides_by_the_training_fold_scale():
    rng = np.random.default_rng(20260805)
    X = rng.normal(size=(80, 6)) * np.array([1.0, 10.0, 0.1, 5.0, 2.0, 0.5])
    y = (X @ np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(max_iter=5000))])
    pipeline.fit(X, y)
    scaler = pipeline.named_steps["scaler"]
    coefficients = pipeline.named_steps["classifier"].coef_[0]

    direction = raw_space_direction(coefficients, scaler.scale_)
    assert np.allclose(direction, coefficients / scaler.scale_)

    # the reconstruction is the one that reproduces the decision function
    raw = X @ direction
    standardised = scaler.transform(X) @ coefficients
    assert np.allclose(raw - raw.mean(), standardised - standardised.mean())


def test_fold_directions_are_kept_separately(problem):
    cube, y, row_ids = problem
    folds = build_stratified_folds(y, row_ids, n_splits=5)
    result = cross_val_out_of_fold(cube[2], y, folds, keep_fold_detail=True)
    directions = np.vstack([raw_space_direction(f.coef, f.scaler_scale)
                            for f in result.folds])
    assert directions.shape == (5, cube.shape[2])
    assert len({tuple(np.round(d, 12)) for d in directions}) == 5


def test_a_zero_scale_is_refused():
    with pytest.raises(ValidationError):
        raw_space_direction(np.ones(3), np.array([1.0, 0.0, 1.0]))


# --------------------------------------------------------- nested selection
def test_nested_selection_never_scores_the_state_it_chose_on_its_own_rows(problem):
    cube, y, row_ids = problem
    outer = build_stratified_folds(y, row_ids, n_splits=5)
    result = nested_state_selection(lambda s: cube[s], y, outer,
                                    states=[1, 2, 3], inner_splits=3)
    assert len(result.records) == 5
    assert set(result.selected_states) <= {1, 2, 3}
    # the informative state is state 2, and the inner search should find it
    assert result.selected_states.count(2) >= 3
    assert np.isfinite(result.scores).all()
    for record in result.records:
        assert record.n_train + record.n_held_out == y.shape[0]


def test_nested_selection_refuses_a_candidate_set_it_cannot_search(problem):
    cube, y, row_ids = problem
    outer = build_stratified_folds(y, row_ids, n_splits=5)
    with pytest.raises(ValidationError):
        nested_state_selection(lambda s: cube[s], y, outer, states=[],
                               inner_splits=3)


# ---------------------------------------------------------- permutation null
def test_a_permutation_searches_every_included_state(problem):
    cube, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=1)
    null = permutation_null(lambda s: cube[s], y, repeated, states=[1, 2, 3],
                            n_permutations=4, base_seed=1234)

    assert null["n_permutations"] == 4
    assert len(null["records"]) == 5                  # observed + 4
    for record in null["records"]:
        assert record.n_states_searched == 3
    assert null["p_value"] == pytest.approx(
        (1 + null["n_at_least_as_extreme"]) / 5)
    assert null["p_value_floor"] == pytest.approx(1 / 5)


def test_the_null_reuses_the_split_assignment_and_only_moves_the_labels(problem):
    cube, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=1)
    before = [f.fold_of_row.copy() for f in repeated.assignments]
    permutation_null(lambda s: cube[s], y, repeated, states=[1, 2],
                     n_permutations=2, base_seed=99)
    for original, folds in zip(before, repeated.assignments):
        assert np.array_equal(original, folds.fold_of_row)

    first = permuted_labels(y, seed=99)
    assert np.array_equal(first, permuted_labels(y, seed=99))
    assert sorted(first.tolist()) == sorted(y.tolist())


def test_the_observed_statistic_and_the_null_use_the_same_function(problem):
    cube, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=1)
    null = permutation_null(lambda s: cube[s], y, repeated, states=[1, 2, 3],
                            n_permutations=2, base_seed=7)
    direct = search_maximum(lambda s: cube[s], y, repeated, states=[1, 2, 3])
    assert null["observed"]["max_mean_roc_auc"] == pytest.approx(
        direct["max_mean_roc_auc"])
    assert null["observed"]["argmax_state"] == direct["argmax_state"]


def test_a_custom_runner_must_return_one_outcome_per_permutation(problem):
    cube, y, row_ids = problem
    repeated = build_repeated_stratified_folds(y, row_ids, n_splits=5, repeats=1)
    with pytest.raises(ValidationError):
        permutation_null(lambda s: cube[s], y, repeated, states=[1, 2],
                         n_permutations=3, base_seed=5,
                         runner=lambda seeds: [])


def test_the_mean_cross_validated_auc_is_the_mean_over_folds(problem):
    from sklearn.metrics import roc_auc_score

    cube, y, row_ids = problem
    folds = build_stratified_folds(y, row_ids, n_splits=5)
    result = cross_val_out_of_fold(cube[2], y, folds, keep_fold_detail=True)
    by_hand = np.mean([roc_auc_score(y[f.test_index], f.scores)
                       for f in result.folds])
    assert mean_cross_validated_roc_auc(y, result) == pytest.approx(by_hand)


# ------------------------------------------------------------ within source
def test_a_subgroup_with_one_example_of_a_class_is_refused():
    y = np.array([1] + [0] * 20, dtype=int)
    with pytest.raises(ValidationError):
        folds_for_subset(y, [f"r{i}" for i in range(y.shape[0])], n_splits=5,
                         seed=1)


def test_the_fold_count_is_capped_by_the_smaller_class():
    y = np.array([1] * 3 + [0] * 40, dtype=int)
    folds, used = folds_for_subset(y, [f"r{i}" for i in range(y.shape[0])],
                                   n_splits=5, seed=1)
    assert used == 3
    assert folds.n_splits == 3
    for train, _ in folds.splits():
        assert int((y[train] == 1).sum()) > 0

    y_big = np.array([1] * 30 + [0] * 60, dtype=int)
    _, used_big = folds_for_subset(y_big, [f"r{i}" for i in range(90)],
                                   n_splits=5, seed=1)
    assert used_big == 5


def test_the_within_source_protocol_is_declared():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    within = config["within_source"]
    assert within["min_class_examples"] == 2
    assert within["column"] == COVARIATE_COLUMNS["source"]
    assert within["unstable_if_smallest_class_below"] >= 2


# ------------------------------------------------------------- the entry point
def test_the_corrected_analysis_still_refuses_to_run_without_approval():
    source = RUN_ANALYSIS.read_text(encoding="utf-8")
    assert "--i-have-approved-the-corrected-protocol" in source

    sys.path.insert(0, str(RUN_ANALYSIS.parent))
    try:
        import importlib

        module = importlib.import_module("run_analysis")
        importlib.reload(module)
        assert module.main([]) == 0             # declines, does not run
    finally:
        sys.path.remove(str(RUN_ANALYSIS.parent))


def test_a_smoke_run_may_not_write_to_the_reportable_directory():
    sys.path.insert(0, str(RUN_ANALYSIS.parent))
    try:
        import importlib

        module = importlib.import_module("run_analysis")
        with pytest.raises(ValidationError, match="smoke"):
            module.analyse(output_dir=REPO_ROOT / "outputs" / "nowhere",
                           smoke=True)
    finally:
        sys.path.remove(str(RUN_ANALYSIS.parent))


def test_a_reportable_run_may_not_write_to_a_smoke_directory():
    sys.path.insert(0, str(RUN_ANALYSIS.parent))
    try:
        import importlib

        module = importlib.import_module("run_analysis")
        with pytest.raises(ValidationError, match="smoke"):
            module.analyse(output_dir=REPO_ROOT / "outputs" / "corrected_smoke",
                           smoke=False)
    finally:
        sys.path.remove(str(RUN_ANALYSIS.parent))


@needs_real_inputs
def test_the_real_covariate_columns_exist_and_are_populated(inputs):
    for role, column in COVARIATE_COLUMNS.items():
        values = inputs.covariate(role)
        assert len(values) == inputs.n
        assert all(v.strip() for v in values)
        assert len(set(values)) >= 2
    assert inputs.saved_token_count_columns == []


@needs_real_inputs
def test_the_within_source_counts_are_reported_exactly(inputs):
    from collections import Counter

    counts = Counter(zip(inputs.covariate("source"), inputs.y.tolist()))
    sources = sorted(set(inputs.covariate("source")))
    assert len(sources) >= 2
    for source in sources:
        n_e = counts[(source, 1)]
        n_b = counts[(source, 0)]
        assert n_e + n_b > 0
        if min(n_e, n_b) >= 2:
            _, used = folds_for_subset(
                np.array([1] * n_e + [0] * n_b),
                [f"r{i}" for i in range(n_e + n_b)], n_splits=5, seed=1)
            assert used <= min(n_e, n_b)
