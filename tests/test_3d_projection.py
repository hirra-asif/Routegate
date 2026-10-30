"""The held-out 3D projection figure, on synthetic data only.

Nothing here touches the real corpus, the activation array or the corrected
run's outputs. The fixtures are a small planted two-class problem and a
hand-written ``cv_assignments.csv`` in ``tmp_path``, which is enough to test
every claim the figure makes about itself:

* it reuses a frozen split rather than drawing one;
* the scaler, the PLS basis and the plane see training rows only;
* no row is in both halves;
* the projection has the dimensions the axes say it has;
* two runs give the same figure;
* the metadata says what was done, including that it is descriptive only;
* the filenames carry the settings;
* it renders headless.

The plotting test writes into ``tmp_path`` under the Agg backend, so it needs no
display and leaves nothing behind.
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO_ROOT / "experiments" / "residual_geometry"

from routegate.validation import ValidationError                   # noqa: E402


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def script():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        module = importlib.import_module("make_3d_projection")
        yield importlib.reload(module)
    finally:
        sys.path.remove(str(EXPERIMENT))


# --------------------------------------------------------------- fixtures
N_ROWS = 60
N_FEATURES = 12
N_SPLITS = 5
N_REPEATS = 2
SEED = 20260805


def synthetic_problem(n_rows: int = N_ROWS, n_features: int = N_FEATURES):
    """A planted, separable two-class problem in a handful of dimensions."""
    rng = np.random.default_rng(20260806)
    y = np.array([1, 0] * (n_rows // 2), dtype=int)
    X = rng.normal(size=(n_rows, n_features))
    direction = rng.normal(size=n_features)
    direction /= np.linalg.norm(direction)
    X[y == 1] += 2.5 * direction
    row_ids = [f"synthetic:{i:05d}" for i in range(n_rows)]
    return X, y, row_ids


@pytest.fixture
def problem():
    return synthetic_problem()


def write_assignments(path: Path, y, row_ids, n_splits: int = N_SPLITS,
                      repeats: int = N_REPEATS, seed: int = SEED) -> Path:
    """A cv_assignments.csv in the shape the corrected run writes one."""
    fieldnames = ["repeat", "seed", "position", "row_id", "y_true", "final_ebh",
                  "fold", "split", "array_row_index"]
    y = np.asarray(y, dtype=int)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for repeat in range(repeats):
            # Stratified and deterministic: each class is shuffled with its own
            # seeded generator and dealt round-robin, so every fold holds out
            # both classes. Written by hand rather than by the package's fold
            # builder -- the point of these tests is that the *file* is obeyed,
            # whatever produced it.
            fold_of_position = np.zeros(len(row_ids), dtype=int)
            for value in (0, 1):
                members = np.flatnonzero(y == value)
                rng = np.random.default_rng(seed + repeat + value)
                shuffled = members[np.argsort(rng.random(members.size))]
                for rank, position in enumerate(shuffled):
                    fold_of_position[position] = rank % n_splits
            for position, row_id in enumerate(row_ids):
                writer.writerow({
                    "repeat": repeat,
                    "seed": seed + repeat,
                    "position": position,
                    "row_id": row_id,
                    "y_true": int(y[position]),
                    "final_ebh": "E" if y[position] == 1 else "B",
                    "fold": int(fold_of_position[position]),
                    "split": "held_out",
                    "array_row_index": position,
                })
    return path


@pytest.fixture
def assignments(tmp_path, problem):
    _, y, row_ids = problem
    return write_assignments(tmp_path / "cv_assignments.csv", y, row_ids)


@pytest.fixture
def split(script, assignments, problem):
    _, _, row_ids = problem
    repeated = script.read_frozen_assignments(assignments, row_ids)
    return script.select_split(repeated, repeat=0, fold=0)


@pytest.fixture
def fitted(script, problem, split):
    X, y, _ = problem
    projection = script.fit_projection(X[split.train_index],
                                       y[split.train_index], n_components=3)
    return projection, X, y


# ------------------------------------------------- 1. reuse of the frozen split
def test_the_frozen_assignment_is_read_back_not_rebuilt(script, assignments,
                                                        problem):
    _, _, row_ids = problem
    repeated = script.read_frozen_assignments(assignments, row_ids)

    on_disk = {(int(row["repeat"]), row["row_id"]): int(row["fold"])
               for row in read_csv(assignments)}

    for repeat, assignment in enumerate(repeated.assignments):
        for position, row_id in enumerate(assignment.row_ids):
            assert int(assignment.fold_of_row[position]) == on_disk[
                (repeat, row_id)]


def test_the_split_matches_the_file_exactly(script, assignments, problem):
    _, _, row_ids = problem
    repeated = script.read_frozen_assignments(assignments, row_ids)
    selected = script.select_split(repeated, repeat=0, fold=0)

    expected = {row["row_id"] for row in read_csv(assignments)
                if row["repeat"] == "0" and row["fold"] == "0"}
    assert set(selected.held_out_row_ids) == expected


def test_the_split_source_names_the_file_it_came_from(script, split,
                                                      assignments):
    assert "repeated_stratified" in split.source
    assert str(assignments) in split.source


def test_a_missing_assignment_file_says_to_run_the_analysis_first(
        script, tmp_path, problem):
    _, _, row_ids = problem
    with pytest.raises(ValidationError, match="does not create one"):
        script.read_frozen_assignments(tmp_path / "absent.csv", row_ids)


def test_an_assignment_for_other_rows_is_refused(script, tmp_path, problem):
    """The gate that stops a figure being drawn against a different corpus."""
    X, y, row_ids = problem
    path = write_assignments(tmp_path / "other.csv", y,
                             [f"other:{i:05d}" for i in range(len(row_ids))])
    with pytest.raises(ValidationError):
        script.read_frozen_assignments(path, row_ids)


def test_a_repeat_or_fold_outside_the_file_is_refused(script, assignments,
                                                      problem):
    _, _, row_ids = problem
    repeated = script.read_frozen_assignments(assignments, row_ids)
    with pytest.raises(ValidationError, match="repeat"):
        script.select_split(repeated, repeat=N_REPEATS, fold=0)
    with pytest.raises(ValidationError, match="fold"):
        script.select_split(repeated, repeat=0, fold=N_SPLITS)


def test_every_fold_of_every_repeat_is_selectable(script, assignments, problem):
    _, _, row_ids = problem
    repeated = script.read_frozen_assignments(assignments, row_ids)
    for repeat in range(N_REPEATS):
        for fold in range(N_SPLITS):
            selected = script.select_split(repeated, repeat, fold)
            assert selected.held_out_index.size > 0
            assert selected.train_index.size > 0


# --------------------------------------------- 2. no train / held-out overlap
def test_train_and_held_out_row_ids_do_not_overlap(script, assignments,
                                                   problem):
    _, _, row_ids = problem
    repeated = script.read_frozen_assignments(assignments, row_ids)
    for repeat in range(N_REPEATS):
        for fold in range(N_SPLITS):
            selected = script.select_split(repeated, repeat, fold)
            assert not (set(selected.train_row_ids)
                        & set(selected.held_out_row_ids))


def test_the_two_halves_partition_every_row(script, split, problem):
    _, _, row_ids = problem
    assert (set(split.train_row_ids) | set(split.held_out_row_ids)
            == set(row_ids))
    assert (split.train_index.size + split.held_out_index.size) == len(row_ids)


def test_an_overlapping_split_is_refused(script):
    """Stated over row identity, so a positional bug cannot slip past it."""
    bad = script.Split(
        repeat=0, fold=0,
        train_index=np.array([0, 1, 2]), held_out_index=np.array([2, 3]),
        train_row_ids=["a", "b", "c"], held_out_row_ids=["c", "d"],
        source="synthetic")
    with pytest.raises(ValidationError, match="both the training and the"):
        bad.audit()


# ------------------------------------------------------- 3. training-only fits
def test_the_scaler_sees_training_rows_only(script, fitted, split):
    projection, X, _ = fitted
    train = X[split.train_index]
    np.testing.assert_allclose(projection.scaler.mean_, train.mean(axis=0))
    assert projection.n_train == train.shape[0]
    # and is not the statistic it would have had if the held-out rows had been
    # included, which is the failure this test exists to catch
    assert not np.allclose(projection.scaler.mean_, X.mean(axis=0))


def test_changing_the_held_out_rows_changes_nothing_that_was_fitted(
        script, problem, split):
    """The decisive leakage test: held-out values cannot reach the fit."""
    X, y, _ = problem
    first = script.fit_projection(X[split.train_index], y[split.train_index], 3)

    corrupted = X.copy()
    corrupted[split.held_out_index] += 1000.0
    second = script.fit_projection(corrupted[split.train_index],
                                   y[split.train_index], 3)

    np.testing.assert_array_equal(first.scaler.mean_, second.scaler.mean_)
    np.testing.assert_array_equal(first.train_scores, second.train_scores)
    np.testing.assert_array_equal(first.classifier.coef_,
                                  second.classifier.coef_)


def test_transform_fits_nothing(script, fitted, split):
    """Projecting held-out rows must not update the fitted objects."""
    projection, X, _ = fitted
    before_mean = projection.scaler.mean_.copy()
    before_coef = projection.classifier.coef_.copy()

    projection.transform(X[split.held_out_index])

    np.testing.assert_array_equal(projection.scaler.mean_, before_mean)
    np.testing.assert_array_equal(projection.classifier.coef_, before_coef)


def test_the_pls_is_unscaled_because_the_scaler_already_scaled(script, fitted):
    projection, _, _ = fitted
    assert projection.pls.scale is False
    assert projection.pls.n_components == 3


def test_a_single_class_training_set_is_refused(script, problem):
    X, y, _ = problem
    only_e = np.flatnonzero(y == 1)
    with pytest.raises(ValidationError, match="both E"):
        script.fit_projection(X[only_e], y[only_e], 3)


def test_fewer_than_three_components_is_refused(script, problem, split):
    X, y, _ = problem
    with pytest.raises(ValidationError, match="at least 3 components"):
        script.fit_projection(X[split.train_index], y[split.train_index], 2)


# ------------------------------------------------------ 4. output dimensions
def test_the_projection_is_three_dimensional(script, fitted, split):
    projection, X, _ = fitted
    scores = projection.transform(X[split.held_out_index])
    assert scores.shape == (split.held_out_index.size, 3)
    assert projection.train_scores.shape == (split.train_index.size, 3)
    assert projection.train_scores_3d.shape[1] == 3


def test_more_components_still_draw_exactly_three(script, problem, split):
    """--n-components is usable, and the figure stays three-dimensional."""
    X, y = problem[0], problem[1]
    five = script.fit_projection(X[split.train_index], y[split.train_index], 5)
    assert five.n_components == 5
    assert five.train_scores.shape == (split.train_index.size, 5)
    assert five.train_scores_3d.shape == (split.train_index.size, 3)
    assert np.asarray(five.classifier.coef_).ravel().shape == (3,)


def test_rows_of_the_wrong_width_are_refused(script, fitted):
    projection, X, _ = fitted
    with pytest.raises(ValidationError, match="fitted on"):
        projection.transform(X[:, :3])


def test_the_plane_is_fitted_in_the_drawn_space(script, fitted):
    projection, _, _ = fitted
    assert np.asarray(projection.classifier.coef_).ravel().shape == (3,)


def test_the_geometry_has_a_centroid_and_covariance_per_class(script, fitted,
                                                              split):
    projection, _, y = fitted
    geometry = script.class_geometry(projection.train_scores_3d,
                                     y[split.train_index])
    assert sorted(geometry) == ["B", "E"]
    for label in ("B", "E"):
        assert geometry[label]["centroid"].shape == (3,)
        assert geometry[label]["covariance"].shape == (3, 3)
        assert geometry[label]["n"] > 3


def test_an_ellipsoid_is_a_closed_surface_around_its_centroid(script):
    centroid = np.array([1.0, -2.0, 0.5])
    covariance = np.diag([4.0, 1.0, 0.25])
    x, y, z = script.ellipsoid_surface(centroid, covariance, n_sd=1.0,
                                       resolution=24)
    assert x.shape == y.shape == z.shape == (24, 24)
    # a diagonal covariance stretches along the axes by exactly its sd
    np.testing.assert_allclose([x.max(), y.max(), z.max()],
                               centroid + np.array([2.0, 1.0, 0.5]), atol=1e-9)
    np.testing.assert_allclose([x.min(), y.min(), z.min()],
                               centroid - np.array([2.0, 1.0, 0.5]), atol=1e-9)


@pytest.mark.parametrize("resolution", [5, 6, 8, 24, 25, 40])
def test_the_angular_grid_hits_every_required_angle(script, resolution):
    """The quarter-turns and the equator, exactly, at any resolution.

    A plain linspace reaches them only when the arithmetic happens to divide
    evenly: at 24 it needs index 11.5 for pi/2 and lands 0.47% short of the
    equator, which shrinks the drawn ellipsoid without ever looking wrong.
    """
    azimuth = script._angles_through(script.AZIMUTH_BREAKPOINTS, resolution)
    polar = script._angles_through(script.POLAR_BREAKPOINTS, resolution)

    assert azimuth.shape == polar.shape == (resolution,)
    assert np.all(np.diff(azimuth) > 0) and np.all(np.diff(polar) > 0)
    for required in script.AZIMUTH_BREAKPOINTS:
        assert np.abs(azimuth - required).min() == 0.0
    for required in script.POLAR_BREAKPOINTS:
        assert np.abs(polar - required).min() == 0.0
    # the equator is what the principal-axis extrema actually depend on
    assert np.sin(polar).max() == 1.0


@pytest.mark.parametrize("resolution", [5, 8, 24, 25, 40])
def test_the_surface_reaches_both_ends_of_every_principal_axis(script,
                                                               resolution):
    """Stated on a rotated covariance, so it also pins the rotation down.

    The endpoints are ``centroid +/- sqrt(eigenvalue) * eigenvector``. Checking
    against those rather than against the drawing axes is what makes this a
    test of the ellipsoid rather than of the diagonal special case.
    """
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    covariance = rotation @ np.diag([4.0, 1.0, 0.25]) @ rotation.T
    centroid = np.array([0.3, -1.1, 2.0])

    x, y, z = script.ellipsoid_surface(centroid, covariance, n_sd=1.0,
                                       resolution=resolution)
    assert x.shape == y.shape == z.shape == (resolution, resolution)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    for index in range(3):
        axis = np.sqrt(eigenvalues[index]) * eigenvectors[:, index]
        for endpoint in (centroid + axis, centroid - axis):
            assert np.linalg.norm(points - endpoint, axis=1).min() < 1e-12


def test_the_surface_is_closed_around_the_azimuth(script):
    """The full turn is sampled, so the first and last rings coincide."""
    x, y, z = script.ellipsoid_surface(np.zeros(3), np.diag([2.0, 3.0, 1.0]),
                                       n_sd=1.0, resolution=24)
    for mesh in (x, y, z):
        np.testing.assert_allclose(mesh[0], mesh[-1], rtol=0.0, atol=1e-12)


def test_the_surface_is_deterministic(script):
    centroid = np.array([0.3, -1.1, 2.0])
    covariance = np.array([[2.0, 0.6, -0.1],
                           [0.6, 1.4, 0.2],
                           [-0.1, 0.2, 0.9]])
    first = script.ellipsoid_surface(centroid, covariance, n_sd=1.5,
                                     resolution=24)
    second = script.ellipsoid_surface(centroid, covariance, n_sd=1.5,
                                      resolution=24)
    for mesh, again in zip(first, second):
        np.testing.assert_array_equal(mesh, again)


def test_a_negligible_negative_eigenvalue_is_clipped(script):
    """Rounding error on a rank-deficient covariance must still draw."""
    covariance = np.diag([4.0, 1.0, -1e-14])
    x, y, z = script.ellipsoid_surface(np.zeros(3), covariance, n_sd=1.0,
                                       resolution=24)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()
    # the flattened axis has no extent, and the other two are untouched
    np.testing.assert_allclose([x.max(), y.max(), z.max()], [2.0, 1.0, 0.0],
                               atol=1e-9)


def test_a_covariance_that_is_not_positive_semidefinite_is_refused(script):
    """Too large to be rounding error, so it is a defect rather than noise."""
    with pytest.raises(ValidationError, match="positive semi-definite"):
        script.ellipsoid_surface(np.zeros(3), np.diag([4.0, 1.0, -0.5]),
                                 n_sd=1.0, resolution=24)


def test_an_asymmetric_covariance_is_refused(script):
    """``eigh`` reads one triangle, so silence here would be a wrong answer."""
    asymmetric = np.array([[2.0, 0.9, 0.0],
                           [0.1, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
    with pytest.raises(ValidationError, match="not symmetric"):
        script.ellipsoid_surface(np.zeros(3), asymmetric, n_sd=1.0,
                                 resolution=24)


def test_a_resolution_too_small_for_the_required_angles_is_refused(script):
    with pytest.raises(ValidationError, match="required"):
        script.ellipsoid_surface(np.zeros(3), np.eye(3), n_sd=1.0,
                                 resolution=script.MIN_ELLIPSOID_RESOLUTION - 1)


def test_the_plane_satisfies_its_own_equation(script):
    coef = np.array([[0.4, -1.3, 0.9]])
    intercept = np.array([0.25])
    limits = [(-2.0, 2.0)] * 3
    x, y, z = script.decision_plane(coef, intercept, limits, resolution=7)
    residual = 0.4 * x - 1.3 * y + 0.9 * z + 0.25
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_a_plane_with_no_orientation_is_not_drawn(script):
    assert script.decision_plane(np.zeros((1, 3)), np.array([0.5]),
                                 [(-1.0, 1.0)] * 3) is None


def test_the_plane_solves_for_the_axis_it_can(script):
    """A weight of zero on one axis must not divide by that zero."""
    x, y, z = script.decision_plane(np.array([[0.0, 0.0, 2.0]]),
                                    np.array([-1.0]), [(-1.0, 1.0)] * 3,
                                    resolution=5)
    np.testing.assert_allclose(z, 0.5, atol=1e-12)


# ------------------------------------------------------------ 5. determinism
def test_two_fits_of_the_same_rows_agree_exactly(script, problem, split):
    X, y, _ = problem
    first = script.fit_projection(X[split.train_index], y[split.train_index], 3)
    second = script.fit_projection(X[split.train_index], y[split.train_index], 3)

    np.testing.assert_array_equal(first.train_scores, second.train_scores)
    np.testing.assert_array_equal(first.classifier.coef_,
                                  second.classifier.coef_)
    np.testing.assert_array_equal(
        first.transform(X[split.held_out_index]),
        second.transform(X[split.held_out_index]))


def test_the_geometry_is_deterministic(script, problem, split):
    X, y, _ = problem
    geometries = []
    for _ in range(2):
        projection = script.fit_projection(X[split.train_index],
                                           y[split.train_index], 3)
        geometries.append(script.class_geometry(projection.train_scores_3d,
                                                y[split.train_index]))
    for label in ("B", "E"):
        np.testing.assert_array_equal(geometries[0][label]["centroid"],
                                      geometries[1][label]["centroid"])
        np.testing.assert_array_equal(geometries[0][label]["covariance"],
                                      geometries[1][label]["covariance"])


def test_selecting_the_same_split_twice_gives_the_same_rows(script,
                                                            assignments,
                                                            problem):
    _, _, row_ids = problem
    first = script.select_split(
        script.read_frozen_assignments(assignments, row_ids), 0, 0)
    second = script.select_split(
        script.read_frozen_assignments(assignments, row_ids), 0, 0)
    assert first.held_out_row_ids == second.held_out_row_ids
    assert first.train_row_ids == second.train_row_ids


# --------------------------------------------------------------- 6. filenames
def test_the_default_settings_produce_the_declared_filenames(script):
    names = script.output_names(45, 0, 0)
    assert names["png"] == "heldout_pls3d_state45_repeat0_fold0.png"
    assert names["svg"] == "heldout_pls3d_state45_repeat0_fold0.svg"
    assert names["metadata"] == (
        "heldout_pls3d_state45_repeat0_fold0_metadata.json")


def test_the_filenames_follow_the_settings(script):
    names = script.output_names(12, 3, 4)
    assert names["png"] == "heldout_pls3d_state12_repeat3_fold4.png"
    assert names["svg"] == "heldout_pls3d_state12_repeat3_fold4.svg"
    assert names["metadata"] == (
        "heldout_pls3d_state12_repeat3_fold4_metadata.json")


def test_different_settings_never_share_a_filename(script):
    seen = {script.output_names(s, r, f)["png"]
            for s in (0, 45, 48) for r in (0, 1) for f in (0, 4)}
    assert len(seen) == 3 * 2 * 2


# --------------------------------------------------------------- 7. metadata
@pytest.fixture
def metadata(script, fitted, split, tmp_path, assignments):
    projection, X, y = fitted
    names = script.output_names(45, 0, 0)
    return script.build_metadata(
        state=45, repeat=0, fold=0, split=split, projection=projection,
        y_train=y[split.train_index], y_held_out=y[split.held_out_index],
        n_sd=1.5,
        input_hashes={"dataset": {"sha256": "a" * 64},
                      "extraction_manifest": {"sha256": "b" * 64},
                      "activation": {"sha256": "c" * 64},
                      "cv_assignments": {"sha256": "d" * 64}},
        split_record={"path": str(assignments), "sha256": "d" * 64,
                      "reused_not_regenerated": True},
        names=names)


def test_the_metadata_carries_every_required_field(metadata):
    for field in ("schema", "figure_title", "descriptive_only", "state",
                  "repeat", "held_out_fold", "counts", "row_ids",
                  "input_hashes", "split", "scaler", "pls", "classifier",
                  "outputs"):
        assert field in metadata, field


def test_the_metadata_marks_the_figure_descriptive_only(metadata):
    assert metadata["descriptive_only"] is True
    assert "descriptive" in metadata["descriptive_only_note"].lower()


def test_the_metadata_records_the_leakage_discipline(metadata):
    assert metadata["leakage_statement"] == (
        "Descriptive visualisation; preprocessing and projection fitted on "
        "training rows only")
    assert metadata["scaler"]["fitted_on"] == "training rows only"
    assert metadata["pls"]["fitted_on"] == "training rows only"
    assert metadata["classifier"]["fitted_on"] == "training rows only"


def test_the_metadata_never_calls_the_ellipsoids_a_confidence_interval(metadata):
    text = json.dumps(metadata).lower()
    assert "not confidence regions" in text
    for forbidden in ("confidence interval", "confidence band",
                      "credible interval"):
        assert forbidden not in text


def test_the_metadata_records_the_input_hashes(metadata):
    for key in ("dataset", "extraction_manifest", "activation",
                "cv_assignments"):
        assert len(metadata["input_hashes"][key]["sha256"]) == 64


def test_the_metadata_counts_both_splits_and_both_classes(metadata, split):
    counts = metadata["counts"]
    assert counts["train"] == split.train_index.size
    assert counts["held_out"] == split.held_out_index.size
    assert counts["train_E"] + counts["train_B"] == counts["train"]
    assert counts["held_out_E"] + counts["held_out_B"] == counts["held_out"]
    assert counts["held_out_E"] > 0 and counts["held_out_B"] > 0


def test_the_metadata_lists_disjoint_row_ids(metadata):
    train = set(metadata["row_ids"]["train"])
    held_out = set(metadata["row_ids"]["held_out"])
    assert not train & held_out
    assert metadata["row_id_overlap"] == []


def test_the_metadata_records_the_estimator_settings(metadata):
    assert metadata["pls"]["class"] == (
        "sklearn.cross_decomposition.PLSRegression")
    assert metadata["pls"]["n_components"] == 3
    assert metadata["pls"]["scale"] is False
    assert metadata["scaler"]["class"] == "sklearn.preprocessing.StandardScaler"
    assert metadata["classifier"]["class"] == (
        "sklearn.linear_model.LogisticRegression")
    assert len(metadata["classifier"]["coef"]) == 3


def test_the_metadata_says_the_split_was_reused(metadata):
    assert metadata["split"]["reused_not_regenerated"] is True


def test_the_metadata_names_the_files_it_belongs_to(metadata):
    assert metadata["outputs"] == {
        "png": "heldout_pls3d_state45_repeat0_fold0.png",
        "svg": "heldout_pls3d_state45_repeat0_fold0.svg",
        "metadata": "heldout_pls3d_state45_repeat0_fold0_metadata.json"}


def test_the_metadata_is_json_serialisable(metadata, tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == 45


# ---------------------------------------------------------------- 8. plotting
def test_the_figure_renders_headless_and_writes_both_formats(script, fitted,
                                                             split, tmp_path):
    pytest.importorskip("matplotlib")
    projection, X, y = fitted
    scores = projection.transform(X[split.held_out_index])
    geometry = script.class_geometry(projection.train_scores_3d,
                                     y[split.train_index])
    names = script.output_names(45, 0, 0)

    written = script.render_figure(
        tmp_path / "figures", names, scores, y[split.held_out_index],
        geometry, projection, state=45, repeat=0, fold=0)

    assert [p.name for p in written] == [names["png"], names["svg"]]
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_rendering_uses_a_non_interactive_backend(script, fitted, split,
                                                  tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    projection, X, y = fitted
    scores = projection.transform(X[split.held_out_index])
    geometry = script.class_geometry(projection.train_scores_3d,
                                     y[split.train_index])

    script.render_figure(tmp_path / "figures", script.output_names(45, 0, 0),
                         scores, y[split.held_out_index], geometry, projection,
                         state=45, repeat=0, fold=0)

    assert matplotlib.get_backend().lower() == "agg"


def test_rendering_twice_overwrites_rather_than_accumulates(script, fitted,
                                                            split, tmp_path):
    pytest.importorskip("matplotlib")
    projection, X, y = fitted
    scores = projection.transform(X[split.held_out_index])
    geometry = script.class_geometry(projection.train_scores_3d,
                                     y[split.train_index])
    names = script.output_names(45, 0, 0)
    figures = tmp_path / "figures"

    for _ in range(2):
        script.render_figure(figures, names, scores, y[split.held_out_index],
                             geometry, projection, state=45, repeat=0, fold=0)

    assert sorted(p.name for p in figures.iterdir()) == [names["png"],
                                                         names["svg"]]


# ------------------------------------------------- 9. the declared wording
def test_the_figure_title_is_the_declared_one(script):
    assert script.FIGURE_TITLE == (
        "Held-out 3D residual landscape of over-refusal")


def test_the_leakage_statement_is_the_declared_one(script):
    assert script.LEAKAGE_STATEMENT == (
        "Descriptive visualisation; preprocessing and projection fitted on "
        "training rows only")


def test_the_ellipsoids_are_never_called_a_confidence_interval(script):
    """The wording is a constant so that it cannot drift figure by figure."""
    caption = script.ELLIPSOID_CAPTION.format(n_sd=1.5).lower()
    assert "covariance" in caption
    assert "not confidence regions" in caption
    for forbidden in ("confidence interval", "confidence region of",
                      "credible interval", "% confidence"):
        assert forbidden not in caption


def test_the_axis_labels_name_the_three_components(script):
    assert script.AXIS_LABELS == ("PLS component 1", "PLS component 2",
                                  "PLS component 3")
    assert script.PLOT_DIMENSIONS == len(script.AXIS_LABELS)


def test_e_and_b_get_different_markers(script):
    assert script.MARKERS["E"] != script.MARKERS["B"]
    assert script.CENTROID_MARKER not in script.MARKERS.values()


def test_importing_the_module_selects_no_backend(script):
    """Only rendering may touch matplotlib, so the import stays cheap."""
    assert not hasattr(script, "matplotlib")
    assert not hasattr(script, "plt")