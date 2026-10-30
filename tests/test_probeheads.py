"""The two probe heads, on synthetic inputs only.

Nothing here touches the corpus, the activation array, the direction bank on
disk, a model or a GPU. Every frame, fold division and bank in this file is
built in ``tmp_path`` from a seeded generator, so the properties under test are
properties of the code rather than of one dataset.

What is under test is the part of this stage that is easy to get silently wrong:
which rows each head is fitted on, which rows it scores, which fold scored a
given row, whether the scaler saw anything it should not have, and whether the
frozen fold division is bound to the analysed rows by position rather than by
set membership. The layer-to-state mapping is checked because it is off by one
by construction, and refusing material outside the development corpus is checked
because a guard that is never exercised is a guard nobody knows is wired up.
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from routegate.data import NEGATIVE_LABEL, POSITIVE_LABEL, WITHHELD_LABEL
from routegate.ebh import (
    EBH_LABELS,
    EBHFoldAssignment,
    EBHFrame,
    assign_withheld_folds,
)
from routegate.probeheads import (
    CONTROLLER_READ_STATES,
    HEAD_ORDER,
    HEAD_SPECIFICATIONS,
    HISTORICAL_LAYER_TO_STATE,
    LABEL_UNDEFINED,
    PREDICTION_COLUMNS,
    PRIMARY_READ_LAYER,
    PRIMARY_READ_STATE,
    REPAIR_HEAD,
    REPAIR_SPECIFICATION,
    SAFETY_HEAD,
    SAFETY_SPECIFICATION,
    SENSITIVITY_STATES,
    ProbeHeadError,
    bind_fold_assignment,
    cross_check_inherited_folds,
    evaluate_head,
    evaluate_state,
    fit_head,
    fit_probe_heads,
    head_targets,
    layer_of_state,
    load_bank_fold_assignment,
    probe_head_specification,
    repair_targets,
    safety_targets,
    state_matrix,
    state_of_layer,
)
from routegate.probes import make_probe, probe_specification
from routegate.steering import (
    BANK_SCHEMA,
    FOLD_ASSIGNMENT_FILENAME,
    INPUT_HASHES_FILENAME,
    MANIFEST_FILENAME,
    manifest_sha256,
)
from routegate.validation import (
    SealedSetError,
    ValidationError,
    assert_config_is_sealed_free,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "dual_direction_controller.json"
EXPERIMENT = REPO_ROOT / "experiments" / "dual_direction_controller"

N_STATES = 49
HIDDEN = 8
N_SPLITS = 5


# --------------------------------------------------------------- synthetic frame
def _frame(n_e: int = 20, n_b: int = 30, n_h: int = 25,
           n_states: int = N_STATES, hidden: int = HIDDEN,
           seed: int = 20260821) -> EBHFrame:
    """A three-class frame with a planted E/B direction and a planted H direction.

    The two plants are strong and near-orthogonal so that both heads are
    unambiguously above chance out of fold; a weak plant would be shrunk by the
    frozen C = 0.1 and the leakage tests would then be measuring the
    regularisation rather than the thing they are about.
    """
    rng = np.random.default_rng(seed)
    labels = ([POSITIVE_LABEL] * n_e + [NEGATIVE_LABEL] * n_b
              + [WITHHELD_LABEL] * n_h)
    n = len(labels)
    row_ids = [f"{label.lower()}:{i:03d}" for i, label in enumerate(labels)]

    activations = rng.normal(size=(n, n_states, hidden))
    eb_direction = np.zeros(hidden)
    eb_direction[0] = 1.0
    h_direction = np.zeros(hidden)
    h_direction[1] = 1.0
    for i, label in enumerate(labels):
        if label == POSITIVE_LABEL:
            activations[i] += 3.0 * eb_direction
        elif label == WITHHELD_LABEL:
            activations[i] += 3.0 * h_direction

    return EBHFrame(
        row_ids=row_ids,
        labels=labels,
        source_family=[f"family_{i % 3}" for i in range(n)],
        topic=[f"topic_{i % 4}" for i in range(n)],
        task_frame=[f"frame_{i % 2}" for i in range(n)],
        prompt_text=[""] * n,
        original_response=[""] * n,
        activations=activations,
        n_states=n_states,
        hidden_size=hidden,
        dataset_path=Path("synthetic/dataset.csv"),
        activation_path=Path("synthetic/activations.npy"),
        extraction_manifest_path=Path("synthetic/manifest.csv"),
        dataset_sha256="0" * 64,
        activation_sha256="1" * 64,
        extraction_manifest_sha256="2" * 64,
    )


def _assignment(frame: EBHFrame, n_splits: int = N_SPLITS) -> EBHFoldAssignment:
    """E and B dealt round-robin within their class; H by the deterministic rule."""
    fold_of_row = {}
    for label in (POSITIVE_LABEL, NEGATIVE_LABEL):
        for k, position in enumerate(frame.positions_of(label)):
            fold_of_row[frame.row_ids[position]] = k % n_splits
    withheld = [(frame.row_ids[i], frame.source_family[i])
                for i in frame.positions_of(WITHHELD_LABEL)]
    fold_of_row.update(assign_withheld_folds(withheld, n_splits))
    return EBHFoldAssignment(
        fold_of_row=fold_of_row,
        label_of_row=dict(zip(frame.row_ids, frame.labels)),
        n_splits=n_splits,
        eb_source="synthetic",
        eb_repeat=0)


@pytest.fixture
def frame():
    return _frame()


@pytest.fixture
def assignment(frame):
    return _assignment(frame)


# --------------------------------------------------------------- synthetic bank
def _write_bank(directory: Path, frame: EBHFrame, assignment: EBHFoldAssignment,
                inherited: dict | None = None,
                order: list[int] | None = None,
                array_row_index: list[int] | None = None) -> Path:
    """A bank directory carrying only what this stage reads back from one."""
    directory.mkdir(parents=True, exist_ok=True)
    order = list(range(frame.n)) if order is None else order
    index = list(range(frame.n)) if array_row_index is None else array_row_index
    rows = [{"array_row_index": index[place],
             "row_id": frame.row_ids[position],
             "final_ebh": frame.labels[position],
             "fold": assignment.fold_of(frame.row_ids[position]),
             "fold_source": "synthetic"}
            for place, position in enumerate(order)]
    with open(directory / FOLD_ASSIGNMENT_FILENAME, "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (directory / INPUT_HASHES_FILENAME).write_text(json.dumps({
        "inherited_fold_assignment": inherited or {
            "path": str(directory / "cv_assignments.csv"),
            "sha256": "3" * 64,
            "repeat": 0},
    }, indent=2), encoding="utf-8")

    manifest = {
        "schema": BANK_SCHEMA,
        "folds": {"n_splits": assignment.n_splits},
        "artifacts": {name: sha256_file(directory / name)
                      for name in (FOLD_ASSIGNMENT_FILENAME,
                                   INPUT_HASHES_FILENAME)},
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def _write_cv_assignments(path: Path, frame: EBHFrame,
                          assignment: EBHFoldAssignment, repeat: int = 0,
                          repeats: int = 3) -> Path:
    """The inherited E/B file, in the shape ``ebh.load_frozen_eb_folds`` expects."""
    rows = []
    for r in range(repeats):
        for i, rid in enumerate(frame.row_ids):
            label = frame.labels[i]
            if label == WITHHELD_LABEL:
                continue
            fold = (assignment.fold_of(rid) if r == repeat
                    else (assignment.fold_of(rid) + 1) % assignment.n_splits)
            rows.append({"repeat": r, "row_id": rid, "fold": fold,
                         "final_ebh": label, "split": "held_out"})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ==================================================== layer / state convention
def test_the_historical_layer_to_state_mapping_is_off_by_one_on_purpose():
    assert state_of_layer(31) == 32
    assert state_of_layer(32) == 33
    assert state_of_layer(33) == 34
    assert HISTORICAL_LAYER_TO_STATE == {31: 32, 32: 33, 33: 34}
    for layer, state in HISTORICAL_LAYER_TO_STATE.items():
        assert layer_of_state(state) == layer
        assert state_of_layer(layer) == state


def test_state_zero_has_no_historical_layer():
    """State 0 is the embedding output; calling it layer -1 would be a fiction."""
    with pytest.raises(ProbeHeadError, match="embedding output"):
        layer_of_state(0)


def test_state_32_is_the_predeclared_primary_and_33_34_are_sensitivity_only():
    assert PRIMARY_READ_LAYER == 31
    assert PRIMARY_READ_STATE == 32
    assert SENSITIVITY_STATES == (33, 34)
    assert CONTROLLER_READ_STATES == (32, 33, 34)
    assert PRIMARY_READ_STATE not in SENSITIVITY_STATES


def test_the_module_offers_no_way_to_rank_the_read_states():
    """No 'best state' is selectable here; the primary state is predeclared."""
    import routegate.probeheads as probeheads

    exported = dir(probeheads)
    for forbidden in ("best_state", "select_state", "rank_states"):
        assert forbidden not in exported


# ============================================================ target orientation
def test_the_repair_head_is_E_positive_B_negative_and_H_untargeted():
    targets = repair_targets([POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL,
                              POSITIVE_LABEL])
    np.testing.assert_array_equal(targets, [1, 0, LABEL_UNDEFINED, 1])
    assert REPAIR_SPECIFICATION.positive_labels == (POSITIVE_LABEL,)
    assert REPAIR_SPECIFICATION.negative_labels == (NEGATIVE_LABEL,)
    assert REPAIR_SPECIFICATION.excluded_from_training == (WITHHELD_LABEL,)
    assert REPAIR_SPECIFICATION.training_labels == (POSITIVE_LABEL, NEGATIVE_LABEL)
    assert REPAIR_SPECIFICATION.score_column == "p_E"


def test_the_safety_head_puts_E_and_B_together_as_one_negative_class():
    targets = safety_targets([POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL])
    np.testing.assert_array_equal(targets, [0, 0, 1])
    assert SAFETY_SPECIFICATION.positive_labels == (WITHHELD_LABEL,)
    assert set(SAFETY_SPECIFICATION.negative_labels) == {POSITIVE_LABEL,
                                                         NEGATIVE_LABEL}
    assert SAFETY_SPECIFICATION.excluded_from_training == ()
    assert SAFETY_SPECIFICATION.score_column == "p_H"
    assert LABEL_UNDEFINED not in set(targets.tolist())


def test_both_heads_score_every_class_even_where_they_have_no_target():
    for spec in HEAD_SPECIFICATIONS.values():
        assert spec.scored_labels == EBH_LABELS


def test_a_label_outside_the_three_class_frame_is_refused():
    with pytest.raises(ProbeHeadError, match="not one of"):
        repair_targets([POSITIVE_LABEL, "X"])


def test_a_head_with_only_one_class_present_is_refused():
    """A missing class is a stop, not a degenerate fit."""
    with pytest.raises(ProbeHeadError, match="needs both classes"):
        repair_targets([POSITIVE_LABEL, POSITIVE_LABEL, WITHHELD_LABEL])
    with pytest.raises(ProbeHeadError, match="needs both classes"):
        safety_targets([POSITIVE_LABEL, NEGATIVE_LABEL])


# ============================================ what each head is actually fitted on
def test_no_H_row_reaches_the_repair_heads_training_set_in_any_fold(frame,
                                                                    assignment):
    X = state_matrix(frame, PRIMARY_READ_STATE)
    result = fit_head(REPAIR_SPECIFICATION, X, frame, assignment,
                      PRIMARY_READ_STATE)
    for fold, counts in enumerate(result.fold_counts):
        train = assignment.training_positions(frame, fold)
        assert counts["n_train"] == (train[POSITIVE_LABEL].size
                                     + train[NEGATIVE_LABEL].size)
        assert counts["n_train_positive"] == train[POSITIVE_LABEL].size
        assert counts["n_train_negative"] == train[NEGATIVE_LABEL].size
        # the H rows exist in the fold's training half and are not used
        assert train[WITHHELD_LABEL].size > 0
        assert counts["n_train"] < sum(train[label].size for label in EBH_LABELS)


def test_the_safety_head_trains_on_all_three_classes_with_E_and_B_negative(
        frame, assignment):
    X = state_matrix(frame, PRIMARY_READ_STATE)
    result = fit_head(SAFETY_SPECIFICATION, X, frame, assignment,
                      PRIMARY_READ_STATE)
    for fold, counts in enumerate(result.fold_counts):
        train = assignment.training_positions(frame, fold)
        assert counts["n_train"] == sum(train[label].size for label in EBH_LABELS)
        assert counts["n_train_positive"] == train[WITHHELD_LABEL].size
        assert counts["n_train_negative"] == (train[POSITIVE_LABEL].size
                                              + train[NEGATIVE_LABEL].size)
        assert counts["n_held_out_scored_without_a_target"] == 0


def test_the_repair_head_scores_the_H_rows_it_never_trained_on(frame, assignment):
    heads = fit_probe_heads(frame, assignment, PRIMARY_READ_STATE)
    repair = heads.heads[REPAIR_HEAD]
    withheld = frame.positions_of(WITHHELD_LABEL)
    assert withheld.size > 0
    assert np.all(repair.targets[withheld] == LABEL_UNDEFINED)
    assert np.isfinite(repair.scores[withheld]).all()
    assert ((repair.scores[withheld] >= 0.0)
            & (repair.scores[withheld] <= 1.0)).all()
    # and those rows carry no target, so they are outside its metrics
    assert evaluate_head(repair)["n_scored_without_a_target"] == withheld.size
    assert evaluate_head(repair)["n_scored"] == frame.n - withheld.size


# ================================================== the scaler is fold-specific
def test_each_folds_scaler_is_fitted_on_that_folds_training_rows_only(frame,
                                                                      assignment):
    """The frozen StandardScaler lives inside the pipeline; this proves the scope."""
    X = state_matrix(frame, PRIMARY_READ_STATE)
    result = fit_head(REPAIR_SPECIFICATION, X, frame, assignment,
                      PRIMARY_READ_STATE)
    for fold, outcome in enumerate(result.folds):
        train = assignment.training_positions(frame, fold)
        eb_rows = np.sort(np.concatenate([train[POSITIVE_LABEL],
                                          train[NEGATIVE_LABEL]]))
        np.testing.assert_allclose(outcome.scaler_mean, X[eb_rows].mean(axis=0),
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(outcome.scaler_scale, X[eb_rows].std(axis=0),
                                   rtol=1e-9, atol=1e-9)


def test_the_repair_heads_scaler_is_not_the_scaler_of_all_the_rows(frame,
                                                                   assignment):
    """If H had leaked into the scaler, these statistics would coincide."""
    X = state_matrix(frame, PRIMARY_READ_STATE)
    result = fit_head(REPAIR_SPECIFICATION, X, frame, assignment,
                      PRIMARY_READ_STATE)
    everything = X.mean(axis=0)
    for outcome in result.folds:
        assert not np.allclose(outcome.scaler_mean, everything, rtol=1e-6,
                               atol=1e-6)


def test_the_folds_do_not_share_one_scaler(frame, assignment):
    X = state_matrix(frame, PRIMARY_READ_STATE)
    result = fit_head(SAFETY_SPECIFICATION, X, frame, assignment,
                      PRIMARY_READ_STATE)
    means = [outcome.scaler_mean for outcome in result.folds]
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            assert not np.array_equal(means[i], means[j])


# ================================================= every row scored exactly once
def test_every_row_receives_one_cross_fitted_p_E_and_one_p_H(frame, assignment):
    heads = fit_probe_heads(frame, assignment, PRIMARY_READ_STATE)
    for scores in (heads.p_E, heads.p_H):
        assert scores.shape == (frame.n,)
        assert np.isfinite(scores).all()
        assert ((scores >= 0.0) & (scores <= 1.0)).all()
    rows = heads.prediction_rows()
    assert len(rows) == frame.n
    assert [tuple(row) for row in rows][0] == PREDICTION_COLUMNS
    assert len({row["row_id"] for row in rows}) == frame.n


def test_a_rows_score_comes_from_the_fold_that_holds_it_out(frame, assignment):
    heads = fit_probe_heads(frame, assignment, PRIMARY_READ_STATE)
    for i, rid in enumerate(frame.row_ids):
        assert int(heads.fold_of_row[i]) == assignment.fold_of(rid)
    for row in heads.prediction_rows():
        assert row["fold"] == assignment.fold_of(row["row_id"])
        assert row["state"] == PRIMARY_READ_STATE


def test_the_repair_score_is_the_one_a_hand_fitted_fold_probe_produces(
        frame, assignment):
    """Cross-fitting, checked against an independent refit of one fold."""
    state = PRIMARY_READ_STATE
    X = state_matrix(frame, state)
    heads = fit_probe_heads(frame, assignment, state)

    fold = 0
    train = assignment.training_positions(frame, fold)
    held_out = assignment.held_out_positions(frame, fold)
    train_rows = np.sort(np.concatenate([train[POSITIVE_LABEL],
                                         train[NEGATIVE_LABEL]]))
    test_rows = np.sort(np.concatenate([held_out[label] for label in EBH_LABELS]))

    pipeline = make_probe()
    y = repair_targets(frame.labels)
    pipeline.fit(X[train_rows], y[train_rows])
    column = int(np.flatnonzero(pipeline.named_steps["classifier"].classes_ == 1)[0])
    expected = pipeline.predict_proba(X[test_rows])[:, column]
    np.testing.assert_allclose(heads.p_E[test_rows], expected,
                               rtol=1e-12, atol=1e-12)


def test_both_heads_are_above_chance_on_the_planted_signal(frame, assignment):
    """Not a result -- a guard that the fixtures are not scoring noise."""
    record = evaluate_state(fit_probe_heads(frame, assignment,
                                            PRIMARY_READ_STATE))
    for name in HEAD_ORDER:
        head = record["heads"][name]
        assert head["roc_auc"] > 0.9
        assert head["pr_auc"] > head["chance"]["pr_auc"]
        assert 0.0 <= head["brier"] <= 1.0
        assert head["calibration_model_fitted"] is False


# ============================================================ metrics and record
def test_the_metrics_record_names_the_classes_it_actually_scored(frame,
                                                                 assignment):
    record = evaluate_state(fit_probe_heads(frame, assignment,
                                            PRIMARY_READ_STATE))
    repair = record["heads"][REPAIR_HEAD]
    safety = record["heads"][SAFETY_HEAD]

    assert repair["positive_class"] == [POSITIVE_LABEL]
    assert repair["negative_class"] == [NEGATIVE_LABEL]
    assert repair["excluded_from_training"] == [WITHHELD_LABEL]
    assert repair["n_positive"] == frame.positions_of(POSITIVE_LABEL).size
    assert repair["n_negative"] == frame.positions_of(NEGATIVE_LABEL).size

    assert safety["positive_class"] == [WITHHELD_LABEL]
    assert set(safety["negative_class"]) == {POSITIVE_LABEL, NEGATIVE_LABEL}
    assert safety["n_positive"] == frame.positions_of(WITHHELD_LABEL).size
    assert safety["n_negative"] == (frame.positions_of(POSITIVE_LABEL).size
                                    + frame.positions_of(NEGATIVE_LABEL).size)
    assert safety["n_scored"] == frame.n

    assert record["role"] == "primary"
    assert record["historical_layer"] == PRIMARY_READ_LAYER


def test_a_sensitivity_state_is_recorded_as_one(frame, assignment):
    record = evaluate_state(fit_probe_heads(frame, assignment,
                                            SENSITIVITY_STATES[0]))
    assert record["role"] == "sensitivity_check"
    assert record["historical_layer"] == layer_of_state(SENSITIVITY_STATES[0])
    for name in HEAD_ORDER:
        assert record["heads"][name]["is_primary_read_state"] is False


def test_the_declaration_reuses_the_frozen_probe_and_adds_no_hyperparameter():
    spec = probe_head_specification()
    assert spec["probe"] == probe_specification()
    assert spec["calibration"]["model_fitted"] is False
    assert spec["layer_state_convention"]["primary_read_state"] == PRIMARY_READ_STATE
    assert spec["layer_state_convention"]["sensitivity_states"] == list(
        SENSITIVITY_STATES)


# ========================================================= missing class / fold
def test_a_fold_whose_training_half_has_no_negative_row_is_refused():
    frame = _frame(n_e=8, n_b=6, n_h=8)
    fold_of_row = {}
    for i, rid in enumerate(frame.row_ids):
        label = frame.labels[i]
        # every B row into fold 0, so fold 0 trains on no B at all
        fold_of_row[rid] = 0 if label == NEGATIVE_LABEL else i % 2
    assignment = EBHFoldAssignment(
        fold_of_row=fold_of_row,
        label_of_row=dict(zip(frame.row_ids, frame.labels)),
        n_splits=2, eb_source="synthetic", eb_repeat=0)
    X = state_matrix(frame, PRIMARY_READ_STATE)
    with pytest.raises(ValidationError,
                       match="training_set_has_both_classes"):
        fit_head(REPAIR_SPECIFICATION, X, frame, assignment, PRIMARY_READ_STATE)


def test_an_assignment_that_does_not_cover_every_row_is_refused(frame,
                                                                assignment):
    partial = EBHFoldAssignment(
        fold_of_row={rid: fold for rid, fold
                     in list(assignment.fold_of_row.items())[:-1]},
        label_of_row=assignment.label_of_row,
        n_splits=assignment.n_splits, eb_source="synthetic", eb_repeat=0)
    X = state_matrix(frame, PRIMARY_READ_STATE)
    with pytest.raises(ProbeHeadError, match="fold assignment covers"):
        fit_head(REPAIR_SPECIFICATION, X, frame, partial, PRIMARY_READ_STATE)


def test_a_design_with_the_wrong_number_of_rows_is_refused(frame, assignment):
    X = state_matrix(frame, PRIMARY_READ_STATE)[:-1]
    with pytest.raises(ProbeHeadError, match="rows and the frame has"):
        fit_head(REPAIR_SPECIFICATION, X, frame, assignment, PRIMARY_READ_STATE)


def test_a_state_outside_the_stored_trajectory_is_refused(frame):
    with pytest.raises(ProbeHeadError, match="outside"):
        state_matrix(frame, frame.n_states)
    with pytest.raises(ProbeHeadError, match="outside"):
        state_matrix(frame, -1)


# ================================================ row order and hash mismatches
def test_the_frozen_division_binds_to_the_frame_by_position(tmp_path, frame,
                                                            assignment):
    bank = _write_bank(tmp_path / "bank", frame, assignment)
    loaded, source = load_bank_fold_assignment(bank)
    bind_fold_assignment(frame, loaded, source)
    assert loaded.fold_of_row == assignment.fold_of_row
    assert source.n_splits == assignment.n_splits
    assert source.fold_assignment_sha256 == sha256_file(
        bank / FOLD_ASSIGNMENT_FILENAME)


def test_a_fold_file_in_a_different_row_order_is_refused(tmp_path, frame,
                                                         assignment):
    """The same 480 row_ids in another order would attach every fold to the
    wrong activation, and a set comparison would not notice."""
    reversed_order = list(range(frame.n))[::-1]
    bank = _write_bank(tmp_path / "bank", frame, assignment,
                       order=reversed_order)
    loaded, source = load_bank_fold_assignment(bank)
    with pytest.raises(ValidationError,
                       match="names_the_analysed_rows_in_the_analysed_order"):
        bind_fold_assignment(frame, loaded, source)


def test_a_fold_file_whose_array_row_index_is_not_0_to_n_minus_1_is_refused(
        tmp_path, frame, assignment):
    shuffled = list(range(1, frame.n)) + [0]
    bank = _write_bank(tmp_path / "bank", frame, assignment,
                       array_row_index=shuffled)
    with pytest.raises(ProbeHeadError, match="does not run 0"):
        load_bank_fold_assignment(bank)


def test_a_fold_file_edited_after_the_bank_was_frozen_is_refused(tmp_path,
                                                                 frame,
                                                                 assignment):
    bank = _write_bank(tmp_path / "bank", frame, assignment)
    path = bank / FOLD_ASSIGNMENT_FILENAME
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ProbeHeadError, match="hashes to"):
        load_bank_fold_assignment(bank)


def test_a_manifest_edited_after_the_bank_was_frozen_is_refused(tmp_path, frame,
                                                                assignment):
    bank = _write_bank(tmp_path / "bank", frame, assignment)
    path = bank / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["folds"]["n_splits"] = 4
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ProbeHeadError, match="modified since it was frozen"):
        load_bank_fold_assignment(bank)


def test_a_bank_that_is_not_the_pinned_one_is_refused(tmp_path, frame,
                                                      assignment):
    bank = _write_bank(tmp_path / "bank", frame, assignment)
    with pytest.raises(ProbeHeadError, match="not the bank this run was told"):
        load_bank_fold_assignment(bank, expected_manifest_sha256="f" * 64)


def test_a_bank_with_no_manifest_is_refused(tmp_path):
    (tmp_path / "empty_bank").mkdir()
    with pytest.raises(ProbeHeadError, match="does not exist"):
        load_bank_fold_assignment(tmp_path / "empty_bank")


# ======================================= the division must reproduce from its rule
def test_the_division_is_cross_checked_against_the_file_it_came_from(tmp_path,
                                                                     frame,
                                                                     assignment):
    cv = _write_cv_assignments(tmp_path / "cv_assignments.csv", frame, assignment)
    bank = _write_bank(tmp_path / "bank", frame, assignment,
                       inherited={"path": str(cv), "sha256": sha256_file(cv),
                                  "repeat": 0})
    loaded, source = load_bank_fold_assignment(bank)
    outcome = cross_check_inherited_folds(frame, loaded, source)
    assert outcome.startswith("agrees")
    assert source.cross_check.startswith("agrees")


def test_a_division_its_stated_rule_does_not_reproduce_is_refused(tmp_path,
                                                                  frame,
                                                                  assignment):
    """Two conflicting historical assignments are never silently reconciled."""
    cv = _write_cv_assignments(tmp_path / "cv_assignments.csv", frame, assignment,
                               repeat=1)
    bank = _write_bank(tmp_path / "bank", frame, assignment,
                       inherited={"path": str(cv), "sha256": sha256_file(cv),
                                  "repeat": 0})
    loaded, source = load_bank_fold_assignment(bank)
    with pytest.raises(ValidationError, match="reproduces_the_bank"):
        cross_check_inherited_folds(frame, loaded, source)


def test_an_inherited_file_whose_hash_moved_is_refused(tmp_path, frame,
                                                       assignment):
    cv = _write_cv_assignments(tmp_path / "cv_assignments.csv", frame, assignment)
    bank = _write_bank(tmp_path / "bank", frame, assignment,
                       inherited={"path": str(cv), "sha256": "9" * 64,
                                  "repeat": 0})
    loaded, source = load_bank_fold_assignment(bank)
    with pytest.raises(ValidationError, match="is_the_one_the_bank_recorded"):
        cross_check_inherited_folds(frame, loaded, source)


def test_an_unreachable_inherited_file_is_reported_rather_than_assumed(tmp_path,
                                                                       frame,
                                                                       assignment):
    bank = _write_bank(tmp_path / "bank", frame, assignment,
                       inherited={"path": str(tmp_path / "absent.csv"),
                                  "sha256": "3" * 64, "repeat": 0})
    loaded, source = load_bank_fold_assignment(bank)
    outcome = cross_check_inherited_folds(frame, loaded, source)
    assert outcome.startswith("not_available")


# ================================================ nothing outside development data
def test_a_fold_source_that_reads_as_held_out_material_is_refused(tmp_path):
    """The guard is generic and matches on the name, before anything is opened."""
    with pytest.raises(SealedSetError):
        load_bank_fold_assignment(tmp_path / "sealed_direction_bank")


def test_the_committed_configuration_names_nothing_outside_development_data():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "dual direction controller config")


def test_the_runner_exposes_no_option_reaching_outside_development_data():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        module = importlib.reload(importlib.import_module("fit_probe_heads"))
    finally:
        sys.path.remove(str(EXPERIMENT))

    from routegate.validation import looks_sealed

    options = [option
               for action in module.build_parser()._actions
               for option in action.option_strings]
    assert options, "the runner must expose a command line"
    for option in options:
        assert looks_sealed(option) is None, option
    assert "--output-dir" in options

    required = [action.dest for action in module.build_parser()._actions
                if getattr(action, "required", False)]
    assert "output_dir" in required


def test_the_runner_refuses_an_output_directory_that_already_holds_something(
        tmp_path):
    sys.path.insert(0, str(EXPERIMENT))
    try:
        module = importlib.reload(importlib.import_module("fit_probe_heads"))
    finally:
        sys.path.remove(str(EXPERIMENT))

    fresh = module.claim_output_dir(tmp_path / "run")
    assert fresh.is_dir()
    module.claim_output_dir(tmp_path / "run")            # still empty, still fine

    (fresh / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError, match="already holds"):
        module.claim_output_dir(fresh)


# ============================================ the committed configuration itself
def test_the_committed_configuration_declares_the_frozen_probe():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = probe_specification()
    assert config["status"] == "DECLARED_NOT_RUN"
    assert set(config["probe"]) <= set(frozen)
    for field in ("penalty", "C", "class_weight", "max_iter", "solver",
                  "random_state", "tol", "decision_threshold", "pipeline"):
        assert config["probe"][field] == frozen[field], field


def test_the_committed_configuration_declares_the_two_heads_as_the_package_does():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in HEAD_ORDER:
        declared = config["targets"][f"{name}_head"]
        spec = HEAD_SPECIFICATIONS[name].as_dict()
        for field in ("positive_class", "negative_class",
                      "excluded_from_training", "trained_on", "scored_rows",
                      "label_encoding", "score_column"):
            assert declared[field] == spec[field], (name, field)


def test_the_committed_configuration_binds_the_read_states_and_no_calibration():
    """The parts of this file the probe heads themselves rest on: where they
    read, and that nothing recalibrates their probabilities afterwards.

    The direction pair is no longer in the open list: it was fixed on
    2026-08-22, before the bank was built. It moved OUT of ``unresolved``
    rather than being given an answer in place, so a reader looking for it
    finds where it went.
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["read_states"]["states"] == list(CONTROLLER_READ_STATES)
    assert config["read_states"]["primary_read_state"] == PRIMARY_READ_STATE
    assert config["read_states"]["primary_read_layer"] == PRIMARY_READ_LAYER
    assert config["calibration"]["model_fitted"] is False

    settled = "direction_normalisation_and_orthogonalisation"
    assert settled not in config["unresolved"]
    assert config["resolved"][settled]["recorded_in"] == "direction_construction"
    assert config["resolved"][settled]["ablation"] == (
        "predeclared_ablations.orthogonalised_safety")


def test_the_two_coefficient_proposal_is_preserved_as_phase_2_metadata():
    """It is history, and it is kept exactly as it was written.

    ``proposed_controller`` records what was proposed BEFORE phase 2 ran: one
    shared controller emitting a coefficient for each of two directions. Phase
    2 then returned NO_STRENGTH_QUALIFIES and retired d_S = H - E as an
    actuator, so this block is no longer the final design. It is not edited to
    match the outcome -- a proposal rewritten after its result is no longer a
    record of what was proposed -- and this test pins it in place, unchanged,
    as phase-2 metadata.

    What enforces the FINAL design is
    ``tests/test_single_direction_controller.py``; what records the decision
    that superseded this block is
    ``manifests/focused_direction_validation_v1_phase2_closure.json``.
    """
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    proposal = config["proposed_controller"]
    assert proposal["status"] == "DECLARED_NOT_IMPLEMENTED"
    assert proposal["outputs"]["parameterisation"] == (
        "two sigmoid values, each in [0, 1]")
    assert proposal["directions"]["repair_sign"] == "B - E"
    assert proposal["directions"]["safety_sign"] == "H - E"
    assert proposal["application"]["single_pass"] is True
    assert "not after inspecting generated-response outcomes" in (
        proposal["unresolved_before_training_rule"])

    unresolved = config["unresolved"]
    for key in ("controller_input", "controller_hidden_size",
                "controller_depth", "controller_activation_function",
                "controller_loss", "controller_regularisation",
                "controller_optimiser", "controller_training_schedule",
                "controller_training_rows",
                "sigmoid_to_maximum_strength_mapping",
                "successful_strength_targets_or_other_supervisory_signal",
                "controller_selection_criterion", "operating_point"):
        assert unresolved[key] == "UNRESOLVED", key
    assert "BEFORE the controller is trained" in unresolved["note"]


def test_this_configuration_is_not_a_final_controller_configuration():
    """Historical metadata is readable and is not loadable as the final design.

    The two-coefficient proposal, the safety coefficient and the H - E
    direction sign all survive in this file as the record of what was proposed.
    The final path refuses every one of them, so the block cannot reach a
    controller by being the configuration that happened to be lying around.
    """
    from routegate.singledirection import (
        SINGLE_DIRECTION_CONFIG,
        SingleDirectionError,
        load_final_controller_config,
    )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(SingleDirectionError):
        load_final_controller_config(CONFIG_PATH)
    assert config["proposed_controller"]["outputs"]["safety_coefficient"]

    final = load_final_controller_config(SINGLE_DIRECTION_CONFIG)
    assert final["controller"]["outputs"] == ["g_R"]
    assert final["supersedes"]["configuration"] == (
        "configs/dual_direction_controller.json")


def test_head_targets_refuses_a_class_that_is_both_positive_and_negative():
    from routegate.probeheads import HeadSpecification

    contradictory = HeadSpecification(
        name="contradictory", positive_labels=(POSITIVE_LABEL,),
        negative_labels=(POSITIVE_LABEL,), excluded_from_training=(),
        score_column="p_X", description="test only")
    with pytest.raises(ProbeHeadError, match="both a positive and a negative"):
        head_targets(contradictory, [POSITIVE_LABEL, NEGATIVE_LABEL])
