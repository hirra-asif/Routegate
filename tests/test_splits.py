"""Fold loading and fold integrity.

Covers requirements 8 and 9 of the test plan: deterministic fold loading, and no
train/test overlap anywhere.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from conftest import needs_pilot_reference
from routegate.splits import (
    FoldAssignment,
    build_stratified_folds,
    load_frozen_folds,
    write_fold_assignments,
)
from routegate.validation import ValidationError


def _write_fold_file(path, row_ids, folds):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "fold", "noise"])
        writer.writeheader()
        for rid, fold in zip(row_ids, folds):
            writer.writerow({"row_id": rid, "fold": fold, "noise": "x"})


# ------------------------------------------------------ 8. deterministic load
def test_loading_the_same_file_twice_gives_the_same_assignment(tmp_path):
    row_ids = [f"r{i:03d}" for i in range(50)]
    folds = [i % 5 for i in range(50)]
    path = tmp_path / "folds.csv"
    _write_fold_file(path, row_ids, folds)
    first = load_frozen_folds(path, row_ids)
    second = load_frozen_folds(path, row_ids)
    np.testing.assert_array_equal(first.fold_of_row, second.fold_of_row)
    assert first.n_splits == second.n_splits == 5


def test_the_assignment_is_rebound_to_the_analysed_row_order(tmp_path):
    """A reference file in a different order must not permute the folds."""
    row_ids = [f"r{i:03d}" for i in range(20)]
    folds = [i % 4 for i in range(20)]
    shuffled = list(reversed(list(zip(row_ids, folds))))
    path = tmp_path / "folds.csv"
    _write_fold_file(path, [r for r, _ in shuffled], [f for _, f in shuffled])
    loaded = load_frozen_folds(path, row_ids)
    np.testing.assert_array_equal(loaded.fold_of_row, np.array(folds))


def test_a_fold_file_that_misses_a_row_is_refused(tmp_path):
    row_ids = [f"r{i}" for i in range(10)]
    path = tmp_path / "folds.csv"
    _write_fold_file(path, row_ids[:-1], [i % 2 for i in range(9)])
    with pytest.raises(ValidationError, match="does not describe the analysed rows"):
        load_frozen_folds(path, row_ids)


def test_a_duplicated_row_is_refused(tmp_path):
    row_ids = [f"r{i}" for i in range(6)]
    path = tmp_path / "folds.csv"
    _write_fold_file(path, row_ids + ["r0"], [0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(ValidationError, match="more than once"):
        load_frozen_folds(path, row_ids)


def test_a_missing_column_is_refused(tmp_path):
    path = tmp_path / "folds.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "group"])
        writer.writeheader()
        writer.writerow({"row_id": "r0", "group": 1})
    with pytest.raises(ValidationError, match="no 'fold' column"):
        load_frozen_folds(path, ["r0"])


# ------------------------------------------------------- 9. no leakage
def test_folds_partition_the_rows_with_no_overlap():
    y = np.array([1] * 89 + [0] * 178)
    row_ids = [f"r{i:03d}" for i in range(len(y))]
    folds = build_stratified_folds(y, row_ids, n_splits=5, seed=20260805)
    held_out = np.zeros(len(y), dtype=int)
    for train, test in folds.splits():
        assert np.intersect1d(train, test).size == 0
        assert train.size + test.size == len(y)
        held_out[test] += 1
    np.testing.assert_array_equal(held_out, np.ones_like(held_out))


def test_the_audit_rejects_an_overlapping_assignment():
    bad = FoldAssignment(fold_of_row=np.array([0, 0, 1, 1]), n_splits=3,
                         source="test", row_ids=["a", "b", "c", "d"])
    with pytest.raises(ValidationError, match="holds out no rows"):
        bad.audit()


def test_the_audit_requires_both_classes_in_every_training_fold():
    y = np.array([1, 1, 0, 0, 0, 0])
    folds = FoldAssignment(fold_of_row=np.array([0, 0, 1, 1, 1, 1]), n_splits=2,
                           source="test", row_ids=[f"r{i}" for i in range(6)])
    with pytest.raises(ValidationError, match="both_classes"):
        folds.audit(y)


def test_stratified_folds_keep_the_class_balance():
    y = np.array([1] * 89 + [0] * 178)
    row_ids = [f"r{i:03d}" for i in range(len(y))]
    folds = build_stratified_folds(y, row_ids, n_splits=5, seed=20260805)
    summary = folds.summary(y)
    assert sum(s["held_out_E"] for s in summary) == 89
    assert sum(s["held_out_B"] for s in summary) == 178
    assert max(s["held_out_E"] for s in summary) - min(s["held_out_E"] for s in summary) <= 1


def test_stratified_folds_are_reproducible_from_the_seed():
    y = np.array([1] * 40 + [0] * 60)
    row_ids = [f"r{i}" for i in range(len(y))]
    a = build_stratified_folds(y, row_ids, seed=20260805)
    b = build_stratified_folds(y, row_ids, seed=20260805)
    c = build_stratified_folds(y, row_ids, seed=20260806)
    np.testing.assert_array_equal(a.fold_of_row, b.fold_of_row)
    assert not np.array_equal(a.fold_of_row, c.fold_of_row)


def test_written_fold_assignments_round_trip(tmp_path):
    y = np.array([1] * 10 + [0] * 20)
    row_ids = [f"r{i:03d}" for i in range(len(y))]
    folds = build_stratified_folds(y, row_ids, n_splits=5, seed=1)
    path = tmp_path / "fold_assignments.csv"
    write_fold_assignments(path, folds, y)
    reloaded = load_frozen_folds(path, row_ids)
    np.testing.assert_array_equal(reloaded.fold_of_row, folds.fold_of_row)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["row_id"] for r in rows] == row_ids
    assert {r["final_ebh"] for r in rows} == {"E", "B"}


# ------------------------------------------------- the real pilot assignment
@needs_pilot_reference
def test_the_pilot_assignment_loads_and_partitions_cleanly(inputs, frozen_folds):
    assert frozen_folds.n == 267
    assert frozen_folds.n_splits == 5
    frozen_folds.audit(inputs.y)
    summary = frozen_folds.summary(inputs.y)
    assert sum(s["held_out_E"] for s in summary) == 89
    assert sum(s["held_out_B"] for s in summary) == 178
    assert sum(s["n_held_out"] for s in summary) == 267


@needs_pilot_reference
def test_the_pilot_assignment_is_not_a_fresh_stratified_split(inputs, frozen_folds):
    """The pilot used a bespoke deal; importing it is required, not optional."""
    fresh = build_stratified_folds(inputs.y, inputs.row_ids, n_splits=5,
                                   seed=20260805)
    assert not np.array_equal(fresh.fold_of_row, frozen_folds.fold_of_row), (
        "if these ever coincide the reproduction gate stops being informative")
