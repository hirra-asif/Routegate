"""Cross-validation folds.

Three sources of folds, kept strictly apart.

``load_frozen_folds``
    Reads a fold assignment that already exists and must be honoured exactly --
    the pilot's assignment, recovered from its saved out-of-fold predictions.
    Reproduction uses this and nothing else: an independently constructed
    stratified split, however reasonable, gives different held-out sets and
    therefore different numbers.

``build_stratified_folds``
    ``sklearn.model_selection.StratifiedKFold`` for new work. It is not used by
    the reproduction, and it does not reproduce the pilot's assignment -- the
    pilot dealt shuffled per-class positions round-robin, which is a different
    algorithm.

``build_repeated_stratified_folds``
    The same splitter repeated over many seeds, for the corrected analysis. One
    split is one draw; repeating the whole cross-validation turns a point
    estimate into a distribution and shows how much of a state ranking is split
    noise. The assignment is built **once**, written out as a first-class
    artefact, and reused unchanged across every state, every activation variant,
    every control and every permutation -- so a difference between two of them
    is never a difference in how the rows were divided.

All three return the same object: a ``FoldAssignment`` carrying one fold index
per row, in the analysed row order, plus the train/test index pairs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .validation import Checklist, ValidationError


@dataclass
class FoldAssignment:
    """One fold index per analysed row, in the analysed row order."""

    fold_of_row: np.ndarray            # shape [n], values 0..n_splits-1
    n_splits: int
    source: str                        # "frozen:<path>" or "stratified:<seed>"
    row_ids: List[str]

    @property
    def n(self) -> int:
        return int(self.fold_of_row.shape[0])

    def splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """(train_idx, test_idx) for each fold, positions into the row order."""
        all_idx = np.arange(self.n)
        out = []
        for k in range(self.n_splits):
            test = all_idx[self.fold_of_row == k]
            train = all_idx[self.fold_of_row != k]
            if test.size == 0:
                raise ValidationError(f"fold {k} holds out no rows")
            if train.size == 0:
                raise ValidationError(f"fold {k} leaves no training rows")
            out.append((train, test))
        return out

    def audit(self, y: Optional[np.ndarray] = None,
              checklist: Optional[Checklist] = None) -> Checklist:
        """Verify the partition. Raises on any leakage."""
        checks = checklist or Checklist("folds")
        splits = self.splits()
        checks.require("fold_count", len(splits) == self.n_splits,
                       f"{len(splits)} folds")
        held_out = np.zeros(self.n, dtype=int)
        for k, (train, test) in enumerate(splits):
            checks.require(
                f"fold_{k}_train_and_test_are_disjoint",
                np.intersect1d(train, test).size == 0,
                f"train {train.size}, held out {test.size}, overlap 0")
            checks.require(
                f"fold_{k}_partitions_the_rows",
                train.size + test.size == self.n,
                f"{train.size} + {test.size} = {self.n}")
            held_out[test] += 1
        checks.require("every_row_is_held_out_exactly_once",
                       bool((held_out == 1).all()),
                       f"all {self.n} rows appear in exactly one held-out fold")
        if y is not None:
            for k, (train, test) in enumerate(splits):
                checks.require(
                    f"fold_{k}_training_set_has_both_classes",
                    int((y[train] == 1).sum()) > 0 and int((y[train] == 0).sum()) > 0,
                    f"train E={int((y[train] == 1).sum())} "
                    f"B={int((y[train] == 0).sum())}")
        return checks

    def summary(self, y: np.ndarray) -> List[Dict[str, int]]:
        rows = []
        for k, (train, test) in enumerate(self.splits()):
            rows.append({"fold": k,
                         "n_train": int(train.size),
                         "n_held_out": int(test.size),
                         "held_out_E": int((y[test] == 1).sum()),
                         "held_out_B": int((y[test] == 0).sum())})
        return rows


# ------------------------------------------------------------- frozen folds
def load_frozen_folds(path: Path | str, row_ids: Sequence[str],
                      row_id_column: str = "row_id",
                      fold_column: str = "fold") -> FoldAssignment:
    """Recover an existing fold assignment and bind it to the analysed rows.

    ``row_ids`` is the analysed row order. The file must name exactly those
    rows, once each; the returned assignment is reindexed into that order, so a
    difference in the reference file's own ordering cannot silently permute the
    folds.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValidationError(f"{path} contains no rows")
    for column in (row_id_column, fold_column):
        if column not in rows[0]:
            raise ValidationError(
                f"{path} has no '{column}' column; columns are {list(rows[0])}")

    fold_by_row_id: Dict[str, int] = {}
    for row in rows:
        rid = row[row_id_column]
        if rid in fold_by_row_id:
            raise ValidationError(f"{path} lists {rid} more than once")
        fold_by_row_id[rid] = int(row[fold_column])

    if set(fold_by_row_id) != set(row_ids):
        missing = sorted(set(row_ids) - set(fold_by_row_id))
        extra = sorted(set(fold_by_row_id) - set(row_ids))
        raise ValidationError(
            f"{path} does not describe the analysed rows: "
            f"{len(missing)} missing, {len(extra)} unexpected")
    if len(rows) != len(row_ids):
        raise ValidationError(
            f"{path} has {len(rows)} rows for {len(row_ids)} analysed rows")

    return _bind_folds(fold_by_row_id, row_ids, f"frozen:{path}", str(path))


def _bind_folds(fold_by_row_id: Dict[str, int], row_ids: Sequence[str],
                source: str, origin: str) -> FoldAssignment:
    """Reindex a row_id -> fold mapping into the analysed row order."""
    fold_of_row = np.array([fold_by_row_id[rid] for rid in row_ids], dtype=int)
    distinct = sorted(set(fold_of_row.tolist()))
    if distinct != list(range(len(distinct))):
        raise ValidationError(
            f"fold indices in {origin} are {distinct}; expected a contiguous "
            f"range starting at 0")
    return FoldAssignment(fold_of_row=fold_of_row, n_splits=len(distinct),
                          source=source, row_ids=list(row_ids))


# --------------------------------------------------------- new stratified CV
def build_stratified_folds(y: np.ndarray, row_ids: Sequence[str],
                           n_splits: int = 5, seed: int = 20260805,
                           shuffle: bool = True) -> FoldAssignment:
    """A fresh label-stratified split from scikit-learn.

    For new analyses only. This does not reproduce the pilot's assignment.
    """
    y = np.asarray(y).astype(int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                               random_state=seed if shuffle else None)
    fold_of_row = np.full(y.shape[0], -1, dtype=int)
    for k, (_, test) in enumerate(splitter.split(np.zeros_like(y), y)):
        fold_of_row[test] = k
    if (fold_of_row < 0).any():
        raise ValidationError("StratifiedKFold left a row unassigned")
    return FoldAssignment(fold_of_row=fold_of_row, n_splits=n_splits,
                          source=f"stratified:sklearn:seed={seed}:shuffle={shuffle}",
                          row_ids=list(row_ids))


# ----------------------------------------------------- repeated stratified CV
@dataclass
class RepeatedFoldAssignment:
    """One ``FoldAssignment`` per repeat, built once and reused everywhere."""

    assignments: List[FoldAssignment]
    n_splits: int
    base_seed: int
    row_ids: List[str]

    @property
    def n_repeats(self) -> int:
        return len(self.assignments)

    @property
    def n(self) -> int:
        return self.assignments[0].n

    def __iter__(self):
        return iter(enumerate(self.assignments))

    def seed_of_repeat(self, repeat: int) -> int:
        return self.base_seed + repeat

    def audit(self, y: np.ndarray,
              checklist: Optional[Checklist] = None) -> Checklist:
        """Audit every repeat, and confirm the repeats are not all the same."""
        checks = checklist or Checklist("repeated_folds")
        for repeat, folds in self:
            sub = Checklist(f"repeat_{repeat}")
            folds.audit(y, sub)
            for entry in sub.entries:
                checks.record(f"repeat_{repeat}_{entry['check']}",
                              entry["status"] == "PASS", entry["detail"])
        distinct = {tuple(int(v) for v in f.fold_of_row) for f in self.assignments}
        checks.require(
            "repeats_are_distinct_draws",
            len(distinct) == self.n_repeats,
            f"{len(distinct)} distinct assignments over {self.n_repeats} repeats")
        return checks


def build_repeated_stratified_folds(y: np.ndarray, row_ids: Sequence[str],
                                    n_splits: int = 5, repeats: int = 20,
                                    base_seed: int = 20260805
                                    ) -> RepeatedFoldAssignment:
    """``repeats`` independent stratified splits, seeded ``base_seed + repeat``."""
    if repeats < 1:
        raise ValidationError(f"repeats must be at least 1, got {repeats}")
    assignments = [
        build_stratified_folds(y, row_ids, n_splits=n_splits,
                               seed=base_seed + repeat, shuffle=True)
        for repeat in range(repeats)
    ]
    return RepeatedFoldAssignment(assignments=assignments, n_splits=n_splits,
                                  base_seed=base_seed, row_ids=list(row_ids))


def write_repeated_fold_assignments(path: Path | str,
                                    repeated: RepeatedFoldAssignment,
                                    y: np.ndarray,
                                    extra_columns: Optional[Dict[str, Sequence]] = None
                                    ) -> None:
    """Persist every (row_id, repeat, fold, split) triple that a run will use.

    Written once, before any probe is fitted, so that every state, variant,
    control and permutation demonstrably shares one division of the rows.
    """
    extra_columns = extra_columns or {}
    fieldnames = ["repeat", "seed", "position", "row_id", "y_true", "final_ebh",
                  "fold", "split"]
    fieldnames += [c for c in extra_columns if c not in fieldnames]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for repeat, folds in repeated:
            for i, rid in enumerate(folds.row_ids):
                row = {"repeat": repeat,
                       "seed": repeated.seed_of_repeat(repeat),
                       "position": i,
                       "row_id": rid,
                       "y_true": int(y[i]),
                       "final_ebh": "E" if y[i] == 1 else "B",
                       "fold": int(folds.fold_of_row[i]),
                       "split": "held_out"}
                for column, values in extra_columns.items():
                    row[column] = values[i]
                writer.writerow(row)


def load_repeated_fold_assignments(path: Path | str, row_ids: Sequence[str]
                                   ) -> RepeatedFoldAssignment:
    """Recover a written repeated assignment and re-bind it to the row order."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValidationError(f"{path} contains no rows")
    for column in ("repeat", "row_id", "fold"):
        if column not in rows[0]:
            raise ValidationError(
                f"{path} has no '{column}' column; columns are {list(rows[0])}")

    by_repeat: Dict[int, Dict[str, int]] = {}
    seeds: Dict[int, int] = {}
    for row in rows:
        repeat = int(row["repeat"])
        folds = by_repeat.setdefault(repeat, {})
        if row["row_id"] in folds:
            raise ValidationError(
                f"{path} lists {row['row_id']} twice in repeat {repeat}")
        folds[row["row_id"]] = int(row["fold"])
        if "seed" in row:
            seeds[repeat] = int(row["seed"])

    ordered = sorted(by_repeat)
    if ordered != list(range(len(ordered))):
        raise ValidationError(
            f"repeat indices in {path} are {ordered}; expected 0..n-1")
    assignments = []
    for repeat in ordered:
        mapping = by_repeat[repeat]
        if set(mapping) != set(row_ids):
            raise ValidationError(
                f"{path} repeat {repeat} does not describe the analysed rows")
        assignments.append(_bind_folds(mapping, row_ids,
                                       f"repeated_stratified:{path}:repeat={repeat}",
                                       f"{path} repeat {repeat}"))
    base_seed = seeds.get(0, 0)
    if seeds and any(seeds[r] != base_seed + r for r in ordered if r in seeds):
        raise ValidationError(f"{path} seeds are not base_seed + repeat: {seeds}")
    return RepeatedFoldAssignment(
        assignments=assignments, n_splits=assignments[0].n_splits,
        base_seed=base_seed, row_ids=list(row_ids))


def folds_for_subset(y: np.ndarray, row_ids: Sequence[str], n_splits: int,
                     seed: int) -> Tuple[FoldAssignment, int]:
    """Stratified folds for a subgroup, with ``n_splits`` capped by the rarer class.

    Used by the within-source analysis, where a subgroup can be too small to
    support the configured number of folds. Returns the assignment and the
    number of folds actually used, so the caller reports the cap rather than
    hiding it.
    """
    y = np.asarray(y).astype(int)
    smallest_class = int(min((y == 1).sum(), (y == 0).sum()))
    if smallest_class < 2:
        raise ValidationError(
            f"a subgroup with only {smallest_class} example(s) of a class "
            f"cannot be cross-validated (E={int((y == 1).sum())}, "
            f"B={int((y == 0).sum())})")
    used = int(min(n_splits, smallest_class))
    return build_stratified_folds(y, row_ids, n_splits=used, seed=seed), used


def write_fold_assignments(path: Path | str, folds: FoldAssignment,
                           y: np.ndarray,
                           extra_columns: Optional[Dict[str, Sequence]] = None) -> None:
    """Persist the fold assignment used by a run."""
    extra_columns = extra_columns or {}
    fieldnames = ["position", "row_id", "y_true", "final_ebh", "fold"]
    fieldnames += [c for c in extra_columns if c not in fieldnames]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, rid in enumerate(folds.row_ids):
            row = {"position": i, "row_id": rid, "y_true": int(y[i]),
                   "final_ebh": "E" if y[i] == 1 else "B",
                   "fold": int(folds.fold_of_row[i])}
            for column, values in extra_columns.items():
                row[column] = values[i]
            writer.writerow(row)
