"""The three-class development frame, and one five-fold division of it.

``data.load_development_inputs`` deliberately projects the corpus down to the
E/B contrast, because everything written before this stage analysed that
contrast and nothing else. The causal steering experiment needs all three
classes at once -- E and H are its intervention targets, and B is what both
population directions are measured against -- so this module widens the view
without touching the loader every earlier result depends on.

State numbering is the package's, unchanged:

    state 0       the embedding output at the final prompt token
    states 1-48   the outputs of transformer blocks 0-47

Folds. The residual-geometry analysis already divided the 267 E and B rows into
five folds, twenty times over; its repeat-0 assignment is the one this stage
inherits, unchanged, so a row that was held out there is held out here. The 213
H rows were never assigned, and are added by a rule with no random number in it:
sort by (source family, row id), then deal round-robin across the folds from a
single running counter. That keeps each source family spread evenly and the
folds within one row of each other in size, and it gives the same answer every
time it is run, on any machine, in any Python.

Nothing here builds a direction, loads a model or writes an output. It produces
the row frame and the fold assignment, both of which are frozen before the first
generation and hashed into the direction bank.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SOURCE_COLUMN,
    TASK_FRAME_COLUMN,
    TOPIC_COLUMN,
    WITHHELD_LABEL,
    DevelopmentInputs,
    load_development_inputs,
)
from .validation import Checklist, ValidationError, assert_not_sealed, assert_not_stale

# The order classes are reported in. E first because it is what the experiment
# is trying to repair, B second because it is the reference both differences are
# taken against, H third because it is what must survive.
EBH_LABELS: Tuple[str, str, str] = (POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL)

N_SPLITS = 5

# Where the E/B assignment is inherited from, and which repeat of it. Repeat 0
# is the first draw of the residual-geometry analysis's twenty; the choice is
# recorded rather than implied, because "the folds" is ambiguous in a file that
# holds twenty of them.
FROZEN_EB_FOLD_FILENAME = "cv_assignments.csv"
FROZEN_EB_FOLD_REPEAT = 0

# The columns that file actually carries, verified rather than assumed.
FROZEN_EB_REQUIRED_COLUMNS = ("repeat", "row_id", "fold", "final_ebh", "split")

RESPONSE_COLUMN = "response_text"

H_ASSIGNMENT_RULE = (
    "H rows sorted by (source_family, row_id) and dealt round-robin across the "
    "five folds from a single running counter; no random number generator, no "
    "seed, no dependence on file order or dictionary iteration order")


class EBHError(ValidationError):
    """The three-class frame or its fold assignment was asked for something wrong."""


# --------------------------------------------------------------------- frame
@dataclass
class EBHFrame:
    """Every analysis-eligible row, in activation-array order.

    ``row_ids[i]`` is the prompt whose activations are ``activations[i]``. That
    identity is not assumed: it is the extraction manifest's own ``row_index``
    column, checked by ``data.load_development_inputs`` before this frame is
    built.
    """

    row_ids: List[str]
    labels: List[str]
    source_family: List[str]
    topic: List[str]
    task_frame: List[str]
    prompt_text: List[str]
    original_response: List[str]

    activations: np.ndarray                  # memory-mapped [480, 49, 2048]
    n_states: int
    hidden_size: int

    dataset_path: Path
    activation_path: Path
    extraction_manifest_path: Path
    dataset_sha256: str
    activation_sha256: str
    extraction_manifest_sha256: str

    checklist: Checklist = field(default_factory=lambda: Checklist("ebh_frame"))

    @property
    def n(self) -> int:
        return len(self.row_ids)

    @property
    def position_of(self) -> Dict[str, int]:
        return {rid: i for i, rid in enumerate(self.row_ids)}

    def positions_of(self, label: str) -> np.ndarray:
        """Array rows carrying one label, ascending."""
        if label not in EBH_LABELS:
            raise EBHError(f"unknown label {label!r}; valid: {list(EBH_LABELS)}")
        return np.array([i for i, value in enumerate(self.labels) if value == label],
                        dtype=int)

    def row_ids_of(self, label: str) -> List[str]:
        return [self.row_ids[i] for i in self.positions_of(label)]

    def counts(self) -> Dict[str, int]:
        counter = Counter(self.labels)
        return {label: int(counter.get(label, 0)) for label in EBH_LABELS}

    def metadata(self, position: int) -> Dict[str, str]:
        return {"row_id": self.row_ids[position],
                "final_ebh": self.labels[position],
                SOURCE_COLUMN: self.source_family[position],
                TOPIC_COLUMN: self.topic[position],
                TASK_FRAME_COLUMN: self.task_frame[position]}


def load_ebh_frame(manifest: Optional[Dict[str, Any]] = None,
                   paths: Optional[Dict[str, str]] = None,
                   manifest_path: Optional[Path | str] = None,
                   paths_path: Optional[Path | str] = None,
                   verify_hashes: bool = True) -> EBHFrame:
    """All three classes, behind the same gates the E/B loader uses.

    ``data.load_development_inputs`` runs first and unchanged: it is what checks
    the dataset hash, the activation hash, the extraction-manifest hash, the row
    counts, the per-row alignment between the manifest and the dataset, and the
    array's shape and dtype. This function only widens the projection afterwards
    and re-reads the dataset for the two columns the E/B path never needed for
    the withheld class.
    """
    kwargs: Dict[str, Any] = {"verify_hashes": verify_hashes}
    if manifest is not None:
        kwargs["manifest"] = manifest
    if paths is not None:
        kwargs["paths"] = paths
    if manifest_path is not None:
        kwargs["manifest_path"] = manifest_path
    if paths_path is not None:
        kwargs["paths_path"] = paths_path
    inputs = load_development_inputs(**kwargs)
    return frame_from_inputs(inputs)


def frame_from_inputs(inputs: DevelopmentInputs) -> EBHFrame:
    """Widen a validated E/B load into the three-class frame."""
    checks = Checklist("ebh_frame")

    dataset_path = assert_not_stale(Path(inputs.dataset_path))
    assert_not_sealed(dataset_path, "ebh dataset")
    with open(dataset_path, newline="", encoding="utf-8") as handle:
        dataset_rows = list(csv.DictReader(handle))
    by_row_id = {row["row_id"]: row for row in dataset_rows}

    for column in (TOPIC_COLUMN, TASK_FRAME_COLUMN, SOURCE_COLUMN, RESPONSE_COLUMN):
        checks.require(f"dataset_has_{column}", column in dataset_rows[0],
                       f"{column!r} is a column of the accepted dataset")

    manifest_rows = inputs.manifest_rows
    labels = [row["final_ebh"] for row in manifest_rows]
    unknown = sorted(set(labels) - set(EBH_LABELS))
    checks.require("every_row_is_E_B_or_H", not unknown,
                   f"labels present: {sorted(set(labels))}")

    counter = Counter(labels)
    for label in EBH_LABELS:
        checks.record(f"count_{label}", True, str(counter.get(label, 0)))
    checks.require("row_count_matches_the_array",
                   len(manifest_rows) == int(inputs.activations.shape[0]),
                   f"{len(manifest_rows)} rows, "
                   f"{int(inputs.activations.shape[0])} array rows")

    blank = [row["row_id"] for row in manifest_rows
             if not by_row_id[row["row_id"]][TOPIC_COLUMN].strip()
             or not by_row_id[row["row_id"]][TASK_FRAME_COLUMN].strip()
             or not by_row_id[row["row_id"]][SOURCE_COLUMN].strip()]
    checks.require(
        "every_eligible_row_carries_its_covariates", not blank,
        f"{len(manifest_rows)} rows carry a source family, topic and task frame")

    frame = EBHFrame(
        row_ids=[row["row_id"] for row in manifest_rows],
        labels=labels,
        source_family=[by_row_id[row["row_id"]][SOURCE_COLUMN]
                       for row in manifest_rows],
        topic=[by_row_id[row["row_id"]][TOPIC_COLUMN] for row in manifest_rows],
        task_frame=[by_row_id[row["row_id"]][TASK_FRAME_COLUMN]
                    for row in manifest_rows],
        prompt_text=[row["prompt_text"] for row in manifest_rows],
        original_response=[by_row_id[row["row_id"]].get(RESPONSE_COLUMN, "")
                           for row in manifest_rows],
        activations=inputs.activations,
        n_states=int(inputs.n_states),
        hidden_size=int(inputs.hidden_size),
        dataset_path=dataset_path,
        activation_path=Path(inputs.activation_path),
        extraction_manifest_path=Path(inputs.extraction_manifest_path),
        dataset_sha256=inputs.dataset_sha256,
        activation_sha256=inputs.activation_sha256,
        extraction_manifest_sha256=inputs.extraction_manifest_sha256,
        checklist=checks,
    )

    # The E/B half of this frame must be the same rows, in the same order, that
    # the E/B loader produced -- otherwise a fold inherited by row_id would be
    # bound to a different activation.
    eb_positions = [i for i, label in enumerate(frame.labels)
                    if label in (POSITIVE_LABEL, NEGATIVE_LABEL)]
    checks.require(
        "eb_rows_agree_with_the_two_class_loader",
        [frame.row_ids[i] for i in eb_positions] == list(inputs.row_ids),
        f"{len(eb_positions)} E/B rows in the same order as "
        f"data.load_development_inputs")

    # Read in row chunks rather than materialising the whole array: the E/B
    # loader has already checked its own 267 rows, and this widens the check to
    # the withheld ones without a 385 MB float64 copy of a memory-mapped file.
    non_finite = 0
    for start in range(0, frame.n, 64):
        chunk = np.asarray(inputs.activations[start:start + 64])
        non_finite += int((~np.isfinite(chunk)).sum())
    checks.require("all_activations_are_finite", non_finite == 0,
                   f"{frame.n} x {frame.n_states} x {frame.hidden_size} array, "
                   f"{non_finite} non-finite value(s)")
    return frame


# ------------------------------------------------------------ fold assignment
@dataclass
class EBHFoldAssignment:
    """One held-out fold per row, over all three classes.

    ``fold_of_row`` is keyed by ``row_id`` rather than by position, because the
    E/B half is inherited from a file written by a different analysis whose row
    order is its own.
    """

    fold_of_row: Dict[str, int]
    label_of_row: Dict[str, str]
    n_splits: int
    eb_source: str
    eb_repeat: int
    h_rule: str = H_ASSIGNMENT_RULE

    def fold_of(self, row_id: str) -> int:
        if row_id not in self.fold_of_row:
            raise EBHError(
                f"{row_id} has no fold; the assignment is frozen before the scan "
                f"and rows are not added to it at run time")
        return self.fold_of_row[row_id]

    def row_ids_in_fold(self, fold: int, label: Optional[str] = None) -> List[str]:
        return sorted(rid for rid, f in self.fold_of_row.items()
                      if f == fold and (label is None
                                        or self.label_of_row[rid] == label))

    def held_out_row_ids(self, fold: int, label: Optional[str] = None) -> List[str]:
        """Rows a fold-``fold`` direction may never see."""
        return self.row_ids_in_fold(fold, label)

    def training_row_ids(self, fold: int, label: Optional[str] = None) -> List[str]:
        """Rows a fold-``fold`` direction is built from: everything outside it."""
        return sorted(rid for rid, f in self.fold_of_row.items()
                      if f != fold and (label is None
                                        or self.label_of_row[rid] == label))

    def training_positions(self, frame: EBHFrame, fold: int
                           ) -> Dict[str, np.ndarray]:
        """Array rows, per class, that fold ``fold``'s direction is built from."""
        position_of = frame.position_of
        return {label: np.array(sorted(position_of[rid] for rid
                                       in self.training_row_ids(fold, label)),
                                dtype=int)
                for label in EBH_LABELS}

    def held_out_positions(self, frame: EBHFrame, fold: int
                           ) -> Dict[str, np.ndarray]:
        position_of = frame.position_of
        return {label: np.array(sorted(position_of[rid] for rid
                                       in self.held_out_row_ids(fold, label)),
                                dtype=int)
                for label in EBH_LABELS}

    def counts(self) -> List[Dict[str, Any]]:
        rows = []
        for fold in range(self.n_splits):
            row: Dict[str, Any] = {"fold": fold}
            for label in EBH_LABELS:
                row[f"held_out_{label}"] = len(self.held_out_row_ids(fold, label))
                row[f"train_{label}"] = len(self.training_row_ids(fold, label))
            rows.append(row)
        return rows

    def audit(self, frame: EBHFrame,
              frozen_eb: Optional[Dict[str, int]] = None,
              checklist: Optional[Checklist] = None) -> Checklist:
        """Every property the direction construction depends on. Raises on any."""
        checks = checklist or Checklist("ebh_folds")

        checks.require(
            "assignment_covers_exactly_the_eligible_rows",
            set(self.fold_of_row) == set(frame.row_ids),
            f"{len(self.fold_of_row)} assigned, {frame.n} eligible")
        checks.require(
            "folds_are_0_to_n_minus_1",
            sorted(set(self.fold_of_row.values())) == list(range(self.n_splits)),
            f"folds {sorted(set(self.fold_of_row.values()))}")

        for fold in range(self.n_splits):
            held_out = set(self.held_out_row_ids(fold))
            training = set(self.training_row_ids(fold))
            checks.require(
                f"fold_{fold}_training_and_held_out_are_disjoint",
                not (held_out & training),
                f"{len(training)} training, {len(held_out)} held out, overlap 0")
            checks.require(
                f"fold_{fold}_partitions_the_rows",
                len(held_out) + len(training) == frame.n,
                f"{len(training)} + {len(held_out)} = {frame.n}")
            for label in EBH_LABELS:
                checks.require(
                    f"fold_{fold}_training_{label}_is_not_empty",
                    len(self.training_row_ids(fold, label)) > 0,
                    f"{len(self.training_row_ids(fold, label))} training "
                    f"{label} row(s)")
                checks.require(
                    f"fold_{fold}_held_out_{label}_is_not_empty",
                    len(self.held_out_row_ids(fold, label)) > 0,
                    f"{len(self.held_out_row_ids(fold, label))} held-out "
                    f"{label} row(s)")

        appearances = Counter()
        for fold in range(self.n_splits):
            appearances.update(self.held_out_row_ids(fold))
        checks.require(
            "every_row_is_held_out_exactly_once",
            all(appearances[rid] == 1 for rid in frame.row_ids),
            f"all {frame.n} rows appear in exactly one held-out fold")

        if frozen_eb is not None:
            changed = [rid for rid, fold in frozen_eb.items()
                       if self.fold_of_row.get(rid) != fold]
            checks.require(
                "inherited_E_and_B_folds_are_unchanged",
                not changed,
                f"{len(frozen_eb)} E/B rows keep their repeat-"
                f"{self.eb_repeat} fold from {self.eb_source}")

        h_sizes = [len(self.held_out_row_ids(fold, WITHHELD_LABEL))
                   for fold in range(self.n_splits)]
        checks.require(
            "withheld_rows_are_evenly_distributed",
            max(h_sizes) - min(h_sizes) <= 1,
            f"held-out {WITHHELD_LABEL} per fold: {h_sizes}")

        by_family: Dict[str, List[int]] = defaultdict(lambda: [0] * self.n_splits)
        for position, rid in enumerate(frame.row_ids):
            if frame.labels[position] != WITHHELD_LABEL:
                continue
            by_family[frame.source_family[position]][self.fold_of(rid)] += 1
        spread = {family: max(counts) - min(counts)
                  for family, counts in sorted(by_family.items())}
        checks.record(
            "withheld_source_families_are_spread_across_the_folds",
            True,
            "; ".join(f"{family}: {by_family[family]}" for family in sorted(by_family))
            + f" (max-min per family: {spread})")
        return checks

    def as_rows(self, frame: EBHFrame) -> List[Dict[str, Any]]:
        """The complete assignment, one row per prompt, in array order."""
        rows = []
        for position, rid in enumerate(frame.row_ids):
            label = frame.labels[position]
            rows.append({
                "array_row_index": position,
                "row_id": rid,
                "final_ebh": label,
                "fold": self.fold_of(rid),
                "fold_source": ("inherited_eb_repeat_"
                                f"{self.eb_repeat}" if label != WITHHELD_LABEL
                                else "deterministic_withheld_assignment"),
                SOURCE_COLUMN: frame.source_family[position],
                TOPIC_COLUMN: frame.topic[position],
                TASK_FRAME_COLUMN: frame.task_frame[position],
            })
        return rows


def load_frozen_eb_folds(path: Path | str, row_ids: Sequence[str],
                         repeat: int = FROZEN_EB_FOLD_REPEAT
                         ) -> Tuple[Dict[str, int], int]:
    """Recover the E/B assignment that already exists, at one repeat.

    The file's schema is checked rather than assumed: it carries twenty repeats
    of a five-fold division, one row per (repeat, row_id), and the column names
    are read off the header before anything is parsed out of it.

    Returns the mapping and the number of distinct folds it uses.
    """
    path = Path(path)
    if not path.exists():
        raise EBHError(
            f"{path} does not exist. The E/B folds are inherited from the "
            f"completed residual-geometry analysis, not rebuilt here: a fresh "
            f"stratified split would hold out different rows and silently "
            f"change which prompts each direction was allowed to see.")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise EBHError(f"{path} contains no rows")
    missing_columns = [c for c in FROZEN_EB_REQUIRED_COLUMNS if c not in rows[0]]
    if missing_columns:
        raise EBHError(
            f"{path} has no {missing_columns} column(s); its columns are "
            f"{list(rows[0])}")

    selected = [row for row in rows if int(row["repeat"]) == int(repeat)]
    if not selected:
        available = sorted({int(row["repeat"]) for row in rows})
        raise EBHError(
            f"{path} carries no repeat {repeat}; repeats present: {available}")

    fold_by_row_id: Dict[str, int] = {}
    for row in selected:
        rid = row["row_id"]
        if rid in fold_by_row_id:
            raise EBHError(f"{path} lists {rid} twice in repeat {repeat}")
        if row["final_ebh"] not in (POSITIVE_LABEL, NEGATIVE_LABEL):
            raise EBHError(
                f"{path} assigns a fold to {rid}, labelled "
                f"{row['final_ebh']!r}; the inherited assignment covers "
                f"{POSITIVE_LABEL} and {NEGATIVE_LABEL} only")
        fold_by_row_id[rid] = int(row["fold"])

    wanted = set(row_ids)
    if set(fold_by_row_id) != wanted:
        missing = sorted(wanted - set(fold_by_row_id))
        extra = sorted(set(fold_by_row_id) - wanted)
        raise EBHError(
            f"{path} repeat {repeat} does not describe the E/B rows of the "
            f"accepted corpus: {len(missing)} missing, {len(extra)} unexpected")

    distinct = sorted(set(fold_by_row_id.values()))
    if distinct != list(range(len(distinct))):
        raise EBHError(
            f"fold indices in {path} repeat {repeat} are {distinct}; expected a "
            f"contiguous range starting at 0")
    return fold_by_row_id, len(distinct)


def assign_withheld_folds(rows: Sequence[Tuple[str, str]],
                          n_splits: int = N_SPLITS) -> Dict[str, int]:
    """Deal the withheld rows across the folds, deterministically.

    ``rows`` is ``(row_id, source_family)``. They are sorted by source family
    and then by row id, and dealt round-robin from one counter that carries
    across family boundaries. The counter is what makes the *global* fold sizes
    balanced; sorting by family first is what keeps each family balanced too,
    since consecutive rows of a family land in consecutive folds.

    No seed, no shuffle, no dependence on the order ``rows`` arrives in.
    """
    if n_splits < 2:
        raise EBHError(f"n_splits must be at least 2, got {n_splits}")
    ordered = sorted(rows, key=lambda item: (item[1], item[0]))
    seen = set()
    assignment: Dict[str, int] = {}
    for counter, (row_id, _family) in enumerate(ordered):
        if row_id in seen:
            raise EBHError(f"{row_id} appears twice among the withheld rows")
        seen.add(row_id)
        assignment[row_id] = counter % n_splits
    return assignment


def build_ebh_fold_assignment(frame: EBHFrame, frozen_path: Path | str,
                              repeat: int = FROZEN_EB_FOLD_REPEAT,
                              n_splits: int = N_SPLITS
                              ) -> Tuple[EBHFoldAssignment, Dict[str, int]]:
    """The complete frozen division: inherited E/B, deterministic H.

    Returns the assignment and the inherited E/B mapping, so a caller can audit
    the second against the first rather than trusting that it was honoured.
    """
    eb_row_ids = [rid for rid, label in zip(frame.row_ids, frame.labels)
                  if label in (POSITIVE_LABEL, NEGATIVE_LABEL)]
    frozen_eb, found_splits = load_frozen_eb_folds(frozen_path, eb_row_ids, repeat)
    if found_splits != n_splits:
        raise EBHError(
            f"the inherited assignment uses {found_splits} folds but this stage "
            f"is configured for {n_splits}; the fold count is not something to "
            f"reconcile silently")

    withheld = [(frame.row_ids[i], frame.source_family[i])
                for i, label in enumerate(frame.labels) if label == WITHHELD_LABEL]
    if not withheld:
        raise EBHError(
            f"the frame carries no {WITHHELD_LABEL} rows; the orthogonalised "
            f"direction cannot be built without them")
    withheld_folds = assign_withheld_folds(withheld, n_splits)

    overlap = set(frozen_eb) & set(withheld_folds)
    if overlap:
        raise EBHError(
            f"{len(overlap)} row(s) are assigned by both the inherited E/B file "
            f"and the withheld rule (first: {sorted(overlap)[0]})")

    fold_of_row = dict(frozen_eb)
    fold_of_row.update(withheld_folds)
    label_of_row = dict(zip(frame.row_ids, frame.labels))

    assignment = EBHFoldAssignment(
        fold_of_row=fold_of_row,
        label_of_row=label_of_row,
        n_splits=n_splits,
        eb_source=str(frozen_path),
        eb_repeat=int(repeat),
    )
    return assignment, frozen_eb
