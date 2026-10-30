"""Two frozen probe heads over one cross-fitted division of the 480-row frame.

The proposed same-region dual-direction controller needs two prompt-level
readings before it can decide anything: how much a prompt looks like an
over-refusal worth repairing, and how much it looks like a harmful request whose
refusal must survive. This module produces exactly those two numbers.

    repair head   positive E, negative B. H is **never** in its training labels
                  -- not as a positive, not as a negative, and not among the
                  rows its scaler is estimated from. Every held-out row is
                  nevertheless scored, H included, so a cross-fitted p(E) exists
                  for all 480 prompts.

    safety head   positive H, negative E and B together as one class.

Two heads, one division of the rows. Both are cross-fitted against the fold
assignment the direction bank already froze, so a row's p(E) and its p(H) come
from probes that were each fitted without it, and both are on the same footing
as the direction that would be applied to it.

Why the repair head is not simply refitted with H as a third class, or with H
folded into its negatives: the quantity the controller needs from it is the
E-versus-B over-refusal signal the completed probe analysis measured. Letting H
rows into its training set -- or into its scaler -- would change what p(E) means
while leaving its name unchanged. Scoring H rows with it afterwards is a
different act: applying a fixed function to new rows does not alter the function.

Layer and state numbering, because every mistake here is a silent off-by-one:

    historical layer 31 = state 32     the predeclared primary read state
    historical layer 32 = state 33     sensitivity check only
    historical layer 33 = state 34     sensitivity check only

States 33 and 34 exist so the primary reading can be shown not to hang on one
state. They are not candidates. No "best" state is selected from their outcomes,
and this module deliberately offers no function that would rank states --
``metrics.best_state`` is for the state *search* the corrected analysis runs, and
this is not one.

Nothing here fits a calibration model. The logistic probabilities are reported
exactly as the frozen pipeline emits them, and the Brier score in the metrics
record measures them rather than changing them.

What this module does not do: build a direction, choose a coefficient, generate
text, load a model, or touch anything outside the development corpus. The
controller is not implemented.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss

from .data import NEGATIVE_LABEL, POSITIVE_LABEL, WITHHELD_LABEL, describe_state
from .ebh import (
    EBH_LABELS,
    EBHFoldAssignment,
    EBHFrame,
    build_ebh_fold_assignment,
)
from .metrics import chance_levels, evaluate_out_of_fold
from .probes import (
    DECISION_THRESHOLD,
    FoldOutcome,
    fit_probe_on_fold,
    make_probe,
    probe_specification,
)
from .steering import (
    BANK_SCHEMA,
    FOLD_ASSIGNMENT_FILENAME,
    INPUT_HASHES_FILENAME,
    MANIFEST_FILENAME,
    manifest_sha256,
)
from .validation import (
    Checklist,
    ValidationError,
    assert_not_sealed,
    assert_not_stale,
    sha256_file,
)


class ProbeHeadError(ValidationError):
    """A probe head was asked for something the protocol does not permit."""


# ------------------------------------------------------- layers and states
# The historical numbering is zero-based decoder blocks; this package numbers
# stored representations, and state 0 is the embedding output. The two differ by
# one, always, and the mapping is written down rather than applied from memory.
def state_of_layer(layer: int) -> int:
    """The stored state that carries the output of historical layer ``layer``."""
    if layer < 0:
        raise ProbeHeadError(f"historical layer {layer} is not a decoder block")
    return layer + 1


def layer_of_state(state: int) -> int:
    """The historical decoder block whose output is stored at ``state``."""
    if state <= 0:
        raise ProbeHeadError(
            f"state {state} is the embedding output, not a transformer block; "
            f"it has no historical layer index")
    return state - 1


HISTORICAL_LAYERS: Tuple[int, ...] = (31, 32, 33)
HISTORICAL_LAYER_TO_STATE: Dict[int, int] = {
    layer: state_of_layer(layer) for layer in HISTORICAL_LAYERS}

CONTROLLER_READ_STATES: Tuple[int, ...] = tuple(
    HISTORICAL_LAYER_TO_STATE[layer] for layer in HISTORICAL_LAYERS)

PRIMARY_READ_LAYER = 31
PRIMARY_READ_STATE = state_of_layer(PRIMARY_READ_LAYER)          # 32
SENSITIVITY_STATES: Tuple[int, ...] = tuple(
    state for state in CONTROLLER_READ_STATES if state != PRIMARY_READ_STATE)

PRIMARY_READ_STATE_STANDING = (
    "predeclared. State 32 is the primary controller read state because it is "
    "the frozen actuator's own first layer, declared before any head was "
    "fitted. States 33 and 34 are sensitivity checks: they are reported beside "
    "it and no state is reselected on their outcomes.")

TOKEN_POSITION = "final prompt token"
TOKEN_POSITION_NOTE = (
    "the last row of the prefill forward pass, which is the position the frozen "
    "activation array stores and the position the actuator edits")


# -------------------------------------------------------------- the two heads
LABEL_UNDEFINED = -1        # a row this head has no target for; never trained on

REPAIR_HEAD = "repair"
SAFETY_HEAD = "safety"
HEAD_ORDER: Tuple[str, str] = (REPAIR_HEAD, SAFETY_HEAD)


@dataclass(frozen=True)
class HeadSpecification:
    """What one head is trained on, what it scores, and what it never sees."""

    name: str
    positive_labels: Tuple[str, ...]
    negative_labels: Tuple[str, ...]
    excluded_from_training: Tuple[str, ...]
    score_column: str
    description: str

    @property
    def training_labels(self) -> Tuple[str, ...]:
        """The classes whose rows may be fitted on -- in ``EBH_LABELS`` order."""
        allowed = set(self.positive_labels) | set(self.negative_labels)
        return tuple(label for label in EBH_LABELS if label in allowed)

    @property
    def scored_labels(self) -> Tuple[str, ...]:
        """Every class this head produces a score for. All three, for both heads."""
        return EBH_LABELS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "head": self.name,
            "positive_class": list(self.positive_labels),
            "negative_class": list(self.negative_labels),
            "excluded_from_training": list(self.excluded_from_training),
            "trained_on": list(self.training_labels),
            "scored_rows": list(self.scored_labels),
            "score_column": self.score_column,
            "label_encoding": {label: 1 for label in self.positive_labels}
                              | {label: 0 for label in self.negative_labels},
            "description": self.description,
        }


REPAIR_SPECIFICATION = HeadSpecification(
    name=REPAIR_HEAD,
    positive_labels=(POSITIVE_LABEL,),
    negative_labels=(NEGATIVE_LABEL,),
    excluded_from_training=(WITHHELD_LABEL,),
    score_column="p_E",
    description=(
        "p(E): unnecessary refusal against benign compliance. H is excluded from "
        "its training labels and from its scaler, and every held-out row is "
        "still scored, so p(E) is available for every prompt including the H "
        "rows this head was never fitted on."),
)

SAFETY_SPECIFICATION = HeadSpecification(
    name=SAFETY_HEAD,
    positive_labels=(WITHHELD_LABEL,),
    negative_labels=(POSITIVE_LABEL, NEGATIVE_LABEL),
    excluded_from_training=(),
    score_column="p_H",
    description=(
        "p(H): appropriately refused harmful request against everything safe. E "
        "and B are one negative class, not two: the question this head answers "
        "is whether a prompt should be refused at all, not why a safe prompt "
        "was or was not answered."),
)

HEAD_SPECIFICATIONS: Dict[str, HeadSpecification] = {
    REPAIR_HEAD: REPAIR_SPECIFICATION,
    SAFETY_HEAD: SAFETY_SPECIFICATION,
}


def head_specification(name: str) -> HeadSpecification:
    if name not in HEAD_SPECIFICATIONS:
        raise ProbeHeadError(
            f"unknown head {name!r}; this stage defines {list(HEAD_ORDER)}")
    return HEAD_SPECIFICATIONS[name]


# ------------------------------------------------------- target construction
def head_targets(spec: HeadSpecification, labels: Sequence[str]) -> np.ndarray:
    """1 for a positive row, 0 for a negative one, -1 where the head has no target.

    The -1 is not a third class and is never fitted on: it marks the rows the
    head is asked to *score* but not to learn from. For the repair head those
    are the H rows; for the safety head there are none.
    """
    unknown = sorted(set(labels) - set(EBH_LABELS))
    if unknown:
        raise ProbeHeadError(
            f"{spec.name} head: labels {unknown} are not one of {list(EBH_LABELS)}")
    positive = set(spec.positive_labels)
    negative = set(spec.negative_labels)
    overlap = positive & negative
    if overlap:
        raise ProbeHeadError(
            f"{spec.name} head: {sorted(overlap)} is both a positive and a "
            f"negative class")
    targets = np.full(len(labels), LABEL_UNDEFINED, dtype=int)
    for i, label in enumerate(labels):
        if label in positive:
            targets[i] = 1
        elif label in negative:
            targets[i] = 0
    for label in spec.excluded_from_training:
        if label in positive or label in negative:
            raise ProbeHeadError(
                f"{spec.name} head: {label} is excluded from training and also "
                f"named as a class")
    if int((targets == 1).sum()) == 0 or int((targets == 0).sum()) == 0:
        raise ProbeHeadError(
            f"{spec.name} head: needs both classes, got "
            f"{int((targets == 1).sum())} positive and "
            f"{int((targets == 0).sum())} negative row(s)")
    return targets


def repair_targets(labels: Sequence[str]) -> np.ndarray:
    """E = 1, B = 0, H = -1 (scored, never trained on)."""
    return head_targets(REPAIR_SPECIFICATION, labels)


def safety_targets(labels: Sequence[str]) -> np.ndarray:
    """H = 1, E and B = 0."""
    return head_targets(SAFETY_SPECIFICATION, labels)


# ------------------------------------------------------------- the fold source
@dataclass
class FoldSource:
    """Where the division of the rows came from, and what it hashed to."""

    directory: Path
    manifest_path: Path
    manifest_sha256: str
    fold_assignment_path: Path
    fold_assignment_sha256: str
    inherited: Dict[str, Any]
    rows: List[Dict[str, str]]
    n_splits: int
    declared_n_splits: Optional[int] = None
    cross_check: str = "not_attempted"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": "cross_fitted_direction_bank",
            "directory": str(self.directory),
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "fold_assignment": str(self.fold_assignment_path),
            "fold_assignment_sha256": self.fold_assignment_sha256,
            "n_splits": self.n_splits,
            "n_splits_declared_by_the_manifest": self.declared_n_splits,
            "rows": len(self.rows),
            "inherited_E_and_B_assignment": dict(self.inherited),
            "H_assignment": "the bank's deterministic withheld rule, read back "
                            "from the frozen file rather than recomputed",
            "cross_check_against_the_inherited_file": self.cross_check,
        }


FOLD_ASSIGNMENT_COLUMNS: Tuple[str, ...] = (
    "array_row_index", "row_id", "final_ebh", "fold")


def load_bank_fold_assignment(directory: Path | str,
                              expected_manifest_sha256: Optional[str] = None
                              ) -> Tuple[EBHFoldAssignment, FoldSource]:
    """Recover the frozen 480-row division from the direction bank.

    The bank is the single source of the folds for this stage, and it is read
    the way it was written: the manifest's own ``manifest_sha256`` is recomputed
    over its canonical form, and the two artefacts this function reads are
    checked against the digests that manifest records. A bank whose
    ``fold_assignment.csv`` has been touched since it was frozen is refused, not
    reconciled.

    Nothing is recomputed from a rule here. ``ebh.build_ebh_fold_assignment``
    can rebuild the same division from the inherited E/B file plus the
    deterministic H rule, and ``cross_check_inherited_folds`` requires the two to
    agree -- but the bank's written assignment is what the directions were built
    against, so it is what is used.
    """
    directory = assert_not_stale(Path(directory))
    assert_not_sealed(directory, "direction bank directory")

    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ProbeHeadError(
            f"{manifest_path} does not exist. The fold assignment for this stage "
            f"is the one the cross-fitted direction bank froze; it is not "
            f"rebuilt here, and a fresh split would hold out different rows.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BANK_SCHEMA:
        raise ProbeHeadError(
            f"{manifest_path} declares schema {manifest.get('schema')!r}, "
            f"expected {BANK_SCHEMA!r}")

    recorded = manifest.get("manifest_sha256")
    recomputed = manifest_sha256(manifest)
    if recorded != recomputed:
        raise ProbeHeadError(
            f"{manifest_path} has been modified since it was frozen:\n"
            f"  recorded   {recorded}\n  recomputed {recomputed}")
    if expected_manifest_sha256 and recorded != expected_manifest_sha256:
        raise ProbeHeadError(
            f"{manifest_path} is not the bank this run was told to use:\n"
            f"  expected {expected_manifest_sha256}\n  actual   {recorded}")

    artifacts = manifest.get("artifacts") or {}
    digests: Dict[str, str] = {}
    for name in (FOLD_ASSIGNMENT_FILENAME, INPUT_HASHES_FILENAME):
        path = directory / name
        if name not in artifacts:
            raise ProbeHeadError(f"{manifest_path} records no digest for {name}")
        if not path.exists():
            raise ProbeHeadError(f"the bank is missing {name}")
        actual = sha256_file(path)
        if actual != artifacts[name]:
            raise ProbeHeadError(
                f"{path} hashes to {actual}, but the manifest records "
                f"{artifacts[name]}. The bank has changed since it was frozen.")
        digests[name] = actual

    fold_path = directory / FOLD_ASSIGNMENT_FILENAME
    with open(fold_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ProbeHeadError(f"{fold_path} contains no rows")
    missing = [c for c in FOLD_ASSIGNMENT_COLUMNS if c not in rows[0]]
    if missing:
        raise ProbeHeadError(
            f"{fold_path} has no {missing} column(s); its columns are "
            f"{list(rows[0])}")

    positions = [int(row["array_row_index"]) for row in rows]
    if positions != list(range(len(rows))):
        raise ProbeHeadError(
            f"{fold_path} does not run 0..{len(rows) - 1} in file order; its "
            f"array_row_index column is the activation row position and a file "
            f"in any other order cannot be bound to the frame by position")

    fold_of_row: Dict[str, int] = {}
    label_of_row: Dict[str, str] = {}
    for row in rows:
        rid = row["row_id"]
        if rid in fold_of_row:
            raise ProbeHeadError(f"{fold_path} lists {rid} more than once")
        if row["final_ebh"] not in EBH_LABELS:
            raise ProbeHeadError(
                f"{fold_path} labels {rid} {row['final_ebh']!r}; the three-class "
                f"frame carries {list(EBH_LABELS)} only")
        fold_of_row[rid] = int(row["fold"])
        label_of_row[rid] = row["final_ebh"]

    distinct = sorted(set(fold_of_row.values()))
    if distinct != list(range(len(distinct))):
        raise ProbeHeadError(
            f"fold indices in {fold_path} are {distinct}; expected a contiguous "
            f"range starting at 0")

    declared = (manifest.get("folds") or {}).get("n_splits")
    if declared is not None and int(declared) != len(distinct):
        raise ProbeHeadError(
            f"{manifest_path} declares {declared} folds but "
            f"{FOLD_ASSIGNMENT_FILENAME} uses {len(distinct)}; a bank whose "
            f"manifest and fold file disagree is refused rather than reconciled")

    inherited = json.loads(
        (directory / INPUT_HASHES_FILENAME).read_text(encoding="utf-8")
    ).get("inherited_fold_assignment")
    if not inherited:
        raise ProbeHeadError(
            f"{directory / INPUT_HASHES_FILENAME} records no "
            f"inherited_fold_assignment; the E/B half of the division cannot be "
            f"traced to the file it came from")

    source = FoldSource(
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256=str(recorded),
        fold_assignment_path=fold_path,
        fold_assignment_sha256=digests[FOLD_ASSIGNMENT_FILENAME],
        inherited=dict(inherited),
        rows=rows,
        n_splits=len(distinct),
        declared_n_splits=None if declared is None else int(declared),
    )
    assignment = EBHFoldAssignment(
        fold_of_row=fold_of_row,
        label_of_row=label_of_row,
        n_splits=len(distinct),
        eb_source=str(inherited.get("path", fold_path)),
        eb_repeat=int(inherited.get("repeat", 0)),
    )
    return assignment, source


def bind_fold_assignment(frame: EBHFrame, assignment: EBHFoldAssignment,
                         source: FoldSource,
                         checklist: Optional[Checklist] = None) -> Checklist:
    """Bind the frozen division to the analysed frame, position by position.

    Row *order* is the check that matters. A file that names the right 480 rows
    in a different order would still pass a set comparison and would then attach
    every fold to the wrong activation.
    """
    checks = checklist or Checklist("probe_head_folds")
    file_row_ids = [row["row_id"] for row in source.rows]
    file_labels = [row["final_ebh"] for row in source.rows]

    checks.require(
        "the_fold_file_names_the_analysed_rows_in_the_analysed_order",
        file_row_ids == list(frame.row_ids),
        f"{len(file_row_ids)} row_id values, position for position against the "
        f"frame's {frame.n} rows")
    checks.require(
        "the_fold_file_agrees_with_the_frame_on_every_label",
        file_labels == list(frame.labels),
        f"{len(file_labels)} rows agree on final_ebh")
    checks.record(
        "the_division_the_bank_froze",
        True,
        f"{assignment.n_splits} folds over {len(source.rows)} rows, from "
        f"{source.fold_assignment_path} (sha256 "
        f"{source.fold_assignment_sha256})")
    assignment.audit(frame, checklist=checks)
    return checks


def cross_check_inherited_folds(frame: EBHFrame, assignment: EBHFoldAssignment,
                                source: FoldSource,
                                checklist: Optional[Checklist] = None) -> str:
    """Rebuild the division from its stated origin and require agreement.

    The bank's written assignment is authoritative. This exists so that the
    *rule* the bank claims to have followed -- the inherited E/B repeat plus the
    deterministic H deal -- can be shown to reproduce it, rather than being
    taken on the manifest's word. A disagreement is a stop, not a preference:
    two conflicting historical assignments are never silently resolved here.

    Returns what happened, so a run record can state it rather than imply it.
    """
    checks = checklist or Checklist("probe_head_folds")
    path = Path(source.inherited.get("path", ""))
    expected = str(source.inherited.get("sha256", ""))
    repeat = int(source.inherited.get("repeat", 0))

    if not path.exists():
        outcome = f"not_available: {path} is not reachable from this machine"
        checks.record("the_inherited_E_and_B_file_was_cross_checked", True, outcome)
        source.cross_check = outcome
        return outcome

    actual = sha256_file(path)
    checks.require(
        "the_inherited_E_and_B_file_is_the_one_the_bank_recorded",
        actual == expected,
        f"{path} sha256 {actual}")

    rebuilt, _frozen_eb = build_ebh_fold_assignment(
        frame, path, repeat=repeat, n_splits=assignment.n_splits)
    disagreements = [rid for rid in frame.row_ids
                     if rebuilt.fold_of_row.get(rid) != assignment.fold_of_row.get(rid)]
    checks.require(
        "rebuilding_the_division_from_its_stated_rule_reproduces_the_bank",
        not disagreements,
        f"{frame.n} rows: inherited repeat {repeat} plus the deterministic "
        f"withheld deal reproduces the bank's fold for every row")
    outcome = (f"agrees: rebuilt from {path} repeat {repeat} (sha256 {actual}) "
               f"and matched on all {frame.n} rows")
    source.cross_check = outcome
    return outcome


# --------------------------------------------------------------- the design
def state_matrix(frame: EBHFrame, state: int) -> np.ndarray:
    """The ``[n, hidden]`` design at one state, over every eligible row.

    ``data.DevelopmentInputs.state_matrix`` returns the E/B rows only, which is
    what every earlier analysis needed. Both heads here are scored over all three
    classes, so the H rows must be in the same array, in the frame's own row
    order. Contiguous float64 for the same reason the E/B loader gives: the
    memory-mapped array is strided on the state axis.
    """
    if not 0 <= state < frame.n_states:
        raise ProbeHeadError(
            f"state {state} is outside 0..{frame.n_states - 1}")
    block = frame.activations[:, state, :]
    return np.ascontiguousarray(np.asarray(block, dtype=np.float64))


# ------------------------------------------------------------ fitting a head
@dataclass
class HeadResult:
    """One head, cross-fitted at one state: every row scored exactly once."""

    spec: HeadSpecification
    state: int
    scores: np.ndarray                  # [n] cross-fitted probability
    targets: np.ndarray                 # [n] 1 / 0 / LABEL_UNDEFINED
    fold_of_row: np.ndarray             # [n]
    folds: List[FoldOutcome] = field(default_factory=list)
    fold_counts: List[Dict[str, Any]] = field(default_factory=list)
    degenerate: bool = False

    @property
    def layer(self) -> int:
        return layer_of_state(self.state)

    @property
    def evaluated(self) -> np.ndarray:
        """Rows this head has a target for, and can therefore be scored on."""
        return self.targets >= 0

    @property
    def all_converged(self) -> bool:
        return all(outcome.converged for outcome in self.folds)

    @property
    def max_iterations(self) -> int:
        return max((outcome.n_iter for outcome in self.folds), default=0)


def fit_head(spec: HeadSpecification, X: np.ndarray, frame: EBHFrame,
             assignment: EBHFoldAssignment, state: int,
             checklist: Optional[Checklist] = None) -> HeadResult:
    """Fit ``spec``'s probe once per fold and score the rows that fold held out.

    Within fold k the head is fitted on the training rows of its own classes and
    on nothing else, and it scores **every** row the fold holds out -- including
    rows of a class it was not fitted on. That asymmetry is the point of the
    repair head: H contributes to no coefficient and to no scaler statistic, and
    still receives a p(E).

    There is no separate scaling step that could see the whole design: the frozen
    ``StandardScaler`` is the first stage of ``probes.make_probe``'s pipeline, so
    its mean and scale are estimated on ``train_index`` alone. That is checked
    afterwards against those rows rather than assumed.
    """
    checks = checklist or Checklist(f"{spec.name}_head_state_{state}")
    X = np.asarray(X, dtype=np.float64)
    n = frame.n
    if X.shape[0] != n:
        raise ProbeHeadError(
            f"{spec.name} head: the design has {X.shape[0]} rows and the frame "
            f"has {n}")
    if X.ndim != 2 or X.shape[1] != frame.hidden_size:
        raise ProbeHeadError(
            f"{spec.name} head: the design is {X.shape}, expected "
            f"({n}, {frame.hidden_size})")
    if set(assignment.fold_of_row) != set(frame.row_ids):
        raise ProbeHeadError(
            f"{spec.name} head: the fold assignment covers "
            f"{len(assignment.fold_of_row)} rows and the frame has {n}")

    targets = head_targets(spec, frame.labels)
    degenerate = bool(X.shape[1] > 0 and float(X.std(axis=0).max()) == 0.0)

    scores = np.full(n, np.nan, dtype=np.float64)
    fold_of_row = np.full(n, -1, dtype=int)
    scored_times = np.zeros(n, dtype=int)
    outcomes: List[FoldOutcome] = []
    fold_counts: List[Dict[str, Any]] = []

    for fold in range(assignment.n_splits):
        training = assignment.training_positions(frame, fold)
        held_out = assignment.held_out_positions(frame, fold)

        train_index = np.sort(np.concatenate(
            [training[label] for label in spec.training_labels]).astype(int))
        test_index = np.sort(np.concatenate(
            [held_out[label] for label in spec.scored_labels]).astype(int))

        if train_index.size == 0:
            raise ProbeHeadError(
                f"{spec.name} head, fold {fold}: no training rows")
        if test_index.size == 0:
            raise ProbeHeadError(
                f"{spec.name} head, fold {fold}: no held-out rows")

        overlap = np.intersect1d(train_index, test_index)
        checks.require(
            f"{spec.name}_fold_{fold}_training_and_held_out_are_disjoint",
            overlap.size == 0,
            f"{train_index.size} training, {test_index.size} held out, overlap 0")

        for label in spec.excluded_from_training:
            excluded = set(frame.positions_of(label).tolist())
            leaked = sorted(excluded & set(train_index.tolist()))
            checks.require(
                f"{spec.name}_fold_{fold}_no_{label}_row_is_trained_on",
                not leaked,
                f"{len(excluded)} {label} row(s) in the corpus, none of the "
                f"{train_index.size} training rows")

        fold_targets = targets[train_index]
        checks.require(
            f"{spec.name}_fold_{fold}_training_targets_are_binary",
            set(np.unique(fold_targets).tolist()) <= {0, 1},
            f"training targets {sorted(set(np.unique(fold_targets).tolist()))}")
        n_pos = int((fold_targets == 1).sum())
        n_neg = int((fold_targets == 0).sum())
        checks.require(
            f"{spec.name}_fold_{fold}_training_set_has_both_classes",
            n_pos > 0 and n_neg > 0,
            f"positive {n_pos}, negative {n_neg}")

        outcome = fit_probe_on_fold(X, targets, train_index, test_index,
                                    fold=fold, probe_factory=make_probe,
                                    keep_fold_detail=True)

        # The scaler is meant to have seen this head's training rows and nothing
        # else. Recomputing the mean over exactly those rows is the direct way to
        # say so, and it would catch a pipeline that had been rearranged.
        expected_mean = X[train_index].mean(axis=0)
        checks.require(
            f"{spec.name}_fold_{fold}_scaler_is_fitted_on_its_own_training_rows",
            (outcome.scaler_mean.shape == expected_mean.shape
             and bool(np.allclose(outcome.scaler_mean, expected_mean,
                                  rtol=1e-9, atol=1e-9))),
            f"StandardScaler mean over the {train_index.size} training rows of "
            f"{list(spec.training_labels)}")

        scores[test_index] = outcome.scores
        fold_of_row[test_index] = fold
        scored_times[test_index] += 1
        outcomes.append(outcome)
        fold_counts.append({
            "fold": fold,
            "n_train": int(train_index.size),
            "n_train_positive": n_pos,
            "n_train_negative": n_neg,
            "n_held_out": int(test_index.size),
            "n_held_out_scored_without_a_target": int(
                (targets[test_index] == LABEL_UNDEFINED).sum()),
            "train_by_class": {label: int(training[label].size)
                               for label in EBH_LABELS},
            "held_out_by_class": {label: int(held_out[label].size)
                                  for label in EBH_LABELS},
            "converged": bool(outcome.converged),
            "solver_iterations": int(outcome.n_iter),
        })

    checks.require(
        f"{spec.name}_every_row_is_scored_exactly_once",
        bool((scored_times == 1).all()),
        f"all {n} rows carry one cross-fitted {spec.score_column}")
    checks.require(
        f"{spec.name}_every_score_is_a_finite_probability",
        (bool(np.isfinite(scores).all())
         and bool(((scores >= 0.0) & (scores <= 1.0)).all())),
        f"{spec.score_column} in [0, 1] for all {n} rows")
    checks.require(
        f"{spec.name}_the_scoring_fold_is_the_assignments_fold",
        all(int(fold_of_row[i]) == assignment.fold_of(rid)
            for i, rid in enumerate(frame.row_ids)),
        f"{n} rows scored by the fold that holds them out")

    return HeadResult(spec=spec, state=int(state), scores=scores,
                      targets=targets, fold_of_row=fold_of_row,
                      folds=outcomes, fold_counts=fold_counts,
                      degenerate=degenerate)


# ------------------------------------------------------- both heads at a state
@dataclass
class CrossFittedHeads:
    """Both heads at one state, over the same rows and the same division."""

    state: int
    row_ids: List[str]
    labels: List[str]
    fold_of_row: np.ndarray
    heads: Dict[str, HeadResult]
    checklist: Checklist

    @property
    def layer(self) -> int:
        return layer_of_state(self.state)

    @property
    def is_primary_read_state(self) -> bool:
        return self.state == PRIMARY_READ_STATE

    @property
    def p_E(self) -> np.ndarray:
        return self.heads[REPAIR_HEAD].scores

    @property
    def p_H(self) -> np.ndarray:
        return self.heads[SAFETY_HEAD].scores

    def prediction_rows(self) -> List[Dict[str, Any]]:
        """One row per prompt: ``row_id, final_ebh, fold, state, p_E, p_H``."""
        return [{"row_id": rid,
                 "final_ebh": self.labels[i],
                 "fold": int(self.fold_of_row[i]),
                 "state": int(self.state),
                 "p_E": float(self.p_E[i]),
                 "p_H": float(self.p_H[i])}
                for i, rid in enumerate(self.row_ids)]


PREDICTION_COLUMNS: Tuple[str, ...] = (
    "row_id", "final_ebh", "fold", "state", "p_E", "p_H")


def fit_probe_heads(frame: EBHFrame, assignment: EBHFoldAssignment, state: int,
                    checklist: Optional[Checklist] = None) -> CrossFittedHeads:
    """Both heads at one state, cross-fitted against one division of the rows."""
    checks = checklist or Checklist(f"probe_heads_state_{state}")
    checks.record("state", True, describe_state(state))
    if state > 0:
        checks.record("historical_layer", True,
                      f"state {state} is historical layer {layer_of_state(state)}")

    X = state_matrix(frame, state)
    checks.require(
        f"state_{state}_design_covers_every_eligible_row",
        X.shape == (frame.n, frame.hidden_size),
        f"{X.shape} design over {frame.n} rows")

    heads = {name: fit_head(head_specification(name), X, frame, assignment,
                            state, checks)
             for name in HEAD_ORDER}

    repair, safety = heads[REPAIR_HEAD], heads[SAFETY_HEAD]
    checks.require(
        "both_heads_used_the_same_division",
        bool(np.array_equal(repair.fold_of_row, safety.fold_of_row)),
        f"one fold index per row, shared by {list(HEAD_ORDER)}")
    checks.require(
        "the_repair_head_scores_the_rows_it_never_trained_on",
        int((repair.targets == LABEL_UNDEFINED).sum()) > 0
        and bool(np.isfinite(repair.scores[repair.targets == LABEL_UNDEFINED]).all()),
        f"{int((repair.targets == LABEL_UNDEFINED).sum())} "
        f"{WITHHELD_LABEL} row(s) carry a cross-fitted p(E)")
    checks.require(
        "the_safety_head_has_a_target_for_every_row",
        int((safety.targets == LABEL_UNDEFINED).sum()) == 0,
        f"{frame.n} rows, {WITHHELD_LABEL} positive and "
        f"{POSITIVE_LABEL}+{NEGATIVE_LABEL} negative")

    return CrossFittedHeads(
        state=int(state),
        row_ids=list(frame.row_ids),
        labels=list(frame.labels),
        fold_of_row=repair.fold_of_row.copy(),
        heads=heads,
        checklist=checks,
    )


# --------------------------------------------------------------------- scoring
def evaluate_head(result: HeadResult) -> Dict[str, Any]:
    """ROC-AUC, PR-AUC and Brier score for one head, on the rows it has targets for.

    ROC-AUC, PR-AUC, balanced accuracy and accuracy come from
    ``metrics.evaluate_out_of_fold`` unchanged, so they are the same guarded
    scikit-learn calls every other stage reports. The Brier score is added here
    because no earlier stage reported one: it is
    ``sklearn.metrics.brier_score_loss`` on the probabilities exactly as the
    frozen pipeline emits them. Nothing is calibrated -- a Brier score is how the
    probabilities are measured, not a step that changes them.

    ``StateMetrics`` names its class counts ``n_E`` and ``n_B`` because it was
    written for the E/B contrast. They are renamed here to positive and negative,
    because for the safety head they are H and E+B.
    """
    spec = result.spec
    mask = result.evaluated
    y = result.targets[mask].astype(int)
    p = result.scores[mask].astype(np.float64)

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ProbeHeadError(
            f"{spec.name} head at state {result.state}: both classes must be "
            f"present to score it (positive {n_pos}, negative {n_neg})")

    record = evaluate_out_of_fold(
        y, p, state=result.state, degenerate=result.degenerate,
        all_folds_converged=result.all_converged,
        max_solver_iterations=result.max_iterations,
        threshold=DECISION_THRESHOLD)
    brier = float(brier_score_loss(y, p))
    if not np.isfinite(brier) or not 0.0 <= brier <= 1.0:
        raise ProbeHeadError(
            f"{spec.name} head at state {result.state}: Brier score {brier} is "
            f"not a probability-scale loss in [0, 1]")

    return {
        "head": spec.name,
        "state": int(result.state),
        "historical_layer": result.layer,
        "is_primary_read_state": result.state == PRIMARY_READ_STATE,
        "role": ("primary" if result.state == PRIMARY_READ_STATE
                 else "sensitivity_check"),
        "positive_class": list(spec.positive_labels),
        "negative_class": list(spec.negative_labels),
        "excluded_from_training": list(spec.excluded_from_training),
        "score_column": spec.score_column,
        "n_scored": int(mask.sum()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_scored_without_a_target": int((~mask).sum()),
        "roc_auc": record.roc_auc,
        "pr_auc": record.pr_auc,
        "brier": brier,
        "balanced_accuracy": record.balanced_accuracy,
        "accuracy": record.accuracy,
        "decision_threshold": DECISION_THRESHOLD,
        "chance": chance_levels(y),
        "calibration_model_fitted": False,
        "brier_note": "sklearn.metrics.brier_score_loss on the logistic "
                      "probabilities as emitted; no calibration model is fitted "
                      "at this stage and the probabilities are unchanged",
        "degenerate": record.degenerate,
        "all_folds_converged": record.all_folds_converged,
        "max_solver_iterations": record.max_solver_iterations,
        "n_folds": len(result.folds),
        "per_fold": list(result.fold_counts),
    }


def evaluate_state(heads: CrossFittedHeads) -> Dict[str, Any]:
    """Both heads at one state, as one record."""
    return {
        "state": int(heads.state),
        "historical_layer": heads.layer,
        "role": ("primary" if heads.is_primary_read_state
                 else "sensitivity_check"),
        "state_meaning": describe_state(heads.state),
        "token_position": TOKEN_POSITION,
        "n_rows": len(heads.row_ids),
        "heads": {name: evaluate_head(heads.heads[name]) for name in HEAD_ORDER},
    }


# ------------------------------------------------------------ the declaration
def probe_head_specification() -> Dict[str, Any]:
    """Everything frozen about this stage, for the run record.

    The probe block is ``probes.probe_specification()`` verbatim. This stage
    introduces no hyperparameter of its own: the two heads differ in their labels
    and in which rows they are fitted on, and in nothing else.
    """
    return {
        "stage": "dual_direction_controller_probe_heads",
        "probe": probe_specification(),
        "heads": {name: HEAD_SPECIFICATIONS[name].as_dict() for name in HEAD_ORDER},
        "cross_fitting": {
            "n_splits_source": "the frozen direction-bank fold assignment",
            "rule": "within fold k each head is fitted on training-fold rows of "
                    "its own classes only and scores every row fold k holds "
                    "out; each head's scaler is fitted on that head's training "
                    "rows alone",
            "every_row_scored_exactly_once": True,
        },
        "layer_state_convention": {
            "mapping": {str(layer): state
                        for layer, state in HISTORICAL_LAYER_TO_STATE.items()},
            "rule": "historical layer L is state L+1; state 0 is the embedding "
                    "output and is not a transformer block",
            "primary_read_layer": PRIMARY_READ_LAYER,
            "primary_read_state": PRIMARY_READ_STATE,
            "primary_read_state_standing": PRIMARY_READ_STATE_STANDING,
            "sensitivity_states": list(SENSITIVITY_STATES),
            "sensitivity_states_are_not_candidates":
                "states 33 and 34 are reported beside state 32 and no state is "
                "reselected on their outcomes",
        },
        "token_position": {"position": TOKEN_POSITION,
                           "meaning": TOKEN_POSITION_NOTE},
        "calibration": {"model_fitted": False,
                        "brier_score_role": "a measurement of the emitted "
                                            "probabilities, not a transform of "
                                            "them"},
        "not_implemented_at_this_stage": [
            "the dual-direction controller itself",
            "any steering direction",
            "any coefficient rule or coefficient network",
            "generation, judging or any behavioural label",
        ],
    }
