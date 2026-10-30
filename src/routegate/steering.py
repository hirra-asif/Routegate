"""Cross-fitted, H-orthogonalised population directions, and the edit they define.

This is activation **steering**, not activation patching. Nothing is copied from
one prompt into another and nothing is replaced. A single direction, estimated
from class means over training rows, is added at one state at the final prompt
token, with a magnitude set as a fraction of the target's own block-input norm.

The arithmetic, per fold ``f`` and stored state ``l``, over training rows only:

    e_mean, b_mean, h_mean          class means of the stored activations

    false_refusal = e_mean − b_mean      what distinguishes an unnecessary
                                          refusal from an answered safe prompt
    true_refusal  = h_mean − b_mean      what distinguishes a justified refusal
                                          from an answered safe prompt

    projection    = (⟨false_refusal, true_refusal⟩ / ⟨true_refusal, true_refusal⟩)
                    · true_refusal
    orthogonal    = false_refusal − projection
    repair        = −orthogonal / ‖orthogonal‖

The orthogonalisation is the point of the design. ``false_refusal`` and
``true_refusal`` are both refusal-associated and are not expected to be
independent; subtracting the component of one along the other leaves the part of
the over-refusal signature that justified refusal does not share, and moving
*away* from it is what "repair" means here. An intervention built from the raw
difference would be free to work by suppressing refusal in general, which is
exactly the confound the H arm of the scan exists to detect.

Cross-fitting is not decoration. A direction used on a row that helped estimate
it is measuring its own training data. Every direction here is built from rows
outside the fold it will be applied in, and the bank stores both the training
and the held-out row identities per fold so that the guarantee can be checked
after the fact rather than believed.

Two degeneracies are handled explicitly rather than by a tolerance that hides
them. State 0 is the embedding output, and on this corpus every rendered chat
prompt ends with the same token, so all three class means coincide and both
differences are exactly zero. That state is retained in the bank as a
descriptive record and is refused as an intervention site. Any *transformer
block* state that comes out degenerate is a failure, not a control, and the
build stops.

The probe coefficient from the completed linear-probe analysis is deliberately
not used here. That analysis established that E and B are decodable; a decoding
direction and a causal direction are different claims, and this experiment is
the one that tests the second.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is needed to apply a direction, not to estimate one
    import torch

    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from .data import NEGATIVE_LABEL, POSITIVE_LABEL, WITHHELD_LABEL
from .intervention import (
    InterventionPlan,
    budget_reference_state,
    unit_scaled,
    vector_norm,
)
from .validation import ValidationError, sha256_text

# ------------------------------------------------------------------ constants
BANK_SCHEMA = "routegate/orthogonalized_direction_bank/1"

# What this stage calls its edit. It is intentionally not one of
# ``intervention.INTERVENTION_TYPES``: those four are the donor-patching
# vocabulary, they are frozen by that stage's tests and configuration, and a
# population direction is not one of them.
INTERVENTION_TYPE = "orthogonalized_population_direction"

# The state axis. State 0 is stored and described; the causal scan is the
# transformer blocks and nothing else.
N_STORED_STATES = 49
STATE_EMBEDDING = 0
CAUSAL_SCAN_STATES: Tuple[int, ...] = tuple(range(1, N_STORED_STATES))
N_TRANSFORMER_BLOCKS = 48

# The operating strength for the block-discovery scan. It is a pre-existing
# pilot value carried forward so that one axis is fixed while the other is
# searched -- not a measured optimum. A strength-sensitivity analysis is a
# separate, later run at whichever state(s) the scan selects.
FROZEN_SCAN_STRENGTH = 0.2

# A difference of class means is "effectively zero" when it is negligible
# against the scale of the means it came from. The relative term is what makes
# this meaningful across 49 states whose norms differ by orders of magnitude;
# the absolute term keeps an all-zero state from dividing by zero.
ZERO_NORM_ABS = 1e-12
ZERO_NORM_REL = 1e-9

# How close to unit length a stored direction must be before it may be applied.
UNIT_NORM_TOLERANCE = 1e-6

STATUS_OK = "ok"
STATUS_ZERO_TRUE_REFUSAL = "degenerate_zero_true_refusal"
STATUS_ZERO_ORTHOGONAL = "degenerate_zero_orthogonal_false_refusal"

# The artefacts a bank directory holds, all non-pickle.
ARRAYS_FILENAME = "direction_bank.npz"
VECTOR_TABLE_FILENAME = "vector_summary.csv"
FOLD_ASSIGNMENT_FILENAME = "fold_assignment.csv"
TRAINING_ROWS_FILENAME = "training_row_ids.json"
HELD_OUT_ROWS_FILENAME = "held_out_row_ids.json"
COUNTS_FILENAME = "counts.json"
INPUT_HASHES_FILENAME = "input_hashes.json"
CONFIG_FILENAME = "config.json"
MANIFEST_FILENAME = "manifest.json"

# Every file a complete bank holds, apart from the manifest that hashes them.
BANK_ARTIFACTS: Tuple[str, ...] = (
    ARRAYS_FILENAME, VECTOR_TABLE_FILENAME, FOLD_ASSIGNMENT_FILENAME,
    TRAINING_ROWS_FILENAME, HELD_OUT_ROWS_FILENAME, COUNTS_FILENAME,
    INPUT_HASHES_FILENAME, CONFIG_FILENAME,
)

# Arrays stored for every fold and every one of the 49 stored representations.
ARRAY_NAMES = ("e_mean", "b_mean", "h_mean", "false_refusal", "true_refusal",
               "orthogonal_false_refusal", "repair_direction")


class SteeringError(ValidationError):
    """The steering direction or its bank was asked for something it must not do."""


# ------------------------------------------------------------------- the maths
def _as_float64(vector: Any, label: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(vector, dtype=np.float64))
    if array.ndim != 1:
        raise SteeringError(
            f"{label} must be a single hidden-state vector, got shape "
            f"{array.shape}")
    if not bool(np.isfinite(array).all()):
        bad = int((~np.isfinite(array)).sum())
        raise SteeringError(f"{label} contains {bad} non-finite value(s)")
    return array


def is_effectively_zero(norm: float, scale: float) -> bool:
    """Whether a difference vector has no usable direction, relative to ``scale``."""
    return float(norm) <= max(ZERO_NORM_ABS, ZERO_NORM_REL * float(scale))


@dataclass
class PopulationVectors:
    """The five vectors and every number needed to argue about them, at one site."""

    false_refusal: np.ndarray
    true_refusal: np.ndarray
    orthogonal_false_refusal: np.ndarray
    repair_direction: np.ndarray

    norm_false_refusal: float
    norm_true_refusal: float
    norm_orthogonal_false_refusal: float
    norm_repair_direction: float

    projection_coefficient: float
    cosine_false_true_refusal: float
    orthogonality_residual_cosine: float
    mean_scale: float
    status: str

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def as_row(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "norm_false_refusal": self.norm_false_refusal,
            "norm_true_refusal": self.norm_true_refusal,
            "norm_orthogonal_false_refusal": self.norm_orthogonal_false_refusal,
            "norm_repair_direction": self.norm_repair_direction,
            "projection_coefficient": self.projection_coefficient,
            "cosine_false_true_refusal": self.cosine_false_true_refusal,
            "orthogonality_residual_cosine": self.orthogonality_residual_cosine,
            "mean_scale": self.mean_scale,
        }


def population_vectors(e_mean: Any, b_mean: Any, h_mean: Any) -> PopulationVectors:
    """The full construction at one (fold, state), from three class means.

    Pure arithmetic on three vectors: it holds no model, reads no file and knows
    nothing about folds, which is what lets every claim about it be tested on
    hand-written numbers.

    A degenerate site returns a zero ``repair_direction`` and says which
    degeneracy it hit. Deciding whether that is a control or a failure depends
    on *which state* it happened at, and that decision belongs to the caller.
    """
    e = _as_float64(e_mean, "e_mean")
    b = _as_float64(b_mean, "b_mean")
    h = _as_float64(h_mean, "h_mean")
    if not (e.shape == b.shape == h.shape):
        raise SteeringError(
            f"the three class means disagree on shape: E {e.shape}, B {b.shape}, "
            f"H {h.shape}")

    false_refusal = e - b
    true_refusal = h - b
    scale = float(max(np.linalg.norm(e), np.linalg.norm(b), np.linalg.norm(h)))

    norm_false = float(np.linalg.norm(false_refusal))
    norm_true = float(np.linalg.norm(true_refusal))
    zeros = np.zeros_like(false_refusal)

    if is_effectively_zero(norm_true, scale):
        # Without a justified-refusal direction there is nothing to orthogonalise
        # against, and projecting would divide by (almost) zero.
        return PopulationVectors(
            false_refusal=false_refusal, true_refusal=true_refusal,
            orthogonal_false_refusal=zeros.copy(), repair_direction=zeros.copy(),
            norm_false_refusal=norm_false, norm_true_refusal=norm_true,
            norm_orthogonal_false_refusal=0.0, norm_repair_direction=0.0,
            projection_coefficient=0.0, cosine_false_true_refusal=0.0,
            orthogonality_residual_cosine=0.0, mean_scale=scale,
            status=STATUS_ZERO_TRUE_REFUSAL)

    coefficient = float(np.dot(false_refusal, true_refusal)
                        / np.dot(true_refusal, true_refusal))
    orthogonal = false_refusal - coefficient * true_refusal
    norm_orthogonal = float(np.linalg.norm(orthogonal))
    cosine = (float(np.dot(false_refusal, true_refusal) / (norm_false * norm_true))
              if norm_false > 0.0 else 0.0)

    if is_effectively_zero(norm_orthogonal, scale):
        # false_refusal lies entirely along true_refusal: the over-refusal
        # signature has no component justified refusal does not share, so there
        # is no direction here that could repair E without moving H.
        return PopulationVectors(
            false_refusal=false_refusal, true_refusal=true_refusal,
            orthogonal_false_refusal=orthogonal, repair_direction=zeros.copy(),
            norm_false_refusal=norm_false, norm_true_refusal=norm_true,
            norm_orthogonal_false_refusal=norm_orthogonal,
            norm_repair_direction=0.0, projection_coefficient=coefficient,
            cosine_false_true_refusal=cosine,
            orthogonality_residual_cosine=0.0, mean_scale=scale,
            status=STATUS_ZERO_ORTHOGONAL)

    repair = -orthogonal / norm_orthogonal
    residual = float(np.dot(orthogonal, true_refusal)
                     / (norm_orthogonal * norm_true))

    return PopulationVectors(
        false_refusal=false_refusal, true_refusal=true_refusal,
        orthogonal_false_refusal=orthogonal, repair_direction=repair,
        norm_false_refusal=norm_false, norm_true_refusal=norm_true,
        norm_orthogonal_false_refusal=norm_orthogonal,
        norm_repair_direction=float(np.linalg.norm(repair)),
        projection_coefficient=coefficient, cosine_false_true_refusal=cosine,
        orthogonality_residual_cosine=residual, mean_scale=scale,
        status=STATUS_OK)


def class_means(activations: Any, positions: Sequence[int]) -> np.ndarray:
    """Mean activation over ``positions``, at every stored state: [n_states, hidden].

    The rows are taken in ascending array order and reduced in float64, so the
    summation order -- and therefore the result, bit for bit -- does not depend
    on the order the caller happened to collect the identities in.
    """
    index = np.asarray(sorted(int(p) for p in positions), dtype=int)
    if index.size == 0:
        raise SteeringError(
            "a class mean over zero rows is undefined; a training class must "
            "never be empty")
    block = np.asarray(activations[index], dtype=np.float64)
    if block.ndim != 3:
        raise SteeringError(
            f"expected [rows, states, hidden] activations, got shape {block.shape}")
    if not bool(np.isfinite(block).all()):
        raise SteeringError(
            f"the {index.size} selected activation row(s) contain non-finite values")
    return block.mean(axis=0)


def direction_sha256(vector: Any) -> str:
    """A stable digest of one direction.

    Little-endian float64 bytes, so the digest of a direction is the same on any
    machine that computed the same numbers, and a record that names a hash names
    exactly one vector.
    """
    array = np.ascontiguousarray(np.asarray(vector, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


# --------------------------------------------------------------- the direction bank
@dataclass
class DirectionBankArrays:
    """Every stored vector, for every fold and every stored state."""

    arrays: Dict[str, np.ndarray]            # each [n_folds, n_states, hidden]
    rows: List[Dict[str, Any]]               # one per (fold, state)
    n_folds: int
    n_states: int
    hidden_size: int

    def repair_direction(self, fold: int, state: int) -> np.ndarray:
        return self.arrays["repair_direction"][fold, state]

    def row(self, fold: int, state: int) -> Dict[str, Any]:
        for row in self.rows:
            if row["fold"] == fold and row["state"] == state:
                return row
        raise SteeringError(f"no record for fold {fold}, state {state}")


def build_direction_bank(activations: Any,
                         training_positions: Sequence[Dict[str, Sequence[int]]],
                         n_states: int = N_STORED_STATES,
                         causal_states: Sequence[int] = CAUSAL_SCAN_STATES,
                         ) -> DirectionBankArrays:
    """Estimate every fold's directions from that fold's training rows only.

    ``training_positions[f]`` maps each of ``E``, ``B``, ``H`` to the array rows
    fold ``f`` may learn from. Held-out rows are simply not passed in, which is
    the only way this function can be told about them -- there is no argument
    here that could accidentally let one through.

    Fails loudly, in this order: an empty training class, a non-finite
    activation, a zero-norm justified-refusal direction at a transformer block,
    a zero-norm orthogonal component at a transformer block.
    """
    n_folds = len(training_positions)
    if n_folds < 2:
        raise SteeringError(
            f"cross-fitting needs at least two folds, got {n_folds}")
    causal = sorted({int(s) for s in causal_states})
    if STATE_EMBEDDING in causal:
        raise SteeringError(
            "state 0 is the embedding output, not a transformer block, and is "
            "not part of the causal scan")
    outside = [state for state in causal if not 0 <= state < n_states]
    if outside:
        raise SteeringError(
            f"state(s) {outside} were declared part of the causal scan but the "
            f"activation array holds states 0..{n_states - 1}; a scanned state "
            f"whose degeneracy is never checked is worse than no check")

    arrays: Dict[str, List[np.ndarray]] = {name: [] for name in ARRAY_NAMES}
    rows: List[Dict[str, Any]] = []
    hidden_size: Optional[int] = None

    for fold, positions in enumerate(training_positions):
        missing = [label for label in (POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL)
                   if label not in positions]
        if missing:
            raise SteeringError(
                f"fold {fold} has no training rows declared for class(es) "
                f"{missing}")
        means = {}
        for label in (POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL):
            if len(positions[label]) == 0:
                raise SteeringError(
                    f"fold {fold} has no training rows of class {label}; a "
                    f"direction cannot be estimated from an empty class")
            means[label] = class_means(activations, positions[label])
            if means[label].shape[0] != n_states:
                raise SteeringError(
                    f"the activation array carries {means[label].shape[0]} "
                    f"states, expected {n_states}")
        hidden_size = int(means[NEGATIVE_LABEL].shape[1])

        per_fold = {name: np.zeros((n_states, hidden_size), dtype=np.float64)
                    for name in ARRAY_NAMES}
        per_fold["e_mean"] = means[POSITIVE_LABEL]
        per_fold["b_mean"] = means[NEGATIVE_LABEL]
        per_fold["h_mean"] = means[WITHHELD_LABEL]

        for state in range(n_states):
            vectors = population_vectors(means[POSITIVE_LABEL][state],
                                         means[NEGATIVE_LABEL][state],
                                         means[WITHHELD_LABEL][state])
            if state in causal and not vectors.ok:
                raise SteeringError(
                    f"fold {fold}, state {state} (output of transformer block "
                    f"{state - 1}) is degenerate: {vectors.status}. A "
                    f"transformer-block state with no usable direction is a "
                    f"failure of the estimate, not a control; the scan will not "
                    f"be run against it.")
            per_fold["false_refusal"][state] = vectors.false_refusal
            per_fold["true_refusal"][state] = vectors.true_refusal
            per_fold["orthogonal_false_refusal"][state] = \
                vectors.orthogonal_false_refusal
            per_fold["repair_direction"][state] = vectors.repair_direction
            rows.append({
                "fold": fold,
                "state": state,
                "site": ("embedding_output" if state == STATE_EMBEDDING
                         else "block_output"),
                "transformer_block_index": ("" if state == STATE_EMBEDDING
                                            else state - 1),
                "in_causal_scan": state in causal,
                "n_train_E": len(positions[POSITIVE_LABEL]),
                "n_train_B": len(positions[NEGATIVE_LABEL]),
                "n_train_H": len(positions[WITHHELD_LABEL]),
                "direction_sha256": direction_sha256(vectors.repair_direction),
                **vectors.as_row(),
            })

        for name in ARRAY_NAMES:
            arrays[name].append(per_fold[name])

    stacked = {name: np.ascontiguousarray(np.stack(values, axis=0))
               for name, values in arrays.items()}
    for name, array in stacked.items():
        if not bool(np.isfinite(array).all()):
            raise SteeringError(f"the stored {name} array contains non-finite values")
    return DirectionBankArrays(arrays=stacked, rows=rows, n_folds=n_folds,
                               n_states=n_states, hidden_size=int(hidden_size or 0))


def canonical_json(payload: Dict[str, Any]) -> str:
    """A manifest as hashed: sorted keys, no whitespace, hash field removed."""
    body = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_sha256(payload: Dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload))


VECTOR_TABLE_COLUMNS: Tuple[str, ...] = (
    "fold", "state", "site", "transformer_block_index", "in_causal_scan",
    "status", "n_train_E", "n_train_B", "n_train_H",
    "norm_false_refusal", "norm_true_refusal", "norm_orthogonal_false_refusal",
    "norm_repair_direction", "projection_coefficient",
    "cosine_false_true_refusal", "orthogonality_residual_cosine", "mean_scale",
    "direction_sha256",
)


def write_direction_bank(directory: Path | str,
                         bank: DirectionBankArrays,
                         fold_assignment_rows: Sequence[Dict[str, Any]],
                         training_row_ids: Dict[int, Dict[str, List[str]]],
                         held_out_row_ids: Dict[int, Dict[str, List[str]]],
                         counts: Dict[str, Any],
                         input_hashes: Dict[str, Any],
                         config: Dict[str, Any],
                         created_utc: str,
                         environment: Optional[Dict[str, Any]] = None,
                         ) -> Dict[str, Any]:
    """Write every artefact, then a manifest that hashes all of them.

    Order matters: the arrays and tables are written first and the manifest last,
    over their actual on-disk bytes, so a manifest can never describe a file that
    was not written. Everything is NPZ or JSON or CSV -- no pickle, so reading a
    bank back cannot execute anything.
    """
    import csv as _csv

    from .runrecord import write_json
    from .validation import sha256_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # np.savez writes every keyword as an array, so it takes no options here;
    # every stored array is plain float64 and no pickle is involved either way.
    np.savez(directory / ARRAYS_FILENAME,
             **{name: bank.arrays[name] for name in ARRAY_NAMES})

    with open(directory / VECTOR_TABLE_FILENAME, "w", newline="",
              encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(VECTOR_TABLE_COLUMNS))
        writer.writeheader()
        for row in bank.rows:
            writer.writerow({column: row[column] for column in VECTOR_TABLE_COLUMNS})

    assignment_columns = list(fold_assignment_rows[0]) if fold_assignment_rows else []
    with open(directory / FOLD_ASSIGNMENT_FILENAME, "w", newline="",
              encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=assignment_columns)
        writer.writeheader()
        for row in fold_assignment_rows:
            writer.writerow(row)

    write_json(directory / TRAINING_ROWS_FILENAME,
               {str(fold): {label: list(ids) for label, ids in sorted(by_label.items())}
                for fold, by_label in sorted(training_row_ids.items())})
    write_json(directory / HELD_OUT_ROWS_FILENAME,
               {str(fold): {label: list(ids) for label, ids in sorted(by_label.items())}
                for fold, by_label in sorted(held_out_row_ids.items())})
    write_json(directory / COUNTS_FILENAME, counts)
    write_json(directory / INPUT_HASHES_FILENAME, input_hashes)
    write_json(directory / CONFIG_FILENAME, config)

    # The artefact list is the declared one, not whatever the directory happens
    # to contain: a stale file left by an earlier build must not be adopted into
    # a manifest as though this build had produced it.
    artifacts = {name: sha256_file(directory / name) for name in BANK_ARTIFACTS}

    manifest: Dict[str, Any] = {
        "schema": BANK_SCHEMA,
        "created_utc": created_utc,
        "purpose": (
            "cross-fitted, H-orthogonalised population repair directions for the "
            "layer-wise causal activation steering scan; activation steering, "
            "not activation patching"),
        "construction": {
            "false_refusal": "mean(E) - mean(B), training rows only",
            "true_refusal": "mean(H) - mean(B), training rows only",
            "orthogonalisation": (
                "the component of false_refusal along true_refusal is removed "
                "before the direction is normalised"),
            "repair_direction": "-unit(orthogonal_false_refusal)",
            "cross_fitting": (
                "a fold's direction is estimated from rows outside that fold "
                "only; the held-out row identities are written out so the "
                "guarantee can be checked rather than assumed"),
            "not_used": [
                "individual B donors",
                "the linear-probe coefficient",
                "any held-out benchmark material",
            ],
        },
        "states": {
            "stored": bank.n_states,
            "numbering": ("state 0 is the embedding output; states 1-48 are the "
                          "outputs of transformer blocks 0-47"),
            "stored_for_description": [STATE_EMBEDDING],
            "causal_scan": list(CAUSAL_SCAN_STATES),
        },
        "folds": {"n_splits": bank.n_folds},
        "hidden_size": bank.hidden_size,
        "config": config,
        "counts": counts,
        "inputs": input_hashes,
        "artifacts": artifacts,
        "sites": [{"fold": row["fold"], "state": row["state"],
                   "status": row["status"],
                   "direction_sha256": row["direction_sha256"]}
                  for row in bank.rows],
        "environment": environment or {},
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    write_json(directory / MANIFEST_FILENAME, manifest)
    return manifest


# ------------------------------------------------------------------ reading it back
@dataclass
class DirectionBank:
    """A frozen bank, read back and verified."""

    directory: Path
    manifest: Dict[str, Any]
    arrays: Dict[str, np.ndarray]
    fold_of_row: Dict[str, int]
    label_of_row: Dict[str, str]
    training_row_ids: Dict[int, Dict[str, List[str]]]
    held_out_row_ids: Dict[int, Dict[str, List[str]]]
    status_by_site: Dict[Tuple[int, int], str] = field(default_factory=dict)
    hash_by_site: Dict[Tuple[int, int], str] = field(default_factory=dict)

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    @property
    def dataset_sha256(self) -> str:
        return str(self.manifest["inputs"]["dataset"]["sha256"])

    @property
    def n_folds(self) -> int:
        return int(self.manifest["folds"]["n_splits"])

    @property
    def strength(self) -> float:
        return float(self.manifest["config"]["strength"])

    def repair_direction(self, fold: int, state: int) -> np.ndarray:
        """The unit direction for a target held out in ``fold``, at ``state``.

        Refuses state 0 and any site the build recorded as degenerate: a caller
        that reaches for one of those has misread the protocol, and returning a
        zero vector would let the scan record an intervention that did nothing.
        """
        if state == STATE_EMBEDDING:
            raise SteeringError(
                "state 0 is the embedding output, not a transformer block. It "
                "is stored for descriptive analysis and is not an intervention "
                "site in the causal block scan.")
        if not 0 <= fold < self.n_folds:
            raise SteeringError(
                f"fold {fold} is outside 0..{self.n_folds - 1}")
        status = self.status_by_site.get((fold, state))
        if status is None:
            raise SteeringError(f"the bank holds no fold {fold}, state {state}")
        if status != STATUS_OK:
            raise SteeringError(
                f"fold {fold}, state {state} was recorded as {status}; it has no "
                f"usable direction and must not be steered with")
        direction = self.arrays["repair_direction"][fold, state]
        recorded = self.hash_by_site[(fold, state)]
        recomputed = direction_sha256(direction)
        if recomputed != recorded:
            raise SteeringError(
                f"fold {fold}, state {state}: the stored direction hashes to "
                f"{recomputed} but the manifest records {recorded}. The bank has "
                f"changed since it was frozen.")
        norm = float(np.linalg.norm(direction))
        if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            raise SteeringError(
                f"fold {fold}, state {state}: the stored direction has norm "
                f"{norm}, which is not a unit vector")
        return direction

    def direction_sha256(self, fold: int, state: int) -> str:
        return self.hash_by_site[(fold, state)]

    def fold_of(self, row_id: str) -> int:
        if row_id not in self.fold_of_row:
            raise SteeringError(
                f"{row_id} is not in the frozen fold assignment; targets are not "
                f"invented at run time")
        return self.fold_of_row[row_id]

    def assert_held_out(self, row_id: str, fold: int) -> None:
        """Refuse to steer a row with a direction its own activations shaped.

        The check is made against the bank's own written record of which rows
        each fold trained on, not against a recomputation of the fold rule, so
        it would catch an assignment that drifted after the bank was built.
        """
        training = self.training_row_ids.get(fold, {})
        for label, ids in training.items():
            if row_id in ids:
                raise SteeringError(
                    f"{row_id} is a fold-{fold} training row of class {label}; "
                    f"steering it with the fold-{fold} direction would apply a "
                    f"direction its own activations helped estimate")
        held_out = self.held_out_row_ids.get(fold, {})
        if not any(row_id in ids for ids in held_out.values()):
            raise SteeringError(
                f"{row_id} is not recorded as held out in fold {fold}; the "
                f"direction applied to a row must be the one built without it")


def load_direction_bank(directory: Path | str,
                        expected_sha256: Optional[str] = None) -> DirectionBank:
    """Read a frozen bank and refuse it if anything about it has moved."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise SteeringError(
            f"{manifest_path} does not exist. The direction bank is built and "
            f"frozen before any generation; there is no run-time estimation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BANK_SCHEMA:
        raise SteeringError(
            f"{manifest_path} declares schema {manifest.get('schema')!r}, "
            f"expected {BANK_SCHEMA!r}")

    recorded = manifest.get("manifest_sha256")
    recomputed = manifest_sha256(manifest)
    if recorded != recomputed:
        raise SteeringError(
            f"{manifest_path} has been modified since it was frozen:\n"
            f"  recorded   {recorded}\n  recomputed {recomputed}")
    if expected_sha256 and recorded != expected_sha256:
        raise SteeringError(
            f"{manifest_path} is not the bank this run was told to use:\n"
            f"  expected {expected_sha256}\n  actual   {recorded}")

    from .validation import sha256_file

    for name, digest in sorted(manifest["artifacts"].items()):
        path = directory / name
        if not path.exists():
            raise SteeringError(f"the bank is missing {name}")
        actual = sha256_file(path)
        if actual != digest:
            raise SteeringError(
                f"{path} hashes to {actual}, but the manifest records {digest}")

    with np.load(directory / ARRAYS_FILENAME, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]) for name in ARRAY_NAMES}

    fold_of_row: Dict[str, int] = {}
    label_of_row: Dict[str, str] = {}
    import csv as _csv

    with open(directory / FOLD_ASSIGNMENT_FILENAME, newline="",
              encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            fold_of_row[row["row_id"]] = int(row["fold"])
            label_of_row[row["row_id"]] = row["final_ebh"]

    training = {int(fold): {label: list(ids) for label, ids in by_label.items()}
                for fold, by_label in json.loads(
                    (directory / TRAINING_ROWS_FILENAME).read_text(
                        encoding="utf-8")).items()}
    held_out = {int(fold): {label: list(ids) for label, ids in by_label.items()}
                for fold, by_label in json.loads(
                    (directory / HELD_OUT_ROWS_FILENAME).read_text(
                        encoding="utf-8")).items()}

    status_by_site = {(int(row["fold"]), int(row["state"])): str(row["status"])
                      for row in manifest["sites"]}
    hash_by_site = {(int(row["fold"]), int(row["state"])):
                    str(row["direction_sha256"]) for row in manifest["sites"]}

    return DirectionBank(directory=directory, manifest=manifest, arrays=arrays,
                         fold_of_row=fold_of_row, label_of_row=label_of_row,
                         training_row_ids=training, held_out_row_ids=held_out,
                         status_by_site=status_by_site, hash_by_site=hash_by_site)


# ------------------------------------------------------------------ the edit
def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised only without torch
        raise SteeringError(
            f"applying a steering direction needs torch, which failed to "
            f"import: {_TORCH_IMPORT_ERROR}")


def build_steering_plan(state: int, direction: Any, strength: float,
                        reference_vector: Any,
                        budget_reference: str = "block_input") -> InterventionPlan:
    """Add ``direction × strength × ‖reference‖`` at one state. Nothing is replaced.

    ``reference_vector`` is the target's own activation at the budget-reference
    state -- for the ``block_input`` rule that is the input of the block being
    steered, which is the preceding state. The budget is therefore a fraction of
    the target's own residual magnitude at that depth, which is what makes an
    edit at state 3 and an edit at state 44 comparable in size.

    Returns an ``intervention.InterventionPlan`` so that the already-tested hook
    controller applies it unchanged.
    """
    _require_torch()
    if state == STATE_EMBEDDING:
        raise SteeringError(
            "state 0 is the embedding output and is not an intervention site in "
            "the causal block scan")
    if strength <= 0.0:
        raise SteeringError(f"strength must be positive, got {strength}")

    vector = torch.as_tensor(
        np.ascontiguousarray(np.asarray(direction, dtype=np.float32)))
    if reference_vector is None:
        raise SteeringError(
            "the budget is a fraction of the target's own activation norm, so a "
            "reference vector is required")
    if tuple(vector.shape) != tuple(reference_vector.shape):
        raise SteeringError(
            f"the direction has shape {tuple(vector.shape)} but the reference "
            f"activation has shape {tuple(reference_vector.shape)}; a mismatch "
            f"means two different states or two different models")
    if not bool(torch.isfinite(vector).all()):
        raise SteeringError("the steering direction contains non-finite values")

    norm = vector_norm(vector)
    if abs(norm - 1.0) > 1e-4:
        raise SteeringError(
            f"the steering direction has norm {norm}; the bank stores unit "
            f"directions and the budget assumes one")

    reference_state, note = budget_reference_state(state, budget_reference)
    plan = InterventionPlan(state=int(state), intervention_type=INTERVENTION_TYPE,
                            strength=float(strength), status="no_intervention",
                            budget_reference_state=reference_state,
                            budget_reference_note=note)
    plan.reference_norm = vector_norm(reference_vector)
    plan.budget = float(strength) * plan.reference_norm
    plan.raw_delta_norm = norm

    scaled = unit_scaled(vector, plan.budget)
    if scaled is None:
        plan.status = "zero_delta"
        return plan
    plan.delta = scaled
    plan.applied_delta_norm = vector_norm(scaled)
    plan.status = "ok"
    return plan
