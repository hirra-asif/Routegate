"""The validated dual-direction bank: repair ``B_d1 - E`` and safety ``H - E``.

Two directions, built from the ten retrospectively validated E/B/H triplets and
from nothing else. This module holds the arithmetic and the gates; the runner
in ``experiments/dual_direction_controller/`` holds the paths and the I/O.

What eligibility means here
---------------------------
``manifests/rq1_triplet_inclusion.json`` is the authoritative record and the
only one consulted. A case enters a direction if, and only if, that manifest
marks the direction's own rule ``validated`` for it:

===========  ==============================  ==================================
direction    arm                             inclusion rule
===========  ==============================  ==================================
``repair``   ``B_d1`` -- the historical d1    ``comparator_actuator_source``
             donor
``safety``   ``H`` -- the harmful contrast    ``h_e_direction_source``
===========  ==============================  ==================================

``primary_actuator_source`` -- the historical B_d2/B_d3 actuator -- is **not**
a rule this module reads. It is validated for 3 of 10 cases; the other seven
were judged and failed. It stays a comparison baseline and cannot enter this
bank: there is no code path here that names its arms.

The construction, at each layer independently
---------------------------------------------
For every eligible case::

    delta          = arm_activation - E_activation
    case_direction = delta / ||delta||_2

then average the ten case directions and renormalise that average to unit L2
norm. Identical for repair and for safety.

The subtraction order is ``arm - E`` and not ``E - arm``; it is fixed here and
checked by ``assert_subtraction_order``, because a sign flip is the one error
in this file that would produce a perfectly well-formed artefact pointing the
wrong way.

Per-case normalisation before averaging is deliberate and is what RQ2 froze: it
gives every case one vote regardless of how far apart its two activations
happen to sit, so one case with a large delta cannot dominate the bank.

What this module does not do
----------------------------
* **No orthogonalisation.** The bank holds the two direct directions. Their
  per-layer cosine is measured and recorded as a diagnostic, and that is all.
  An orthogonalised variant is a later declared ablation, not this artefact.
* **No application strength.** No coefficient is chosen, embedded or defaulted.
  This phase constructs directions; scaling one is a different decision, taken
  elsewhere, on its own terms.
* **No controller.** Nothing here is fitted or trained.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .intervention import historical_layer_to_state
from .validation import Checklist, ValidationError

# ------------------------------------------------------------------ geometry
#: The historical zero-based decoder blocks this bank covers, and this
#: package's state numbering for them. The mapping is not restated as a
#: literal: it comes from ``intervention.historical_layer_to_state`` so that
#: one convention cannot drift from the other.
HISTORICAL_LAYERS: Tuple[int, ...] = (31, 32, 33)
BANK_STATES: Tuple[int, ...] = tuple(
    historical_layer_to_state(layer) for layer in HISTORICAL_LAYERS)

HIDDEN_SIZE = 2048
N_LAYERS = len(HISTORICAL_LAYERS)
N_CASES = 10

#: The activation block every arm arrives as: ten cases, three layers, 2048.
CAPTURE_SHAPE: Tuple[int, int, int] = (N_CASES, N_LAYERS, HIDDEN_SIZE)
CAPTURE_DTYPE = np.dtype("float32")

#: Where in the forward pass these activations were read. Recorded so the
#: artefact states its own provenance rather than relying on a reader knowing.
TOKEN_POSITION = "final prompt token"
TOKEN_POSITION_NOTE = (
    "window=1, slot 0: the final prefill row, the position that decides "
    "generated token 1. Decode positions are not read.")
SITE = "residual_stream"
SITE_KIND = "block_output"

#: The model whose activations these are. Pinned, and checked against the
#: caller's own record rather than assumed from the capture, which does not
#: stamp a revision of its own.
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

# --------------------------------------------------------------- vocabulary
BASELINE_ARM = "E"

#: Which tensor in the historical capture carries which arm. The capture names
#: its tensors by role, not by arm, and the correspondence is written down once
#: here rather than inferred at each call site.
ARM_TENSOR: Dict[str, str] = {
    "E": "safe_residual_stream",
    "B_d1": "donor_residual_stream",
    "H": "harmful_residual_stream",
}

#: The two directions, each bound to its arm and to the inclusion rule that
#: makes a case eligible for it. Exhaustive: a direction not in this table
#: cannot be built, which is what keeps the B_d2/B_d3 arms out.
DIRECTIONS: Dict[str, Dict[str, str]] = {
    "repair": {"arm": "B_d1", "rule": "comparator_actuator_source"},
    "safety": {"arm": "H", "rule": "h_e_direction_source"},
}

#: The rule this bank must never read. Named so the refusal can say why.
BASELINE_ONLY_RULE = "primary_actuator_source"
BASELINE_ONLY_ARMS: Tuple[str, ...] = ("B_d2", "B_d3")

ELIGIBLE_STATUS = "validated"

# ------------------------------------------------------------- tolerances
#: A delta shorter than this is treated as zero. Two activations that close
#: together carry no direction, and normalising them would amplify noise into
#: a unit vector that looks exactly like a real one.
MIN_DELTA_NORM = 1e-6

#: How far a finished direction may sit from unit norm. Float32 accumulation
#: over 2048 dimensions and ten cases lands well inside this.
UNIT_NORM_TOLERANCE = 1e-5

#: The observed float32 agreement between a bank rebuilt from the capture and
#: the historical frozen d1 comparator, measured 2026-08-22 across layers
#: 31/32/33: cosine 0.999999880791 / 1.000000000000 / 1.000000119209, maximum
#: elementwise difference 2.235e-08 / 1.490e-08 / 1.490e-08. The gates below
#: sit just outside that, so they pass on agreement and fail on a different
#: construction rather than on rounding.
PARITY_MIN_COSINE = 1.0 - 1e-6
PARITY_MAX_ABS_DIFF = 1e-7

#: The frozen variant the repair direction must reproduce.
PARITY_VARIANT = "historical_d1_global_frozen"

#: Fixed archive timestamp for the NPZ, so the same inputs give the same bytes.
#: Zip records a modification time per member; left to the clock it would make
#: two identical banks hash differently.
ZIP_TIMESTAMP: Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)


class DirectionBankError(ValidationError):
    """The direction bank was asked for something it must not produce."""


# ------------------------------------------------------------------ checks
def assert_activation_block(array: Any, label: str) -> np.ndarray:
    """One arm's activations: ``[10, 3, 2048]`` float32, finite.

    Returned as a fresh float32 array. The dtype is checked rather than cast:
    the capture stores float32, and an arm arriving as float16 or bfloat16
    would mean it came from somewhere else.
    """
    if not isinstance(array, np.ndarray):
        raise DirectionBankError(
            f"{label}: expected a numpy array, got {type(array).__name__}")
    if array.dtype != CAPTURE_DTYPE:
        raise DirectionBankError(
            f"{label}: dtype is {array.dtype}, expected {CAPTURE_DTYPE}. The "
            f"capture stores float32; another dtype is a different capture, "
            f"not one to cast.")
    if array.shape != CAPTURE_SHAPE:
        raise DirectionBankError(
            f"{label}: shape is {array.shape}, expected {CAPTURE_SHAPE} "
            f"({N_CASES} cases x {N_LAYERS} layers x {HIDDEN_SIZE})")
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise DirectionBankError(
            f"{label}: {bad} non-finite value(s). A NaN or an infinity in an "
            f"activation is refused, never zeroed or dropped.")
    return np.array(array, dtype=CAPTURE_DTYPE)


def assert_layers_and_states(layers: Sequence[int], states: Sequence[int],
                             label: str = "the capture") -> None:
    """The band is the historical 31/32/33, numbered here as states 32/33/34."""
    if tuple(int(layer) for layer in layers) != HISTORICAL_LAYERS:
        raise DirectionBankError(
            f"{label} covers decoder blocks {list(layers)}; this bank is the "
            f"historical {list(HISTORICAL_LAYERS)} and does not remap them.")
    if tuple(int(state) for state in states) != BANK_STATES:
        raise DirectionBankError(
            f"{label} declares states {list(states)}; blocks "
            f"{list(HISTORICAL_LAYERS)} are states {list(BANK_STATES)} in this "
            f"package's numbering (state = block + 1).")


def assert_subtraction_order(direction: str, arm: str) -> None:
    """``arm - E``, never ``E - arm``, and never a baseline-only arm.

    Both halves matter. A reversed subtraction yields a unit vector of the
    right shape pointing the wrong way, and a B_d2/B_d3 arm yields a direction
    that looks validated and is backed by 3 of 10 cases.
    """
    if direction not in DIRECTIONS:
        raise DirectionBankError(
            f"unknown direction {direction!r}; this bank holds "
            f"{sorted(DIRECTIONS)}")
    expected = DIRECTIONS[direction]["arm"]
    if arm != expected:
        raise DirectionBankError(
            f"{direction} is {expected} - {BASELINE_ARM}; it was given arm "
            f"{arm!r}")
    if arm in BASELINE_ONLY_ARMS:
        raise DirectionBankError(
            f"{arm} belongs to the historical primary actuator, which is "
            f"{BASELINE_ONLY_RULE} and validated for 3 of {N_CASES} cases. It "
            f"is a comparison baseline and does not enter this bank.")


def eligible_case_ids(inclusion: Mapping[str, Any], rule: str) -> Tuple[str, ...]:
    """The case ids a rule marks ``validated``, in the manifest's own order.

    Order comes from ``expected_case_ids`` and not from iterating ``cases``:
    the bank averages ten vectors, and while addition is commutative in
    principle it is not in float32, so a stable order is what makes the bytes
    reproducible.
    """
    if rule == BASELINE_ONLY_RULE:
        raise DirectionBankError(
            f"{BASELINE_ONLY_RULE} is the historical B_d2/B_d3 actuator's "
            f"rule. It is a comparison baseline and is not an eligibility "
            f"source for this bank.")
    expected = inclusion.get("expected_case_ids")
    if not isinstance(expected, list) or not expected:
        raise DirectionBankError(
            "the inclusion manifest carries no expected_case_ids")
    cases = inclusion.get("cases")
    if not isinstance(cases, Mapping):
        raise DirectionBankError("the inclusion manifest carries no cases")

    eligible = []
    for case_id in expected:
        record = cases.get(case_id)
        if not isinstance(record, Mapping):
            raise DirectionBankError(
                f"{case_id} is in expected_case_ids but has no case record")
        rules = record.get("rules")
        if not isinstance(rules, Mapping) or rule not in rules:
            raise DirectionBankError(
                f"{case_id} has no {rule} record; eligibility cannot be "
                f"assumed from its absence")
        status = rules[rule].get("inclusion_status")
        if status == ELIGIBLE_STATUS:
            eligible.append(case_id)
    if len(eligible) != N_CASES:
        raise DirectionBankError(
            f"{rule} marks {len(eligible)} case(s) {ELIGIBLE_STATUS}; this "
            f"bank is built from {N_CASES}. A direction averaged over a "
            f"different number of cases is a different direction and is not "
            f"silently produced.")
    return tuple(eligible)


def eligibility_record_sha256(inclusion: Mapping[str, Any], case_id: str,
                              rule: str) -> str:
    """A digest of the one rule record that made this case eligible.

    The whole manifest is hashed separately as the gate. This is finer: it
    pins the specific decision, so a later manifest that changed only this
    case's status is visibly a different input.
    """
    record = inclusion["cases"][case_id]["rules"][rule]
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False)
        .encode("utf-8")).hexdigest()


# -------------------------------------------------------------- arithmetic
def case_directions(arm_block: np.ndarray, baseline_block: np.ndarray,
                    label: str) -> np.ndarray:
    """Per-case unit deltas: ``unit(arm - E)``, ``[10, 3, 2048]``.

    Each case is normalised on its own, at each layer independently, before
    anything is averaged. A case whose delta is degenerate is refused rather
    than normalised: dividing a near-zero vector by its near-zero norm returns
    a unit vector indistinguishable from a real one.
    """
    arm = assert_activation_block(arm_block, f"{label} arm")
    baseline = assert_activation_block(baseline_block, f"{label} {BASELINE_ARM}")

    delta = arm - baseline
    if not np.isfinite(delta).all():
        raise DirectionBankError(
            f"{label}: the delta holds non-finite values even though both "
            f"activations are finite; refusing rather than repairing")

    norms = np.linalg.norm(delta, axis=-1)
    degenerate = np.argwhere(norms < MIN_DELTA_NORM)
    if degenerate.size:
        first = [(int(c), HISTORICAL_LAYERS[int(l)]) for c, l in degenerate[:3]]
        raise DirectionBankError(
            f"{label}: {len(degenerate)} (case, layer) delta(s) shorter than "
            f"{MIN_DELTA_NORM} -- first (case index, layer) {first}. Two "
            f"activations that close together carry no direction; "
            f"normalising them would turn rounding noise into a unit vector.")
    return (delta / norms[..., None]).astype(CAPTURE_DTYPE)


def aggregate(per_case: np.ndarray, label: str) -> np.ndarray:
    """The mean of the per-case unit directions, renormalised. ``[3, 2048]``."""
    if per_case.shape != CAPTURE_SHAPE:
        raise DirectionBankError(
            f"{label}: per-case directions have shape {per_case.shape}, "
            f"expected {CAPTURE_SHAPE}")
    mean = per_case.mean(axis=0)
    norms = np.linalg.norm(mean, axis=-1)
    if (norms < MIN_DELTA_NORM).any():
        raise DirectionBankError(
            f"{label}: the averaged direction is degenerate at layer(s) "
            f"{[HISTORICAL_LAYERS[i] for i in np.argwhere(norms < MIN_DELTA_NORM).ravel()]}"
            f". The ten case directions cancelled; there is no bank to "
            f"normalise.")
    direction = (mean / norms[..., None]).astype(CAPTURE_DTYPE)
    assert_unit_norm(direction, label)
    return direction


def assert_unit_norm(direction: np.ndarray, label: str) -> Tuple[float, ...]:
    """Every layer of a finished direction is unit length."""
    if direction.shape != (N_LAYERS, HIDDEN_SIZE):
        raise DirectionBankError(
            f"{label}: shape {direction.shape}, expected "
            f"{(N_LAYERS, HIDDEN_SIZE)}")
    norms = tuple(float(n) for n in np.linalg.norm(direction, axis=-1))
    off = [(HISTORICAL_LAYERS[i], n) for i, n in enumerate(norms)
           if abs(n - 1.0) > UNIT_NORM_TOLERANCE]
    if off:
        raise DirectionBankError(
            f"{label}: layer(s) {off} are not unit norm within "
            f"{UNIT_NORM_TOLERANCE}")
    return norms


def build_direction(direction: str, arm_block: np.ndarray,
                    baseline_block: np.ndarray) -> np.ndarray:
    """One finished ``[3, 2048]`` direction, by the one recipe."""
    assert_subtraction_order(direction, DIRECTIONS[direction]["arm"])
    per_case = case_directions(arm_block, baseline_block, direction)
    return aggregate(per_case, direction)


def per_layer_cosine(left: np.ndarray, right: np.ndarray) -> Tuple[float, ...]:
    """Cosine between two ``[3, 2048]`` banks, layer by layer."""
    if left.shape != right.shape:
        raise DirectionBankError(
            f"cannot compare shapes {left.shape} and {right.shape}")
    out = []
    for i in range(left.shape[0]):
        a, b = left[i].astype(np.float64), right[i].astype(np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            raise DirectionBankError(f"layer index {i} has a zero vector")
        out.append(float(np.dot(a, b) / denom))
    return tuple(out)


def parity_against_frozen(rebuilt: np.ndarray, reference: np.ndarray
                          ) -> Dict[str, Any]:
    """Does the rebuilt repair bank reproduce the historical frozen d1?

    It must, and to float32 rounding: the frozen comparator was built from the
    same ten cases, the same capture and the same recipe. A disagreement means
    this build is not the construction RQ2 froze, whatever else it may be, and
    that is a stop rather than a note.
    """
    if reference.shape != (N_LAYERS, HIDDEN_SIZE):
        raise DirectionBankError(
            f"the frozen comparator has shape {reference.shape}, expected "
            f"{(N_LAYERS, HIDDEN_SIZE)}")
    cosines = per_layer_cosine(rebuilt, reference)
    diffs = tuple(float(np.abs(rebuilt[i].astype(np.float64)
                               - reference[i].astype(np.float64)).max())
                  for i in range(N_LAYERS))
    failures = [
        (HISTORICAL_LAYERS[i], cosines[i], diffs[i]) for i in range(N_LAYERS)
        if cosines[i] < PARITY_MIN_COSINE or diffs[i] > PARITY_MAX_ABS_DIFF]
    if failures:
        raise DirectionBankError(
            f"the rebuilt repair direction does not reproduce "
            f"{PARITY_VARIANT} within float32 tolerance (cosine >= "
            f"{PARITY_MIN_COSINE}, max|diff| <= {PARITY_MAX_ABS_DIFF}). "
            f"Failing (layer, cosine, max|diff|): {failures}. The frozen "
            f"comparator was built from these same ten cases by this same "
            f"recipe; a disagreement means this is a different construction.")
    return {
        "variant": PARITY_VARIANT,
        "per_layer_cosine": {str(HISTORICAL_LAYERS[i]): cosines[i]
                             for i in range(N_LAYERS)},
        "per_layer_max_abs_diff": {str(HISTORICAL_LAYERS[i]): diffs[i]
                                   for i in range(N_LAYERS)},
        "min_cosine_required": PARITY_MIN_COSINE,
        "max_abs_diff_allowed": PARITY_MAX_ABS_DIFF,
        "agrees": True,
    }


# ----------------------------------------------------------------- artefact
#: The two arrays the NPZ carries, in the order it stores them.
ARTIFACT_ARRAYS: Tuple[str, ...] = ("repair_direction", "safety_direction")


def _npy_bytes(array: np.ndarray) -> bytes:
    """One array in ``.npy`` format, as bytes. No pickle: version 1.0 only."""
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array),
                              version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """A deterministic, pickle-free NPZ.

    ``numpy.savez`` stamps each member with the current time, so two identical
    banks would hash differently. This writes the same members in a fixed order
    with a fixed timestamp, which is what lets the manifest's digest of this
    file mean anything.
    """
    missing = [name for name in ARTIFACT_ARRAYS if name not in arrays]
    extra = sorted(set(arrays) - set(ARTIFACT_ARRAYS))
    if missing or extra:
        raise DirectionBankError(
            f"the artefact holds exactly {list(ARTIFACT_ARRAYS)}; missing "
            f"{missing}, unexpected {extra}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name in ARTIFACT_ARRAYS:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, _npy_bytes(arrays[name]))
    return buffer.getvalue()


def read_npz(path: Path | str) -> Dict[str, np.ndarray]:
    """Read the artefact back with pickle disabled."""
    with np.load(str(path), allow_pickle=False) as handle:
        return {name: np.array(handle[name]) for name in handle.files}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """A digest of the raw float32 values, independent of any container.

    The NPZ has its own digest. This one survives re-framing: it is what makes
    two artefacts comparable when one was written by a later version of this
    code.
    """
    return sha256_bytes(
        np.ascontiguousarray(array, dtype=CAPTURE_DTYPE).tobytes())


# -------------------------------------------------------------- the capture
def capture_arrays(path: Path | str,
                   arms: Iterable[str] = tuple(ARM_TENSOR)
                   ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """The arm activations and metadata from the historical capture.

    ``torch.load`` is called with ``weights_only=True``, so the file is read as
    tensors and primitives and no code in it can execute. torch is imported
    lazily: everything above this point is numpy, and reading a bank should not
    require a deep-learning runtime.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is present in-container
        raise DirectionBankError(
            f"reading the capture needs torch, which failed to import: {exc}"
        ) from exc

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise DirectionBankError(
            f"{path}: expected a dict of tensors, got {type(payload).__name__}")

    out: Dict[str, np.ndarray] = {}
    for arm in arms:
        if arm not in ARM_TENSOR:
            raise DirectionBankError(
                f"unknown arm {arm!r}; the capture carries {sorted(ARM_TENSOR)}")
        key = ARM_TENSOR[arm]
        if key not in payload:
            raise DirectionBankError(
                f"{path}: no tensor {key!r} for arm {arm}. A missing arm is "
                f"refused; no other tensor is substituted for it.")
        tensor = payload[key]
        if not torch.is_tensor(tensor):
            raise DirectionBankError(
                f"{path}: {key!r} is {type(tensor).__name__}, not a tensor")
        out[arm] = assert_activation_block(
            tensor.detach().cpu().numpy(), f"{path.name if isinstance(path, Path) else path} {key}")

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise DirectionBankError(f"{path}: the capture carries no meta record")
    return out, dict(meta)


@dataclass(frozen=True)
class DirectionBank:
    """Two unit directions over three layers, and how they were made."""

    repair: np.ndarray                       # [3, 2048] float32
    safety: np.ndarray                       # [3, 2048] float32
    case_ids: Dict[str, Tuple[str, ...]]     # direction -> the ten used
    layers: Tuple[int, ...] = HISTORICAL_LAYERS
    states: Tuple[int, ...] = BANK_STATES
    checklist: Checklist = field(
        default_factory=lambda: Checklist("validated_direction_bank"))

    def arrays(self) -> Dict[str, np.ndarray]:
        return {"repair_direction": self.repair,
                "safety_direction": self.safety}

    def norms(self) -> Dict[str, Tuple[float, ...]]:
        return {"repair": assert_unit_norm(self.repair, "repair"),
                "safety": assert_unit_norm(self.safety, "safety")}

    def cosine(self) -> Tuple[float, ...]:
        """Repair against safety, layer by layer. A diagnostic, not a step.

        Nothing downstream reads this to alter either direction; it is measured
        so the relationship between them is on the record before anyone decides
        whether an orthogonalised ablation is worth running.
        """
        return per_layer_cosine(self.repair, self.safety)


def build_bank(activations: Mapping[str, np.ndarray],
               case_ids: Mapping[str, Sequence[str]],
               checks: Optional[Checklist] = None) -> DirectionBank:
    """Both directions from one capture, by the one recipe."""
    checks = checks or Checklist("validated_direction_bank")
    if BASELINE_ARM not in activations:
        raise DirectionBankError(
            f"the {BASELINE_ARM} activations are the baseline of both "
            f"directions and are absent")
    built: Dict[str, np.ndarray] = {}
    for direction in sorted(DIRECTIONS):
        arm = DIRECTIONS[direction]["arm"]
        if arm not in activations:
            raise DirectionBankError(
                f"{direction} needs the {arm} activations and they are absent")
        ids = tuple(case_ids[direction])
        if len(ids) != N_CASES or len(set(ids)) != N_CASES:
            raise DirectionBankError(
                f"{direction}: {len(ids)} case id(s), {len(set(ids))} "
                f"distinct; expected {N_CASES} distinct")
        built[direction] = build_direction(
            direction, activations[arm], activations[BASELINE_ARM])
        checks.record(f"{direction}_direction", True,
                      f"{arm} - {BASELINE_ARM} over {len(ids)} validated "
                      f"case(s), per-case unit, averaged, renormalised")
    return DirectionBank(repair=built["repair"], safety=built["safety"],
                         case_ids={d: tuple(case_ids[d]) for d in DIRECTIONS},
                         checklist=checks)
