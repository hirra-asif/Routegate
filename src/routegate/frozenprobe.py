"""The frozen probe heads, reconstructed exactly and made differentiable.

The completed probe run stores its out-of-fold probabilities and its metrics.
It does **not** store the fitted parameters: ``probeheads.PREDICTION_COLUMNS``
is ``(row_id, final_ebh, fold, state, p_E, p_H)``, and ``probes.FoldOutcome``
holds ``coef``, ``intercept``, ``scaler_mean`` and ``scaler_scale`` in memory
only. A differentiable probe therefore has to be rebuilt.

Rebuilding is legitimate here, and only here, because every input to the fit is
pinned and the estimator is deterministic: the corpus, the activation capture
and the extraction manifest are hash-gated; the fold division is the frozen
one; and the estimator is ``l2, C=0.1, class_weight=balanced, max_iter=5000,
solver=lbfgs, random_state=20260805, tol=1e-6``. Nothing is re-chosen. The
reconstruction is checked against the saved probabilities to 1e-12 and refuses
to proceed otherwise -- so this is a *reproduction*, and a failure to reproduce
is a failure of the whole stage rather than something to work around.

The differentiable form
-----------------------
Each head is ``StandardScaler`` then ``LogisticRegression``, so its logit is
affine in the activation::

    z(h) = w . ((h - mu) / sigma) + b

Written in torch with ``w``, ``mu``, ``sigma`` and ``b`` all
``requires_grad=False``, that is differentiable with respect to ``h`` and to
nothing else. Gradients reach the controller through the steered activation and
stop at the probe's parameters, which is exactly the required path.

Why logits, never probabilities
-------------------------------
The probe run fitted no calibration model -- ``metrics.json`` records
``calibration.model_fitted: false`` and states that the Brier score measures
the emitted probabilities rather than transforming them. So ``sigma(z)`` is a
monotone reparameterisation of a decision score, not a posterior. Every loss
and every threshold in this stage operates on ``z``, so nothing here assumes a
calibration that was never fitted.

What this module does not do
----------------------------
It loads no language model, generates nothing, judges nothing and trains
nothing. It reconstructs two frozen linear readers and hands them over as
tensors.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from threadpoolctl import threadpool_limits

from .ebh import EBHFoldAssignment, EBHFrame
from .probeheads import (
    HEAD_ORDER,
    PRIMARY_READ_STATE,
    REPAIR_HEAD,
    SAFETY_HEAD,
    head_specification,
    head_targets,
    state_matrix,
)
from .probes import make_probe, probe_specification
from .validation import Checklist, ValidationError

SCHEMA = "routegate/frozen_probe_reconstruction/1"

#: The committed tolerance. A reconstruction that does not reproduce the saved
#: probabilities to this is not the probe behind the committed metrics, and the
#: stage stops rather than adopting it.
REPRODUCTION_TOLERANCE = 1e-12

#: The saved run this is reproduced against.
PROBE_RUN_DIRECTORY = "outputs/dual_direction_controller/probe_heads_v1"
PROBE_PREDICTIONS = "oof_predictions.csv"
PROBE_ARTIFACTS: Dict[str, str] = {
    "oof_predictions.csv":
        "6923919c99b17dcd840202e7eda63a83fc5e98bee26b660ea9002e9a5eafd278",
    "metrics.json":
        "9557be27b97a65532da6d76a8ed7f28e8bbd141517b60bedf7f51fe9b7e65744",
    "validation.json":
        "45eef137f572fb12fb6f3994b2a8776c96ae14ba56c0daef1637a0c3a9ad243e",
    "fold_assignments.csv":
        "3692fd99cca4521e88176c5fa3e09253b27e9ae80251d92f4d8af8b65739fc6f",
}

SCORE_COLUMN = {REPAIR_HEAD: "p_E", SAFETY_HEAD: "p_H"}


class FrozenProbeError(ValidationError):
    """A frozen probe could not be reproduced, or was asked to move."""


@dataclass(frozen=True)
class FrozenProbe:
    """One head, one fold: an affine reader with no trainable parameter.

    ``weight``, ``mean``, ``scale`` and ``intercept`` are the fitted values.
    They are held as float64 arrays and converted to tensors on demand, and
    every tensor this produces carries ``requires_grad=False``.
    """

    head: str
    fold: int
    state: int
    weight: np.ndarray                  # [hidden]
    mean: np.ndarray                    # [hidden]
    scale: np.ndarray                   # [hidden]
    intercept: float
    n_train: int

    def logit_numpy(self, activation: np.ndarray) -> np.ndarray:
        """``z(h)``, for checking the reconstruction without torch."""
        h = np.atleast_2d(np.asarray(activation, dtype=np.float64))
        return ((h - self.mean) / self.scale) @ self.weight + self.intercept

    def as_tensors(self, device: Any = None, dtype: Any = None) -> Dict[str, Any]:
        """The same parameters as frozen tensors, for the differentiable path."""
        import torch

        dtype = dtype or torch.float32
        out = {}
        for name, value in (("weight", self.weight), ("mean", self.mean),
                            ("scale", self.scale)):
            tensor = torch.as_tensor(np.ascontiguousarray(value), dtype=dtype)
            if device is not None:
                tensor = tensor.to(device)
            tensor.requires_grad_(False)
            out[name] = tensor
        bias = torch.as_tensor(float(self.intercept), dtype=dtype)
        if device is not None:
            bias = bias.to(device)
        bias.requires_grad_(False)
        out["intercept"] = bias
        return out

    def kappa(self, direction: np.ndarray) -> float:
        """``<w, d / sigma>``: how one unit of ``d`` moves this logit.

        The whole design turns on the sign of this for the repair head. It is a
        scalar, independent of the row, because both the scaler and the
        classifier are affine.
        """
        d = np.asarray(direction, dtype=np.float64)
        if d.shape != self.weight.shape:
            raise FrozenProbeError(
                f"the direction has shape {d.shape} and this probe reads "
                f"{self.weight.shape}")
        return float(np.dot(self.weight, d / self.scale))


class TorchProbe:
    """A frozen probe as a differentiable function of the activation only."""

    def __init__(self, probe: FrozenProbe, device: Any = None,
                 dtype: Any = None) -> None:
        self.probe = probe
        tensors = probe.as_tensors(device, dtype)
        self.weight = tensors["weight"]
        self.mean = tensors["mean"]
        self.scale = tensors["scale"]
        self.intercept = tensors["intercept"]

    @property
    def parameters_require_grad(self) -> bool:
        return any(t.requires_grad for t in (self.weight, self.mean,
                                             self.scale, self.intercept))

    def logit(self, activation: Any) -> Any:
        """``z(h)``, differentiable with respect to ``h`` and nothing else."""
        scaled = (activation - self.mean) / self.scale
        return scaled @ self.weight + self.intercept


# ------------------------------------------------------------ reconstruction
def reconstruct_fold_probe(spec_name: str, frame: EBHFrame,
                           assignment: EBHFoldAssignment, state: int,
                           fold: int) -> FrozenProbe:
    """Refit one head on one fold's training rows, exactly as the run did.

    The estimator, the rows and the fold all come from the frozen sources. The
    only thing this adds is keeping the fitted parameters, which the original
    run computed and did not persist.
    """
    spec = head_specification(spec_name)
    design = state_matrix(frame, state)
    targets = head_targets(spec, frame.labels)
    training = assignment.training_positions(frame, fold)
    index = np.sort(np.concatenate(
        [training[label] for label in spec.training_labels]).astype(int))
    if index.size == 0:
        raise FrozenProbeError(
            f"{spec_name} head, fold {fold}: no training rows")

    # BLAS threading changes the order of a reduction, and this
    # reconstruction is gated at 1e-12 against the saved probabilities. On a
    # 36-core node the unpinned order drifts to ~1.6e-12 and the gate fails --
    # correctly, because a different reduction order is a different
    # computation. Pinning to one thread restores the order the saved run was
    # produced under. It changes no estimator, no row, no fold and no
    # hyperparameter: only how many partial sums the BLAS splits the dot
    # products into.
    pipeline = make_probe()
    with threadpool_limits(limits=1):
        pipeline.fit(design[index], targets[index])
    classifier = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]
    return FrozenProbe(
        head=spec_name, fold=int(fold), state=int(state),
        weight=np.array(classifier.coef_[0], dtype=np.float64),
        mean=np.array(scaler.mean_, dtype=np.float64),
        scale=np.array(scaler.scale_, dtype=np.float64),
        intercept=float(classifier.intercept_[0]),
        n_train=int(index.size))


def read_saved_predictions(path: Path | str, state: int = PRIMARY_READ_STATE
                           ) -> Dict[str, Dict[str, float]]:
    """The saved out-of-fold probabilities at one state, by row id."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["state"]) == state]
    if not rows:
        raise FrozenProbeError(f"{path} holds no rows at state {state}")
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["row_id"] in out:
            raise FrozenProbeError(
                f"{path} lists {row['row_id']} more than once at state {state}")
        out[row["row_id"]] = {"fold": int(row["fold"]),
                              "p_E": float(row["p_E"]),
                              "p_H": float(row["p_H"])}
    return out


def reconstruct_and_verify(frame: EBHFrame, assignment: EBHFoldAssignment,
                           saved: Mapping[str, Mapping[str, float]],
                           state: int = PRIMARY_READ_STATE,
                           tolerance: float = REPRODUCTION_TOLERANCE,
                           checks: Optional[Checklist] = None
                           ) -> Dict[str, Dict[int, FrozenProbe]]:
    """Reconstruct both heads on every fold and prove they reproduce.

    Fail-closed: every head, every fold, every row. A reconstruction that is
    close but not within ``tolerance`` is refused, because a probe that is
    nearly the committed one is not the committed one and every number
    downstream would inherit the difference.
    """
    checks = checks or Checklist("frozen_probe_reconstruction")
    spec = probe_specification()
    checks.record(
        "frozen_estimator", True,
        " ".join(f"{k}={spec[k]}" for k in ("penalty", "C", "class_weight",
                                            "max_iter", "solver",
                                            "random_state", "tol")))

    probes: Dict[str, Dict[int, FrozenProbe]] = {name: {} for name in HEAD_ORDER}
    errors: Dict[str, List[float]] = {name: [] for name in HEAD_ORDER}
    design = state_matrix(frame, state)

    for name in HEAD_ORDER:
        column = SCORE_COLUMN[name]
        for fold in range(assignment.n_splits):
            probe = reconstruct_fold_probe(name, frame, assignment, state, fold)
            probes[name][fold] = probe
            held_out = [i for i, rid in enumerate(frame.row_ids)
                        if saved[rid]["fold"] == fold]
            if not held_out:
                raise FrozenProbeError(
                    f"{name} head: fold {fold} holds out no row")
            z = probe.logit_numpy(design[held_out])
            reproduced = 1.0 / (1.0 + np.exp(-z))
            expected = np.array([saved[frame.row_ids[i]][column]
                                 for i in held_out], dtype=np.float64)
            error = np.abs(reproduced - expected)
            errors[name].append(float(error.max()))
            checks.require(
                f"{name}_fold_{fold}_reproduces",
                float(error.max()) <= tolerance,
                f"{name} head, fold {fold}: max |reproduced - saved| = "
                f"{error.max():.6e} over {len(held_out)} held-out row(s), "
                f"tolerance {tolerance:g}. The reconstruction is not the probe "
                f"behind the committed metrics; it is refused, not adopted.")
        checks.record(
            f"{name}_reproduces", True,
            f"max |error| {max(errors[name]):.6e} over "
            f"{assignment.n_splits} fold(s), tolerance {tolerance:g}")
    return probes


def assert_repair_direction_reduces_the_repair_logit(
        probes: Mapping[str, Mapping[int, FrozenProbe]],
        direction: np.ndarray,
        checks: Optional[Checklist] = None) -> Dict[str, Dict[int, float]]:
    """The mandatory sign gate, and the safety diagnostic beside it.

    ``kappa_R < 0`` is required in every fold: the repair loss is minimised by
    *lowering* the E-positive logit, so a non-negative kappa would mean the
    objective is minimised at ``g_R = 0`` and the whole design is inverted.

    ``kappa_H`` is reported and **not** gated. A negative value means steering
    reduces the H-positive logit, which the one-sided safety loss then
    penalises; a non-negative value means the direction preserves or increases
    that logit, which is a pass for the safety term rather than a fault. The
    direction is never altered to make either come out a particular way.
    """
    checks = checks or Checklist("frozen_probe_reconstruction")
    out: Dict[str, Dict[int, float]] = {}
    for name in HEAD_ORDER:
        out[name] = {fold: probe.kappa(direction)
                     for fold, probe in sorted(probes[name].items())}
    negative = [f for f, v in out[REPAIR_HEAD].items() if v < 0.0]
    checks.require(
        "kappa_R_is_negative_in_every_fold",
        len(negative) == len(out[REPAIR_HEAD]),
        f"kappa_R = {[round(v, 6) for v in out[REPAIR_HEAD].values()]}; it must "
        f"be negative in every fold, because the repair loss is minimised by "
        f"lowering the E-positive logit. A non-negative value means the edit "
        f"raises it and the objective is inverted.")
    kappa_h = out[SAFETY_HEAD]
    checks.record(
        "kappa_H_recorded_not_gated", True,
        f"kappa_H = {[round(v, 6) for v in kappa_h.values()]}; negative in "
        f"{sum(1 for v in kappa_h.values() if v < 0.0)} of {len(kappa_h)} "
        f"fold(s). Reported as a diagnostic: a non-negative value passes the "
        f"safety term rather than failing it, and d_R is never altered either "
        f"way.")
    return out
