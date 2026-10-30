"""The deployable coefficient controller: checkpoints, and the live edit path.

WHY THIS EXISTS
---------------
``gate_oof_v1`` produced 480 out-of-fold coefficients and **no weights**.
Its
own manifest says ``final_refit_performed: false``, and its
``output_hashes.json`` lists seven files, none of them a checkpoint. A
column of per-row coefficients is not a controller: using it at generation
time would be a row-id lookup, which is not inference and which cannot be
applied to a prompt the table has never seen.

So this module defines what a *deployable* controller is -- weights, the
standardisation they were fitted with, and the veto threshold derived from the
same outer-training rows -- and the live path that applies one in a single
forward pass.

WHAT A CHECKPOINT MUST CARRY
----------------------------
A controller is not its weights alone. The input is per-fold standardised, so
``mu`` and ``sd`` are part of the function; the safety brake is derived from
outer-training rows, so ``tau_hi`` is too. A checkpoint missing any of them
would silently compute a different ``g_R`` from the same activation.

None of it is sensitive: weights, two 2048-vectors and a scalar. No prompt, no
response, no per-row coefficient and no corpus row id is stored, and
``assert_checkpoint_is_metadata_only`` refuses one that tries.

THE FOLD RULE, ENFORCED
-----------------------
Five fold controllers, each trained on the four folds it was not held out on.
A row is scored by the controller that never saw it -- ``controller_for_row``
raises rather than guess -- so the development comparison stays out-of-fold.
The final all-development refit is a **deployment** artifact and carries
``is_out_of_fold: false``; asking it to score a development row is refused.

THE LIVE EDIT
-------------
One forward pass, read before write:

1. read the unsteered activation at package state 32 -- the output of
   historical block 31 -- at the final prompt token;
2. compute ``g_R = sigmoid(f_theta((h - mu) / sd))`` once;
3. apply ``0.15 * g_R * r_l * d_R`` at historical blocks 31 and 32, reusing
   **the same** ``g_R`` at both sites.

The read happens before either write, so the controller input is the unsteered
state and not something an earlier edit produced. Historical block 33 is not a
site: an edit at 31 changes the activation entering 33, so 33's unsteered
budget does not exist in the same pass.

Nothing here is differentiable. The language model and both probes are frozen
and never see a gradient; this module never calls ``backward``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .gatecontroller import (
    ALPHA_R,
    CONTROLLER_PARAMETERS,
    N_SPLITS,
    READ_STATE,
    build_controller,
    seed_for,
)
from .triplets import sha256_text
from .validation import ValidationError

CHECKPOINT_SCHEMA = "routegate/gate_controller_checkpoint/1"

#: The historical blocks the same coefficient is applied at, in order.
APPLY_BLOCKS: Tuple[int, ...] = (31, 32)

#: The fields a checkpoint carries. Anything else is refused.
CHECKPOINT_FIELDS: Tuple[str, ...] = (
    "schema", "kind", "fold", "is_out_of_fold", "read_state",
    "apply_historical_blocks", "alpha_R", "parameters", "seed",
    "state_dict", "mu", "sd", "tau_hi", "selected_step", "trained_on_rows",
    "contains_prompt_or_response_text", "contains_per_row_coefficients",
    "weights_sha256",
)

#: Key fragments that would mean a checkpoint had picked up something it must
#: never carry.
FORBIDDEN_FRAGMENTS: Tuple[str, ...] = (
    "prompt", "response", "text", "row_id", "coefficient", "g_r", "screen_row",
    "completion", "answer")

KIND_FOLD = "fold"
KIND_REFIT = "all_development_refit"


class LiveControllerError(ValidationError):
    """The live controller path was asked for something it must not do."""


# ================================================================ checkpoints
def assert_checkpoint_is_metadata_only(payload: Mapping[str, Any]) -> None:
    """A checkpoint carries weights and calibration, and nothing sensitive."""
    extra = sorted(set(payload) - set(CHECKPOINT_FIELDS))
    if extra:
        raise LiveControllerError(
            f"checkpoint carries {extra}; a checkpoint carries exactly "
            f"{list(CHECKPOINT_FIELDS)}")
    for name in payload.get("state_dict", {}):
        low = str(name).lower()
        if any(fragment in low for fragment in FORBIDDEN_FRAGMENTS):
            raise LiveControllerError(
                f"state_dict key {name!r} names something a checkpoint never "
                f"carries")
    if payload.get("contains_prompt_or_response_text") is not False:
        raise LiveControllerError(
            "a checkpoint must state contains_prompt_or_response_text: false")
    if payload.get("contains_per_row_coefficients") is not False:
        raise LiveControllerError(
            "a checkpoint must state contains_per_row_coefficients: false. "
            "Per-row coefficients are not a controller and are never shipped "
            "as one.")


def weights_digest(state_dict: Mapping[str, Any]) -> str:
    """A digest over the weights alone, in sorted key order."""
    body = {str(k): np.asarray(v, dtype=np.float64).round(12).tolist()
            for k, v in sorted(state_dict.items())}
    return sha256_text(json.dumps(body, sort_keys=True,
                                  separators=(",", ":")))


def build_checkpoint(kind: str, fold: Optional[int], state_dict: Mapping,
                     mu: Sequence[float], sd: Sequence[float],
                     tau_hi: float, selected_step: int,
                     trained_on_rows: int) -> Dict[str, Any]:
    """One controller, in the only shape this package will load."""
    if kind not in (KIND_FOLD, KIND_REFIT):
        raise LiveControllerError(f"unknown checkpoint kind {kind!r}")
    if kind == KIND_FOLD and fold is None:
        raise LiveControllerError("a fold checkpoint must name its fold")
    plain = {str(k): np.asarray(v, dtype=np.float64).tolist()
             for k, v in state_dict.items()}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "kind": kind,
        "fold": fold,
        # A fold controller never saw its own held-out fold, so it may score
        # it. The refit saw every development row and may not score any.
        "is_out_of_fold": kind == KIND_FOLD,
        "read_state": READ_STATE,
        "apply_historical_blocks": list(APPLY_BLOCKS),
        "alpha_R": ALPHA_R,
        "parameters": CONTROLLER_PARAMETERS,
        "seed": seed_for(fold) if fold is not None else None,
        "state_dict": plain,
        "mu": [float(v) for v in mu],
        "sd": [float(v) for v in sd],
        "tau_hi": float(tau_hi),
        "selected_step": int(selected_step),
        "trained_on_rows": int(trained_on_rows),
        "contains_prompt_or_response_text": False,
        "contains_per_row_coefficients": False,
    }
    payload["weights_sha256"] = weights_digest(plain)
    assert_checkpoint_is_metadata_only(payload)
    return payload


def load_checkpoint(path: Path | str) -> Dict[str, Any]:
    """Read one checkpoint and refuse it unless it verifies."""
    path = Path(path)
    if not path.is_file():
        raise LiveControllerError(f"{path} is not a file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise LiveControllerError(
            f"{path.name} declares schema {payload.get('schema')!r}, not "
            f"{CHECKPOINT_SCHEMA!r}")
    assert_checkpoint_is_metadata_only(payload)
    recorded = payload.get("weights_sha256")
    derived = weights_digest(payload["state_dict"])
    if recorded != derived:
        raise LiveControllerError(
            f"{path.name} records weights_sha256 {recorded} and hashes to "
            f"{derived}. A checkpoint that does not verify has been edited.")
    if int(payload["read_state"]) != READ_STATE:
        raise LiveControllerError(
            f"{path.name} reads state {payload['read_state']}, and this "
            f"controller reads state {READ_STATE}")
    if tuple(payload["apply_historical_blocks"]) != APPLY_BLOCKS:
        raise LiveControllerError(
            f"{path.name} applies at {payload['apply_historical_blocks']}, "
            f"and this actuator applies at {list(APPLY_BLOCKS)}")
    if len(payload["mu"]) != len(payload["sd"]):
        raise LiveControllerError(f"{path.name}: mu and sd differ in length")
    return payload


@dataclass(frozen=True)
class ControllerBundle:
    """Every fold controller, plus the deployment refit if one exists."""

    folds: Mapping[int, Mapping[str, Any]]
    refit: Optional[Mapping[str, Any]] = None

    def controller_for_row(self, outer_fold: int) -> Mapping[str, Any]:
        """The controller that did **not** train on this row's fold.

        Controller ``k`` is held out on fold ``k``, so a row in fold ``k`` is
        scored by controller ``k`` and by no other. Asking for anything else
        raises: an in-fold score is not an out-of-fold score, and the two must
        never be mixed in one comparison.
        """
        if outer_fold not in self.folds:
            raise LiveControllerError(
                f"no controller for outer fold {outer_fold}; this bundle "
                f"holds {sorted(self.folds)}")
        checkpoint = self.folds[outer_fold]
        if not checkpoint.get("is_out_of_fold"):
            raise LiveControllerError(
                f"the controller for fold {outer_fold} is not out-of-fold; "
                f"the all-development refit never scores a development row")
        return checkpoint

    def deployment_controller(self) -> Mapping[str, Any]:
        """The all-development refit, for a prompt outside development."""
        if self.refit is None:
            raise LiveControllerError(
                "this bundle carries no all-development refit; gate_oof_v1 "
                "performed none, and a deployment needs one")
        return self.refit


def load_bundle(directory: Path | str) -> ControllerBundle:
    """Load five fold checkpoints and, if present, the deployment refit."""
    directory = Path(directory)
    folds: Dict[int, Mapping[str, Any]] = {}
    for k in range(N_SPLITS):
        path = directory / f"controller_fold_{k}.json"
        if not path.is_file():
            raise LiveControllerError(
                f"{path} is absent; the bundle needs all {N_SPLITS} fold "
                f"controllers, because a row is scored only by the one that "
                f"never saw it")
        checkpoint = load_checkpoint(path)
        if checkpoint["kind"] != KIND_FOLD or checkpoint["fold"] != k:
            raise LiveControllerError(
                f"{path.name} is not the fold-{k} controller")
        folds[k] = checkpoint
    refit_path = directory / "controller_all_development_refit.json"
    refit = load_checkpoint(refit_path) if refit_path.is_file() else None
    if refit is not None and refit["kind"] != KIND_REFIT:
        raise LiveControllerError(
            f"{refit_path.name} is not the all-development refit")
    return ControllerBundle(folds=folds, refit=refit)


# =================================================================== the gate
def standardise(activation: np.ndarray, checkpoint: Mapping[str, Any]
                ) -> np.ndarray:
    """``(h - mu) / sd`` with the checkpoint's own fitted statistics."""
    mu = np.asarray(checkpoint["mu"], dtype=np.float64)
    sd = np.asarray(checkpoint["sd"], dtype=np.float64)
    activation = np.asarray(activation, dtype=np.float64)
    if activation.shape[-1] != mu.shape[0]:
        raise LiveControllerError(
            f"activation is {activation.shape[-1]}-wide and this controller "
            f"was standardised on {mu.shape[0]}")
    return (activation - mu) / sd


def gate_from_activation(activation: np.ndarray,
                         checkpoint: Mapping[str, Any]) -> np.ndarray:
    """``g_R`` for one or more unsteered state-32 activations. No gradient.

    Runs the frozen module itself, in ``eval`` mode and under ``no_grad``, in
    float32 -- the arithmetic training used. Re-deriving the forward pass in
    another dtype or another GELU form would reproduce the coefficients only
    approximately, and the whole point of this path is that it reproduces them
    exactly.

    ``no_grad`` is what makes this non-differentiable: no graph is built, so
    nothing downstream can call ``backward`` through the controller and reach
    the language model or a probe.
    """
    import torch

    x = standardise(activation, checkpoint)
    net = build_controller(int(checkpoint["seed"] or 0))
    net.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32)
                         for k, v in checkpoint["state_dict"].items()})
    net.eval()
    with torch.no_grad():
        out = net(torch.as_tensor(np.asarray(x), dtype=torch.float32))
        g = torch.sigmoid(out).reshape(-1)
    return g.numpy().astype(np.float64)


def applied_coefficient(g_r: np.ndarray) -> np.ndarray:
    """``alpha_R * g_R``. The ceiling is arithmetic, not a clamp."""
    return ALPHA_R * np.asarray(g_r, dtype=np.float64)


def apply_hard_veto(g_r: np.ndarray, z_safety: np.ndarray,
                    tau_hi: float) -> np.ndarray:
    """Zero the coefficient where the frozen safety veto fires.

    The veto can only lower a gate, never raise one, and it is not trained.
    This is the *same* ``g_R``; the two dynamic arms share one controller.
    """
    g_r = np.asarray(g_r, dtype=np.float64)
    z_safety = np.asarray(z_safety, dtype=np.float64)
    return np.where(z_safety >= tau_hi, 0.0, g_r)


# ================================================================== live edit
@dataclass(frozen=True)
class LivePlan:
    """What the live path will do for one row, before any tensor is written."""

    row_fold: int
    g_r: float
    applied_alpha: float
    apply_at_historical_blocks: Tuple[int, ...]
    read_state: int
    same_coefficient_at_every_site: bool
    vetoed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"row_fold": self.row_fold, "g_R": self.g_r,
                "applied_alpha_R": self.applied_alpha,
                "apply_at_historical_blocks":
                    list(self.apply_at_historical_blocks),
                "read_state": self.read_state,
                "same_coefficient_at_every_site":
                    self.same_coefficient_at_every_site,
                "vetoed": self.vetoed}


def plan_live_edit(activation: np.ndarray, outer_fold: int,
                   bundle: ControllerBundle, z_safety: Optional[float] = None,
                   hard_veto: bool = False) -> LivePlan:
    """Decide the coefficient for one row. Reads; writes nothing.

    ``activation`` is the **unsteered** state-32 final-prompt-token vector,
    read before either site is written. One ``g_R`` is computed and reused at
    both historical blocks; there is no second controller call and no second
    coefficient.
    """
    checkpoint = bundle.controller_for_row(outer_fold)
    g = float(gate_from_activation(np.asarray(activation).reshape(1, -1),
                                   checkpoint)[0])
    vetoed = False
    if hard_veto:
        if z_safety is None:
            raise LiveControllerError(
                "the hard veto needs the unsteered safety logit; without it "
                "the brake cannot be evaluated and is not assumed to pass")
        gated = float(apply_hard_veto(np.array([g]), np.array([z_safety]),
                                      float(checkpoint["tau_hi"]))[0])
        vetoed = gated != g
        g = gated
    return LivePlan(row_fold=outer_fold, g_r=g,
                    applied_alpha=float(ALPHA_R * g),
                    apply_at_historical_blocks=APPLY_BLOCKS,
                    read_state=READ_STATE,
                    same_coefficient_at_every_site=True,
                    vetoed=vetoed)


def assert_no_parameter_requires_grad(modules: Iterable[Any]) -> None:
    """Neither the language model nor a probe may receive a gradient.

    The live path never calls ``backward``, so this is belt and braces: it
    proves the tensors are not even *able* to accumulate one.
    """
    offending = []
    for module in modules:
        params = getattr(module, "parameters", None)
        if params is None:
            continue
        for name, parameter in getattr(module, "named_parameters")():
            if getattr(parameter, "requires_grad", False):
                offending.append(f"{type(module).__name__}.{name}")
    if offending:
        raise LiveControllerError(
            f"{len(offending)} parameter(s) would receive a gradient, first "
            f"{offending[:3]}. The language model and both probes are frozen; "
            f"only the controller was ever trained, and not here.")


__all__ = [
    "APPLY_BLOCKS", "CHECKPOINT_FIELDS", "CHECKPOINT_SCHEMA",
    "ControllerBundle", "KIND_FOLD", "KIND_REFIT", "LiveControllerError",
    "LivePlan", "applied_coefficient", "apply_hard_veto",
    "assert_checkpoint_is_metadata_only", "assert_no_parameter_requires_grad",
    "build_checkpoint", "gate_from_activation", "load_bundle",
    "load_checkpoint", "plan_live_edit", "standardise", "weights_digest",
]
