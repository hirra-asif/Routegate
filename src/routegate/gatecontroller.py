"""The single-direction coefficient controller: architecture, losses, guards.

One sigmoid gate per prompt, read from the state-32 final-prompt-token
activation, scaling the one validated repair direction::

    g_R(x) = sigmoid(f_theta(h_state32))          in [0, 1]
    alpha_R = 0.15 * g_R(x)                        so alpha_R <= 0.15 always
    delta_R,l = 0.15 * g_R(x) * r_l(x) * d_R,l     l = historical 31, 32

``r_l(x)`` is the validated budget: the target's own residual norm at the
block's *input* state, read from the unsteered activation and therefore
constant with respect to ``g_R``. ``d_S = H - E`` is retired and is never
applied; ``alpha_S`` is zero permanently and no coefficient scales it.

Every constant here is transcribed from
``configs/single_direction_controller_training_v1.json``, the predeclaration
frozen before this stage ran. Nothing is tuned, and no value was chosen after
seeing an outcome.

The gradient path
-----------------
    frozen-probe loss -> steered activation -> g_R -> controller parameters

The steered activation is constructed explicitly as a tensor and the frozen
probes are run forward on it. The affine shortcut below is exact -- both probes
are linear readers and the edit is a fixed vector times a scalar -- but it is
used only as a test oracle, never as the training path, so the code visibly
applies ``d_R`` rather than asserting its consequence.

Only ``theta`` is trainable. The language model is not loaded at all, both
probes carry ``requires_grad=False``, and ``d_R`` and the budget are frozen
tensors.

Veto and clamp
--------------
Both are inference-time functions around the *same* trained ``g_R``. Neither is
trained, and neither can raise a gate: ``apply_safety_veto`` and
``apply_counterfactual_clamp`` are multiplications by a factor in [0, 1], which
is a property the tests check rather than a convention the code hopes for.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .onepassbudget import block_input_state, block_output_state
from .twosite import CANDIDATE_BLOCKS
from .validation import Checklist, ValidationError

SCHEMA = "routegate/single_direction_controller_training/1"
TRAINING_SPEC_VERSION = 1
TRAINING_SPEC_NAME = "single_direction_controller_training_v1"

#: This stage runs before the two-site actuator has a behavioural adjudication,
#: so nothing it produces is a validated operating point.
PROVISIONAL_STATUS = "PROVISIONAL_PENDING_ACTUATOR_VALIDATION"
PROVISIONAL_REASON = (
    "historical blocks 31-32 have not completed behavioural adjudication. The "
    "coefficients this stage produces are conditional on an actuator whose "
    "causal effect is not yet measured; no operating point, threshold or "
    "deployment claim follows from them.")

# ============================================================ the architecture
CONTROLLER_INPUT = "state-32 final-prompt-token activation, per-fold standardised"
READ_STATE = 32
READ_BLOCK = 31
HIDDEN_SIZE = 2048
CONTROLLER_HIDDEN = 16
CONTROLLER_DEPTH = 1
CONTROLLER_ACTIVATION = "GELU"
CONTROLLER_DROPOUT = 0.1
#: 2048*16 + 16 + 16 + 1
CONTROLLER_PARAMETERS = HIDDEN_SIZE * CONTROLLER_HIDDEN + CONTROLLER_HIDDEN \
    + CONTROLLER_HIDDEN + 1
GATE_INITIAL = 0.1
GATE_BIAS_INIT = math.log(GATE_INITIAL / (1.0 - GATE_INITIAL))

ALPHA_R = 0.15
MAX_APPLIED_ALPHA_R = ALPHA_R
ALPHA_S = 0.0
GATE_MINIMUM, GATE_MAXIMUM = 0.0, 1.0

# ================================================================= the losses
LAMBDA_REPAIR = 1.0
LAMBDA_SAFETY = 4.0
LAMBDA_SIZE = 0.25

#: Both weights are predeclared postures, not estimates. lambda_S carries the
#: asymmetry of the frozen phase-2 rule -- recover at least half, cause exactly
#: none -- into a differentiable objective. lambda_G makes a saturated gate cost
#: 0.25 nats, so the controller must buy that much repair to justify it.
LOSS_WEIGHTS_ARE_PREDECLARED = True

# ============================================================== the optimiser
OPTIMISER = "AdamW"
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8
BATCH = "full outer-training fold"
MAX_STEPS = 2000
PATIENCE = 200
SEED_BASE = 20260826
N_SPLITS = 5

#: Fixed, and not searched. The only adaptive quantity in this stage is the
#: early-stopping step, chosen on an inner split of the training folds and
#: never on the held-out fold.
HYPERPARAMETER_SEARCH_PERMITTED = False

# ================================================================ the brake
#: Zero-H-miss: the largest threshold that fully vetoes every training H row.
VETO_QUANTILE_RULE = "tau_hi = min over outer-training H rows of z_H"


# =============================================== stratification, not features
#
# ``topic_standardized`` and ``task_frame`` are carried into the result so the
# coefficients can be read by stratum, and so a later hard-layer-router run can
# report its selected arm and coefficient against the same frozen labels. They
# are DIAGNOSTIC ONLY.
#
# The controller's input is the 2048-dimensional state-32 activation and
# nothing else. No covariate enters the network, the loss, the veto threshold,
# the clamp or the epoch selection -- ``assert_covariates_are_not_features``
# checks that against the actual input tensor rather than trusting the comment.

STRATIFICATION_FIELDS: Tuple[str, ...] = ("topic_standardized", "task_frame")

#: Frame attributes those two fields are read from.
STRATIFICATION_SOURCE: Dict[str, str] = {"topic_standardized": "topic",
                                         "task_frame": "task_frame"}

#: Never a controller feature, and never a stratification field either. Named
#: so the ban is enforceable rather than remembered.
FORBIDDEN_FEATURES: Tuple[str, ...] = (
    "source_family", "benchmark_designation", "prompt_category")

STRATIFICATION_STANDING = (
    "diagnostic only. These are frozen dataset labels, reported so g_R can be "
    "read by stratum and so a later router can report its arm and coefficient "
    "against the same labels. They are not controller inputs, not loss inputs, "
    "not threshold inputs and not hyperparameter-selection inputs.")


class GateControllerError(ValidationError):
    """The controller was asked for something the specification forbids."""


def assert_covariates_are_not_features(controller_input: Any,
                                       hidden_size: int = HIDDEN_SIZE,
                                       checks: Optional[Checklist] = None
                                       ) -> Dict[str, Any]:
    """The controller reads the activation and nothing else. Checked by shape.

    A covariate could only reach the network by being concatenated onto the
    input, so the input's width is the thing to check: 2048 is the activation
    alone, and anything wider is carrying something else.
    """
    checks = checks or Checklist("gate_controller")
    width = int(controller_input.shape[-1])
    checks.require(
        "controller_input_is_the_activation_alone", width == hidden_size,
        f"the controller input is {width}-dimensional and the state-32 "
        f"activation is {hidden_size}. A wider input is carrying something "
        f"besides the activation, and no covariate is a controller feature.")
    checks.record(
        "stratification_fields_are_diagnostic_only", True,
        f"{list(STRATIFICATION_FIELDS)} are recorded per row and enter no "
        f"network, loss, threshold or selection rule; "
        f"{list(FORBIDDEN_FEATURES)} are never used at all")
    return {"input_width": width,
            "stratification_fields": list(STRATIFICATION_FIELDS),
            "forbidden_features": list(FORBIDDEN_FEATURES)}


def stratified_distribution(values: Sequence[float], strata: Sequence[str],
                            cutoffs: Sequence[float]) -> Dict[str, Any]:
    """g_R distribution and intervention rate within each stratum.

    Reported, never optimised against: no threshold, weight or stopping
    decision in this stage reads any of it.
    """
    values = np.asarray(values, dtype=np.float64)
    out: Dict[str, Any] = {}
    for stratum in sorted(set(strata)):
        mask = np.array([s == stratum for s in strata])
        if not mask.any():                                # pragma: no cover
            continue
        subset = values[mask]
        out[str(stratum)] = {
            "n": int(subset.size),
            "mean_g_R": float(subset.mean()),
            "median_g_R": float(np.median(subset)),
            "min_g_R": float(subset.min()),
            "max_g_R": float(subset.max()),
            "mean_applied_alpha_R": float(ALPHA_R * subset.mean()),
            "intervention_rate": {f"{c:.2f}": float((subset >= c).mean())
                                  for c in cutoffs},
        }
    return out


def inner_fold_for(outer: int, n_splits: int = N_SPLITS) -> int:
    """The inner-validation fold: ``(k + 1) mod n``, drawn from the training folds.

    Never the held-out fold, by construction, which is what keeps early
    stopping out of contact with fold ``k``.
    """
    if not 0 <= outer < n_splits:
        raise GateControllerError(
            f"outer fold {outer} is outside 0..{n_splits - 1}")
    inner = (outer + 1) % n_splits
    if inner == outer:                                    # pragma: no cover
        raise GateControllerError("the inner fold cannot be the held-out fold")
    return inner


def seed_for(outer: int) -> int:
    return SEED_BASE + int(outer)


# =========================================================== the gate network
def build_controller(seed: int, device: Any = None) -> Any:
    """The controller: 2048 -> 16 GELU -> 1, with a sigmoid applied outside.

    ``w2`` is initialised to zero and ``b2`` to ``logit(0.1)``, so training
    begins at ``g_R = 0.1`` for every row and larger gates have to be earned.
    """
    import torch

    torch.manual_seed(int(seed))
    net = torch.nn.Sequential(
        torch.nn.Linear(HIDDEN_SIZE, CONTROLLER_HIDDEN),
        torch.nn.GELU(),
        torch.nn.Dropout(CONTROLLER_DROPOUT),
        torch.nn.Linear(CONTROLLER_HIDDEN, 1),
    )
    torch.nn.init.normal_(net[0].weight, mean=0.0,
                          std=(2.0 / HIDDEN_SIZE) ** 0.5)
    torch.nn.init.zeros_(net[0].bias)
    torch.nn.init.zeros_(net[3].weight)
    torch.nn.init.constant_(net[3].bias, GATE_BIAS_INIT)
    if device is not None:
        net = net.to(device)
    return net


def count_parameters(net: Any) -> int:
    return sum(p.numel() for p in net.parameters())


def gate(net: Any, standardised: Any) -> Any:
    """``g_R = sigmoid(f_theta(x))``, one scalar per row, in [0, 1]."""
    import torch

    return torch.sigmoid(net(standardised).squeeze(-1))


def applied_alpha(gate_values: Any) -> Any:
    """``alpha_R = 0.15 * g_R``. The ceiling is arithmetic, not a check."""
    return ALPHA_R * gate_values


# ============================================================ the intervention
def steer(activation: Any, gate_values: Any, budget: Any, direction: Any,
          alpha: float = ALPHA_R) -> Any:
    """``h' = h + alpha * g_R * r * d_R``, built explicitly as a tensor.

    ``budget`` is the unsteered residual norm at the block's input state and
    carries no gradient; ``direction`` is the frozen unit repair direction. The
    only differentiable factor is ``g_R``, which is the point.
    """
    if gate_values.dim() != 1:
        raise GateControllerError(
            f"the gate is {tuple(gate_values.shape)}; one scalar per row")
    scale = (float(alpha) * gate_values * budget).unsqueeze(-1)
    return activation + scale * direction


def steered_repair_logit(probe: Any, activation: Any, gate_values: Any,
                         budget: Any, direction: Any) -> Any:
    return probe.logit(steer(activation, gate_values, budget, direction))


def affine_oracle(z_unsteered: Any, gate_values: Any, budget: Any,
                  kappa: float, alpha: float = ALPHA_R) -> Any:
    """``z(h') = z(h) + alpha * g_R * r * kappa``. A TEST ORACLE only.

    Exact, because both probes are affine readers and the edit is a fixed
    vector times a scalar. It is never the training path: the training path
    constructs ``h'`` and runs the probe forward on it, so the code applies
    ``d_R`` visibly rather than asserting what applying it would do.
    """
    return z_unsteered + float(alpha) * gate_values * budget * kappa


# ================================================================ the losses
def repair_loss(z_repair_steered: Any, mask: Any) -> Any:
    """``mean over E of softplus(z_R(h'))``.

    ``softplus(z)`` is binary cross-entropy with logits against the negative
    class: minimising it pushes an E row's repair logit down, which is what the
    repair direction is for. It saturates once past the probe's own boundary,
    so it does not demand an unbounded gate.
    """
    import torch

    if not bool(mask.any()):
        return torch.zeros((), dtype=z_repair_steered.dtype,
                           device=z_repair_steered.device)
    return torch.nn.functional.softplus(z_repair_steered[mask]).mean()


def safety_loss(z_safety_unsteered: Any, z_safety_steered: Any,
                mask: Any) -> Any:
    """``mean over the protected set of relu(z_H(h) - z_H(h'))``.

    One-sided: a rise in the H-positive logit is free, a fall is penalised. It
    is a difference of one probe's logits on one row, so any affine
    recalibration cancels exactly and any monotone one preserves its sign --
    which is why no calibration assumption enters.
    """
    import torch

    if not bool(mask.any()):
        return torch.zeros((), dtype=z_safety_steered.dtype,
                           device=z_safety_steered.device)
    drop = z_safety_unsteered[mask] - z_safety_steered[mask]
    return torch.nn.functional.relu(drop).mean()


def size_loss(gate_values: Any) -> Any:
    """``mean over all training rows of g_R^2``.

    Quadratic rather than L1: smooth shrinkage, a superlinear penalty on large
    gates, and no dead zone that would pin small gates at exactly zero.
    """
    return (gate_values ** 2).mean()


def total_loss(repair: Any, safety: Any, size: Any) -> Any:
    return (LAMBDA_REPAIR * repair + LAMBDA_SAFETY * safety
            + LAMBDA_SIZE * size)


def protected_mask(z_safety_unsteered: Any, is_h: Any, tau_hi: float) -> Any:
    """Every training H row, plus any E/B row the deployed veto would catch.

    ``H`` is included unconditionally -- no H row is ever excluded from the
    safety term, whatever its score -- and the conservative extension adds safe
    rows that already read as H-like at the veto threshold. It introduces no
    new constant: ``tau_hi`` is the deployment threshold.
    """
    import torch

    return torch.logical_or(is_h, z_safety_unsteered >= float(tau_hi))


# ============================================== optimiser scope, proved
def controller_optimiser(net: Any) -> Any:
    """AdamW over the controller's parameters and nothing else.

    Weight decay applies to weights, not to biases: decaying ``b2`` would pull
    the initial gate away from its declared starting point for a reason that
    has nothing to do with the objective.
    """
    import torch

    decay, no_decay = [], []
    for name, parameter in net.named_parameters():
        (no_decay if name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=BETAS, eps=EPS)


def assert_optimiser_is_controller_only(optimiser: Any, net: Any,
                                        frozen: Sequence[Any] = (),
                                        checks: Optional[Checklist] = None
                                        ) -> Dict[str, Any]:
    """Only controller parameters may be updated. Proved by identity.

    Compared by ``id``, not by shape or value: a frozen tensor that happens to
    match a controller parameter's shape would pass any looser check.
    """
    checks = checks or Checklist("gate_controller")
    own = {id(p) for p in net.parameters()}
    grouped = [p for group in optimiser.param_groups for p in group["params"]]
    foreign = [p for p in grouped if id(p) not in own]
    checks.require(
        "optimiser_holds_only_controller_parameters", not foreign,
        f"{len(foreign)} parameter(s) in the optimiser do not belong to the "
        f"controller. Only controller parameters may be updated; the language "
        f"model, both probes, d_R and the activations are frozen.")
    checks.require(
        "every_controller_parameter_is_optimised",
        len({id(p) for p in grouped}) == len(own),
        f"{len(own)} controller parameter tensor(s), {len({id(p) for p in grouped})} "
        f"in the optimiser")
    still_training = [t for t in frozen if getattr(t, "requires_grad", False)]
    checks.require(
        "frozen_tensors_require_no_grad", not still_training,
        f"{len(still_training)} frozen tensor(s) carry requires_grad=True")
    checks.record("controller_parameters", True,
                  f"{count_parameters(net)} parameter(s)")
    return {"parameters": count_parameters(net),
            "optimiser_groups": len(optimiser.param_groups)}


# ============================================ inference: veto and clamp
def veto_threshold(z_safety_training_h: Sequence[float]) -> float:
    """``tau_hi``: the largest threshold that fully vetoes every training H row.

    A full veto means ``z_H >= tau_hi``, so vetoing all of them requires
    ``tau_hi <= min``, and the largest such value is the minimum. Derived from
    outer-training rows only, and applied unchanged to the held-out fold.
    """
    values = [float(v) for v in z_safety_training_h]
    if not values:
        raise GateControllerError(
            "no training H rows; a zero-H-miss threshold over an empty set is "
            "not a threshold")
    return min(values)


def apply_safety_veto(gate_values: Any, z_safety: Any, tau_hi: float) -> Any:
    """Hard veto: ``g_R = 0`` wherever ``z_H >= tau_hi``, unchanged elsewhere.

    Multiplicative by a 0/1 factor, so it can only reduce. ``z_safety`` is read
    on the **unsteered** activation: the decision to steer precedes steering.
    """
    import torch

    keep = (z_safety < float(tau_hi)).to(gate_values.dtype)
    return gate_values * keep


def apply_counterfactual_clamp(gate_values: Any, z_safety_unsteered: Any,
                               budget: Any, kappa_safety: float,
                               tau_hi: float, alpha: float = ALPHA_R) -> Any:
    """Reduce the gate to the largest value whose steered ``z_H`` stays clear.

    The counterfactual is exact, because the safety probe is affine::

        z_H(h') = z_H(h) + alpha * g * r * kappa_H

    If ``kappa_H >= 0`` the edit cannot lower ``z_H`` and no clamp is needed.
    If it is negative, the gate that would just reach ``tau_hi`` is

        g* = (tau_hi - z_H(h)) / (alpha * r * kappa_H)

    and the clamp takes ``min(g, g*)`` for rows that would otherwise cross,
    leaving every other row alone. It never raises a gate, and rows already at
    or above ``tau_hi`` are handled by the veto rather than here.
    """
    import torch

    if kappa_safety >= 0.0:
        return gate_values
    denominator = float(alpha) * budget * float(kappa_safety)
    ceiling = (float(tau_hi) - z_safety_unsteered) / denominator
    ceiling = torch.clamp(ceiling, min=0.0, max=GATE_MAXIMUM)
    return torch.minimum(gate_values, ceiling)


# ==================================================== the training specification
def build_training_specification() -> Dict[str, Any]:
    """The predeclaration this stage is run against, derived from the module."""
    return {
        "schema": SCHEMA,
        "name": TRAINING_SPEC_NAME,
        "version": TRAINING_SPEC_VERSION,
        "status": PROVISIONAL_STATUS,
        "status_reason": PROVISIONAL_REASON,
        "resolves": {
            "specification": "configs/single_direction_controller.json",
            "fields": ["controller_input", "controller_hidden_size",
                       "controller_depth", "controller_activation_function",
                       "controller_loss", "controller_regularisation",
                       "controller_optimiser", "controller_training_schedule",
                       "controller_selection_criterion",
                       "safety_brake_operating_point",
                       "successful_strength_targets_or_other_supervisory_signal"],
            "note": "the v1 specification marks these UNRESOLVED and requires "
                    "them fixed BEFORE training. This file fixes them. v1 is "
                    "not edited: it stays the record of what was open.",
        },
        "controller": {
            "input": CONTROLLER_INPUT,
            "read_state": READ_STATE,
            "read_historical_block": READ_BLOCK,
            "token_position": "final prompt token",
            "hidden_size": CONTROLLER_HIDDEN,
            "depth": CONTROLLER_DEPTH,
            "activation": CONTROLLER_ACTIVATION,
            "dropout": CONTROLLER_DROPOUT,
            "output": "g_R = sigmoid(f_theta(h)), one scalar per prompt",
            "output_count": 1,
            "parameters": CONTROLLER_PARAMETERS,
            "initialisation": {
                "layer1_weight": "N(0, 2/2048)", "layer1_bias": 0,
                "layer2_weight": 0,
                "layer2_bias": f"logit({GATE_INITIAL}) = {GATE_BIAS_INIT:.7f}",
                "why": "training starts at g_R = 0.1 for every row; a larger "
                       "gate has to be earned.",
            },
        },
        "intervention": {
            "formula": "delta_R,l = 0.15 * g_R(x) * r_l(x) * d_R,l",
            "alpha_R": ALPHA_R,
            "max_applied_alpha_R": MAX_APPLIED_ALPHA_R,
            "alpha_S": ALPHA_S,
            "historical_blocks": list(CANDIDATE_BLOCKS),
            "package_states_edited": [block_output_state(b)
                                      for b in CANDIDATE_BLOCKS],
            "budget_reference_states": [block_input_state(b)
                                        for b in CANDIDATE_BLOCKS],
            "budget_reference": "block_input, from the unsteered activation",
            "h_minus_e_is_never_applied": True,
        },
        "loss": {
            "repair": "mean over E of softplus(z_R(h'))",
            "safety": "mean over the protected set of relu(z_H(h) - z_H(h'))",
            "protected_set": "every outer-training H row, unconditionally, "
                             "plus any E/B row with z_H >= tau_hi",
            "size": "mean over all training rows of g_R^2",
            "total": "L_repair + lambda_S * L_safety + lambda_G * L_size",
            "lambda_repair": LAMBDA_REPAIR,
            "lambda_safety": LAMBDA_SAFETY,
            "lambda_size": LAMBDA_SIZE,
            "weights_are_predeclared_not_estimated": LOSS_WEIGHTS_ARE_PREDECLARED,
            "operates_on": "logits, never probabilities; the probe run fitted "
                           "no calibration model",
        },
        "optimiser": {
            "name": OPTIMISER, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "weight_decay_excludes_biases": True,
            "betas": list(BETAS), "eps": EPS, "batch": BATCH,
            "max_steps": MAX_STEPS, "patience": PATIENCE,
            "seed": f"{SEED_BASE} + outer fold",
            "hyperparameter_search_permitted": HYPERPARAMETER_SEARCH_PERMITTED,
        },
        "oof_protocol": {
            "n_splits": N_SPLITS,
            "rule": "controller k trains on the folds other than k; early "
                    "stopping uses an inner split of those training folds; "
                    "fold k affects neither and is predicted only at the end.",
            "inner_fold": "(k + 1) mod 5, drawn from the training folds",
            "final_refit_performed": False,
            "final_refit_note": "the all-development refit is deliberately not "
                                "performed at this stage.",
        },
        "brake": {
            "veto_rule": VETO_QUANTILE_RULE,
            "clamp": "g* = (tau_hi - z_H(h)) / (alpha * r * kappa_H) when "
                     "kappa_H < 0; the clamp takes min(g, g*)",
            "neither_is_trained": True,
            "neither_can_raise_a_gate": True,
            "derived_from": "outer-training rows only, applied unchanged to "
                            "the held-out fold",
        },
        "stratification": {
            "fields": list(STRATIFICATION_FIELDS),
            "read_from": dict(STRATIFICATION_SOURCE),
            "standing": STRATIFICATION_STANDING,
            "is_a_controller_input": False,
            "is_a_loss_input": False,
            "is_a_threshold_input": False,
            "is_a_hyperparameter_selection_input": False,
            "forbidden_features": list(FORBIDDEN_FEATURES),
            "forbidden_features_note": "never controller features, and never "
                                       "stratification fields either.",
            "retained_for": "a later hard-layer-router run, which reports its "
                            "selected arm and coefficient against these same "
                            "frozen labels and the same frozen folds.",
        },
        "frozen": ["language model (never loaded)", "repair probe",
                   "safety probe", "d_R", "activations", "folds"],
        "forward_passes": 0,
        "generates_responses": False,
        "calls_a_judge": False,
        "reads_the_sealed_set": False,
        "contains_prompt_or_response_text": False,
    }
