"""The one-direction controller: one gate, one direction, one ceiling.

Phase 2 decided ``NO_STRENGTH_QUALIFIES`` and retired ``d_S = H - E`` as an
actuator (see ``routegate.phase2closure``). What continues is a controller that
steers the validated repair direction and nothing else:

    g_R      one sigmoid gate, in [0, 1], emitted once per prompt
    alpha_R  = 0.15 * g_R, the strength actually applied

so the applied strength cannot exceed **0.15**, the repair strength phase 1
validated and phase 2 pinned ``alpha_R`` to across all four combined arms. The
ceiling is arithmetic, not a convention a later caller can relax: there is no
coefficient in the parameterisation that can carry it past 0.15, because the
sigmoid saturates at 1 and 0.15 x 1 is 0.15.

The safety reading is retained and is a **brake**. It may scale ``g_R`` down or
drive it to zero, and it may do nothing else: it produces no direction, no
second coefficient, and no output of its own. A gate the safety head has
touched is never larger than the gate it was given, which is checked rather
than intended.

Where it reads
--------------
State **32** -- the output of historical decoder block **31** -- at the **final
prompt token**, during prefill. That is the frozen actuator's own first layer
and the state the probe heads declared as primary before either head was
fitted. States 33 and 34 stay what they always were: sensitivity diagnostics
reported beside state 32. They are not candidates, and ``assert_read_state``
refuses one being promoted to the controller's read state after the fact.

What this module does not do
----------------------------
It does not implement, train, initialise or evaluate a controller. There is no
optimiser here, no loss, no parameter tensor and no training loop. It holds the
architecture, the protocol both will be run under, and the gates that refuse a
configuration the phase-2 decision has ruled out.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .dualsteer import COMBINED_ALPHA_REPAIR
from .phase2closure import (
    CLOSURE_NAME,
    PHASE2_OUTCOME,
    PHASE2_READINGS_SHA256,
)
from .probeheads import (
    PRIMARY_READ_LAYER,
    PRIMARY_READ_STATE,
    SENSITIVITY_STATES,
    TOKEN_POSITION,
    TOKEN_POSITION_NOTE,
    layer_of_state,
    state_of_layer,
)
from .validation import Checklist, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "routegate/single_direction_controller/1"

#: The specification's version. The superseded two-coefficient proposal lives
#: in ``configs/dual_direction_controller.json`` and is not edited into this
#: shape: it is phase-2 history and stays readable as what was proposed then.
SPECIFICATION_VERSION = 1

CONTROLLER_NAME = "single_direction_repair_controller_v1"

#: The committed specification this module is checked against.
SINGLE_DIRECTION_CONFIG = (
    REPO_ROOT / "configs" / "single_direction_controller.json")

#: The historical configuration, named here ONLY so it can be refused as a
#: final-controller configuration. It is preserved, never rewritten.
HISTORICAL_DUAL_CONFIG = "dual_direction_controller.json"


class SingleDirectionError(ValidationError):
    """The one-direction controller was asked for something it must not do."""


# ==================================================================== the gate
GATE_NAME = "g_R"

#: Every output the controller emits. One. The tuple is the specification, and
#: a second entry is not an extension of it -- it is the design phase 2 ruled
#: out returning under a new name.
CONTROLLER_OUTPUTS: Tuple[str, ...] = (GATE_NAME,)

GATE_MINIMUM = 0.0
GATE_MAXIMUM = 1.0
GATE_ACTIVATION = "sigmoid"

#: The frozen validated repair strength. Not re-chosen here: it is the value
#: phase 1 measured H damage at and phase 2 held alpha_R at in every combined
#: arm, and this module reads it from the screen rather than restating it.
FROZEN_VALIDATED_REPAIR_STRENGTH = COMBINED_ALPHA_REPAIR       # 0.15

#: The ceiling on the applied repair strength. Identical to the frozen value by
#: construction: alpha_R = 0.15 * g_R and g_R <= 1.
MAX_APPLIED_ALPHA_REPAIR = FROZEN_VALIDATED_REPAIR_STRENGTH

APPLIED_STRENGTH_FORMULA = "alpha_R = 0.15 * g_R"

DIRECTION_REPAIR = "d_R = B_d1 - E"

#: Named so it can be refused. Nothing in the final path applies it.
DIRECTION_SAFETY_RETIRED = "d_S = H - E"


def _validated_gate(gate: float) -> float:
    """``g_R`` as a number in [0, 1], or a refusal naming what it was."""
    if isinstance(gate, bool) or not isinstance(gate, (int, float)):
        raise SingleDirectionError(
            f"{GATE_NAME} must be a number, got {gate!r}")
    value = float(gate)
    if not math.isfinite(value):
        raise SingleDirectionError(f"{GATE_NAME} is {value}; a gate is finite")
    if not GATE_MINIMUM <= value <= GATE_MAXIMUM:
        raise SingleDirectionError(
            f"{GATE_NAME} is {value}, outside [{GATE_MINIMUM}, "
            f"{GATE_MAXIMUM}]. The gate is a sigmoid output; a value outside "
            f"its range did not come from one, and multiplying it by "
            f"{FROZEN_VALIDATED_REPAIR_STRENGTH} would apply a strength "
            f"nothing validated.")
    return value


def applied_repair_strength(gate: float) -> float:
    """``alpha_R = 0.15 * g_R``, with the gate's own range enforced.

    The ceiling is not asserted after the fact by a caller who remembers to:
    a gate outside [0, 1] is not a sigmoid output and is refused here, before
    it can be multiplied into a strength nobody validated.
    """
    strength = FROZEN_VALIDATED_REPAIR_STRENGTH * _validated_gate(gate)
    if strength > MAX_APPLIED_ALPHA_REPAIR:                # pragma: no cover
        raise SingleDirectionError(
            f"applied alpha_R {strength} exceeds the frozen validated repair "
            f"strength {MAX_APPLIED_ALPHA_REPAIR}")
    return strength


# ============================================================== the safety brake
SAFETY_BRAKE: Dict[str, Any] = {
    "reading": "p_H, from the H-versus-(E+B) probe head",
    "role": "sensor",
    "may_reduce_the_gate": True,
    "may_veto_the_gate": True,
    "may_increase_the_gate": False,
    "may_produce_a_steering_direction": False,
    "may_produce_a_controller_output": False,
    "veto_means": f"{GATE_NAME} = 0, and therefore alpha_R = 0: the repair "
                  f"edit is not applied at all",
    "statement": "the safety head brakes the repair gate. It never steers. "
                 "There is no direction it scales and no coefficient it "
                 "emits, and the retired d_S = H - E is not reachable from it.",
}


def apply_safety_brake(gate: float, brake: float) -> float:
    """The braked gate: ``g_R * brake``, and never larger than ``g_R``.

    ``brake`` is in [0, 1]: 1 leaves the gate alone, 0 is a veto. Written as a
    multiplication rather than a general function of the two readings because
    a general function could raise the gate, and a safety reading that can
    raise a repair gate is a steering input wearing a brake's name.
    """
    if isinstance(brake, bool) or not isinstance(brake, (int, float)):
        raise SingleDirectionError(f"the brake must be a number, got {brake!r}")
    value = float(brake)
    if not math.isfinite(value):
        raise SingleDirectionError(f"the brake is {value}; a brake is finite")
    if not 0.0 <= value <= 1.0:
        raise SingleDirectionError(
            f"the brake is {value}, outside [0, 1]. A brake scales the gate "
            f"down or vetoes it; a factor outside [0, 1] would let the safety "
            f"reading raise the repair gate, which is steering.")
    original = _validated_gate(gate)
    braked = original * value
    if braked > original:                                  # pragma: no cover
        raise SingleDirectionError(
            f"the braked gate {braked} exceeds the gate {original} it was "
            f"given; a brake reduces or vetoes and never raises")
    return braked


# ============================================================== where it reads
READ_LAYER = PRIMARY_READ_LAYER                                     # 31
READ_STATE = PRIMARY_READ_STATE                                     # 32
DIAGNOSTIC_STATES: Tuple[int, ...] = tuple(SENSITIVITY_STATES)      # (33, 34)

READ_STATE_STANDING = (
    f"state {READ_STATE} is the output of historical decoder block "
    f"{READ_LAYER}. It was predeclared as the primary controller read state "
    f"before either probe head was fitted, because it is the frozen actuator's "
    f"own first layer. It is not selected from the sensitivity states' "
    f"outcomes.")

DIAGNOSTIC_STATES_STANDING = (
    f"states {list(DIAGNOSTIC_STATES)} remain probe diagnostics: they are "
    f"reported beside state {READ_STATE} as a sensitivity check on the primary "
    f"reading. They are not candidates for the controller's read state, and "
    f"none of them may be promoted to it after its metrics are known.")


def assert_read_state(state: int, checks: Optional[Checklist] = None) -> int:
    """The controller reads state 32 and refuses to read anywhere else.

    A diagnostic state is refused with its own message, because reaching for
    one is a specific mistake: it is the post-hoc reselection this protocol
    exists to make impossible, not a typo.
    """
    if int(state) in DIAGNOSTIC_STATES:
        raise SingleDirectionError(
            f"state {state} is a probe DIAGNOSTIC (historical layer "
            f"{layer_of_state(int(state))}), reported beside state "
            f"{READ_STATE} as a sensitivity check. It is not a candidate read "
            f"state and is not promoted to one after the fact: the primary "
            f"read state was predeclared before either head was fitted, and "
            f"reselecting it on the diagnostics' outcomes is the choice this "
            f"protocol refuses.")
    if int(state) != READ_STATE:
        raise SingleDirectionError(
            f"the controller reads state {READ_STATE} (the output of "
            f"historical block {READ_LAYER}); state {state} is not it")
    if state_of_layer(READ_LAYER) != READ_STATE:           # pragma: no cover
        raise SingleDirectionError(
            f"the layer-to-state mapping has moved: historical block "
            f"{READ_LAYER} is state {state_of_layer(READ_LAYER)}, not "
            f"{READ_STATE}")
    if checks is not None:
        checks.record("controller_read_state", True,
                      f"state {READ_STATE} = output of historical block "
                      f"{READ_LAYER}, final prompt token")
    return READ_STATE


def assert_token_position(position: str) -> str:
    """Final-prompt-token activations only. Nothing decoded is read."""
    if position != TOKEN_POSITION:
        raise SingleDirectionError(
            f"the controller reads the {TOKEN_POSITION!r}; {position!r} is not "
            f"it. {TOKEN_POSITION_NOTE}. A reading taken anywhere else is not "
            f"the reading the probe heads were fitted on.")
    return TOKEN_POSITION


# ============================================================== the OOF protocol
#: The frozen division the probe heads and the direction bank already share.
N_SPLITS = 5


def oof_protocol_for_fold(fold: int, n_splits: int = N_SPLITS) -> Dict[str, Any]:
    """What the held-out evaluation of one fold is made of.

    Everything that touches fold ``k`` is fitted without fold ``k``: the repair
    head, the safety head and the controller. Only fold ``k`` is generated and
    evaluated. Anything less than all three is a controller scored partly on
    rows something in its own pipeline was fitted on.
    """
    if not isinstance(fold, int) or isinstance(fold, bool):
        raise SingleDirectionError(f"fold must be an int, got {fold!r}")
    if not 0 <= fold < n_splits:
        raise SingleDirectionError(
            f"fold {fold} is outside 0..{n_splits - 1}")
    training = [k for k in range(n_splits) if k != fold]
    return {
        "held_out_fold": fold,
        "n_splits": n_splits,
        "repair_probe": {"fitted_without_fold": fold,
                         "training_folds": list(training)},
        "safety_probe": {"fitted_without_fold": fold,
                         "training_folds": list(training)},
        "controller": {"trained_without_fold": fold,
                       "training_folds": list(training)},
        "generated_folds": [fold],
        "evaluated_folds": [fold],
        "rule": f"for held-out fold {fold}, use the fold-{fold} repair and "
                f"safety probes -- each trained without fold {fold} -- train "
                f"the corresponding controller without fold {fold}, and "
                f"generate and evaluate fold {fold} only.",
    }


def assert_oof_protocol(plan: Mapping[str, Any], fold: int,
                        n_splits: int = N_SPLITS,
                        checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """One held-out fold's plan is out-of-fold in every component.

    Checked component by component rather than as a whole, so a leak names the
    thing that leaked: "the controller was trained on fold 3" is actionable
    where "the plan is not out of fold" is not.
    """
    expected = oof_protocol_for_fold(fold, n_splits)
    for component in ("repair_probe", "safety_probe", "controller"):
        declared = dict(plan.get(component) or {})
        training = [int(k) for k in declared.get("training_folds", [])]
        if fold in training:
            raise SingleDirectionError(
                f"the {component.replace('_', ' ')} for held-out fold {fold} "
                f"declares training folds {sorted(training)}, which include "
                f"fold {fold}. Every component that touches a held-out row "
                f"must have been fitted without it.")
        if sorted(training) != expected[component]["training_folds"]:
            raise SingleDirectionError(
                f"the {component.replace('_', ' ')} for held-out fold {fold} "
                f"declares training folds {sorted(training)}, the protocol is "
                f"{expected[component]['training_folds']}")
    for key in ("generated_folds", "evaluated_folds"):
        declared = sorted(int(k) for k in (plan.get(key) or []))
        if declared != [fold]:
            raise SingleDirectionError(
                f"{key} is {declared}; held-out fold {fold} generates and "
                f"evaluates fold {fold} and no other fold. Generating a "
                f"training fold would score the controller on rows it was "
                f"fitted on.")
    if checks is not None:
        checks.record("oof_protocol", True,
                      f"fold {fold}: probes and controller fitted without it, "
                      f"fold {fold} generated and evaluated")
    return expected


OOF_PROTOCOL: Dict[str, Any] = {
    "n_splits": N_SPLITS,
    "fold_source": "the cross-fitted direction bank's own frozen 480-row "
                   "division, as the probe heads already use",
    "rule": "for held-out fold k: use the fold-k repair and safety probes, "
            "each trained without fold k; train the corresponding controller "
            "without fold k; generate and evaluate fold k only.",
    "every_component_is_out_of_fold": ["repair_probe", "safety_probe",
                                       "controller"],
    "why": "a controller reading two probe scores is a three-stage pipeline. "
           "Cross-fitting two of the three stages and not the third scores it "
           "partly on rows its own pipeline was fitted on, and the leak is "
           "invisible in the number that comes out.",
    "implemented_by": "routegate.singledirection.assert_oof_protocol",
}


# ======================================================= the deployment protocol
DEPLOYMENT_PROTOCOL: Dict[str, Any] = {
    "runs_only_after": "architecture freeze",
    "steps": [
        "refit the two probe heads on ALL development rows, with every "
        "hyperparameter fixed at the frozen specification -- nothing is "
        "re-tuned at this step",
        "train the final controller on ALL development rows, with the "
        "architecture and every training-time choice already fixed",
        "freeze both, and record their digests, BEFORE anything reads the "
        "sealed set",
    ],
    "order_is_not_advisory": "each step is a precondition of the next. A "
                             "controller trained before the heads are refit "
                             "is trained on scores it will not be deployed "
                             "with.",
    "nothing_is_retuned_here": True,
    "sealed_set": {
        "accessed_before_the_freeze": False,
        "statement": "the sealed set is read once, after the heads and the "
                     "controller are frozen and their digests recorded. A "
                     "choice made after reading it is not a choice this "
                     "protocol permits.",
    },
    "implemented_by": "routegate.singledirection.assert_deployment_preconditions",
}

DEPLOYMENT_STEPS: Tuple[str, ...] = (
    "architecture_frozen",
    "probe_heads_refit_on_all_development_rows",
    "final_controller_trained_on_all_development_rows",
    "both_frozen_and_digested",
    "sealed_set_read",
)


def assert_deployment_preconditions(state: Mapping[str, Any],
                                    checks: Optional[Checklist] = None
                                    ) -> Dict[str, Any]:
    """The final deployment path is walked in order, and never past the seal.

    ``state`` carries one boolean per step of ``DEPLOYMENT_STEPS``. A step
    whose predecessor is not done is refused, and reading the sealed set before
    the freeze is refused whatever else is true.
    """
    done = {step: bool(state.get(step, False)) for step in DEPLOYMENT_STEPS}
    if done["sealed_set_read"] and not done["both_frozen_and_digested"]:
        raise SingleDirectionError(
            "the sealed set is recorded as read while the probe heads and the "
            "controller are not frozen. The freeze comes first, always: a "
            "choice made after reading held-out material is not a choice this "
            "protocol permits.")
    for index, step in enumerate(DEPLOYMENT_STEPS[1:], start=1):
        if done[step] and not done[DEPLOYMENT_STEPS[index - 1]]:
            raise SingleDirectionError(
                f"{step} is recorded as done and {DEPLOYMENT_STEPS[index - 1]} "
                f"is not. The deployment steps are preconditions of one "
                f"another, not a checklist to tick in any order.")
    if checks is not None:
        checks.record(
            "deployment_preconditions", True,
            ", ".join(f"{step}={done[step]}" for step in DEPLOYMENT_STEPS))
    return done


# ================================================ what a final config may not be
#
# Requirement, not taste. Phase 2 removed d_S as an actuator; a final
# configuration that carries a safety steering coefficient, an alpha_S above
# zero, an active H - E actuator or a second controller output is not a
# variation on the decision -- it is the design the decision ruled out,
# reaching the final path under a different name.

def _normalise_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


#: Keys that ARE a safety steering coefficient, whatever they are set to. The
#: presence of the field is the refusal: a final configuration with a safety
#: coefficient set to zero still has one, and the next edit sets it.
SAFETY_COEFFICIENT_KEYS = frozenset({
    "alphas", "alphasafety", "safetyalpha", "safetycoefficient",
    "safetysteeringcoefficient", "safetystrength", "safetyscale",
    "safetysteeringstrength", "as", "gs", "safetygate", "safetyoutput",
})

#: The retired direction as an expression, matched as a standalone token so it
#: is found inside "d_S = H - E" and not inside "THE-END".
_RETIRED_DIRECTION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[Hh]\s*[-‐-―−]\s*[Ee](?![A-Za-z0-9_])")

#: Keys a configuration declares an ACTUATOR under. The retired direction may
#: be described in prose -- a specification that cannot say what it retired is
#: worse, not safer -- but it may not sit in one of these slots.
DIRECTION_SLOT_KEYS = frozenset({
    "direction", "directions", "actuator", "actuators",
    "steereddirection", "steereddirections", "activedirections",
    "safetydirection", "safetysign", "safetyactuator", "steeringdirection",
    "appliedirection", "applieddirection", "applieddirections", "ds",
})

#: The one block a final configuration may name the retired direction in as a
#: direction: the one that says it is retired.
RETIRED_ACTUATOR_DECLARATION_KEY = "retired_actuator"

#: The superseded parameterisation, refused by name so it cannot be carried
#: forward verbatim from the historical configuration.
SUPERSEDED_PARAMETERISATION = "two sigmoid"


def _walk(node: Any, trail: Tuple[str, ...] = ()):
    """Every (trail, key, value) in a configuration tree, depth first.

    List items are yielded too, under a ``[i]`` key: a direction hidden in a
    list is still a direction, and a walk that only visited mapping values
    would step straight over it.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield trail, str(key), value
            yield from _walk(value, trail + (str(key),))
    elif isinstance(node, (list, tuple)) and not isinstance(node, (str, bytes)):
        for index, value in enumerate(node):
            yield trail, f"[{index}]", value
            yield from _walk(value, trail + (f"[{index}]",))


def _slot(trail: Tuple[str, ...], key: str) -> str:
    """The nearest named key, skipping list indices.

    ``directions[0]`` is a direction slot; its key is ``[0]`` and its slot is
    ``directions``.
    """
    for name in reversed(trail + (key,)):
        if not name.startswith("["):
            return _normalise_key(name)
    return ""


def assert_final_controller_config(
        config: Mapping[str, Any],
        context: str = "final controller configuration",
        checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """Refuse a configuration the phase-2 decision has ruled out.

    Four refusals, each with its own message, because they are four different
    mistakes and "invalid configuration" tells a reader nothing about which one
    was made.
    """
    checks = checks or Checklist("single_direction_controller")
    findings: List[str] = []

    for trail, key, value in _walk(config):
        where = ".".join(trail + (str(key),)) or str(key)
        normalised = _normalise_key(key)

        # alpha_S first, and by value: it is the more specific mistake, and a
        # message that names the strength somebody set is more use than one
        # that names the field it sits in.
        if (normalised in {"alphas", "alphasafety"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool) and float(value) > 0.0):
            raise SingleDirectionError(
                f"{context} sets alpha_S = {value} at {where!r}. Phase 2 "
                f"returned {PHASE2_OUTCOME} and retired "
                f"{DIRECTION_SAFETY_RETIRED} as an actuator: no strength "
                f"applies it, on the final path, at any layer.")

        if normalised in SAFETY_COEFFICIENT_KEYS:
            raise SingleDirectionError(
                f"{context} declares a safety steering coefficient at "
                f"{where!r} (= {value!r}). Phase 2 returned {PHASE2_OUTCOME} "
                f"and retired {DIRECTION_SAFETY_RETIRED} as an actuator: the "
                f"final controller emits {list(CONTROLLER_OUTPUTS)} and has no "
                f"safety coefficient at all. A field set to zero is still the "
                f"field, and the next edit sets it.")

        if isinstance(value, str):
            if _RETIRED_DIRECTION_PATTERN.search(value):
                slot = _slot(trail, key)
                retired_block = (RETIRED_ACTUATOR_DECLARATION_KEY
                                 in trail + (str(key),))
                if slot in DIRECTION_SLOT_KEYS and not retired_block:
                    raise SingleDirectionError(
                        f"{context} declares {value!r} in the actuator slot "
                        f"{where!r}. {DIRECTION_SAFETY_RETIRED} was removed as "
                        f"an actuator by {PHASE2_OUTCOME}; it may be RECORDED "
                        f"as retired, under a "
                        f"{RETIRED_ACTUATOR_DECLARATION_KEY!r} block, and it "
                        f"may not be declared anywhere a final configuration "
                        f"reads an actuator from.")
                if retired_block:
                    findings.append(f"retired direction recorded at {where}")
            if SUPERSEDED_PARAMETERISATION in value.lower():
                raise SingleDirectionError(
                    f"{context} carries the superseded parameterisation "
                    f"({SUPERSEDED_PARAMETERISATION!r}) at {where!r}. That is "
                    f"the two-coefficient design phase 2 ruled out; it is "
                    f"preserved as history in "
                    f"configs/{HISTORICAL_DUAL_CONFIG} and is not the final "
                    f"design.")

    controller = config.get("controller")
    if not isinstance(controller, Mapping):
        raise SingleDirectionError(
            f"{context} declares no 'controller' block; the final path reads "
            f"its outputs from there and does not infer them")
    outputs = controller.get("outputs")
    if not isinstance(outputs, (list, tuple)) or [str(name) for name in outputs] \
            != list(CONTROLLER_OUTPUTS):
        raise SingleDirectionError(
            f"{context} declares controller outputs {outputs!r}; the final "
            f"controller emits exactly {list(CONTROLLER_OUTPUTS)}. "
            f"{len(outputs) if isinstance(outputs, (list, tuple)) else '?'} "
            f"output(s) is not a variation on the one-direction design -- a "
            f"second output is the retired safety coefficient returning under "
            f"another name.")

    maximum = controller.get("max_applied_alpha_R")
    if maximum is None or float(maximum) != MAX_APPLIED_ALPHA_REPAIR:
        raise SingleDirectionError(
            f"{context} declares max_applied_alpha_R {maximum!r}; the frozen "
            f"validated repair strength is {MAX_APPLIED_ALPHA_REPAIR} and the "
            f"applied strength is {APPLIED_STRENGTH_FORMULA}, which cannot "
            f"exceed it")

    checks.record(
        "final_controller_config", True,
        f"{context}: one output {list(CONTROLLER_OUTPUTS)}, no safety "
        f"coefficient, no active {DIRECTION_SAFETY_RETIRED}, alpha_R capped at "
        f"{MAX_APPLIED_ALPHA_REPAIR}"
        + (f"; {len(findings)} retired-direction record(s)" if findings else ""))
    return {"checks": checks, "recorded_retired_direction": findings}


def load_final_controller_config(
        path: Path | str = SINGLE_DIRECTION_CONFIG,
        checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The only way a configuration reaches the final controller.

    Every final-path caller loads through here, so the refusals above are not a
    check somebody remembers to run: they are the load.
    """
    path = Path(path)
    if not path.is_file():
        raise SingleDirectionError(
            f"{path} does not exist; it is the one-direction controller's "
            f"specification. Write it with "
            f"experiments/dual_direction_controller/close_phase2.py")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert_final_controller_config(config, str(path.name), checks)
    return config


# ============================================================ the specification
def build_single_direction_specification() -> Dict[str, Any]:
    """The committed specification, derived from the package and not retyped."""
    return {
        "schema": SCHEMA,
        "name": CONTROLLER_NAME,
        "version": SPECIFICATION_VERSION,
        "status": "SPECIFIED_NOT_IMPLEMENTED",
        "status_meaning": "the architecture, the read position, the strength "
                          "ceiling, the safety brake's role, the out-of-fold "
                          "protocol and the deployment protocol are fixed "
                          "here. No controller is implemented, initialised or "
                          "trained by this file or by the package.",
        "supersedes": {
            "configuration": f"configs/{HISTORICAL_DUAL_CONFIG}",
            "superseded_block": "proposed_controller",
            "superseded_design": "one shared controller emitting a pair of "
                                 "sigmoid coefficients, one per direction",
            "why": f"phase 2 returned {PHASE2_OUTCOME} and retired "
                   f"{DIRECTION_SAFETY_RETIRED} as an actuator",
            "the_historical_file_is_not_rewritten": True,
            "historical_file_standing": "phase-2 metadata. It records what was "
                                        "proposed before phase 2 ran and is "
                                        "kept exactly as it was; it is not a "
                                        "final-controller configuration and "
                                        "the final path refuses it.",
        },
        "phase2": {
            "outcome": PHASE2_OUTCOME,
            "closure": CLOSURE_NAME,
            "readings_sha256": PHASE2_READINGS_SHA256,
            "continuation_selected_by": "the frozen phase-2 decision rule's "
                                        "if_no_strength_qualifies branch, "
                                        "written before any phase-2 output "
                                        "existed",
        },
        "controller": {
            "outputs": list(CONTROLLER_OUTPUTS),
            "output_count": len(CONTROLLER_OUTPUTS),
            "gate": {
                "name": GATE_NAME,
                "activation": GATE_ACTIVATION,
                "range": [GATE_MINIMUM, GATE_MAXIMUM],
                "emitted": "once per prompt, single pass",
            },
            "applied_strength_formula": APPLIED_STRENGTH_FORMULA,
            "frozen_validated_repair_strength":
                FROZEN_VALIDATED_REPAIR_STRENGTH,
            "max_applied_alpha_R": MAX_APPLIED_ALPHA_REPAIR,
            "ceiling_is_arithmetic": f"g_R saturates at {GATE_MAXIMUM} and "
                                     f"{FROZEN_VALIDATED_REPAIR_STRENGTH} x "
                                     f"{GATE_MAXIMUM} = "
                                     f"{MAX_APPLIED_ALPHA_REPAIR}. There is no "
                                     f"coefficient in the parameterisation "
                                     f"that can carry the applied strength "
                                     f"past the validated one.",
            "steered_direction": DIRECTION_REPAIR,
            "steered_direction_count": 1,
            "emits_a_safety_coefficient": False,
            "coefficient_sharing": "the single gate is reused unchanged at "
                                   "every intervention site; it is predicted "
                                   "once per prompt",
        },
        "retired_actuator": {
            "direction": DIRECTION_SAFETY_RETIRED,
            "active": False,
            "retired_by": PHASE2_OUTCOME,
            "recorded_in": CLOSURE_NAME,
            "note": "recorded here as retired so that a reader of this file "
                    "alone knows the second direction was screened and "
                    "removed, rather than never considered.",
        },
        "read_position": {
            "state": READ_STATE,
            "historical_layer": READ_LAYER,
            "state_is": f"the output of historical decoder block {READ_LAYER}",
            "standing": READ_STATE_STANDING,
            "token_position": TOKEN_POSITION,
            "token_position_note": TOKEN_POSITION_NOTE,
            "prefill_only": True,
            "decode_positions_are_not_read": True,
            "diagnostic_states": list(DIAGNOSTIC_STATES),
            "diagnostic_states_are_not_candidates": DIAGNOSTIC_STATES_STANDING,
        },
        "safety_brake": dict(SAFETY_BRAKE),
        "oof_protocol": dict(OOF_PROTOCOL),
        "deployment_protocol": dict(DEPLOYMENT_PROTOCOL),
        "probe_heads": {
            "repair_head": "p_E, E versus B, H excluded from its training "
                           "labels and from its scaler",
            "safety_head": "p_H, H versus (E + B)",
            "definitions_unchanged_by_phase_2": True,
            "run_retained": "outputs/dual_direction_controller/probe_heads_v1",
            "refit_in_this_patch": False,
        },
        "not_implemented_here": [
            "the controller itself: no parameters, no optimiser, no loss, no "
            "training loop",
            "any generation, judging or model load",
            "any reading of the sealed set",
        ],
        "unresolved": {
            "note": "each field below is open and has no default in this file, "
                    "in code, or anywhere else in this repository. Every one "
                    "must be fixed BEFORE the controller is trained, not after "
                    "inspecting generated-response outcomes.",
            "controller_input": "UNRESOLVED",
            "controller_hidden_size": "UNRESOLVED",
            "controller_depth": "UNRESOLVED",
            "controller_activation_function": "UNRESOLVED",
            "controller_loss": "UNRESOLVED",
            "controller_regularisation": "UNRESOLVED",
            "controller_optimiser": "UNRESOLVED",
            "controller_training_schedule": "UNRESOLVED",
            "successful_strength_targets_or_other_supervisory_signal":
                "UNRESOLVED",
            "controller_selection_criterion": "UNRESOLVED",
            "safety_brake_operating_point": "UNRESOLVED",
        },
        "resolved_by_phase_2": {
            "number_of_directions": "one: d_R = B_d1 - E",
            "number_of_controller_outputs": len(CONTROLLER_OUTPUTS),
            "sigmoid_to_maximum_strength_mapping": APPLIED_STRENGTH_FORMULA,
            "controller_training_rows": "the development rows outside the "
                                        "held-out fold, per the out-of-fold "
                                        "protocol; all development rows for "
                                        "the final controller",
        },
        "contains_prompt_or_response_text": False,
    }


SPECIFICATION_COMPARED_FIELDS = (
    "schema", "name", "version", "status", "supersedes", "phase2",
    "controller", "retired_actuator", "read_position", "safety_brake",
    "oof_protocol", "deployment_protocol", "probe_heads", "unresolved",
    "resolved_by_phase_2",
)


def verify_single_direction_specification(
        path: Path | str = SINGLE_DIRECTION_CONFIG,
        checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The committed specification is what this package declares."""
    checks = checks or Checklist("single_direction_controller")
    committed = load_final_controller_config(path, checks)
    derived = build_single_direction_specification()
    for field in SPECIFICATION_COMPARED_FIELDS:
        checks.require(
            f"specification_{field}", committed.get(field) == derived[field],
            f"{Path(path).name}: '{field}' differs from "
            f"routegate.singledirection. A specification that has moved since "
            f"it was written is refused; neither copy is preferred over the "
            f"other.")
    checks.require(
        "specification_holds_no_prompt_or_response_text",
        committed.get("contains_prompt_or_response_text") is False,
        f"{Path(path).name} must state contains_prompt_or_response_text: false")
    return {"committed": committed, "checks": checks}
