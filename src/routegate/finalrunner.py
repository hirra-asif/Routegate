"""Condition dispatch for the final evaluation: one call site, three paths.

WHY THIS MODULE EXISTS
----------------------
Protocol v1's runner built a prepared condition and then threw it away: every
condition reached ``generate_response(..., mode="off")`` with no controller, so
all three would have produced byte-identical baseline text and the twelve
telemetry fields would have been blank. Nothing had been generated when that
was found.

The correction is structural rather than careful. :func:`execute_item` is the
only path to generation and it **requires** a :class:`ConditionRuntime`. There
is no argument by which a caller can ask for a steered condition and get an
unsteered generation, and :func:`assert_record_is_complete` refuses a record
that claims an intervention it cannot evidence.

THE THREE PATHS
---------------
``baseline``
    no controller is constructed, no hook is installed, ``mode="off"``,
    ``hooks_expected == hooks_fired == 0``.

``dynamic_raw``
    a capture pass reads package state 32 at the final prompt token; ``g_R``
    comes from the all-development refit through
    ``livecontroller.gate_from_activation``; ``twosite.build_two_site_plans``
    builds the edit at historical blocks 31 and 32 through the validated
    actuator's own arithmetic; generation runs in ``mode="patch"`` and both
    hooks must fire.

``safety_first_routegate``
    a capture pass reads package state 45; ``p_E`` and ``p_H`` come from the
    two deployment sensors; the frozen thresholds decide. If the gate fires the
    frozen static actuator is installed at historical blocks 31, 32 and 33 and
    all three hooks must fire. If it does not fire the row is a **declared**
    no-op: ``gate_fires`` is False, ``hooks_expected`` is 0 and
    ``noop_reason`` says so -- which is a different record from a row whose
    wiring was missing, and the two cannot be confused.

Every path uses the existing tested runtime, controller and hook
implementation. Nothing here re-implements an edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .validation import ValidationError

# The three condition ids, imported by name from the frozen protocol.
COND_BASELINE = "baseline"
COND_DYNAMIC = "dynamic_raw"
COND_ROUTEGATE = "safety_first_routegate"

#: dynamic_raw: read state 32, apply at historical blocks 31 and 32.
DYNAMIC_READ_STATE = 32
DYNAMIC_BLOCKS: Tuple[int, ...] = (31, 32)
DYNAMIC_STATES: Tuple[int, ...] = (32, 33)
DYNAMIC_HOOKS = len(DYNAMIC_BLOCKS)

#: safety_first_routegate: read state 45, apply at historical blocks 31-33.
GATE_READ_STATE = 45
STATIC_BLOCKS: Tuple[int, ...] = (31, 32, 33)
STATIC_STATES: Tuple[int, ...] = (32, 33, 34)
STATIC_HOOKS = len(STATIC_BLOCKS)
STATIC_ALPHA = 0.15

NOOP_GATE_DID_NOT_FIRE = "gate_did_not_fire"

#: The telemetry every record must carry, and which conditions must fill each.
TELEMETRY_FIELDS: Tuple[str, ...] = (
    "gate_read_state", "gate_p_E", "gate_p_H", "gate_fires",
    "threshold_E", "threshold_H", "controller_read_state", "g_R",
    "applied_alpha_R", "applied_at_historical_blocks",
    "hooks_expected", "hooks_fired",
)
GATE_TELEMETRY: Tuple[str, ...] = (
    "gate_read_state", "gate_p_E", "gate_p_H", "gate_fires",
    "threshold_E", "threshold_H")
CONTROLLER_TELEMETRY: Tuple[str, ...] = (
    "controller_read_state", "g_R", "applied_alpha_R")


class DispatchError(ValidationError):
    """A condition was asked to generate in a way it must not."""


@dataclass(frozen=True)
class ConditionRuntime:
    """Everything one condition needs, verified, before a model is loaded."""

    condition: str
    steers: bool
    hook_states: Tuple[int, ...] = ()
    capture_state: Optional[int] = None
    apply_blocks: Tuple[int, ...] = ()
    hooks_expected: int = 0
    bundle: Any = None            # gate sensors (safety_first_routegate)
    thresholds: Any = None
    controller_checkpoint: Any = None      # refit controller (dynamic_raw)
    direction_bank: Any = None             # B_d1 - E bank (dynamic_raw)
    actuator: Any = None                   # RQ2 static actuator (routegate)

    def __post_init__(self) -> None:
        if self.condition == COND_BASELINE:
            if self.steers or self.hook_states or self.hooks_expected:
                raise DispatchError(
                    "baseline installs no hook and expects none")
        elif self.condition == COND_DYNAMIC:
            if self.controller_checkpoint is None or self.direction_bank is None:
                raise DispatchError(
                    "dynamic_raw needs the refit controller and the direction "
                    "bank; it does not fall back to an unsteered generation")
        elif self.condition == COND_ROUTEGATE:
            if self.bundle is None or self.thresholds is None:
                raise DispatchError(
                    "safety_first_routegate needs both sensors and both "
                    "thresholds; it does not fall back to an unsteered "
                    "generation")
            if self.actuator is None:
                raise DispatchError(
                    "safety_first_routegate needs the frozen static actuator")
        else:
            raise DispatchError(f"unknown condition {self.condition!r}")

    @property
    def needs_controller(self) -> bool:
        """Whether a PatchController must be constructed for this condition."""
        return self.condition != COND_BASELINE


def blank_telemetry() -> Dict[str, Any]:
    """Every telemetry field present and explicitly null."""
    return {name: None for name in TELEMETRY_FIELDS}


# ================================================================= execution
def execute_item(runtime: ConditionRuntime, *, model: Any, tokenizer: Any,
                 controller: Any, prompt: str, item_id: str,
                 generation: Any, prompt_mode: str,
                 stop_strings: Optional[Sequence[str]] = None
                 ) -> Dict[str, Any]:
    """Generate one item under one condition. The ONLY path to generation.

    ``runtime`` is required, so a steered condition cannot reach an unsteered
    generation by omission. What comes back is the produced text plus the full
    telemetry block, already consistent with what actually fired.
    """
    from .runtime import capture_prompt_states, generate_response

    if not isinstance(runtime, ConditionRuntime):
        raise DispatchError(
            "execute_item requires the prepared ConditionRuntime; a condition "
            "that is prepared and then discarded is exactly the v1 defect")

    telemetry = blank_telemetry()

    # ---------------------------------------------------------- baseline
    if runtime.condition == COND_BASELINE:
        if controller is not None:
            raise DispatchError(
                "baseline installs no controller; one was supplied")
        produced = generate_response(model, tokenizer, prompt, generation,
                                     controller=None, mode="off",
                                     example_id=item_id,
                                     prompt_mode=prompt_mode,
                                     stop_strings=stop_strings)
        telemetry.update({"hooks_expected": 0, "hooks_fired": 0})
        return {**produced, **telemetry, "mode": "off",
                "noop_reason": None, "condition": runtime.condition}

    if controller is None:
        raise DispatchError(
            f"{runtime.condition} steers and was given no controller; an "
            f"unsteered generation is not a substitute")

    # ------------------------------------------------------- read the state
    capture_prompt_states(model, tokenizer, prompt, controller,
                          capture_name="target", example_id=item_id,
                          config=generation, prompt_mode=prompt_mode)
    capture = dict(controller.capture_for("target"))
    if runtime.capture_state not in capture:
        raise DispatchError(
            f"{runtime.condition}: the capture pass produced no state "
            f"{runtime.capture_state}; the condition cannot be decided")
    activation = np.asarray(capture[runtime.capture_state], dtype=np.float64)

    # ----------------------------------------------------- dynamic_raw
    if runtime.condition == COND_DYNAMIC:
        from .livecontroller import gate_from_activation
        from .twosite import build_two_site_plans
        from .gatecontroller import ALPHA_R

        g_r = float(gate_from_activation(activation.reshape(1, -1),
                                         runtime.controller_checkpoint)[0])
        applied = float(ALPHA_R * g_r)
        telemetry.update({
            "controller_read_state": runtime.capture_state,
            "g_R": g_r, "applied_alpha_R": applied,
            "applied_at_historical_blocks": list(runtime.apply_blocks),
            "hooks_expected": runtime.hooks_expected})
        if not applied > 0.0:
            raise DispatchError(
                f"{item_id}: dynamic_raw computed an applied coefficient of "
                f"{applied!r}. A row with no coefficient is not silently "
                f"downgraded to a baseline generation.")
        plans = build_two_site_plans(runtime.direction_bank, capture,
                                     alpha=ALPHA_R, g_r=g_r,
                                     blocks=runtime.apply_blocks)
        produced = generate_response(model, tokenizer, prompt, generation,
                                     controller=controller, mode="patch",
                                     plans=plans, example_id=item_id,
                                     prompt_mode=prompt_mode,
                                     stop_strings=stop_strings)
        telemetry["hooks_fired"] = int(produced.get("hooks_fired", 0))
        return {**produced, **telemetry, "mode": "patch",
                "noop_reason": None, "condition": runtime.condition}

    # ------------------------------------------ safety_first_routegate
    decision = runtime.bundle.decide(activation)
    telemetry.update({
        "gate_read_state": runtime.capture_state,
        "gate_p_E": decision.p_E, "gate_p_H": decision.p_H,
        "gate_fires": bool(decision.fires),
        "threshold_E": runtime.thresholds.threshold_E,
        "threshold_H": runtime.thresholds.threshold_H})

    if not decision.fires:
        # A DECLARED no-op. Distinct from missing wiring: the gate was
        # evaluated, its probabilities are recorded, and it said no.
        telemetry.update({"hooks_expected": 0, "hooks_fired": 0,
                          "applied_at_historical_blocks": []})
        produced = generate_response(model, tokenizer, prompt, generation,
                                     controller=controller, mode="off",
                                     example_id=item_id,
                                     prompt_mode=prompt_mode,
                                     stop_strings=stop_strings)
        return {**produced, **telemetry, "mode": "off",
                "noop_reason": NOOP_GATE_DID_NOT_FIRE,
                "condition": runtime.condition}

    from .onepassbudget import block_input_state
    from .steering import build_steering_plan

    plans = {}
    for block in runtime.apply_blocks:
        state = block + 1
        reference_state = block_input_state(block)
        if reference_state not in capture:
            raise DispatchError(
                f"{item_id}: state {state} is budgeted against state "
                f"{reference_state}, which the capture does not hold")
        plans[state] = build_steering_plan(
            state=state,
            direction=runtime.actuator.direction_for_state(state),
            strength=STATIC_ALPHA,
            reference_vector=capture[reference_state],
            budget_reference="block_input")
    telemetry.update({
        "applied_alpha_R": STATIC_ALPHA,
        "applied_at_historical_blocks": list(runtime.apply_blocks),
        "hooks_expected": runtime.hooks_expected})
    produced = generate_response(model, tokenizer, prompt, generation,
                                 controller=controller, mode="patch",
                                 plans=plans, example_id=item_id,
                                 prompt_mode=prompt_mode,
                                 stop_strings=stop_strings)
    telemetry["hooks_fired"] = int(produced.get("hooks_fired", 0))
    return {**produced, **telemetry, "mode": "patch",
            "noop_reason": None, "condition": runtime.condition}


# ================================================================= refusals
def assert_record_is_complete(record: Mapping[str, Any]) -> None:
    """Refuse a record that claims an intervention it cannot evidence."""
    condition = record.get("condition")
    if condition not in (COND_BASELINE, COND_DYNAMIC, COND_ROUTEGATE):
        raise DispatchError(f"unknown condition {condition!r}")
    for name in TELEMETRY_FIELDS:
        if name not in record:
            raise DispatchError(
                f"{condition}: the record carries no {name}; the twelve "
                f"telemetry fields are written for every row, explicitly null "
                f"where a condition does not use them")

    expected = record.get("hooks_expected")
    fired = record.get("hooks_fired")
    if expected is None or fired is None:
        raise DispatchError(f"{condition}: hook counts are not recorded")
    if int(expected) != int(fired):
        raise DispatchError(
            f"{condition}: hooks_expected {expected} and hooks_fired {fired} "
            f"disagree. A run that did not patch is never recorded as patched.")

    if condition == COND_BASELINE:
        if record.get("mode") != "off" or int(expected) != 0:
            raise DispatchError(
                "baseline runs with mode 'off' and zero hooks")
        return

    if record.get("mode") == "off" and condition == COND_DYNAMIC:
        raise DispatchError(
            "dynamic_raw may never generate with mode 'off'; that is the v1 "
            "defect and it is refused rather than recorded")

    if condition == COND_DYNAMIC:
        for name in CONTROLLER_TELEMETRY:
            if record.get(name) is None:
                raise DispatchError(f"dynamic_raw: {name} is not recorded")
        if not float(record["applied_alpha_R"]) > 0.0:
            raise DispatchError(
                "dynamic_raw: no applied coefficient was recorded")
        if int(fired) != DYNAMIC_HOOKS:
            raise DispatchError(
                f"dynamic_raw: {fired} hook(s) fired, expected "
                f"{DYNAMIC_HOOKS}")
        return

    # safety_first_routegate
    for name in GATE_TELEMETRY:
        if record.get(name) is None:
            raise DispatchError(
                f"safety_first_routegate: {name} is not recorded; the gate "
                f"decision must be evidenced whether or not it fired")
    if record.get("gate_fires"):
        if record.get("mode") != "patch":
            raise DispatchError(
                "safety_first_routegate: the gate fired but generation ran "
                "unsteered")
        if int(fired) != STATIC_HOOKS:
            raise DispatchError(
                f"safety_first_routegate: the gate fired and {fired} hook(s) "
                f"fired, expected {STATIC_HOOKS}. A gate-fired row with no "
                f"hooks is missing wiring, not a no-op.")
    else:
        if int(expected) != 0 or int(fired) != 0:
            raise DispatchError(
                "safety_first_routegate: the gate did not fire, so no hook "
                "may fire")
        if record.get("noop_reason") != NOOP_GATE_DID_NOT_FIRE:
            raise DispatchError(
                "safety_first_routegate: a gate no-op is declared with "
                f"noop_reason {NOOP_GATE_DID_NOT_FIRE!r}, so it can never be "
                f"read as missing wiring")


__all__ = [
    "COND_BASELINE", "COND_DYNAMIC", "COND_ROUTEGATE",
    "CONTROLLER_TELEMETRY", "ConditionRuntime", "DYNAMIC_BLOCKS",
    "DYNAMIC_HOOKS", "DYNAMIC_READ_STATE", "DYNAMIC_STATES", "DispatchError",
    "GATE_READ_STATE", "GATE_TELEMETRY", "NOOP_GATE_DID_NOT_FIRE",
    "STATIC_ALPHA", "STATIC_BLOCKS", "STATIC_HOOKS", "STATIC_STATES",
    "TELEMETRY_FIELDS", "assert_record_is_complete", "blank_telemetry",
    "execute_item",
]
