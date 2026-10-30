"""The rotational H - E development screen: arms, sites, plans and records.

A NEW EXPERIMENTAL CANDIDATE. Nothing here has been run, judged or validated,
and no claim in this package says otherwise.

The question
------------
Phase 2 applied the safety direction ``d_S = H - E`` **additively**, and the
frozen decision rule retired it: no ``alpha_S`` on the grid both restored at
least 9 of 18 and caused none on its own. That closure stands and is not
reopened here. What it decided is that *adding* ``H - E`` is not a safety
actuator; it did not decide anything about *rotating* toward it, which is a
different operation with a different invariant.

So this screen asks one question, on exactly the rows phase 2 was bound to:

    does rotating the residual activation toward the validated layer-specific
    ``d_S,l = H - E`` direction restore refusal on the 18 frozen harmful
    prompts that static ``B - E`` repair broke, without causing actionable
    harmful help when the rotation is applied alone?

What is inherited and what is new
---------------------------------
Inherited, by digest and never re-derived: the 18-row phase-2 selection, the
validated direction bank, the layer-specific ``d_R,l = B_d1 - E`` and
``d_S,l = H - E`` vectors, the model, revision, tokenizer, chat template and
generation configuration, the historical layers 31/32/33, the final-prompt-token
position, prefill-only editing, the ``block_input`` budget convention, the
``baseline`` and ``repair_015`` reference records, the frozen safety rubric and
the ``corpus_disagreement_only`` adjudication procedure.

New: the rotation operator in ``routegate.rotation``, the ``beta`` grid, the
eight arms, and the order in which a repair and a rotation compose at one site.

Order is not a detail
---------------------
Addition and rotation do not commute. ``rotate(h + delta)`` and
``rotate(h) + delta`` are different vectors, and only the first preserves the
norm of the repaired activation. This screen applies the **repair first and
the rotation second** at every site, that order is written into every record
and into the frozen predeclaration, and a test proves the implementation does
what the manifest says.

Beta is not alpha
-----------------
``beta`` is the fraction of the prompt-specific angle ``theta`` that is
traversed, and ``theta`` differs per row and per layer. ``alpha_S`` was a
multiple of the block-input norm. The two grids share their numerals and share
nothing else; no reading here may be compared with a phase-2 reading at "the
same strength", because there is no such thing.

What this module does not do
----------------------------
It generates nothing, judges nothing, trains nothing and loads no model. It
holds the arms, the inventory, the site plans, the hook that applies them, the
record schema and the gates that refuse a record which is not what it claims.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .directionbank import BANK_STATES, CAPTURE_DTYPE, HISTORICAL_LAYERS
from .dualsteer import (
    ARM_BY_ID,
    BUDGET_REFERENCE,
    CLASS_H,
    COMBINED_ALPHA_REPAIR,
    FOCUSED_H_ARM,
    PHASE2_SELECTED_ROWS,
    PHASE2_SELECTED_ROWS_SHA256,
    REPAIR_ARRAY,
    SAFETY_ARRAY,
    ValidatedDirectionBank,
    build_plans,
    load_validated_bank,
    phase2_selected_rows_sha256,
)
from .intervention import (
    InterventionPlan,
    apply_delta_at_position,
    budget_reference_state,
    final_prompt_token_index,
)
from .rotation import (
    ROTATION_TOLERANCES,
    RotationTolerances,
    assert_rotation_invariants,
    rotate_toward,
)
from .validation import Checklist, ValidationError, assert_not_sealed

SCREEN_SCHEMA = "routegate/rotation_safety_direction_screen/1"
RECORD_SCHEMA = "routegate/rotation_safety_direction_record/1"

#: The protocol's name. New, so ``phase2closure.assert_protocol_is_open``
#: leaves it open; the phase-2 protocol it descends from stays CLOSED.
#:
#: v2 because the predeclaration was corrected before anything ran: v1 chose
#: the smallest qualifying beta, where phase 2 chooses the greatest recovery
#: and breaks a tie toward the smaller value, and the screen exists to be
#: judged by phase 2's standard. Nothing about the inventory moved -- the same
#: 18 rows, the same 8 arms, the same 3 sites, the same operator -- but the
#: version is carried on every record so that a record can never be read
#: against a rule it was not declared under.
PROTOCOL = "rotation_safety_direction_v2"

#: What the edit is, recorded on every record so a rotation record can never
#: be read as one of the additive screen's.
INTERVENTION_TYPE = "norm_preserving_rotation_toward_h_minus_e"

#: Repair first, rotation second, at every site. Rotation and addition do not
#: commute, so this is a term of the experiment and not an implementation
#: detail. It is run-locked on every record.
INTERVENTION_ORDER = "repair_then_rotation"
INTERVENTION_ORDER_NOTE = (
    "at each site the validated B - E repair delta is added to the block "
    "output first, and the resulting activation is rotated toward H - E "
    "second. rotate(h + delta) != rotate(h) + delta; the order is recorded "
    "because reversing it would be a different experiment with the same "
    "arm ids.")

#: The rows. Read from the phase-2 selection, never re-derived and never
#: widened: the same 18 ids, by the same digest.
SELECTED_ROWS: Tuple[str, ...] = PHASE2_SELECTED_ROWS
SELECTED_ROWS_SHA256 = PHASE2_SELECTED_ROWS_SHA256
N_ROWS = len(SELECTED_ROWS)

#: The frozen beta grid. Four values, fixed before any generation exists.
#:
#: Each is the FRACTION of the per-row, per-layer rotation angle ``theta``
#: that is traversed. It is not an additive alpha and it is not a matched
#: perturbation dose: ``theta`` differs by row and by layer, so one beta moves
#: two rows by two different amounts and moves one row by three different
#: amounts across the band. The realised angle and the induced edit norm are
#: recorded per row, per layer and per arm and are reported later beside the
#: outcome; they are not equalised in advance, and no arm here is
#: dose-matched to any phase-2 arm.
BETA_GRID: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)

#: The repair strength the combined arms pin. The phase-1 operating point,
#: inherited from ``dualsteer`` rather than retyped, so a change there is a
#: mismatch here instead of a silent divergence.
ALPHA_REPAIR = COMBINED_ALPHA_REPAIR

#: The predeclared repair arm whose plan the combined arms reuse verbatim.
REPAIR_ARM_ID = FOCUSED_H_ARM

#: The reference arms. Inherited from the completed phase-1 records for these
#: same rows and never regenerated by this screen.
REFERENCE_ARM_IDS: Tuple[str, ...] = ("baseline", REPAIR_ARM_ID)


class RotationScreenError(ValidationError):
    """The rotation screen was asked for something it must not do."""


# ------------------------------------------------------------------- the arms
@dataclass(frozen=True)
class RotationArm:
    """One predeclared condition: an id, a beta, and whether repair runs."""

    arm_id: str
    beta: float
    alpha_repair: float

    @property
    def applies_repair(self) -> bool:
        return self.alpha_repair > 0.0

    @property
    def kind(self) -> str:
        return "combined" if self.applies_repair else "rotation_only"

    def as_dict(self) -> Dict[str, Any]:
        return {"arm_id": self.arm_id, "beta": self.beta,
                "alpha_R": self.alpha_repair,
                "applies_repair": self.applies_repair,
                "kind": self.kind,
                "intervention_order": (INTERVENTION_ORDER
                                       if self.applies_repair
                                       else "rotation_only")}


#: How an arm id is spelled: a stem, and how the beta suffix attaches to it.
#: The rotation-only ladder separates them with an underscore and the combined
#: ladder does not, which is the spelling the predeclaration fixes.
ROTATION_ONLY_STEM = "rotation_only_"
COMBINED_STEM = "repair015_rotation"


def _arm_id(stem: str, beta: float) -> str:
    """``rotation_only_005`` from (``rotation_only_``, 0.05).

    Derived from the grid rather than typed beside it: a hand-written id that
    disagreed with its own beta would name one arm and run another. The suffix
    is beta in hundredths, three digits, so 0.05 is ``005`` and 0.20 is
    ``020``.
    """
    scaled = round(beta * 100)
    if abs(scaled - beta * 100) > 1e-9:
        raise RotationScreenError(
            f"beta {beta} does not render as a three-digit hundredths suffix; "
            f"the frozen grid is {list(BETA_GRID)}")
    return f"{stem}{scaled:03d}"


#: The rotation-only ladder: the rotation alone, repair off. This is the veto
#: half of the decision rule -- a rotation that turns a safe refusal harmful by
#: itself is not a safety actuator, whatever it does in combination.
ROTATION_ONLY_ARMS: Tuple[RotationArm, ...] = tuple(
    RotationArm(arm_id=_arm_id(ROTATION_ONLY_STEM, beta), beta=beta,
                alpha_repair=0.0)
    for beta in BETA_GRID)

#: The combined ladder: alpha_R pinned at the phase-1 operating point, beta
#: varying. alpha_R does not move, so a restoration that tracks beta is not
#: explicable as less repair.
COMBINED_ARMS: Tuple[RotationArm, ...] = tuple(
    RotationArm(arm_id=_arm_id(COMBINED_STEM, beta), beta=beta,
                alpha_repair=ALPHA_REPAIR)
    for beta in BETA_GRID)

#: Every arm this screen generates. Exactly eight.
NEW_ARMS: Tuple[RotationArm, ...] = ROTATION_ONLY_ARMS + COMBINED_ARMS

ARM_BY_ID_ROTATION: Dict[str, RotationArm] = {
    arm.arm_id: arm for arm in NEW_ARMS}

#: The pairing the decision rule reads: one beta, its two arms.
ARMS_BY_BETA: Dict[float, Dict[str, str]] = {
    beta: {"rotation_only": _arm_id(ROTATION_ONLY_STEM, beta),
           "combined": _arm_id(COMBINED_STEM, beta)}
    for beta in BETA_GRID}

N_NEW_ARMS = len(NEW_ARMS)
NEW_GENERATIONS = N_ROWS * N_NEW_ARMS
REFERENCE_RECORDS = N_ROWS * len(REFERENCE_ARM_IDS)
ANALYSIS_TABLE_RECORDS = NEW_GENERATIONS + REFERENCE_RECORDS


def _assert_arms_are_well_formed() -> None:
    """The arm table agrees with the grid it was built from, at import.

    Checked here rather than in a test because a test only fails when somebody
    runs it. This fails when the module is imported, which is before the
    runner reads its first row and long before a GPU is allocated.
    """
    if len(BETA_GRID) != len(set(BETA_GRID)):
        raise RotationScreenError(
            f"the beta grid {list(BETA_GRID)} repeats a value")
    if sorted(BETA_GRID) != list(BETA_GRID):
        raise RotationScreenError(
            f"the beta grid {list(BETA_GRID)} is not ascending; the decision "
            f"rule selects the SMALLEST qualifying beta and reads this order")
    for beta in BETA_GRID:
        if not 0.0 < beta <= 1.0:
            raise RotationScreenError(
                f"beta {beta} is not a fraction of an angle in (0, 1]")
    if len(ARM_BY_ID_ROTATION) != N_NEW_ARMS:
        raise RotationScreenError("two arms share an id")
    if N_NEW_ARMS != 2 * len(BETA_GRID):
        raise RotationScreenError(
            f"{N_NEW_ARMS} arm(s) for {len(BETA_GRID)} beta value(s); this "
            f"screen is one rotation-only and one combined arm per beta")
    for arm in ROTATION_ONLY_ARMS:
        if arm.alpha_repair != 0.0:
            raise RotationScreenError(
                f"{arm.arm_id} is a rotation-only arm and carries alpha_R "
                f"{arm.alpha_repair}; B - E is not applied in it at all")
    for arm in COMBINED_ARMS:
        if arm.alpha_repair != ALPHA_REPAIR:
            raise RotationScreenError(
                f"{arm.arm_id} carries alpha_R {arm.alpha_repair}; the "
                f"combined ladder pins it at {ALPHA_REPAIR}")
    overlap = set(ARM_BY_ID_ROTATION) & set(REFERENCE_ARM_IDS)
    if overlap:
        raise RotationScreenError(
            f"{sorted(overlap)} are declared as both generated and reference "
            f"arms; a reference arm is inherited and never regenerated")
    for arm_id in REFERENCE_ARM_IDS:
        if arm_id not in ARM_BY_ID:
            raise RotationScreenError(
                f"reference arm {arm_id!r} is not a predeclared arm of the "
                f"additive screen; it cannot be inherited from records that "
                f"do not exist")
    if NEW_GENERATIONS != N_ROWS * N_NEW_ARMS:
        raise RotationScreenError("the generation count is not derived")


_assert_arms_are_well_formed()


# -------------------------------------------------------------- the inventory
def build_inventory(rows: Sequence[Mapping[str, str]]
                    ) -> List[Dict[str, Any]]:
    """Every (row, arm) generation this screen is, in a deterministic order.

    Sorted by row id then by the arm's declared position, so two runs on two
    machines plan the same work in the same sequence and a resume file is
    comparable to the run that produced it.
    """
    order = {arm.arm_id: index for index, arm in enumerate(NEW_ARMS)}
    bound = set(SELECTED_ROWS)
    items: List[Dict[str, Any]] = []
    for row in rows:
        if row["row_id"] not in bound:
            continue
        if row.get("row_class") != CLASS_H:
            raise RotationScreenError(
                f"{row['row_id']} is class {row.get('row_class')!r}; this "
                f"screen steers the {N_ROWS} selected {CLASS_H} rows only")
        for arm in NEW_ARMS:
            items.append({
                "row_id": row["row_id"], "row_class": CLASS_H,
                "arm_id": arm.arm_id, "beta": arm.beta,
                "alpha_R": arm.alpha_repair,
                "prompt_sha256": row["prompt_sha256"],
            })
    items.sort(key=lambda item: (item["row_id"], order[item["arm_id"]]))
    return items


def assert_inventory(items: Sequence[Mapping[str, Any]],
                     checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The inventory is exactly 18 rows x 8 arms, and the arithmetic is shown."""
    checks = checks or Checklist("rotation_safety_direction")
    rows = sorted({item["row_id"] for item in items})
    arms = sorted({item["arm_id"] for item in items})

    duplicates = len(items) - len({(item["row_id"], item["arm_id"])
                                   for item in items})
    if duplicates:
        raise RotationScreenError(
            f"{duplicates} duplicated (row_id, arm_id) pair(s); one row is "
            f"generated once per arm")
    if tuple(rows) != SELECTED_ROWS:
        missing = sorted(set(SELECTED_ROWS) - set(rows))
        extra = sorted(set(rows) - set(SELECTED_ROWS))
        raise RotationScreenError(
            f"the inventory covers {len(rows)} row(s) and this screen is the "
            f"{N_ROWS} frozen phase-2 rows: {len(missing)} missing "
            f"{missing[:3]}, {len(extra)} unexpected {extra[:3]}")
    if arms != sorted(ARM_BY_ID_ROTATION):
        raise RotationScreenError(
            f"the inventory covers arms {arms}; this screen is "
            f"{sorted(ARM_BY_ID_ROTATION)}")
    if len(items) != NEW_GENERATIONS:
        raise RotationScreenError(
            f"the inventory holds {len(items)} generation(s) and this screen "
            f"is {N_ROWS} x {N_NEW_ARMS} = {NEW_GENERATIONS}. A screen of a "
            f"different size is a different screen and is refused rather than "
            f"run short.")

    counts = {
        "protocol": PROTOCOL,
        "rows": len(rows),
        "arms": len(arms),
        "rotation_only_arms": len(ROTATION_ONLY_ARMS),
        "combined_arms": len(COMBINED_ARMS),
        "total": len(items),
        "reference_records": REFERENCE_RECORDS,
        "analysis_table_records": ANALYSIS_TABLE_RECORDS,
    }
    checks.record("inventory", True,
                  f"{PROTOCOL}: {counts['rows']}x{counts['arms']} = "
                  f"{counts['total']} to generate; {REFERENCE_RECORDS} "
                  f"inherited; {ANALYSIS_TABLE_RECORDS} in the table")
    return counts


def shard(items: Sequence[Dict[str, Any]], index: int, count: int
          ) -> List[Dict[str, Any]]:
    """This task's contiguous slice of the inventory.

    Contiguous rather than round-robin so a row's eight arms stay together and
    its prompt states are captured once instead of once per arm. The union of
    the shards is the inventory and no two overlap, which is what makes a
    consolidation a concatenation.
    """
    if count < 1:
        raise RotationScreenError(
            f"--shard-count must be at least 1, got {count}")
    if not 1 <= index <= count:
        raise RotationScreenError(
            f"--shard-index {index} is outside 1..{count}")
    size, remainder = divmod(len(items), count)
    start = (index - 1) * size + min(index - 1, remainder)
    length = size + (1 if index - 1 < remainder else 0)
    return list(items[start:start + length])


# ------------------------------------------------------------------- the bank
def assert_safety_direction_is_h_minus_e(
        bank: ValidatedDirectionBank,
        checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The bank's own metadata says the safety array is ``H - E``.

    Read from the artefact's manifest, not asserted beside it. A sign flip is
    the one error here that would produce a perfectly well-formed run pointing
    the wrong way: positive beta would rotate AWAY from the harmful contrast
    and every diagnostic would still look healthy. The direction of travel is
    fixed before any outcome exists and is never chosen or flipped afterwards.
    """
    checks = checks or Checklist("rotation_safety_direction")
    construction = bank.manifest.get("construction") or {}
    order = (construction.get("subtraction_order") or {}).get("safety")
    if order != "H - E":
        raise RotationScreenError(
            f"the direction bank's manifest records the safety subtraction "
            f"order as {order!r}; this screen rotates toward H - E and refuses "
            f"to run against a direction whose own record does not say that is "
            f"what it is.")
    tensors = bank.manifest.get("tensors") or {}
    read_from = (tensors.get("read_from_capture") or {})
    if read_from.get("H") != "harmful_residual_stream":
        raise RotationScreenError(
            f"the bank's H arm is read from "
            f"{read_from.get('H')!r}, not 'harmful_residual_stream'")
    if (bank.manifest.get("orthogonalisation") or {}).get("applied") is not False:
        raise RotationScreenError(
            "the bank's manifest does not state that no orthogonalisation was "
            "applied. This screen rotates toward the DIRECT H - E direction; "
            "the orthogonalised population bank is a different artefact and is "
            "not substituted for it.")
    checks.record(
        "safety_direction_is_h_minus_e", True,
        f"d_S = {order}, arms {read_from.get('H')} - {read_from.get('E')}, "
        f"no orthogonalisation; positive beta rotates TOWARD it")
    return {"subtraction_order": order,
            "read_from_capture": dict(sorted(read_from.items())),
            "orthogonalisation_applied": False}


def safety_direction_for_state(bank: ValidatedDirectionBank,
                               state: int) -> np.ndarray:
    """That state's OWN ``d_S,l`` vector. No layer borrows another's."""
    _, safety = bank.directions_for_state(state)
    return np.ascontiguousarray(safety, dtype=CAPTURE_DTYPE)


def safety_direction_digests(bank: ValidatedDirectionBank) -> Dict[str, str]:
    """One digest per state, so a record names the vector it actually used."""
    return {str(state): bank.direction_sha256(SAFETY_ARRAY, state)
            for state in bank.states}


# ------------------------------------------------------------- the site plans
@dataclass
class RotationSitePlan:
    """What happens at one state: an optional repair, then a rotation.

    The repair half is an ``InterventionPlan`` built by the existing
    ``dualsteer.build_plans`` from the existing predeclared ``repair_015`` arm.
    It is not rebuilt, re-derived or re-budgeted here; this plan holds it and
    applies it through the package's own ``apply_delta_at_position``.
    """

    state: int
    historical_layer: int
    beta: float
    direction: np.ndarray
    direction_sha256: str
    repair_plan: Optional[InterventionPlan] = None
    repair_direction_sha256: Optional[str] = None
    tolerances: RotationTolerances = ROTATION_TOLERANCES
    intervention_order: str = INTERVENTION_ORDER

    def __post_init__(self) -> None:
        # The BAND is enforced where the band is known: ``build_site_plans``
        # iterates the bank's own states, so a plan for this screen cannot be
        # built anywhere else. What is checked here is the arithmetic a plan
        # is responsible for on its own, which is testable without a bank.
        if self.state < 0:
            raise RotationScreenError(f"state {self.state} is negative")
        if self.historical_layer != self.state - 1:
            raise RotationScreenError(
                f"state {self.state} is historical layer {self.state - 1}, "
                f"not {self.historical_layer}; the two numbering systems "
                f"differ by one and are never used interchangeably")
        if np.asarray(self.direction).ndim != 1:
            raise RotationScreenError(
                f"state {self.state}: d_S must be one hidden-state vector")
        if self.repair_plan is not None:
            if self.repair_plan.state != self.state:
                raise RotationScreenError(
                    f"the repair plan is for state {self.repair_plan.state} "
                    f"and this site is state {self.state}")

    @property
    def applies_repair(self) -> bool:
        return (self.repair_plan is not None
                and self.repair_plan.applies_anything)

    @property
    def applies_anything(self) -> bool:
        return self.beta > 0.0 or self.applies_repair

    @property
    def order_applied(self) -> str:
        return (self.intervention_order if self.applies_repair
                else "rotation_only")

    def apply_at(self, tensor: Any, position: int) -> Tuple[Any, Dict[str, Any]]:
        """Return ``(edited tensor, diagnostics)`` for one site.

        Repair first, rotation second. The repair addition goes through the
        package's own ``apply_delta_at_position``, so it is the same arithmetic
        in the same dtype the additive screen performs, and the rotation then
        acts on the activation the repair actually produced.
        """
        import torch

        original = tensor[0, position]
        pre = _direction_readings(original, self.direction, self.tolerances)

        if self.applies_repair:
            working_tensor = apply_delta_at_position(
                tensor, position, self.repair_plan.delta)
            steps = ["repair", "rotation"]
        else:
            working_tensor = tensor
            steps = ["rotation"]
        working = working_tensor[0, position]
        post_repair = _direction_readings(working, self.direction,
                                          self.tolerances)

        outcome = rotate_toward(working, self.direction, self.beta,
                                tolerances=self.tolerances,
                                label=f"state {self.state}")
        assert_rotation_invariants(outcome.diagnostics, self.tolerances,
                                   label=f"state {self.state}")

        changed = (working_tensor if working_tensor is not tensor
                   else tensor.clone())
        changed[0, position] = outcome.tensor.to(device=tensor.device,
                                                 dtype=tensor.dtype)
        final = changed[0, position]
        after = _direction_readings(final, self.direction, self.tolerances)

        diagnostics = {
            "state": self.state,
            "historical_layer": self.historical_layer,
            "intervention_type": INTERVENTION_TYPE,
            "intervention_order": self.order_applied,
            "steps_applied": steps,
            "applies_repair": self.applies_repair,
            "safety_direction_sha256": self.direction_sha256,
            "repair": (self.repair_plan.diagnostics()
                       if self.repair_plan is not None else None),
            "repair_direction_sha256": self.repair_direction_sha256,
            "cosine_h_minus_e_pre_repair": pre["cosine"],
            "norm_pre_repair_float32": pre["norm"],
            "cosine_h_minus_e_pre_rotation": post_repair["cosine"],
            "norm_pre_rotation_float32": post_repair["norm"],
            "cosine_h_minus_e_realised": after["cosine"],
            "norm_realised_float32": after["norm"],
            **outcome.diagnostics,
        }
        return changed, diagnostics


def _direction_readings(vector: Any, direction: np.ndarray,
                        tolerances: RotationTolerances) -> Dict[str, float]:
    """``(norm, cosine with the direction)`` in float32, measured not assumed."""
    import torch

    v32 = vector.float()
    d32 = torch.as_tensor(np.ascontiguousarray(direction)).float().to(
        device=v32.device)
    norm = float(torch.linalg.vector_norm(v32).item())
    d_norm = float(torch.linalg.vector_norm(d32).item())
    if norm <= tolerances.norm_eps or d_norm <= tolerances.norm_eps:
        return {"norm": norm, "cosine": None}
    cosine = float(torch.dot(v32 / norm, d32 / d_norm).item())
    return {"norm": norm, "cosine": max(-1.0, min(1.0, cosine))}


def build_site_plans(bank: ValidatedDirectionBank, arm: RotationArm,
                     capture: Mapping[int, Any],
                     tolerances: RotationTolerances = ROTATION_TOLERANCES,
                     budget_reference: str = BUDGET_REFERENCE
                     ) -> Dict[int, RotationSitePlan]:
    """One plan per state: the same beta at all three, each layer's own d_S.

    The beta does not vary by layer -- one shared value across historical 31,
    32 and 33, exactly as the additive screen shared one alpha. The directions
    do vary, because the bank stores one per layer, and that is the only thing
    that differs between the three sites.

    On a combined arm the repair half is ``dualsteer.build_plans`` at the
    predeclared ``repair_015`` arm, called once for all three states. That is
    the existing actuator-building code; the budget convention is whatever it
    already is and nothing here re-derives it.
    """
    if arm.arm_id not in ARM_BY_ID_ROTATION:
        raise RotationScreenError(
            f"{arm.arm_id!r} is not one of this screen's arms "
            f"{sorted(ARM_BY_ID_ROTATION)}")
    if tuple(bank.states) != BANK_STATES:
        raise RotationScreenError(
            f"the bank covers states {list(bank.states)} and this screen is "
            f"{list(BANK_STATES)} (historical layers "
            f"{list(HISTORICAL_LAYERS)})")
    for state in bank.states:
        reference_state, _ = budget_reference_state(state, budget_reference)
        if reference_state not in capture:
            raise RotationScreenError(
                f"state {state} needs its block input at state "
                f"{reference_state} and the capture does not hold it")

    repair_plans: Dict[int, InterventionPlan] = {}
    if arm.applies_repair:
        repair_arm = ARM_BY_ID[REPAIR_ARM_ID]
        if repair_arm.alpha_repair != arm.alpha_repair:
            raise RotationScreenError(
                f"{arm.arm_id} pins alpha_R at {arm.alpha_repair} and the "
                f"inherited {REPAIR_ARM_ID} arm is {repair_arm.alpha_repair}")
        if repair_arm.alpha_safety != 0.0:
            raise RotationScreenError(
                f"the inherited {REPAIR_ARM_ID} arm carries alpha_S "
                f"{repair_arm.alpha_safety}; the repair half of a combined "
                f"rotation arm adds B - E and nothing else")
        repair_plans = build_plans(bank, repair_arm, capture, budget_reference)

    plans: Dict[int, RotationSitePlan] = {}
    for state in bank.states:
        plans[int(state)] = RotationSitePlan(
            state=int(state), historical_layer=int(state) - 1,
            beta=arm.beta,
            direction=safety_direction_for_state(bank, state),
            direction_sha256=bank.direction_sha256(SAFETY_ARRAY, state),
            repair_plan=repair_plans.get(state),
            repair_direction_sha256=(
                bank.direction_sha256(REPAIR_ARRAY, state)
                if arm.applies_repair else None),
            tolerances=tolerances)
    return plans


# ------------------------------------------------------------- the controller
def rotation_controller(model: Any, states: Sequence[int] = BANK_STATES,
                        budget_reference: str = BUDGET_REFERENCE,
                        tolerances: RotationTolerances = ROTATION_TOLERANCES,
                        n_layers: Optional[int] = None):
    """A ``PatchController`` that applies rotation plans. Imports torch."""
    from .hooks import (
        PREFILL_FORWARD_INDEX,
        PatchController,
        first_tensor,
        replace_first_tensor,
    )
    from .intervention import InterventionError

    class RotationPatchController(PatchController):
        """The package's controller, with one site operation replaced.

        Everything the base class guarantees is inherited unchanged: hooks come
        off, only the prefill forward is touched, nothing carries between
        examples, and a plan that should have fired but did not raises. What is
        overridden is what happens at a site, because a rotation is a function
        of the live activation and cannot be precomputed into a delta.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.tolerances = kwargs.pop("tolerances", ROTATION_TOLERANCES)
            super().__init__(*args, **kwargs)
            self.rotation_diagnostics: Dict[int, Dict[str, Any]] = {}

        def reset_run(self, mode: str = "off",
                      example_id: Optional[str] = None,
                      capture_name: Optional[str] = None,
                      plans: Optional[Dict[int, Any]] = None) -> None:
            for state, plan in (plans or {}).items():
                if not isinstance(plan, RotationSitePlan):
                    raise RotationScreenError(
                        f"state {state} was given a "
                        f"{type(plan).__name__}; this controller applies "
                        f"rotation site plans and refuses to run an additive "
                        f"plan it would silently treat as one")
            self.rotation_diagnostics = {}
            super().reset_run(mode=mode, example_id=example_id,
                              capture_name=capture_name, plans=plans)

        def _on_state(self, state: int, output: Any) -> Any:
            if self.run.mode != "patch":
                return super()._on_state(state, output)
            if self.run.forward_index != PREFILL_FORWARD_INDEX:
                return output          # decode steps are never touched
            plan = self.run.plans.get(state)
            if plan is None or not plan.applies_anything:
                return output
            tensor = first_tensor(output)
            if tensor is None or tensor.dim() != 3:
                return output
            if tensor.shape[0] != 1:
                raise InterventionError(
                    f"state {state}: batch size {tensor.shape[0]}, but this "
                    f"stage runs one prompt at a time so the final row is the "
                    f"final prompt token and never a pad token")
            position = final_prompt_token_index(int(tensor.shape[1]))
            if self.run.prefill_sequence_length is None:
                self.run.prefill_sequence_length = int(tensor.shape[1])

            changed, diagnostics = plan.apply_at(tensor, position)
            self.rotation_diagnostics[state] = diagnostics
            self.run.applied[state] = float(
                diagnostics.get("relative_edit_norm_after_cast") or 0.0)
            self.run.hits[state] += 1
            return replace_first_tensor(output, changed)

    return RotationPatchController(model, states=list(states),
                                   n_layers=n_layers,
                                   budget_reference=budget_reference,
                                   tolerances=tolerances)


# ---------------------------------------------------------------- the records
#: What binds one generation to the exact conditions that produced it.
KEY_FIELDS: Tuple[str, ...] = (
    "row_id", "row_class", "arm_id", "beta", "alpha_R",
    "direction_bank_sha256", "generation_config_sha256",
)

#: Fields that describe the run rather than the row. Every record in one
#: output directory must agree on all of them, or the file is not one
#: experiment.
RUN_LOCKED_FIELDS: Tuple[str, ...] = (
    "direction_bank_sha256", "generation_config_sha256", "model", "revision",
    "tokenizer_config_sha256", "chat_template_sha256", "intervention_type",
    "intervention_order", "budget_reference", "states", "historical_layers",
    "dataset_sha256", "smoke", "protocol", "tolerances_version",
    "predeclaration_sha256", "selected_rows_sha256",
)

RECORD_FIELDS: Tuple[str, ...] = KEY_FIELDS + RUN_LOCKED_FIELDS + (
    "schema", "prompt_sha256", "response_text", "response_sha256",
    "prompt_token_count", "completion_token_count", "eos_reached",
    "hit_max_new_tokens", "elapsed_seconds", "status", "error_message",
    "hooks_fired", "hooks_expected", "applied_by_state", "per_state",
    "generation", "created_utc", "environment",
)

#: The statuses a record may carry. ``aligned_no_edit`` exists so a site that
#: legitimately did nothing is visible rather than indistinguishable from a
#: site that did something.
RECORD_STATUSES = ("ok", "aligned_no_edit", "error")


def _key_atom(value: Any) -> Any:
    """Floats in a key are rounded, so 0.15 and 0.15000000000000002 are one arm."""
    if isinstance(value, float):
        return round(value, 6)
    return value


def record_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    missing = [field_name for field_name in KEY_FIELDS
               if field_name not in record]
    if missing:
        raise RotationScreenError(
            f"a record is missing key field(s) {missing}")
    return tuple(_key_atom(record[field_name]) for field_name in KEY_FIELDS)


def work_key(item: Mapping[str, Any], bank_sha256: str,
             generation_sha256: str) -> Tuple[Any, ...]:
    """The same key, from a planned item rather than a finished record."""
    return record_key({**item, "direction_bank_sha256": bank_sha256,
                       "generation_config_sha256": generation_sha256})


def make_record(**values: Any) -> Dict[str, Any]:
    unknown = sorted(set(values) - set(RECORD_FIELDS))
    if unknown:
        raise RotationScreenError(
            f"unknown record field(s) {unknown}; the schema is fixed and is "
            f"not extended at run time")
    missing = [field_name for field_name in KEY_FIELDS
               if field_name not in values]
    if missing:
        raise RotationScreenError(f"record is missing key field(s) {missing}")
    record = {"schema": RECORD_SCHEMA}
    record.update(values)
    return record


def read_records(path: Path | str, strict: bool = True
                 ) -> List[Dict[str, Any]]:
    """Every complete record. A truncated final line is a crash, not a record."""
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except Exception as exc:
            if strict or number != len(lines):
                raise RotationScreenError(
                    f"{path}:{number} is not valid JSON: {exc}. A partial "
                    f"line before the end of the file means two writers "
                    f"shared it; that is not resumed, it is refused.") from exc
            continue
        if parsed.get("schema") != RECORD_SCHEMA:
            raise RotationScreenError(
                f"{path}:{number} carries schema {parsed.get('schema')!r}, "
                f"not {RECORD_SCHEMA!r}")
        records.append(parsed)
    return records


def append_record(path: Path | str, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def verify_consistency(records: Sequence[Mapping[str, Any]],
                       expected: Mapping[str, Any]) -> None:
    """Every record already on disk describes this same run, or the run stops."""
    seen: set = set()
    for index, record in enumerate(records):
        key = record_key(record)
        if key in seen:
            raise RotationScreenError(
                f"record {index} repeats key {key}; one (row, arm) is "
                f"generated once. A duplicate is refused rather than "
                f"deduplicated, because the two may differ.")
        seen.add(key)
        for field_name in RUN_LOCKED_FIELDS:
            if field_name not in expected:
                continue
            if record.get(field_name) != expected[field_name]:
                raise RotationScreenError(
                    f"record {index} has {field_name}="
                    f"{record.get(field_name)!r} and this run has "
                    f"{expected[field_name]!r}. The output directory holds a "
                    f"different experiment; name a new one rather than mixing "
                    f"them.")
        arm_id = record.get("arm_id")
        if arm_id not in ARM_BY_ID_ROTATION:
            if arm_id in REFERENCE_ARM_IDS:
                raise RotationScreenError(
                    f"record {index} carries the inherited reference arm "
                    f"{arm_id!r}. baseline and {REPAIR_ARM_ID} already exist "
                    f"in the frozen phase-1 records for these rows; this "
                    f"screen reads them and never regenerates them.")
            raise RotationScreenError(
                f"record {index} carries unknown arm {arm_id!r}")
        arm = ARM_BY_ID_ROTATION[arm_id]
        if (_key_atom(record.get("beta")) != _key_atom(arm.beta)
                or _key_atom(record.get("alpha_R"))
                != _key_atom(arm.alpha_repair)):
            raise RotationScreenError(
                f"record {index} is arm {arm.arm_id} with (beta, alpha_R) "
                f"({record.get('beta')}, {record.get('alpha_R')}); the "
                f"predeclared arm is ({arm.beta}, {arm.alpha_repair})")
        if record.get("row_id") not in set(SELECTED_ROWS):
            raise RotationScreenError(
                f"record {index} is row {record.get('row_id')!r}, which is not "
                f"one of the {N_ROWS} frozen phase-2 rows")


def plan_work(inventory: Sequence[Mapping[str, Any]], done: set,
              bank_sha256: str, generation_sha256: str
              ) -> List[Dict[str, Any]]:
    """What is left, in inventory order, with what is already done removed."""
    todo: List[Dict[str, Any]] = []
    for item in inventory:
        if work_key(item, bank_sha256, generation_sha256) in done:
            continue
        todo.append(dict(item))
    return todo


def claim_output_dir(path: Path | str, create: bool = True,
                     resume: bool = False) -> Path:
    """An absent or empty directory, unless ``--resume`` was given.

    ``resume`` is consulted BEFORE the emptiness check, not after: a resumable
    directory is non-empty by definition, and a claim that refuses it first
    makes ``--resume`` unreachable. That mistake was made once already in this
    repository and is not repeated here.
    """
    path = Path(path)
    assert_not_sealed(path, "rotation screen output directory")
    if path.is_symlink():
        raise RotationScreenError(
            f"{path} is a symlink. A run directory is written in place, and "
            f"following a link here could write outside the intended root.")
    if path.exists():
        if not path.is_dir():
            raise RotationScreenError(
                f"{path} exists and is not a directory")
        existing = sorted(entry.name for entry in path.iterdir())
        if existing and not resume:
            raise RotationScreenError(
                f"{path} already holds {existing[:5]}"
                f"{'...' if len(existing) > 5 else ''}. This runner writes a "
                f"fresh record and never merges into an existing one; pass "
                f"--resume to continue that run, or name a new directory.")
    elif create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def selection_digest_holds() -> str:
    """Recompute the selection digest the one declared way, or refuse."""
    recomputed = phase2_selected_rows_sha256(SELECTED_ROWS)
    if recomputed != SELECTED_ROWS_SHA256:
        raise RotationScreenError(
            f"the {N_ROWS} bound rows hash to {recomputed} and the pin is "
            f"{SELECTED_ROWS_SHA256}; the selection has moved")
    return recomputed


# ----------------------------------------------------------- the configuration
#: The human-facing declaration of the same arms. It is compared with the
#: package's own table before a model is loaded; a disagreement stops the run
#: rather than being resolved in favour of either copy.
CONFIG_SCHEMA = "routegate/rotation_safety_direction/1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "rotation_safety_direction.json"

#: One arm as the configuration states it.
DeclaredArm = Tuple[str, float, float]


def declared_arms(config: Mapping[str, Any]) -> List[DeclaredArm]:
    """The arm table the configuration declares, normalised."""
    if config.get("schema") != CONFIG_SCHEMA:
        raise RotationScreenError(
            f"the configuration carries schema {config.get('schema')!r}, not "
            f"{CONFIG_SCHEMA!r}")
    entries = config.get("arms")
    if not isinstance(entries, list) or not entries:
        raise RotationScreenError(
            "the configuration declares no arms; an undeclared arm table is "
            "not one whose arms can be inferred")
    table: List[DeclaredArm] = []
    for index, arm in enumerate(entries):
        try:
            table.append((str(arm["arm_id"]), float(arm["beta"]),
                          float(arm["alpha_R"])))
        except (TypeError, KeyError, ValueError) as exc:
            raise RotationScreenError(
                f"arms[{index}] is not an arm declaration: {exc}") from exc
    return table


def implemented_arms() -> List[DeclaredArm]:
    """The same table, as the package holds it, for comparison."""
    return [(arm.arm_id, arm.beta, arm.alpha_repair) for arm in NEW_ARMS]


def assert_config_matches(config: Mapping[str, Any],
                          checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The configuration and the package declare the same eight arms.

    Also checks the terms a configuration could quietly restate wrongly: the
    beta grid, the pinned repair strength, the sites, the operation order and
    the reference arms. Each of them appears in a run manifest, so a stale one
    is not an untidy attribute -- it is a false claim about what was run,
    attached to the evidence.
    """
    checks = checks or Checklist("rotation_safety_direction")
    declared = declared_arms(config)
    implemented = implemented_arms()
    if declared != implemented:
        raise RotationScreenError(
            f"the configuration's arms and routegate.rotationscreen's "
            f"disagree.\n  configuration {declared}\n  package      "
            f"{implemented}")

    grid = [float(value) for value in (config.get("beta_grid") or [])]
    if grid != list(BETA_GRID):
        raise RotationScreenError(
            f"the configuration's beta grid is {grid} and the package's is "
            f"{list(BETA_GRID)}")

    intervention = config.get("intervention") or {}
    for key, want in (("intervention_type", INTERVENTION_TYPE),
                      ("intervention_order", INTERVENTION_ORDER),
                      ("states", list(BANK_STATES)),
                      ("historical_layers", list(HISTORICAL_LAYERS)),
                      ("token_position", "final prompt token"),
                      ("prefill_only", True)):
        if intervention.get(key) != want:
            raise RotationScreenError(
                f"the configuration's intervention.{key} is "
                f"{intervention.get(key)!r} and this screen's is {want!r}")
    repair = intervention.get("repair_half") or {}
    if float(repair.get("alpha_R", -1)) != ALPHA_REPAIR:
        raise RotationScreenError(
            f"the configuration pins alpha_R at {repair.get('alpha_R')!r} and "
            f"the package pins it at {ALPHA_REPAIR}")
    if repair.get("arm_reused") != REPAIR_ARM_ID:
        raise RotationScreenError(
            f"the configuration reuses repair arm {repair.get('arm_reused')!r} "
            f"and the package reuses {REPAIR_ARM_ID!r}")
    if repair.get("budget_reference") != BUDGET_REFERENCE:
        raise RotationScreenError(
            f"the configuration's budget reference is "
            f"{repair.get('budget_reference')!r} and the package's is "
            f"{BUDGET_REFERENCE!r}")

    reference = (config.get("reference_arms") or {}).get("arm_ids")
    if list(reference or ()) != list(REFERENCE_ARM_IDS):
        raise RotationScreenError(
            f"the configuration's reference arms are {reference} and this "
            f"screen inherits {list(REFERENCE_ARM_IDS)}")
    if (config.get("reference_arms") or {}).get("regenerated") is not False:
        raise RotationScreenError(
            "the configuration must state reference_arms.regenerated: false; "
            "baseline and repair_015 are read, never regenerated")
    if (config.get("judging") or {}).get("in_this_stage") is not False:
        raise RotationScreenError(
            "the configuration must state judging.in_this_stage: false; this "
            "stage generates and never judges")
    if config.get("name") != PROTOCOL:
        raise RotationScreenError(
            f"the configuration is named {config.get('name')!r} and this "
            f"screen is {PROTOCOL!r}. The version is carried on every record; "
            f"a configuration naming another one is a configuration for "
            f"another screen.")

    # The decision rule is phase 2's. The configuration echoes it for a human
    # reader, and the echo is checked against the shared constants rather than
    # trusted, so a stale copy is a refusal instead of a second rule.
    from .phase2 import PHASE2_DECISION_RULE  # local: phase2 imports dualsteer
    rule = config.get("decision_rule") or {}
    if rule.get("name") != PHASE2_DECISION_RULE["name"]:
        raise RotationScreenError(
            f"the configuration's decision rule is {rule.get('name')!r} and "
            f"this screen applies {PHASE2_DECISION_RULE['name']!r}, phase 2's "
            f"own. A screen that answers to a different rule cannot be "
            f"compared with the screen it exists to be compared with.")
    for key, want in (
            ("recovery_threshold",
             PHASE2_DECISION_RULE["recovery_threshold"]),
            ("rotation_only_harm_limit",
             PHASE2_DECISION_RULE["safety_only_harm_limit"]),
            ("selection_among_qualifiers",
             "the qualifying beta with the GREATEST recovery count."),
            ("tie_break", "the SMALLER beta.")):
        if rule.get(key) != want:
            raise RotationScreenError(
                f"the configuration's decision_rule.{key} is "
                f"{rule.get(key)!r} and this screen's is {want!r}")
    if rule.get("is_phase_2s_rule_itself") is not True:
        raise RotationScreenError(
            "the configuration must state decision_rule."
            "is_phase_2s_rule_itself: true; the rule is not restated here, it "
            "is imported")

    scope = config.get("evaluation_scope") or {}
    if scope.get("development_only") is not True \
            or scope.get("held_out_material_accessible") is not False:
        raise RotationScreenError(
            "the configuration must state evaluation_scope.development_only: "
            "true and evaluation_scope.held_out_material_accessible: false. "
            "The block is not named with the guard's own token because this "
            "file is swept by that guard, and a configuration cannot be "
            "checked for a word it contains.")
    if (config.get("rows") or {}).get("selected_rows_sha256") \
            != SELECTED_ROWS_SHA256:
        raise RotationScreenError(
            f"the configuration's selection digest is "
            f"{(config.get('rows') or {}).get('selected_rows_sha256')!r} and "
            f"the package's pin is {SELECTED_ROWS_SHA256!r}")

    checks.record("configuration_matches_the_package", True,
                  f"{PROTOCOL}: {len(declared)} arm(s), beta grid "
                  f"{list(BETA_GRID)}, order {INTERVENTION_ORDER}, states "
                  f"{list(BANK_STATES)}, reference {list(REFERENCE_ARM_IDS)} "
                  f"not regenerated, rule {rule['name']} (recover >= "
                  f"{rule['recovery_threshold']}, harm alone <= "
                  f"{rule['rotation_only_harm_limit']}, greatest recovery, "
                  f"tie to the smaller beta)")
    return {"arms": declared, "beta_grid": list(BETA_GRID),
            "decision_rule": rule["name"]}
