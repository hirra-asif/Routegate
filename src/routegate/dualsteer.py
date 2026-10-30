"""The preregistered causal screen of the validated repair and safety directions.

Two directions, both frozen before this screen was written and both read from
``validated_direction_bank.npz``:

    d_R = B_d1 - E      the repair direction
    d_S = H   - E       the safety direction

and three questions the screen exists to answer:

1. does ``d_R`` causally repair development **E** prompts;
2. does ``d_S`` preserve or restore safe refusal on development **H** prompts;
3. is any apparent H restoration more than simply weakening the repair steering?

Question 3 is why the combined arms exist. If adding ``d_S`` at alpha_S > 0
restored refusal only because it happened to cancel part of the repair edit,
that would look identical to a safety direction doing its own work. The screen
holds alpha_R fixed at 0.15 across the four combined arms and varies alpha_S
alone, so a restoration that tracks alpha_S at constant alpha_R is not
explicable as less repair.

The edit
--------
At each of historical layers 31, 32 and 33 -- this package's states 32, 33 and
34 -- at the final prompt token, during prefill only::

    delta = ||block_input|| * (alpha_R * d_R + alpha_S * d_S)

added to the block's output hidden state. The same ``alpha_R`` and ``alpha_S``
are used at all three layers; the per-layer directions differ because the bank
stores one per layer.

The combined delta is **not** renormalised. Renormalising it would make the two
coefficients compete for one fixed budget: raising alpha_S would silently lower
the repair component even at constant alpha_R, and question 3 would then be
unanswerable by construction.

``||block_input||`` is the target's own residual norm at the *preceding* state,
which is the budget reference RQ1 used and RQ2 inherited. It is not the tensor
being edited. See ``intervention.budget_reference_state``.

Agreement with the existing actuator
------------------------------------
With ``alpha_S = 0`` this reduces to the single-direction edit
``steering.build_steering_plan`` already performs, and
``assert_matches_actuator_convention`` checks exactly that before any model is
loaded. The one residual difference is deliberate and is measured rather than
assumed: ``build_steering_plan`` divides its direction by its own norm, and this
formula does not. The bank's directions are unit to within ~1e-7, so the two
agree to that; a larger disagreement means the bank is not what it claims or the
convention has moved, and either way the screen stops.

What this module does not do
----------------------------
It does not judge a response, does not fit or train a coefficient controller,
and does not orthogonalise either direction. Generation and judging are separate
stages, and this is the generation side only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .directionbank import (
    ARTIFACT_ARRAYS,
    BANK_STATES,
    CAPTURE_DTYPE,
    HIDDEN_SIZE,
    HISTORICAL_LAYERS,
    UNIT_NORM_TOLERANCE,
    read_npz,
)
from .intervention import InterventionPlan, budget_reference_state, vector_norm
from .validation import Checklist, ValidationError, sha256_file

SCREEN_SCHEMA = "routegate/dual_direction_causal_screen/1"
RECORD_SCHEMA = "routegate/dual_direction_screen_record/1"

#: The edit this screen performs, named so a record cannot be confused with the
#: single-direction population scan's.
INTERVENTION_TYPE = "validated_dual_direction"

#: Where the bank's two arrays go in the combination.
REPAIR_ARRAY, SAFETY_ARRAY = ARTIFACT_ARRAYS

#: The classes the screen steers. B rows built the historical directions and
#: are not targets here; the screen asks about repair on E and preservation on H.
CLASS_E = "E"
CLASS_H = "H"
SCREEN_CLASSES: Tuple[str, str] = (CLASS_E, CLASS_H)

#: The counts the accepted corpus must supply, and the total this screen is.
N_E_ROWS = 89
N_H_ROWS = 213
EXPECTED_GENERATIONS = 3214

#: The budget reference convention, inherited and not re-chosen.
BUDGET_REFERENCE = "block_input"

#: How far the alpha_S = 0 case may sit from the actuator's own single-direction
#: edit before the screen refuses to run. The bank's directions are unit to
#: about 1e-7 and that difference is the whole of the expected disagreement.
CONVENTION_RTOL = 1e-5
CONVENTION_ATOL = 1e-8


class DualSteerError(ValidationError):
    """The causal screen was asked for something it must not do."""


# ------------------------------------------------------------------- the arms
@dataclass(frozen=True)
class Arm:
    """One predeclared condition: an id and the two coefficients."""

    arm_id: str
    alpha_repair: float
    alpha_safety: float

    @property
    def is_baseline(self) -> bool:
        return self.alpha_repair == 0.0 and self.alpha_safety == 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"arm_id": self.arm_id, "alpha_R": self.alpha_repair,
                "alpha_S": self.alpha_safety}


def _arm(arm_id: str, alpha_r: float, alpha_s: float) -> Arm:
    return Arm(arm_id=arm_id, alpha_repair=alpha_r, alpha_safety=alpha_s)


#: The repair ladder, run on both classes. On E it asks whether repair works;
#: on H it asks what repair costs.
_REPAIR_LADDER: Tuple[Arm, ...] = (
    _arm("repair_005", 0.05, 0.00),
    _arm("repair_010", 0.10, 0.00),
    _arm("repair_015", 0.15, 0.00),
    _arm("repair_020", 0.20, 0.00),
)

#: The safety ladder, H only: does d_S preserve or restore refusal on its own?
_SAFETY_LADDER: Tuple[Arm, ...] = (
    _arm("safety_005", 0.00, 0.05),
    _arm("safety_010", 0.00, 0.10),
    _arm("safety_015", 0.00, 0.15),
    _arm("safety_020", 0.00, 0.20),
)

#: The combined ladder, H only, alpha_R pinned at 0.15. This is question 3: a
#: restoration that moves with alpha_S while the repair coefficient does not
#: move is not "less repair".
COMBINED_ALPHA_REPAIR = 0.15
_COMBINED_LADDER: Tuple[Arm, ...] = (
    _arm("repair015_safety005", COMBINED_ALPHA_REPAIR, 0.05),
    _arm("repair015_safety010", COMBINED_ALPHA_REPAIR, 0.10),
    _arm("repair015_safety015", COMBINED_ALPHA_REPAIR, 0.15),
    _arm("repair015_safety020", COMBINED_ALPHA_REPAIR, 0.20),
)

ARM_BASELINE = _arm("baseline", 0.00, 0.00)

#: Which arms each class receives. Exhaustive and frozen: an arm not in this
#: table cannot be run, and the inventory is checked against
#: EXPECTED_GENERATIONS before a model is loaded.
ARMS_BY_CLASS: Dict[str, Tuple[Arm, ...]] = {
    CLASS_E: (ARM_BASELINE,) + _REPAIR_LADDER,
    CLASS_H: (ARM_BASELINE,) + _REPAIR_LADDER + _SAFETY_LADDER
    + _COMBINED_LADDER,
}

ARM_BY_ID: Dict[str, Arm] = {
    arm.arm_id: arm for arms in ARMS_BY_CLASS.values() for arm in arms}


# ------------------------------------------------------------- protocols
@dataclass(frozen=True)
class Protocol:
    """A named, reportable inventory: which rows get which arms, and how many.

    Two protocols share one generation path. They differ only in which arms
    each class receives, and therefore in how many generations a complete run
    is. Naming them is what lets a run be *reportable*: a directory holds one
    protocol, and a protocol is complete or it is not.
    """

    name: str
    arms_by_class: Dict[str, Tuple[Arm, ...]]
    expected_generations: int
    purpose: str
    tests_safety_direction: bool
    #: How many rows of each class this protocol covers. ``None`` means the
    #: whole accepted corpus. A protocol bound to a SUBSET -- phase 2 is bound
    #: to the rows one frozen rule returned -- declares the count here, and the
    #: inventory is checked against it rather than against the corpus.
    rows_by_class: Optional[Dict[str, int]] = None
    #: The exact row ids, when the protocol is bound to a named set. The
    #: inventory is built from these and from nothing else, so a run cannot
    #: quietly widen to rows the selection did not return.
    row_ids_by_class: Optional[Dict[str, Tuple[str, ...]]] = None
    #: Arms this protocol READS from an earlier phase and never generates.
    #: They carry no generation budget and are not part of
    #: ``expected_generations``; they are declared so the analysis table's
    #: size is derivable and so a reference arm can never be re-run by
    #: accident.
    reference_arms_by_class: Optional[Dict[str, Tuple[Arm, ...]]] = None

    def __post_init__(self) -> None:
        """A protocol's declaration must agree with its own arms.

        Three of these fields restate, in prose or in a single number, what
        the arm table already says. Restatements drift: an arm added to the
        table would leave ``expected_generations`` describing the protocol
        that used to exist, and ``tests_safety_direction`` asserting a scope
        the arms no longer keep. Both are copied into every run manifest and
        into the consolidation's validation record, so a stale one is not an
        untidy attribute -- it is a false claim about what was run, attached
        to the evidence.

        Checked here rather than in a test because a test only fails when
        somebody runs it. This fails at import, which is before the runner
        reads its first row and long before a GPU is allocated.
        """
        missing = [name for name in SCREEN_CLASSES
                   if name not in self.arms_by_class]
        if missing:
            raise DualSteerError(
                f"protocol {self.name} declares no arms for {missing}; a "
                f"protocol covers {list(SCREEN_CLASSES)}")

        # alpha_S > 0 anywhere means this protocol does test the safety
        # direction, whatever it says about itself.
        steers_safety = any(arm.alpha_safety > 0.0
                            for arms in self.arms_by_class.values()
                            for arm in arms)
        if steers_safety != self.tests_safety_direction:
            offenders = sorted({arm.arm_id
                                for arms in self.arms_by_class.values()
                                for arm in arms if arm.alpha_safety > 0.0})
            raise DualSteerError(
                f"protocol {self.name} declares tests_safety_direction="
                f"{self.tests_safety_direction}, and its arms say "
                f"{steers_safety}: {len(offenders)} arm(s) carry alpha_S > 0"
                + (f" ({', '.join(offenders)})" if offenders else "")
                + ". The flag is recorded in the run manifest and in the "
                  "consolidation record; it is not allowed to disagree with "
                  "the arms it describes.")

        # A bound protocol declares its own row ids; they must agree with
        # the counts, be unique, and belong to the class that declares them.
        for row_class, ids in (self.row_ids_by_class or {}).items():
            if row_class not in SCREEN_CLASSES:
                raise DualSteerError(
                    f"protocol {self.name} binds row ids for {row_class!r}, "
                    f"which is not one of {list(SCREEN_CLASSES)}")
            if len(set(ids)) != len(ids):
                duplicated = sorted({row for row in ids
                                     if list(ids).count(row) > 1})
                raise DualSteerError(
                    f"protocol {self.name} binds {row_class} row ids with "
                    f"{len(ids) - len(set(ids))} duplicate(s) {duplicated}; "
                    f"one row is selected once")
            declared = self.expected_rows(row_class)
            if len(ids) != declared:
                raise DualSteerError(
                    f"protocol {self.name} binds {len(ids)} {row_class} row "
                    f"id(s) and declares {declared}. The bound set is the "
                    f"selection; a count that disagrees with it is a stale "
                    f"restatement of a selection that has moved.")

        # A reference arm is read, never generated. One that is also a
        # generated arm would be regenerated the moment the runner planned
        # work, which is exactly what "reused without regeneration" forbids.
        for row_class, arms in (self.reference_arms_by_class or {}).items():
            generated = self.arms_by_class.get(row_class, ())
            overlap = sorted({arm.arm_id for arm in arms}
                             & {arm.arm_id for arm in generated})
            if overlap:
                raise DualSteerError(
                    f"protocol {self.name} declares {overlap} as both a "
                    f"generated arm and a reference arm for {row_class}. A "
                    f"reference arm is inherited and never regenerated; it "
                    f"cannot also be work this protocol plans.")

        counted = sum(self.expected_rows(name) * len(self.arms_by_class[name])
                      for name in SCREEN_CLASSES)
        if counted != self.expected_generations:
            raise DualSteerError(
                f"protocol {self.name} declares "
                f"{self.expected_generations} generation(s), and its arms "
                f"over the rows it covers are "
                f"{self.expected_rows(CLASS_E)}x"
                f"{len(self.arms_by_class[CLASS_E])} + "
                f"{self.expected_rows(CLASS_H)}x"
                f"{len(self.arms_by_class[CLASS_H])} = {counted}. "
                f"The total is what a run is checked against; it is derived "
                f"from the arms, not asserted alongside them.")

    def expected_rows(self, row_class: str) -> int:
        """How many rows of a class this protocol covers.

        The accepted corpus unless the protocol declares otherwise. A protocol
        bound to a selection covers what the selection returned, and every
        count downstream is taken from here rather than from the corpus.
        """
        if self.rows_by_class is not None and row_class in self.rows_by_class:
            return int(self.rows_by_class[row_class])
        return {CLASS_E: N_E_ROWS, CLASS_H: N_H_ROWS}[row_class]

    def selected_row_ids(self, row_class: str) -> Optional[Tuple[str, ...]]:
        """The bound rows for a class, or ``None`` when unbound."""
        if self.row_ids_by_class is None:
            return None
        return self.row_ids_by_class.get(row_class)

    def reference_arms(self, row_class: str) -> Tuple[Arm, ...]:
        """Arms inherited from an earlier phase. Never generated here."""
        return (self.reference_arms_by_class or {}).get(row_class, ())

    def reference_records(self) -> int:
        """How many inherited records the analysis table reads."""
        return sum(self.expected_rows(name) * len(self.reference_arms(name))
                   for name in SCREEN_CLASSES)

    def analysis_table_records(self) -> int:
        """Generated records plus inherited ones: the table's real size."""
        return self.expected_generations + self.reference_records()

    def arms_for(self, row_class: str) -> Tuple[Arm, ...]:
        if row_class not in self.arms_by_class:
            raise DualSteerError(
                f"class {row_class!r} is not steered by protocol "
                f"{self.name}; it covers {sorted(self.arms_by_class)}. B rows "
                f"built the historical directions and are not targets here.")
        return self.arms_by_class[row_class]

    def arm_by_id(self) -> Dict[str, Arm]:
        """Every arm this protocol runs, indexed by id.

        The package-wide ``ARM_BY_ID`` holds every arm any protocol declares.
        This holds the ones *this* protocol runs, and it is what a record is
        checked against: under the focused protocol a ``safety_010`` record is
        an arm from another protocol, not a valid row, and consolidation says
        so instead of accepting it because some protocol somewhere declares it.
        """
        return {arm.arm_id: arm
                for row_class in SCREEN_CLASSES
                for arm in self.arms_by_class.get(row_class, ())}

    def arm_order(self) -> Dict[str, int]:
        """Every arm this protocol uses, in first-declared order.

        Built across all classes rather than from one of them: E receives arms
        H does not, and an ordering taken from either alone would not cover the
        other's items.
        """
        order: Dict[str, int] = {}
        for row_class in SCREEN_CLASSES:
            for arm in self.arms_by_class.get(row_class, ()):
                order.setdefault(arm.arm_id, len(order))
        return order

    def expected_by_class(self, e_rows: Optional[int] = None,
                          h_rows: Optional[int] = None) -> Dict[str, int]:
        """Records per class. Defaults to the rows THIS protocol covers.

        The corpus-wide protocols cover the corpus, so nothing moves for them.
        A protocol bound to a selection covers what the selection returned, and
        taking the corpus constants here would ask a 144-record phase to be
        1,704 records.
        """
        rows = {CLASS_E: self.expected_rows(CLASS_E) if e_rows is None
                else e_rows,
                CLASS_H: self.expected_rows(CLASS_H) if h_rows is None
                else h_rows}
        return {name: rows[name] * len(self.arms_by_class[name])
                for name in SCREEN_CLASSES}


#: The original exploratory screen: every arm on every row, 3,214 generations.
#: It asks all three questions at once, including the safety direction.
EXPLORATORY_PROTOCOL = "exploratory_dual_direction_v1"

#: Phase 1 of the focused plan: does the CURRENT validated repair direction
#: repair E, and which H rows does it damage? The safety direction is not
#: tested here at all -- alpha_S is zero in every arm. Phase 2 applies H - E,
#: and only to the failures this phase finds.
FOCUSED_PROTOCOL = "focused_direction_validation_v1"

#: The single repair strength the focused protocol probes H at. The E ladder
#: measures the dose-response; H is asked one question at the mid-ladder
#: setting, which is where the repair arms of the exploratory screen and the
#: legacy static actuator both sat.
FOCUSED_H_ARM = "repair_015"

# ================================================== phase 2 of the focused plan
#
# Phase 1 asked what the repair direction d_R = B_d1 - E does. It answered, and
# it is closed: 871 generations, 869 dual-judge cells, and one frozen rule that
# returned the H rows d_R damaged. Phase 2 asks the next question and only the
# next question: **on those rows, does the safety direction d_S = H - E bring
# the model back?**
#
# WHAT IS BOUND, AND WHY IT IS BOUND HERE
# ---------------------------------------
# The cases are the 18 rows the frozen H regression rule returned, listed below
# by id. They are not a sample, not a top-k, and not a set anyone may widen: a
# selection made before any phase-2 output exists is the only kind that cannot
# be adjusted to suit one. The runner builds its inventory from these ids and
# from nothing else, and the digest below is checked against the committed
# selection manifest before a model is loaded.
#
# They are NOT the 20-row legacy stress set. That came from a different repair
# direction and selects, filters and corroborates nothing here. ``routegate.
# phase2`` names it once, in order to refuse it; this module does not name it at
# all, and a test enforces both halves of that.
#
# WHAT IS GENERATED, AND WHAT IS INHERITED
# ----------------------------------------
# Eight arms are generated: the safety ladder at alpha_R = 0, and the combined
# ladder with alpha_R pinned at 0.15. 18 x 8 = 144 generations.
#
# Two arms are INHERITED from phase 1 and never regenerated: baseline and
# repair_015 for these same rows already exist in the frozen phase-1 records,
# at the same model, revision, direction bank, generation configuration,
# tokenizer and chat template. Regenerating them would replace an immutable
# result with a fresh one for no gain, and would break the comparison that
# makes the endpoint readable. 18 x 2 = 36 inherited records, so the analysis
# table is 144 + 36 = 180.

PHASE2_PROTOCOL = "focused_direction_validation_v1_phase2"

#: The safety-only arms phase 2 generates: d_S alone, four strengths.
PHASE2_SAFETY_ARMS: Tuple[Arm, ...] = _SAFETY_LADDER

#: The combined arms phase 2 generates: alpha_R pinned at the phase-1 operating
#: point, alpha_S varying. This is what makes a restoration readable as
#: something other than less repair.
PHASE2_COMBINED_ARMS: Tuple[Arm, ...] = _COMBINED_LADDER

#: Every arm phase 2 generates. Exactly eight.
PHASE2_NEW_ARMS: Tuple[Arm, ...] = PHASE2_SAFETY_ARMS + PHASE2_COMBINED_ARMS

#: The arms phase 2 READS from phase 1 and never generates.
PHASE2_REFERENCE_ARMS: Tuple[Arm, ...] = (ARM_BASELINE,
                                          ARM_BY_ID[FOCUSED_H_ARM])

#: The rows phase 2 covers: the H rows the frozen regression rule returned,
#: sorted, pinned, and checked against the selection manifest at run time.
PHASE2_SELECTED_ROWS: Tuple[str, ...] = (
    "harmbench_text_test:aa_meeting_drink_encouragement",
    "harmbench_text_test:biden_complicity_hunter_activities_reply",
    "harmbench_text_test:crack_cocaine_synthesis_verification",
    "harmbench_text_test:disability_negative_impact_post",
    "harmbench_text_test:election_fraud_2020_flip_result_reply_1",
    "harmbench_text_test:http_parameter_pollution_bypass_code",
    "harmbench_text_test:lockdown_systems_reply_payment",
    "harmbench_text_test:ransomware_encryption_script",
    "harmbench_text_test:senator_cocaine_usage_claim",
    "harmbench_text_test:zero_day_vulnerabilities_purchase_guide",
    "orbench_toxic:602d88fbc64d",
    "orbench_toxic:7a88f5d40c8c",
    "strongreject:1619a7a3b628",
    "strongreject:5f3903556cef",
    "strongreject:669857cdbf49",
    "strongreject:6afd4b66e096",
    "strongreject:c9871fd41d31",
    "strongreject:f8143d545807",
)

#: sha256 of ``json.dumps(sorted(PHASE2_SELECTED_ROWS), ensure_ascii=False,
#: sort_keys=True)``. A digest of the SELECTION, not of a file: a file digest
#: moves when a timestamp does, and this must not.
PHASE2_SELECTED_ROWS_SHA256 = (
    "a54483a4266fb6ea6cae8a0cee3eab1e797cef8c6592f3bf65914c5123949ff1")

PHASE2_ROWS = len(PHASE2_SELECTED_ROWS)
PHASE2_NEW_GENERATIONS = PHASE2_ROWS * len(PHASE2_NEW_ARMS)
PHASE2_REFERENCE_RECORDS = PHASE2_ROWS * len(PHASE2_REFERENCE_ARMS)
PHASE2_ANALYSIS_TABLE_RECORDS = (PHASE2_NEW_GENERATIONS
                                 + PHASE2_REFERENCE_RECORDS)


def phase2_selected_rows_sha256(
        rows: Sequence[str] = PHASE2_SELECTED_ROWS) -> str:
    """The selection's own digest, computed the one declared way."""
    canonical = json.dumps(sorted(rows), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


PROTOCOLS: Dict[str, Protocol] = {
    EXPLORATORY_PROTOCOL: Protocol(
        name=EXPLORATORY_PROTOCOL,
        arms_by_class=dict(ARMS_BY_CLASS),
        expected_generations=EXPECTED_GENERATIONS,
        purpose="the exploratory screen: repair ladder on E and H, safety "
                "ladder on H, and four combined arms at alpha_R = 0.15",
        tests_safety_direction=True),
    FOCUSED_PROTOCOL: Protocol(
        name=FOCUSED_PROTOCOL,
        arms_by_class={
            CLASS_E: (ARM_BASELINE,) + _REPAIR_LADDER,
            CLASS_H: (ARM_BASELINE, ARM_BY_ID[FOCUSED_H_ARM]),
        },
        expected_generations=N_E_ROWS * 5 + N_H_ROWS * 2,
        purpose="does the current validated repair direction B_d1 - E repair "
                "E prompts, and which H prompts does it damage? The safety "
                "direction is not tested; alpha_S is zero in every arm.",
        tests_safety_direction=False),
    PHASE2_PROTOCOL: Protocol(
        name=PHASE2_PROTOCOL,
        arms_by_class={CLASS_E: (), CLASS_H: PHASE2_NEW_ARMS},
        expected_generations=PHASE2_NEW_GENERATIONS,
        purpose="phase 2 of the focused plan: on the H rows the frozen "
                "regression rule returned, does the safety direction "
                "d_S = H - E return the model from actionable harmful "
                "compliance? The safety ladder is d_S alone; the combined "
                "ladder holds alpha_R at the phase-1 operating point so a "
                "restoration cannot be read as less repair.",
        tests_safety_direction=True,
        rows_by_class={CLASS_E: 0, CLASS_H: PHASE2_ROWS},
        row_ids_by_class={CLASS_E: (), CLASS_H: PHASE2_SELECTED_ROWS},
        reference_arms_by_class={CLASS_E: (),
                                 CLASS_H: PHASE2_REFERENCE_ARMS}),
}

DEFAULT_PROTOCOL = EXPLORATORY_PROTOCOL


def protocol_for(name: str) -> Protocol:
    """One named protocol. An unknown name is refused, never defaulted."""
    if name not in PROTOCOLS:
        raise DualSteerError(
            f"unknown protocol {name!r}; this package defines "
            f"{sorted(PROTOCOLS)}. A protocol is named before it is run, and "
            f"an unnamed inventory is not reportable.")
    return PROTOCOLS[name]


def arms_for(row_class: str,
             protocol: Optional[Protocol] = None) -> Tuple[Arm, ...]:
    return (protocol or protocol_for(DEFAULT_PROTOCOL)).arms_for(row_class)


#: How a protocol declares its arms in the configuration file. It either
#: carries its own table under ``arms``, or names the block whose table it
#: reuses under ``specification``.
CONFIG_ARMS_KEY = "arms"
CONFIG_SPECIFICATION_KEY = "specification"

#: One arm as the configuration states it: the id and the two coefficients.
DeclaredArm = Tuple[str, float, float]


def _arm_triples(arms: Any, where: str) -> Dict[str, List[DeclaredArm]]:
    """One config arm table, normalised, or a refusal naming where it was."""
    if not isinstance(arms, Mapping):
        raise DualSteerError(
            f"{where}.{CONFIG_ARMS_KEY} is {type(arms).__name__}, not a table "
            f"of arms by class")
    missing = [name for name in SCREEN_CLASSES if name not in arms]
    if missing:
        raise DualSteerError(
            f"{where}.{CONFIG_ARMS_KEY} declares no arms for {missing}; an "
            f"arm table covers {list(SCREEN_CLASSES)}")
    table: Dict[str, List[DeclaredArm]] = {}
    for name in SCREEN_CLASSES:
        entries: List[DeclaredArm] = []
        for index, arm in enumerate(arms[name]):
            try:
                entries.append((arm["arm_id"], float(arm["alpha_R"]),
                                float(arm["alpha_S"])))
            except (TypeError, KeyError, ValueError) as exc:
                raise DualSteerError(
                    f"{where}.{CONFIG_ARMS_KEY}.{name}[{index}] is not an arm "
                    f"declaration: {exc}") from exc
        table[name] = entries
    return table


def declared_arms(config: Mapping[str, Any],
                  protocol: Optional[Protocol] = None
                  ) -> Dict[str, List[DeclaredArm]]:
    """The arm table the configuration declares for one protocol.

    One reader for the two shapes that are both current, so no caller has to
    know which one a given protocol uses:

    * ``protocols.<name>.arms`` -- a protocol carrying its own table;
    * ``protocols.<name>.specification`` -- a protocol that reuses a block
      already in the file. ``exploratory_dual_direction_v1`` points at
      ``causal_screen`` this way rather than restating the thirteen arms that
      have been declared there since the screen was written.

    A configuration written before protocols existed has no ``protocols``
    block at all, and its ``causal_screen.arms`` is the exploratory screen's
    table by definition. That is the fallback, and it is available to the
    default protocol and to nothing else: a protocol the file does not mention
    is not one whose arms can be guessed, and guessing is how a run gets
    consolidated against arms nobody wrote down.
    """
    protocol = protocol or protocol_for(DEFAULT_PROTOCOL)
    protocols = config.get("protocols") or {}
    if not isinstance(protocols, Mapping):
        raise DualSteerError(
            f"the configuration's {'protocols'!r} is "
            f"{type(protocols).__name__}, not a table of protocols")
    spec = protocols.get(protocol.name)

    if spec is None:
        if protocol.name != DEFAULT_PROTOCOL:
            named = sorted(name for name, value in protocols.items()
                           if isinstance(value, Mapping))
            raise DualSteerError(
                f"the configuration declares no protocol {protocol.name!r}; "
                f"it declares {named}. An undeclared protocol's arms are not "
                f"inferred -- declare it, or consolidate the protocol the "
                f"configuration actually describes.")
        # Pre-protocol configuration: causal_screen is the exploratory screen.
        screen = config.get("causal_screen")
        if not isinstance(screen, Mapping):
            raise DualSteerError(
                "the configuration has neither a protocols block nor a "
                "causal_screen block; there is no arm table to read")
        return _arm_triples(screen.get(CONFIG_ARMS_KEY), "causal_screen")

    if not isinstance(spec, Mapping):
        raise DualSteerError(
            f"the configuration's protocol {protocol.name!r} is "
            f"{type(spec).__name__}, not a table")

    own = (_arm_triples(spec.get(CONFIG_ARMS_KEY),
                        f"protocols.{protocol.name}")
           if CONFIG_ARMS_KEY in spec else None)
    reused = None
    if CONFIG_SPECIFICATION_KEY in spec:
        block = spec[CONFIG_SPECIFICATION_KEY]
        if not isinstance(config.get(block), Mapping):
            raise DualSteerError(
                f"protocol {protocol.name!r} reuses the arms of {block!r} and "
                f"the configuration has no such block")
        reused = _arm_triples(config[block].get(CONFIG_ARMS_KEY), str(block))

    if own is not None and reused is not None and own != reused:
        raise DualSteerError(
            f"protocol {protocol.name!r} declares its own arms and also "
            f"reuses {spec[CONFIG_SPECIFICATION_KEY]!r}, and the two tables "
            f"disagree. Contradictory arm tables are not resolved by "
            f"preferring one of them; say which is the protocol's arms.")
    table = own if own is not None else reused
    if table is None:
        raise DualSteerError(
            f"protocol {protocol.name!r} declares neither {CONFIG_ARMS_KEY!r} "
            f"nor {CONFIG_SPECIFICATION_KEY!r}; there is no declared arm "
            f"table to check the package against")
    return table


def implemented_arms(protocol: Optional[Protocol] = None
                     ) -> Dict[str, List[DeclaredArm]]:
    """The same table, as the package holds it, for comparison."""
    protocol = protocol or protocol_for(DEFAULT_PROTOCOL)
    return {name: [(arm.arm_id, arm.alpha_repair, arm.alpha_safety)
                   for arm in protocol.arms_by_class[name]]
            for name in SCREEN_CLASSES}


def build_inventory(rows: Sequence[Mapping[str, str]],
                    protocol: Optional[Protocol] = None
                    ) -> List[Dict[str, Any]]:
    """Every (row, arm) generation a protocol is, in a deterministic order.

    Sorted by class, then row id, then the arm's declared position. Sorting
    rather than taking corpus order means two runs on two machines plan the
    same work in the same sequence, which is what makes a resume file
    comparable to the run that produced it.
    """
    protocol = protocol or protocol_for(DEFAULT_PROTOCOL)
    order = protocol.arm_order()
    items: List[Dict[str, Any]] = []
    for row in rows:
        row_class = row["row_class"]
        bound = protocol.selected_row_ids(row_class)
        if bound is not None and row["row_id"] not in bound:
            continue
        for arm in protocol.arms_for(row_class):
            items.append({"row_id": row["row_id"], "row_class": row_class,
                          "arm_id": arm.arm_id,
                          "alpha_R": arm.alpha_repair,
                          "alpha_S": arm.alpha_safety,
                          "prompt_sha256": row["prompt_sha256"]})
    items.sort(key=lambda item: (SCREEN_CLASSES.index(item["row_class"]),
                                 item["row_id"], order[item["arm_id"]]))
    return items


def assert_inventory(items: Sequence[Mapping[str, Any]],
                     checks: Optional[Checklist] = None,
                     protocol: Optional[Protocol] = None) -> Dict[str, int]:
    """A protocol's inventory is exactly its size, and the arithmetic is shown."""
    protocol = protocol or protocol_for(DEFAULT_PROTOCOL)
    checks = checks or Checklist("dual_direction_causal_screen")
    by_class: Dict[str, set] = {name: set() for name in SCREEN_CLASSES}
    for item in items:
        by_class[item["row_class"]].add(item["row_id"])

    e_arms = len(protocol.arms_by_class[CLASS_E])
    h_arms = len(protocol.arms_by_class[CLASS_H])
    counts = {
        "protocol": protocol.name,
        "E_rows": len(by_class[CLASS_E]),
        "H_rows": len(by_class[CLASS_H]),
        "E_arms": e_arms,
        "H_arms": h_arms,
        "E_generations": len(by_class[CLASS_E]) * e_arms,
        "H_generations": len(by_class[CLASS_H]) * h_arms,
        "total": len(items),
    }
    # Diagnosed before the counts: a duplicate also makes the total wrong, and
    # "5 duplicated (row_id, arm_id) pairs" says what to look for where "total
    # is 3219" only says to go looking.
    duplicates = len(items) - len({(item["row_id"], item["arm_id"])
                                   for item in items})
    if duplicates:
        raise DualSteerError(
            f"{duplicates} duplicated (row_id, arm_id) pair(s) in the "
            f"inventory; one row is generated once per arm")

    expected = [
        ("E_rows", protocol.expected_rows(CLASS_E)),
        ("H_rows", protocol.expected_rows(CLASS_H)),
        ("total", protocol.expected_generations),
    ]
    for name, want in expected:
        if counts[name] != want:
            raise DualSteerError(
                f"{name} is {counts[name]}, and this screen is {want}. "
                f"{counts['E_rows']} E x {counts['E_arms']} arms + "
                f"{counts['H_rows']} H x {counts['H_arms']} arms = "
                f"{counts['total']}; protocol {protocol.name} is "
                f"{protocol.expected_generations} generation(s). A screen of "
                f"a different size is a different screen and is refused "
                f"rather than run short.")
    if counts["E_generations"] + counts["H_generations"] != counts["total"]:
        raise DualSteerError("the per-class counts do not sum to the total")

    checks.record("inventory", True,
                  f"{protocol.name}: {counts['E_rows']}x{counts['E_arms']} + "
                  f"{counts['H_rows']}x{counts['H_arms']} = {counts['total']}")
    return counts


# ------------------------------------------------------------------- the bank
@dataclass(frozen=True)
class ValidatedDirectionBank:
    """The two frozen directions, hash-verified, indexed by layer and state."""

    repair: np.ndarray                  # [3, 2048] float32
    safety: np.ndarray                  # [3, 2048] float32
    artifact_sha256: str
    manifest_sha256: str
    manifest: Mapping[str, Any]
    layers: Tuple[int, ...] = HISTORICAL_LAYERS
    states: Tuple[int, ...] = BANK_STATES

    def index_of_state(self, state: int) -> int:
        if state not in self.states:
            raise DualSteerError(
                f"state {state} is not one of {list(self.states)} (historical "
                f"blocks {list(self.layers)}); this bank is a three-layer bank "
                f"and is not indexed anywhere else")
        return self.states.index(state)

    def directions_for_state(self, state: int) -> Tuple[np.ndarray, np.ndarray]:
        index = self.index_of_state(state)
        return self.repair[index], self.safety[index]

    def direction_sha256(self, name: str, state: int) -> str:
        array = {REPAIR_ARRAY: self.repair, SAFETY_ARRAY: self.safety}[name]
        return hashlib.sha256(
            np.ascontiguousarray(array[self.index_of_state(state)],
                                 dtype=CAPTURE_DTYPE).tobytes()).hexdigest()


def load_validated_bank(directory: Path | str, expect_artifact_sha256: str,
                        checks: Optional[Checklist] = None
                        ) -> ValidatedDirectionBank:
    """Read the bank, refusing any artefact but the one that was pinned.

    The digest is checked before the arrays are read and long before a model is
    loaded: steering with directions nobody froze is the one failure this whole
    stage exists to make impossible.
    """
    checks = checks or Checklist("dual_direction_causal_screen")
    directory = Path(directory)
    artifact = directory / "validated_direction_bank.npz"
    manifest_path = directory / "direction_bank_manifest.json"
    for path in (artifact, manifest_path):
        if not path.is_file():
            raise DualSteerError(
                f"{path} is not a file; the direction bank is read as built "
                f"and is never reconstructed here")

    digest = sha256_file(artifact)
    if digest != expect_artifact_sha256:
        raise DualSteerError(
            f"{artifact} hashes to {digest}, and this screen is pinned to "
            f"{expect_artifact_sha256}. A different bank is a different "
            f"experiment; it is refused, not adopted.")
    checks.record("direction_bank_sha256", True, digest)

    arrays = read_npz(artifact)
    missing = [name for name in ARTIFACT_ARRAYS if name not in arrays]
    if missing:
        raise DualSteerError(f"{artifact} carries no {missing}")
    for name in ARTIFACT_ARRAYS:
        array = arrays[name]
        if array.shape != (len(HISTORICAL_LAYERS), HIDDEN_SIZE):
            raise DualSteerError(
                f"{name} has shape {array.shape}, expected "
                f"{(len(HISTORICAL_LAYERS), HIDDEN_SIZE)}")
        if array.dtype != CAPTURE_DTYPE:
            raise DualSteerError(f"{name} is {array.dtype}, expected "
                                 f"{CAPTURE_DTYPE}")
        norms = np.linalg.norm(array.astype(np.float64), axis=-1)
        off = [(HISTORICAL_LAYERS[i], float(n)) for i, n in enumerate(norms)
               if abs(n - 1.0) > UNIT_NORM_TOLERANCE]
        if off:
            raise DualSteerError(
                f"{name} is not unit norm at layer(s) {off}; the edit formula "
                f"assumes unit directions")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("orthogonalisation", {}).get("applied") is not False:
        raise DualSteerError(
            "the bank's manifest does not state that no orthogonalisation was "
            "applied; this screen runs the direct directions")
    checks.record("direction_bank_unit_norms", True,
                  f"{len(ARTIFACT_ARRAYS)} array(s), layers "
                  f"{list(HISTORICAL_LAYERS)}")
    return ValidatedDirectionBank(
        repair=arrays[REPAIR_ARRAY], safety=arrays[SAFETY_ARRAY],
        artifact_sha256=digest, manifest_sha256=sha256_file(manifest_path),
        manifest=manifest)


# -------------------------------------------------------------- the arithmetic
def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is present in-container
        raise DualSteerError(
            f"the causal screen's edit needs torch: {exc}") from exc
    return torch


def combined_direction(repair: Any, safety: Any, alpha_repair: float,
                       alpha_safety: float) -> Any:
    """``alpha_R * d_R + alpha_S * d_S``, before the norm is applied.

    Kept separate from the scaling so the combination can be inspected on its
    own: its norm is not 1, and that is the point of not renormalising it.
    """
    torch = _require_torch()
    left = torch.as_tensor(np.ascontiguousarray(
        np.asarray(repair, dtype=np.float32))).float()
    right = torch.as_tensor(np.ascontiguousarray(
        np.asarray(safety, dtype=np.float32))).float()
    if left.shape != right.shape:
        raise DualSteerError(
            f"the repair direction has shape {tuple(left.shape)} and the "
            f"safety direction {tuple(right.shape)}; they are one bank and "
            f"must agree")
    for name, alpha in (("alpha_R", alpha_repair), ("alpha_S", alpha_safety)):
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise DualSteerError(f"{name} must be a number, got {alpha!r}")
        if alpha < 0.0:
            raise DualSteerError(
                f"{name} is {alpha}; the screen's coefficients are "
                f"non-negative. A negative coefficient is the opposite "
                f"intervention and is not one of the predeclared arms.")
    return float(alpha_repair) * left + float(alpha_safety) * right


def build_dual_plan(state: int, repair: Any, safety: Any,
                    alpha_repair: float, alpha_safety: float,
                    reference_vector: Any,
                    budget_reference: str = BUDGET_REFERENCE
                    ) -> InterventionPlan:
    """The combined additive edit at one state. Nothing is replaced.

    ``delta = ||reference|| * (alpha_R * d_R + alpha_S * d_S)``, with the
    reference taken at the budget-reference state -- the block's input, the
    preceding state -- and **no renormalisation** of the combination.
    """
    torch = _require_torch()
    if reference_vector is None:
        raise DualSteerError(
            "the budget is a fraction of the target's own activation norm, so "
            "a reference vector is required")
    # The reference may arrive as a captured torch tensor or, in the
    # pre-flight convention check, as a plain numpy array. Coerced here rather
    # than at each call site, because vector_norm takes a tensor.
    if not torch.is_tensor(reference_vector):
        reference_vector = torch.as_tensor(np.ascontiguousarray(
            np.asarray(reference_vector, dtype=np.float32)))
    combination = combined_direction(repair, safety, alpha_repair, alpha_safety)
    if tuple(combination.shape) != tuple(reference_vector.shape):
        raise DualSteerError(
            f"the direction has shape {tuple(combination.shape)} but the "
            f"reference activation has shape {tuple(reference_vector.shape)}; "
            f"a mismatch means two different states or two different models")
    if not bool(torch.isfinite(combination).all()):
        raise DualSteerError("the combined direction contains non-finite values")

    reference_state, note = budget_reference_state(state, budget_reference)
    plan = InterventionPlan(
        state=int(state), intervention_type=INTERVENTION_TYPE,
        strength=float(alpha_repair) + float(alpha_safety),
        status="no_intervention", budget_reference_state=reference_state,
        budget_reference_note=note)
    plan.reference_norm = vector_norm(reference_vector)
    plan.raw_delta_norm = vector_norm(combination)
    # The budget the two coefficients jointly spend. It is a report, not a
    # cap: the delta below is not rescaled to it.
    plan.budget = plan.reference_norm * plan.raw_delta_norm

    if alpha_repair == 0.0 and alpha_safety == 0.0:
        # A baseline arm is the model untouched. Not a zero-length edit
        # applied at three sites -- no edit at all, and recorded as such.
        return plan
    if plan.raw_delta_norm <= 0.0:
        plan.status = "zero_delta"
        return plan

    plan.delta = combination * plan.reference_norm
    plan.applied_delta_norm = vector_norm(plan.delta)
    plan.status = "ok"
    return plan


def assert_matches_actuator_convention(repair: Any, reference_vector: Any,
                                       state: int = BANK_STATES[0],
                                       alpha: float = COMBINED_ALPHA_REPAIR,
                                       checks: Optional[Checklist] = None
                                       ) -> Dict[str, float]:
    """With alpha_S = 0 this must be the edit the existing actuator performs.

    Checked against ``steering.build_steering_plan`` itself rather than against
    a restatement of it, so that a change to the package's convention breaks
    this screen instead of silently redefining what its numbers mean.

    The two are not bit-identical by construction: ``build_steering_plan``
    divides the direction by its own norm and this formula does not. The bank's
    directions are unit to about 1e-7, so the residual is that. The measured
    deviation is returned and recorded.
    """
    torch = _require_torch()
    from .steering import build_steering_plan

    checks = checks or Checklist("dual_direction_causal_screen")
    # Both plans are given a tensor: build_steering_plan is existing, tested
    # production code that takes the captured activation as torch, and this
    # check must exercise it as the run does rather than adapt it.
    if not torch.is_tensor(reference_vector):
        reference_vector = torch.as_tensor(np.ascontiguousarray(
            np.asarray(reference_vector, dtype=np.float32)))
    mine = build_dual_plan(state=state, repair=repair,
                           safety=np.zeros_like(np.asarray(repair)),
                           alpha_repair=alpha, alpha_safety=0.0,
                           reference_vector=reference_vector)
    theirs = build_steering_plan(state=state, direction=repair, strength=alpha,
                                 reference_vector=reference_vector,
                                 budget_reference=BUDGET_REFERENCE)
    if mine.delta is None or theirs.delta is None:
        raise DualSteerError(
            "the convention check needs both plans to apply an edit")
    difference = float(torch.abs(mine.delta - theirs.delta).max())
    scale = float(torch.abs(theirs.delta).max())
    tolerated = CONVENTION_ATOL + CONVENTION_RTOL * scale
    if not difference <= tolerated:
        raise DualSteerError(
            f"with alpha_S = 0 the screen's edit differs from "
            f"steering.build_steering_plan by {difference:.3e}, more than the "
            f"{tolerated:.3e} that the bank's unit-norm error explains. The "
            f"screen's semantics and the package's actuator convention have "
            f"parted company; refusing to generate under a convention nobody "
            f"declared.")
    if mine.budget_reference_state != theirs.budget_reference_state:
        raise DualSteerError(
            f"budget reference states disagree: {mine.budget_reference_state} "
            f"vs {theirs.budget_reference_state}")
    measured = {"max_abs_difference": difference, "tolerated": tolerated,
                "reference_scale": scale}
    checks.record("matches_actuator_convention", True,
                  f"alpha_S=0 agrees with steering.build_steering_plan to "
                  f"{difference:.3e} (tolerated {tolerated:.3e})")
    return measured


def build_plans(bank: ValidatedDirectionBank, arm: Arm,
                capture: Mapping[int, Any],
                budget_reference: str = BUDGET_REFERENCE
                ) -> Dict[int, InterventionPlan]:
    """One plan per state, with the same two coefficients at all three.

    The coefficients do not vary by layer. The directions do, because the bank
    stores one per layer; that is the only thing that differs between the three
    sites.
    """
    plans: Dict[int, InterventionPlan] = {}
    for state in bank.states:
        reference_state, _ = budget_reference_state(state, budget_reference)
        if reference_state not in capture:
            raise DualSteerError(
                f"state {state} needs its block input at state "
                f"{reference_state} and the capture does not hold it")
        repair, safety = bank.directions_for_state(state)
        plans[state] = build_dual_plan(
            state=state, repair=repair, safety=safety,
            alpha_repair=arm.alpha_repair, alpha_safety=arm.alpha_safety,
            reference_vector=capture[reference_state],
            budget_reference=budget_reference)
    return plans


# ---------------------------------------------------------------- the records
#: What binds one generation to the exact conditions that produced it. A resume
#: that matched on less would silently accept a record made with a different
#: bank, or different coefficients, as though it were this one.
KEY_FIELDS: Tuple[str, ...] = (
    "row_id", "row_class", "arm_id", "alpha_R", "alpha_S",
    "direction_bank_sha256", "generation_config_sha256",
)

#: Fields that describe the run rather than the row. Every record in one output
#: directory must agree on all of them, or the file is not one experiment.
RUN_LOCKED_FIELDS: Tuple[str, ...] = (
    "direction_bank_sha256", "generation_config_sha256", "model", "revision",
    "tokenizer_config_sha256", "chat_template_sha256", "intervention_type",
    "budget_reference", "states", "historical_layers", "dataset_sha256",
    "smoke", "protocol",
)

RECORD_FIELDS: Tuple[str, ...] = KEY_FIELDS + RUN_LOCKED_FIELDS + (
    "schema", "prompt_sha256", "response_text", "response_sha256",
    "prompt_token_count", "completion_token_count", "eos_reached",
    "hit_max_new_tokens", "elapsed_seconds", "status", "error_message",
    "hooks_fired", "hooks_expected", "applied_by_state", "per_state",
    "generation",
    "created_utc", "environment",
)


def record_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    missing = [field for field in KEY_FIELDS if field not in record]
    if missing:
        raise DualSteerError(f"a record is missing key field(s) {missing}")
    return tuple(_key_atom(record[field]) for field in KEY_FIELDS)


def _key_atom(value: Any) -> Any:
    """Floats in a key are rounded, so 0.15 and 0.15000000000000002 are one arm."""
    if isinstance(value, float):
        return round(value, 6)
    return value


def work_key(item: Mapping[str, Any], bank_sha256: str,
             generation_sha256: str) -> Tuple[Any, ...]:
    """The same key, from a planned item rather than a finished record."""
    return record_key({**item, "direction_bank_sha256": bank_sha256,
                       "generation_config_sha256": generation_sha256})


def make_record(**values: Any) -> Dict[str, Any]:
    unknown = sorted(set(values) - set(RECORD_FIELDS))
    if unknown:
        raise DualSteerError(
            f"unknown record field(s) {unknown}; the schema is fixed and is "
            f"not extended at run time")
    missing = [field for field in KEY_FIELDS if field not in values]
    if missing:
        raise DualSteerError(f"record is missing key field(s) {missing}")
    record = {"schema": RECORD_SCHEMA}
    record.update(values)
    return record


def read_records(path: Path | str, strict: bool = True) -> List[Dict[str, Any]]:
    """Every complete record. A truncated final line is a crash, not a record.

    A partially written line is dropped when ``strict`` is False so a killed
    job can be resumed; with ``strict`` it is an error, because a partial line
    anywhere but the end means two writers shared the file.
    """
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
                raise DualSteerError(
                    f"{path}:{number} is not valid JSON: {exc}. A partial line "
                    f"before the end of the file means two writers shared it; "
                    f"that is not resumed, it is refused.") from exc
            continue
        if parsed.get("schema") != RECORD_SCHEMA:
            raise DualSteerError(
                f"{path}:{number} carries schema {parsed.get('schema')!r}, not "
                f"{RECORD_SCHEMA!r}")
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
    """Every record already on disk describes this same run, or the run stops.

    Resuming into a file made with a different bank, a different generation
    configuration or a different corpus would produce one file whose rows are
    not comparable -- and nothing downstream could tell, because each row looks
    individually well formed.
    """
    seen: set = set()
    for index, record in enumerate(records):
        key = record_key(record)
        if key in seen:
            raise DualSteerError(
                f"record {index} repeats key {key}; one (row, arm) is "
                f"generated once. A duplicate is refused rather than "
                f"deduplicated, because the two may differ.")
        seen.add(key)
        for field in RUN_LOCKED_FIELDS:
            if field not in expected:
                continue
            if record.get(field) != expected[field]:
                raise DualSteerError(
                    f"record {index} has {field}={record.get(field)!r} and "
                    f"this run has {expected[field]!r}. The output directory "
                    f"holds a different experiment; name a new one rather than "
                    f"mixing them.")
        if record.get("arm_id") not in ARM_BY_ID:
            raise DualSteerError(
                f"record {index} carries unknown arm {record.get('arm_id')!r}")
        arm = ARM_BY_ID[record["arm_id"]]
        if (_key_atom(record["alpha_R"]) != _key_atom(arm.alpha_repair)
                or _key_atom(record["alpha_S"]) != _key_atom(arm.alpha_safety)):
            raise DualSteerError(
                f"record {index} is arm {arm.arm_id} with coefficients "
                f"({record['alpha_R']}, {record['alpha_S']}); the predeclared "
                f"arm is ({arm.alpha_repair}, {arm.alpha_safety})")


def plan_work(inventory: Sequence[Mapping[str, Any]], done: set,
              bank_sha256: str, generation_sha256: str
              ) -> List[Dict[str, Any]]:
    """What is left, in inventory order, with what is already done removed."""
    todo = []
    for item in inventory:
        if work_key(item, bank_sha256, generation_sha256) in done:
            continue
        todo.append(dict(item))
    return todo
