"""Phase 2 is closed: what it decided, and what that decision retires.

Phase 2 ran. Its 144 generations were judged, joined to the 36 inherited
phase-1 reference cells, adjudicated under the frozen safety rubric, and read
by the decision rule that ``routegate.phase2`` froze before any phase-2 token
existed. The rule returned **NO_STRENGTH_QUALIFIES**: no ``alpha_S`` both
restored at least 9 of the 18 selected rows and caused none on its own.

That is the predeclared failure branch, and it is a result. This module is
where it is written down once, in a form nothing downstream can quietly
disagree with:

* ``d_S = H - E`` is **retired as an actuator**. Nothing steers along it.
* ``d_R = B_d1 - E`` is **retained**, at the strength phase 1 validated.
* the **H-versus-(E+B) probe is retained**, as a sensor -- a brake or a veto --
  and never as a steering direction. Dropping H - E as an ACTUATOR is not
  dropping the safety READING.
* the controller that continues emits **one** output. There is no ``a_S``.
* the phase-2 causal-screen protocol is **closed** against ordinary reruns.

What this module does not do
----------------------------
It generates nothing, judges nothing, trains nothing, loads no model and
re-derives no verdict. It reads one completed result file to check that it says
what this closure claims it says, and it holds the gates that refuse a tree
which has drifted from that claim.

Why the closure is a separate artifact from the predeclaration
--------------------------------------------------------------
``manifests/focused_direction_validation_v1_phase2.json`` is what phase 2 said
it would do, frozen before it ran. It is history and is never rewritten -- not
to record the outcome, not to mark the protocol closed, not for anything. A
predeclaration edited after its result is no longer a predeclaration, and no
later reader could tell which parts were written before the run and which
after. The outcome therefore lands in a second, separately versioned file that
declares itself post-run, names the readings it was taken from by digest, and
carries its own tamper digest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .dualsteer import (
    COMBINED_ALPHA_REPAIR,
    PHASE2_ANALYSIS_TABLE_RECORDS,
    PHASE2_NEW_GENERATIONS,
    PHASE2_PROTOCOL,
    PHASE2_REFERENCE_RECORDS,
    PHASE2_ROWS,
    PHASE2_SELECTED_ROWS_SHA256,
)
from .phase2 import (
    PHASE2_PREDECLARATION,
    PHASE2_RECOVERY_THRESHOLD,
    PHASE2_SAFETY_ONLY_HARM_LIMIT,
    PHASE2_SELECTION_MANIFEST,
)
from .validation import (
    Checklist,
    ValidationError,
    sha256_file,
    sha256_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "routegate/focused_direction_phase2_closure/1"

#: The closure's own version. A later revision of what phase 2 decided does not
#: edit this file: it is a new version beside it, and this one stays readable.
CLOSURE_VERSION = 1

CLOSURE_NAME = "focused_direction_validation_v1_phase2_closure"

#: The committed closure artifact.
PHASE2_CLOSURE_MANIFEST = (
    REPO_ROOT / "manifests"
    / "focused_direction_validation_v1_phase2_closure.json")

#: The field holding the artifact's digest of itself, excluded from that digest.
CLOSURE_DIGEST_FIELD = "closure_sha256"


class Phase2ClosureError(ValidationError):
    """The phase-2 closure does not hold, or a closed protocol was re-run."""


# ================================================================ the outcome
#
# One string, pinned. Everything below is downstream of it: if the readings say
# something else, this whole file is describing a phase 2 that did not happen,
# and the gates say so rather than letting the tree half-agree with itself.

PHASE2_OUTCOME = "NO_STRENGTH_QUALIFIES"

PHASE2_READINGS_SCHEMA = "routegate/focused_direction_phase2_readings/1"
PHASE2_READINGS_FILENAME = "phase2_readings.json"

#: SHA-256 of the completed phase-2 readings file this closure was taken from.
#: The file itself lives outside the repository, under the private data root,
#: because it is a result rather than a declaration; the digest is what binds
#: this closure to it, and a readings file that hashes to anything else is a
#: different result and is refused.
PHASE2_READINGS_SHA256 = (
    "b10b19ff78b1721e390ce4510627a335dcfbaf10f3fef843c618b5c41e02a011")

#: Where it was read from, recorded for provenance and never opened by import.
PHASE2_READINGS_PATH = (
    "focused_direction_validation_v1/phase2/judging/readings/"
    "phase2_readings.json")

#: The digests the readings themselves pin, restated here so that a readings
#: file which hashes correctly but whose inputs have moved is still caught.
PHASE2_ADJUDICATED_CELLS_SHA256 = (
    "4d807a02f132ef498c8ccb4518bdb040751295720361a85092aacf18d5df4cda")
PHASE1_ADJUDICATED_CELLS_SHA256 = (
    "4e6be82bfaf934fa0248d9a7c35ab4205d99f4ad37305fed553f1bbc040d1e73")


# ============================================================ what it retires
RETIRED_ACTUATOR: Dict[str, Any] = {
    "direction": "d_S = H - E",
    "was": "the candidate safety steering actuator phase 2 screened",
    "status": "RETIRED",
    "active": False,
    "may_be_applied": False,
    "may_carry_a_coefficient": False,
    "retired_by": PHASE2_OUTCOME,
    "reason": f"no alpha_S on the frozen grid both restored at least "
              f"{PHASE2_RECOVERY_THRESHOLD} of {PHASE2_ROWS} and caused "
              f"{PHASE2_SAFETY_ONLY_HARM_LIMIT} on its own. The veto is "
              f"absolute: a direction that turns a safe refusal harmful by "
              f"itself is not a safety direction, whatever it does in "
              f"combination.",
    "what_retirement_means": "no final controller emits a coefficient for it, "
                             "no final configuration declares it as an "
                             "actuator, and no steering plan applies it. It "
                             "remains in the frozen direction bank, which is "
                             "an immutable phase-1 artifact and is not edited "
                             "to remove it.",
    "the_bank_is_not_rewritten": True,
}

RETAINED_ACTUATOR: Dict[str, Any] = {
    "direction": "d_R = B_d1 - E",
    "status": "RETAINED",
    "active": True,
    "validated_strength": COMBINED_ALPHA_REPAIR,
    "validated_strength_standing": "the strength phase 1 measured H damage at "
                                   "and phase 2 pinned alpha_R to across all "
                                   "four combined arms. It is a ceiling for "
                                   "the continuation, not a new claim.",
    "continues_as": "the single steered direction of the one-direction "
                    "controller",
}

#: The safety reading survives the actuator's retirement, and is the whole
#: point of saying so out loud: a reader who sees "H - E removed" and concludes
#: "safety dropped" has read the opposite of the decision.
SAFETY_PROBE_DISPOSITION: Dict[str, Any] = {
    "probe": "H versus (E + B)",
    "status": "RETAINED",
    "role": "sensor",
    "may_reduce_the_repair_gate": True,
    "may_veto_the_repair_gate": True,
    "may_produce_a_steering_direction": False,
    "may_produce_a_controller_output": False,
    "statement": "the H-versus-(E+B) probe is still trained and still used, as "
                 "a brake or a veto on the repair gate rather than as a "
                 "steering direction. Dropping H - E as an ACTUATOR is not "
                 "dropping the safety READING.",
    "trained_by": "src/routegate/probeheads.py, unchanged",
}

CONTROLLER_OUTPUT_DISPOSITION: Dict[str, Any] = {
    "outputs": ["g_R"],
    "output_count": 1,
    "a_S_is_emitted": False,
    "a_S_statement": "no safety coefficient is emitted, by any head, at any "
                     "layer, at any strength. The superseded two-coefficient "
                     "proposal is phase-2 history and is not the final design.",
    "specified_in": "configs/single_direction_controller.json",
    "specified_by": "src/routegate/singledirection.py",
}


# ======================================================== the closed protocol
#
# Phase 2 generated its 144 records once. Re-running the protocol would not
# reproduce them -- it would produce a second set beside them, at the same
# arms, on the same rows, and any difference between the two copies would be
# unattributable. The ordinary path is therefore closed, and the reproduction
# path is a separate, explicitly named request that says out loud what it is.

#: The named flag a caller must pass to run a closed protocol at all. Spelled
#: out rather than inferred: an accidental rerun is a mistyped ``--protocol``,
#: and a mistyped flag is not a reproduction request.
REPRODUCTION_ONLY_FLAG = "--reproduce-closed-protocol"

CLOSED_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    PHASE2_PROTOCOL: {
        "protocol": PHASE2_PROTOCOL,
        "status": "CLOSED",
        "closed_by": CLOSURE_NAME,
        "generations_run": PHASE2_NEW_GENERATIONS,
        "reference_records": PHASE2_REFERENCE_RECORDS,
        "analysis_table_records": PHASE2_ANALYSIS_TABLE_RECORDS,
        "outcome": PHASE2_OUTCOME,
        "why_closed": "the protocol completed and was decided. An ordinary "
                      "rerun would write a second set of records beside the "
                      "first, at the same arms and on the same rows, and no "
                      "difference between the two copies would be "
                      "attributable to anything.",
        "reproduction_is_preserved": True,
        "reproduction_only_flag": REPRODUCTION_ONLY_FLAG,
        "reproduction_note": "nothing about the protocol, its runner, its "
                             "predeclaration or its selection is deleted. A "
                             "reproduction runs the same frozen inventory, "
                             "into an output directory of its own, and says "
                             "so on the command line.",
    },
}


def assert_protocol_is_open(name: str, reproduction_only: bool = False,
                            checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """Refuse an ordinary run of a protocol that has already been decided.

    ``reproduction_only`` is the explicit request, and it is never inferred
    from anything else: not from an empty output directory, not from a
    ``--resume``, and not from the protocol's own name.
    """
    closed = CLOSED_PROTOCOLS.get(name)
    if closed is None:
        return {"protocol": name, "status": "OPEN"}
    if not reproduction_only:
        raise Phase2ClosureError(
            f"protocol {name!r} is CLOSED: it ran, produced "
            f"{closed['generations_run']} record(s), and was decided "
            f"({closed['outcome']}). An ordinary rerun is refused, because it "
            f"would write a second set of records beside the first at the "
            f"same arms on the same rows. Historical reproducibility is not "
            f"deleted: pass {REPRODUCTION_ONLY_FLAG} to run it as a named "
            f"reproduction, into an output directory of its own.")
    if checks is not None:
        checks.record(
            "closed_protocol_reproduction", True,
            f"{name} is closed and was run as an explicit reproduction "
            f"({REPRODUCTION_ONLY_FLAG}); this is not the phase-2 result and "
            f"does not replace it")
    return {"protocol": name, "status": "CLOSED",
            "run_as": "reproduction_only"}


# ==================================================== the probe-head identity
#
# The two heads were fitted once, on 2026-08-21, and that run passed all 254 of
# its gates. It is retained exactly as it is: the retirement of d_S changes
# what the safety head is USED for, not what it is, so refitting it now would
# replace a gated artifact with an ungated one for no gain. Its identity is
# pinned here so the closure names a specific run rather than "the probe heads".

PROBE_HEAD_RUN_DIRECTORY = "outputs/dual_direction_controller/probe_heads_v1"

PROBE_HEAD_RUN: Dict[str, Any] = {
    "run": "probe_heads_v1",
    "directory": PROBE_HEAD_RUN_DIRECTORY,
    "schema": "routegate/dual_direction_probe_heads/1",
    "stage": "dual_direction_controller_probe_heads",
    "started_utc": "2026-08-21T21:38:17Z",
    "created_utc": "2026-08-21T21:38:27Z",
    "checks_run": 254,
    "checks_passed": 254,
    "all_passed": True,
    "status": "RETAINED_UNCHANGED",
    "refit_in_this_patch": False,
    "artifacts": {
        "run_manifest.json":
            "7b4b9d84b92a4a07c382dbff16a662cacc1087a281cd12c1a5500b0de6089ce2",
        "validation.json":
            "45eef137f572fb12fb6f3994b2a8776c96ae14ba56c0daef1637a0c3a9ad243e",
        "metrics.json":
            "9557be27b97a65532da6d76a8ed7f28e8bbd141517b60bedf7f51fe9b7e65744",
        "oof_predictions.csv":
            "6923919c99b17dcd840202e7eda63a83fc5e98bee26b660ea9002e9a5eafd278",
        "fold_assignments.csv":
            "3692fd99cca4521e88176c5fa3e09253b27e9ae80251d92f4d8af8b65739fc6f",
    },
    "not_committed": "the run directory is an output and is gitignored, per "
                     "the repository's output policy. The digests above are "
                     "the committed identity of it.",
    "why_it_is_not_refit_here": "the safety head's DEFINITION is unchanged by "
                                "phase 2. What changed is what its score is "
                                "allowed to do -- brake, never steer -- and "
                                "that is a controller decision, not a probe "
                                "one. Refitting a completed 254/254 run to "
                                "record a decision taken elsewhere would "
                                "trade a gated artifact for an ungated one.",
}


def verify_probe_head_run(directory: Path | str | None = None,
                          checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The pinned probe-head run is the one on disk, when it is reachable.

    The run directory is an output and is not committed, so its absence is a
    fact about this machine and not a failure. Its PRESENCE with a different
    digest is a failure, and is refused: a closure that names a run must name
    the run it names.
    """
    checks = checks or Checklist("phase2_closure")
    path = Path(directory) if directory is not None else (
        REPO_ROOT / PROBE_HEAD_RUN_DIRECTORY)
    if not path.is_dir():
        checks.record(
            "probe_head_run_reachable", True,
            f"{PROBE_HEAD_RUN_DIRECTORY} is not present on this machine; it is "
            f"an output and is not committed. The closure pins its digests.")
        return {"reachable": False, "directory": str(path)}

    observed: Dict[str, str] = {}
    for name, expected in sorted(PROBE_HEAD_RUN["artifacts"].items()):
        artifact = path / name
        checks.require(
            f"probe_head_artifact_present_{name}", artifact.is_file(),
            f"{artifact} is missing from the pinned probe-head run")
        actual = sha256_file(artifact)
        observed[name] = actual
        checks.require(
            f"probe_head_artifact_sha256_{name}", actual == expected,
            f"{name} hashes to {actual}, the closure pins {expected}. The "
            f"254/254 run named by this closure has changed since it was "
            f"pinned; it is refused, not re-pinned.")
    validation = json.loads(
        (path / "validation.json").read_text(encoding="utf-8"))
    checks.require(
        "probe_head_run_gates", validation.get("checks_run") == 254
        and validation.get("all_passed") is True,
        f"the pinned run records {validation.get('checks_run')} check(s), "
        f"all_passed={validation.get('all_passed')}; the closure names a "
        f"254/254 run")
    return {"reachable": True, "directory": str(path), "artifacts": observed}


# ======================================================= reading the readings
def verify_phase2_readings(path: Path | str,
                           checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The completed readings are the ones this closure was taken from.

    Digest first, then content. The digest alone would accept a file that is
    byte-identical to something this closure never saw; the content alone would
    accept a file that says ``NO_STRENGTH_QUALIFIES`` and is not the result.
    Both, and the summary returned carries no prompt or response text.
    """
    checks = checks or Checklist("phase2_closure")
    path = Path(path)
    if not path.is_file():
        raise Phase2ClosureError(
            f"{path} does not exist. The phase-2 closure is taken from the "
            f"completed readings and from nothing else; an outcome typed by "
            f"hand is not one.")

    digest = sha256_file(path)
    checks.require(
        "readings_sha256", digest == PHASE2_READINGS_SHA256,
        f"{path.name} hashes to {digest}, this closure is pinned to "
        f"{PHASE2_READINGS_SHA256}. A different readings file is a different "
        f"result; it is refused, not adopted.")

    readings = json.loads(path.read_text(encoding="utf-8"))
    checks.require(
        "readings_schema", readings.get("schema") == PHASE2_READINGS_SCHEMA,
        f"{path.name} declares schema {readings.get('schema')!r}, expected "
        f"{PHASE2_READINGS_SCHEMA!r}")
    checks.require(
        "readings_phase", readings.get("phase") == 2,
        f"{path.name} declares phase {readings.get('phase')!r}, expected 2")
    checks.require(
        "readings_protocol", readings.get("protocol") == PHASE2_PROTOCOL,
        f"{path.name} declares protocol {readings.get('protocol')!r}, expected "
        f"{PHASE2_PROTOCOL!r}")

    decision = readings.get("d_decision") or {}
    checks.require(
        "readings_outcome", decision.get("outcome") == PHASE2_OUTCOME,
        f"{path.name} records outcome {decision.get('outcome')!r}; this "
        f"closure exists only for {PHASE2_OUTCOME!r}. Nothing here is edited "
        f"to fit another outcome -- a different outcome is a different "
        f"closure.")
    checks.require(
        "readings_select_no_strength", decision.get("selected") is None
        and list(decision.get("qualifying_strengths") or []) == [],
        f"{path.name} selects {decision.get('selected')!r} with qualifying "
        f"strengths {decision.get('qualifying_strengths')!r}; "
        f"{PHASE2_OUTCOME} selects nothing")
    checks.require(
        "readings_thresholds_are_the_frozen_ones",
        decision.get("recovery_threshold") == PHASE2_RECOVERY_THRESHOLD
        and decision.get("safety_only_harm_limit")
        == PHASE2_SAFETY_ONLY_HARM_LIMIT,
        f"{path.name} decided against recovery threshold "
        f"{decision.get('recovery_threshold')!r} and harm limit "
        f"{decision.get('safety_only_harm_limit')!r}; the frozen rule is "
        f"{PHASE2_RECOVERY_THRESHOLD} and {PHASE2_SAFETY_ONLY_HARM_LIMIT}")
    checks.require(
        "readings_denominator", decision.get("denominator") == PHASE2_ROWS,
        f"{path.name} decided over {decision.get('denominator')!r} row(s), "
        f"expected {PHASE2_ROWS}")
    checks.require(
        "readings_are_not_population_reportable",
        decision.get("population_reportable") is False
        and readings.get("withheld", {}).get("population_reportable") is False,
        f"{path.name} must state population_reportable: false; the 18 rows "
        f"were selected because they regressed")
    checks.require(
        "readings_complete_adjudication",
        readings.get("complete_adjudication") is True
        and readings.get("unresolved_cells") == 0,
        f"{path.name} records {readings.get('unresolved_cells')!r} unresolved "
        f"cell(s); {PHASE2_OUTCOME} is a decided outcome and an undecided row "
        f"would have made it UNDETERMINED instead")
    checks.require(
        "readings_analysis_table",
        readings.get("analysis_table_records")
        == PHASE2_ANALYSIS_TABLE_RECORDS
        and readings.get("cells_judged_in_phase2") == PHASE2_NEW_GENERATIONS
        and readings.get("cells_inherited_from_phase1")
        == PHASE2_REFERENCE_RECORDS,
        f"{path.name} reports {readings.get('cells_judged_in_phase2')!r} "
        f"judged + {readings.get('cells_inherited_from_phase1')!r} inherited = "
        f"{readings.get('analysis_table_records')!r}; the frozen counts are "
        f"{PHASE2_NEW_GENERATIONS} + {PHASE2_REFERENCE_RECORDS} = "
        f"{PHASE2_ANALYSIS_TABLE_RECORDS}")
    checks.require(
        "readings_selection_digest",
        readings.get("selection", {}).get("selected_rows_sha256")
        == PHASE2_SELECTED_ROWS_SHA256,
        f"{path.name} was read over selection "
        f"{readings.get('selection', {}).get('selected_rows_sha256')!r}, the "
        f"package pins {PHASE2_SELECTED_ROWS_SHA256}")
    checks.require(
        "readings_adjudicated_cells_digest",
        readings.get("inputs", {}).get("phase2_adjudicated_cells_sha256")
        == PHASE2_ADJUDICATED_CELLS_SHA256,
        f"{path.name} was read from adjudicated cells "
        f"{readings.get('inputs', {}).get('phase2_adjudicated_cells_sha256')!r}"
        f", this closure pins {PHASE2_ADJUDICATED_CELLS_SHA256}")

    return {
        "path": path.name,
        "sha256": digest,
        "schema": readings.get("schema"),
        "outcome": decision.get("outcome"),
        "qualifying_strengths": list(decision.get("qualifying_strengths") or []),
        "selected": decision.get("selected"),
        "rows": readings.get("rows"),
        "analysis_table_records": readings.get("analysis_table_records"),
        "unresolved_cells": readings.get("unresolved_cells"),
        "checks": checks,
    }


# ============================================================ the closure itself
def closure_canonical_json(payload: Mapping[str, Any]) -> str:
    """The closure as hashed: sorted keys, no whitespace, digest field removed."""
    body = {key: value for key, value in payload.items()
            if key != CLOSURE_DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def closure_sha256(payload: Mapping[str, Any]) -> str:
    """The closure's digest of itself."""
    return sha256_text(closure_canonical_json(payload))


def build_phase2_closure() -> Dict[str, Any]:
    """The committed closure, derived from the package rather than retyped.

    Carries no timestamp on purpose: every field is derivable, so re-deriving
    it on any machine at any time produces the same bytes, and "the committed
    file is what the package declares" is a byte comparison rather than a
    judgement call.
    """
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "name": CLOSURE_NAME,
        "version": CLOSURE_VERSION,
        "status": "CLOSED",
        "phase": 2,
        "immutable": True,
        "written_after_the_run": True,
        "is_not_the_predeclaration": {
            "statement": "this file records what phase 2 DECIDED. "
                         f"{PHASE2_PREDECLARATION.name} records what phase 2 "
                         f"said it WOULD DO, frozen before it ran. The "
                         f"predeclaration is history: it is not edited to "
                         f"carry an outcome, not edited to mark the protocol "
                         f"closed, and not edited at all.",
            "predeclaration": PHASE2_PREDECLARATION.name,
            "selection": PHASE2_SELECTION_MANIFEST.name,
            "both_preserved_unchanged": True,
        },
        "outcome": {
            "outcome": PHASE2_OUTCOME,
            "decision_rule": "phase2_safety_direction_qualification",
            "decision_rule_lives_in": "routegate.phase2."
                                      "evaluate_phase2_decision",
            "qualifying_strengths": [],
            "selected": None,
            "recovery_threshold": PHASE2_RECOVERY_THRESHOLD,
            "safety_only_harm_limit": PHASE2_SAFETY_ONLY_HARM_LIMIT,
            "denominator": PHASE2_ROWS,
            "is_a_conditional_development_set_decision": True,
            "is_a_population_estimate": False,
            "population_reportable": False,
            "statement": f"no alpha_S on the frozen grid both restored at "
                         f"least {PHASE2_RECOVERY_THRESHOLD} of "
                         f"{PHASE2_ROWS} and caused "
                         f"{PHASE2_SAFETY_ONLY_HARM_LIMIT} on its own. This is "
                         f"the predeclared failure branch and it is a result, "
                         f"not a failure of the experiment.",
        },
        "readings": {
            "file": PHASE2_READINGS_FILENAME,
            "path_under_the_data_root": PHASE2_READINGS_PATH,
            "sha256": PHASE2_READINGS_SHA256,
            "schema": PHASE2_READINGS_SCHEMA,
            "phase2_adjudicated_cells_sha256": PHASE2_ADJUDICATED_CELLS_SHA256,
            "phase1_adjudicated_cells_sha256": PHASE1_ADJUDICATED_CELLS_SHA256,
            "selected_rows_sha256": PHASE2_SELECTED_ROWS_SHA256,
            "not_committed": "the readings are a result under the private data "
                             "root, not a declaration. The digest is what "
                             "binds this closure to them.",
        },
        "counts": {
            "rows": PHASE2_ROWS,
            "generated": PHASE2_NEW_GENERATIONS,
            "inherited": PHASE2_REFERENCE_RECORDS,
            "analysis_table_records": PHASE2_ANALYSIS_TABLE_RECORDS,
            "unresolved_cells": 0,
        },
        "actuators": {
            "retired_actuator": dict(RETIRED_ACTUATOR),
            "retained_actuator": dict(RETAINED_ACTUATOR),
        },
        "safety_probe": dict(SAFETY_PROBE_DISPOSITION),
        "controller_outputs": dict(CONTROLLER_OUTPUT_DISPOSITION),
        "protocol_closed": dict(CLOSED_PROTOCOLS[PHASE2_PROTOCOL]),
        "probe_head_run": dict(PROBE_HEAD_RUN),
        "continuation": {
            "selected": "single_direction_repair_controller_v1",
            "statement": "the one-direction controller continues, on the "
                         "validated repair direction d_R = B_d1 - E alone, "
                         "with the H-versus-(E+B) probe as a brake or veto.",
            "specification": "configs/single_direction_controller.json",
            "implemented_by": "src/routegate/singledirection.py",
            "trainer_implemented": False,
            "trainer_note": "the controller TRAINER is not implemented by this "
                            "closure. What is fixed here is the architecture "
                            "and the protocol it will be trained under.",
        },
        "contains_prompt_or_response_text": False,
    }
    payload[CLOSURE_DIGEST_FIELD] = closure_sha256(payload)
    return payload


#: The fields compared byte for byte between the committed file and the
#: package. A difference in any of them is a closure that has moved since it
#: was taken, and neither copy is preferred over the other.
CLOSURE_COMPARED_FIELDS = (
    "schema", "name", "version", "status", "phase", "immutable",
    "is_not_the_predeclaration", "outcome", "readings", "counts", "actuators",
    "safety_probe", "controller_outputs", "protocol_closed", "probe_head_run",
    "continuation",
)


def verify_phase2_closure(path: Path | str = PHASE2_CLOSURE_MANIFEST,
                          checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The committed closure is what this package declares, and is untampered.

    Two separate questions, and both are asked. Its self-digest recomputes --
    so an edit anywhere in the file is visible without a second copy to compare
    against -- and every field that decides something matches the package.
    """
    checks = checks or Checklist("phase2_closure")
    path = Path(path)
    if not path.is_file():
        raise Phase2ClosureError(
            f"{path} does not exist; it is the post-run record of what phase 2 "
            f"decided. Write it with "
            f"experiments/dual_direction_controller/close_phase2.py")
    committed = json.loads(path.read_text(encoding="utf-8"))

    recorded = committed.get(CLOSURE_DIGEST_FIELD)
    recomputed = closure_sha256(committed)
    checks.require(
        "closure_digest_recomputes", recorded == recomputed,
        f"{path.name} records {CLOSURE_DIGEST_FIELD} {recorded}, its own "
        f"contents hash to {recomputed}. The closure has been edited since it "
        f"was written; an immutable artifact is not repaired in place.")

    derived = build_phase2_closure()
    for field in CLOSURE_COMPARED_FIELDS:
        checks.require(
            f"closure_{field}", committed.get(field) == derived[field],
            f"{path.name}: '{field}' differs from routegate.phase2closure. A "
            f"closure that has moved since it was taken is refused; neither "
            f"copy is preferred over the other.")
    checks.require(
        "closure_digest_matches_the_package",
        recorded == derived[CLOSURE_DIGEST_FIELD],
        f"{path.name} hashes to {recorded}, the package derives "
        f"{derived[CLOSURE_DIGEST_FIELD]}")
    checks.require(
        "closure_holds_no_prompt_or_response_text",
        committed.get("contains_prompt_or_response_text") is False,
        f"{path.name} must state contains_prompt_or_response_text: false")
    return {"committed": committed, "checks": checks}


def assert_historical_artifacts_are_preserved(
        checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The phase-2 predeclaration and the dual-direction config still exist.

    Closing a phase does not delete what it declared. A tree where the closure
    is present and the predeclaration is gone has lost the only record of what
    was decided in advance, and the closure alone cannot show it.
    """
    checks = checks or Checklist("phase2_closure")
    preserved = {
        "phase2_predeclaration": PHASE2_PREDECLARATION,
        "phase2_selection": PHASE2_SELECTION_MANIFEST,
        "dual_direction_config":
            REPO_ROOT / "configs" / "dual_direction_controller.json",
    }
    out: Dict[str, Any] = {}
    for name, target in preserved.items():
        checks.require(
            f"preserved_{name}", target.is_file(),
            f"{target} is missing. Phase 2 is closed, not erased: its "
            f"predeclaration, its selection and the dual-direction "
            f"configuration are historical artifacts and are kept as they "
            f"are.")
        out[name] = {"path": str(target.relative_to(REPO_ROOT)),
                     "sha256": sha256_file(target)}
    return out


def closure_summary_lines(payload: Optional[Mapping[str, Any]] = None
                          ) -> Sequence[str]:
    """The closure in seven lines, for a runner's stdout and a log."""
    payload = payload or build_phase2_closure()
    return (
        f"outcome                  {payload['outcome']['outcome']}",
        f"readings sha256          {payload['readings']['sha256']}",
        f"d_S = H - E              {payload['actuators']['retired_actuator']['status']}"
        f" (active={payload['actuators']['retired_actuator']['active']})",
        f"d_R = B_d1 - E           {payload['actuators']['retained_actuator']['status']}",
        f"H vs (E+B) probe         {payload['safety_probe']['status']}, "
        f"role={payload['safety_probe']['role']}",
        f"controller outputs       {payload['controller_outputs']['outputs']}",
        f"{PHASE2_PROTOCOL}  {payload['protocol_closed']['status']}",
    )
