"""Real out-of-fold generation for the one-direction dynamic controller.

WHAT THIS ARM IS, AND WHY IT EXISTS EITHER WAY
----------------------------------------------
``dynamic_raw`` is the one-direction controller applied for real: one
prompt-specific coefficient per row, computed by the fold controller that never
trained on that row, applied at historical blocks 31 and 32.

It is required whatever the rotational H - E screen concludes. If the rotation
fails, this is the final method. If the rotation succeeds, this is the
one-direction ablation the rotational result is measured against. Nothing about
this arm is conditional on that screen, and nothing here reads it.

THE INFERENCE MECHANISM, STATED BEFORE ANYTHING ELSE
----------------------------------------------------
``gate_oof_v1`` left a column of 480 coefficients keyed by row id. Reading that
column at generation time would be a **lookup**, not inference: it cannot score
a prompt the table has never seen, and every development row's answer would be
inside the arm that used it. This module never opens that column.

Instead, in the same forward pass that generates the response:

1. the unsteered state-32 final-prompt-token activation is captured, **before**
   either site is written;
2. ``livecontroller.plan_live_edit`` computes ``g_R`` once, from that vector,
   through the fold checkpoint that never saw the row;
3. ``0.15 * g_R * ||h^(l)|| * d_R,l`` is applied at historical blocks 31 and 32
   -- the same ``g_R`` at both sites.

Historical block 33 is not a site. An edit at 31 changes the activation
entering 33, so 33's unsteered budget does not exist in the same pass; the
one-pass budget result measured 7.455261e-02 against a 1e-2 gate and dropped it.

THE FROZEN COEFFICIENT TABLE IS STILL USED -- AS A WITNESS
-----------------------------------------------------------
The frozen extraction holds every row's state-32 activation. Scoring those
vectors through the same out-of-fold checkpoints gives a **reference**
coefficient per row, offline, on CPU, with no model loaded. That reference is
not what is applied. It is what the applied coefficient is checked against:

* before any model is loaded, the five per-fold held-out mean coefficients are
  re-derived from the frozen activations and compared with the means
  ``gate_oof_v1`` itself recorded. A bundle, a fold assignment or an activation
  file that is not the one the reproduction ran against fails here, on CPU, in
  seconds, rather than eight GPU-hours later;
* during generation, every row's live coefficient is compared with its own
  reference and the run **stops** if the two differ by more than
  ``reproequivalence.GATE_ABS_TOLERANCE`` -- the same 32-float32-epsilon budget
  the reproduction itself was gated at, because it is the same quantity in the
  same dtype. A checkpoint that cannot score the corpus faithfully does not
  produce an arm; it produces a refusal.

Both captures are float32 widenings of the same bfloat16 forward at the same
site (``hooks.clone_cpu`` stores them exactly as the historical extraction did),
so bit-identity is the expected outcome and is recorded per row rather than
assumed.

WHAT IS FROZEN
--------------
The language model, the tokenizer, the chat template, the generation
configuration, both probes, the direction bank and every controller weight.
``alpha_R`` is 0.15 and is not a parameter of this stage. ``alpha_S`` is 0.0
permanently and ``d_S = H - E`` is never applied. Nothing here trains, judges,
or reads the sealed set, and no gradient is taken anywhere on this path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .directionbank import BASELINE_ARM, DIRECTIONS
from .gatecontroller import ALPHA_R, ALPHA_S, N_SPLITS, READ_STATE
from .livecontroller import (
    APPLY_BLOCKS,
    ControllerBundle,
    gate_from_activation,
)
from .reproequivalence import (
    FLOAT32_EPS,
    GATE_ABS_TOLERANCE,
    GATE_EPSILON_BUDGET,
    RULE_NAME,
)
from .twosite import CANDIDATE_BLOCKS, CANDIDATE_BUDGET_STATES, CANDIDATE_STATES
from .twositecontinuation import LOCKED_FIELDS as TWO_SITE_LOCKED_FIELDS
from .twositecontinuation import read_jsonl_strict
from .validation import (
    Checklist,
    ValidationError,
    assert_not_sealed,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "routegate/dynamic_controller_arm/1"
PROTOCOL = "dynamic_controller_arm_v1"
ARM_VERSION = 1

ARM_RAW = "dynamic_raw"
ARM_VETO = "dynamic_hard_veto"
ARMS: Tuple[str, ...] = (ARM_RAW, ARM_VETO)

#: The reference arm a vetoed row reuses. A vetoed row applies exactly 0.0, so
#: its response is the unsteered one.
REFERENCE_ARM = "baseline"

EXPECTED_ROWS = 480

OUTPUT_MANIFEST = "run_manifest.json"
OUTPUT_VALIDATION = "validation.json"
COEFFICIENTS_FILENAME = "applied_coefficients.csv"

#: What the controller bundle must carry beside its six checkpoints, and the
#: schemas those two files must declare.
SUMMARY_FILENAME = "reproduction_summary.json"
DIAGNOSTIC_FILENAME = "reproduction_diagnostic.json"
REPRODUCTION_SCHEMA = "routegate/gate_controller_reproduction/2"
EQUIVALENCE_SCHEMA = "routegate/gate_controller_equivalence/1"

#: The clause of the equivalence rule that says these very files score the
#: corpus the way the run that wrote them did. Without it the bundle is a set
#: of weights with no evidence attached.
CHECKPOINT_CLAUSE = "the_written_checkpoints_score_the_corpus_the_same_way"

#: The frozen 480-row division, read from the artefact the directions and the
#: probes were built against. It is not recomputed from a rule here: a fresh
#: split would hold out different rows and the arm would stop being
#: out-of-fold.
FOLD_SOURCE_DIR = ("outputs/causal_repair/orthogonalized_population_scan/"
                   "direction_bank")

#: ``d_R``. Stated as the bank's own manifest states it, so a reversed
#: subtraction -- a well-formed unit vector pointing the wrong way -- is
#: refused rather than applied.
REPAIR_SUBTRACTION_ORDER = f"{DIRECTIONS['repair']['arm']} - {BASELINE_ARM}"

#: A row's applied coefficient is its own, so ``alpha_R`` is not a run-locked
#: field here. Everything else the two-site arm locks is locked identically,
#: and the list is derived from that one rather than restated beside it.
RUN_LOCKED_FIELDS: Tuple[str, ...] = tuple(
    field for field in TWO_SITE_LOCKED_FIELDS if field != "alpha_R")

#: How closely the live coefficient must reproduce the frozen-activation one.
#: Not a new number: ``g_R`` is float32 in [0, 1] here exactly as it was in the
#: reproduction, so it carries the reproduction's own budget.
GATE_TOLERANCE = GATE_ABS_TOLERANCE
GATE_TOLERANCE_EXPRESSION = f"{GATE_EPSILON_BUDGET} * float32_eps"

#: The per-fold held-out mean is an average of ~96 float32 gates accumulated in
#: float64. Re-deriving it cannot move it by more than the gate budget itself.
MEAN_GATE_TOLERANCE = GATE_ABS_TOLERANCE


class DynamicArmError(ValidationError):
    """The dynamic arm was asked for something it must not do."""


# ============================================================== the checkpoints
def load_reproduction(directory: Path | str) -> Tuple[Dict[str, Any],
                                                      Dict[str, Any]]:
    """The bundle's summary and diagnostic, or a refusal.

    A directory of checkpoints with no reproduction record beside them is a set
    of weights whose provenance nobody can check. It is not loaded.
    """
    directory = Path(directory)
    out: List[Dict[str, Any]] = []
    for name, schema in ((SUMMARY_FILENAME, REPRODUCTION_SCHEMA),
                         (DIAGNOSTIC_FILENAME, EQUIVALENCE_SCHEMA)):
        path = directory / name
        if not path.is_file():
            raise DynamicArmError(
                f"{path} is absent. This arm applies weights only where the "
                f"run that produced them recorded what it reproduced and to "
                f"what tolerance.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != schema:
            raise DynamicArmError(
                f"{name} declares schema {payload.get('schema')!r}, not "
                f"{schema!r}")
        out.append(payload)
    return out[0], out[1]


def verify_bundle(directory: Path | str, bundle: ControllerBundle,
                  checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """Every checkpoint hash, and the equivalence the bundle claims for itself.

    ``livecontroller.load_bundle`` already refuses a checkpoint whose weights do
    not hash to what it records. That proves internal consistency and nothing
    about identity: a different but self-consistent bundle would pass it. This
    binds the six checkpoints to the weight digests the reproduction wrote
    down, and binds the run to the clauses that reproduction passed --
    including the one that says these very files score the corpus the way the
    training run did.

    The summary records each checkpoint's ``weights_sha256``, not its file
    digest, so that is what is compared. The chain is complete either way:
    ``load_checkpoint`` recomputes the weight digest from the state dict it
    actually loaded, so a file edited anywhere that matters fails there, and a
    file whose weights are someone else's fails here. The on-disk digest is
    recorded beside it as a fingerprint of the artefact, not as a gate --
    nothing wrote it down to be checked against.
    """
    checks = checks or Checklist(PROTOCOL)
    directory = Path(directory)
    summary, diagnostic = load_reproduction(directory)

    recorded = summary.get("checkpoints") or {}
    if not recorded:
        raise DynamicArmError(
            f"{SUMMARY_FILENAME} records no checkpoint digests")
    loaded: Dict[str, Mapping[str, Any]] = {
        f"controller_fold_{k}.json": checkpoint
        for k, checkpoint in bundle.folds.items()}
    if bundle.refit is not None:
        loaded["controller_all_development_refit.json"] = bundle.refit
    digests: Dict[str, Dict[str, str]] = {}
    for name in sorted(recorded):
        path = directory / name
        if not path.is_file():
            raise DynamicArmError(
                f"{SUMMARY_FILENAME} records {name} and the bundle does not "
                f"carry it")
        if name not in loaded:
            raise DynamicArmError(
                f"{SUMMARY_FILENAME} records {name} and the loaded bundle does "
                f"not hold it; a checkpoint the reproduction wrote is not "
                f"quietly dropped from the bundle that applies it")
        weights = str(loaded[name]["weights_sha256"])
        digests[name] = {"weights_sha256": weights,
                         "file_sha256": sha256_file(path)}
        checks.require(f"checkpoint_{name}", weights == recorded[name],
                       f"{name} weights hash {weights[:12]}, recorded "
                       f"{recorded[name][:12]}")
    missing = sorted(set(loaded) - set(recorded))
    checks.require(
        "no_unrecorded_checkpoint_in_the_bundle", not missing,
        f"{len(missing)} checkpoint(s) the reproduction did not record: "
        f"{missing}" if missing
        else f"{len(loaded)} checkpoint(s), every one recorded")

    equivalence = summary.get("equivalence") or {}
    checks.require("reproduction_passed", equivalence.get("passed") is True,
                   f"{RULE_NAME} v{equivalence.get('rule_version')}")
    checks.require("reproduction_rows",
                   int(summary.get("rows_compared", 0)) == EXPECTED_ROWS,
                   f"{summary.get('rows_compared')} row(s) compared, expected "
                   f"{EXPECTED_ROWS}")
    tolerance = (equivalence.get("tolerance") or {}).get("gate_abs_tolerance")
    checks.require("reproduction_tolerance_is_this_package_s",
                   tolerance == GATE_TOLERANCE,
                   f"{tolerance} == {GATE_TOLERANCE} "
                   f"({GATE_TOLERANCE_EXPRESSION})")

    failed = [c["clause"] for c in (equivalence.get("clauses") or [])
              if c.get("status") != "PASS"]
    checks.require("every_equivalence_clause_passed", not failed,
                   f"{len(equivalence.get('clauses') or [])} clause(s), "
                   f"{len(failed)} failing")
    clauses = {c["clause"] for c in (equivalence.get("clauses") or [])}
    checks.require("checkpoints_score_the_corpus_the_same_way",
                   CHECKPOINT_CLAUSE in clauses,
                   f"{CHECKPOINT_CLAUSE} is among the clauses this bundle "
                   f"passed; without it the weights carry no evidence that "
                   f"they reproduce the run that trained them")

    agreement = diagnostic.get("checkpoint_agreement") or {}
    checks.require("checkpoints_scored_out_of_fold_only",
                   agreement.get("scored_out_of_fold_only") is True,
                   "every reproduced coefficient came from the controller that "
                   "never saw its row")
    checks.require("checkpoint_rows_scored",
                   int(agreement.get("rows_scored", 0)) == EXPECTED_ROWS,
                   f"{agreement.get('rows_scored')} row(s) scored through the "
                   f"written checkpoints")
    error = float(agreement.get("max_abs_error_against_saved", float("inf")))
    checks.require("checkpoint_agreement_with_the_source_run",
                   error <= GATE_TOLERANCE,
                   f"max |checkpoint g_R - gate_oof_v1 g_R| = {error:.6e}, "
                   f"budget {GATE_TOLERANCE:.6e}")
    checks.require("no_per_row_coefficients_in_the_bundle",
                   summary.get("contains_per_row_coefficients") is False
                   and all(c.get("contains_per_row_coefficients") is False
                           for c in bundle.folds.values()),
                   "a checkpoint carries weights and calibration; the "
                   "coefficient column is not shipped as a controller and is "
                   "not the inference mechanism here")

    return {
        "directory": str(directory),
        "checkpoint_sha256": digests,
        "reproduces": summary.get("reproduces"),
        "rows_compared": int(summary.get("rows_compared", 0)),
        "equivalence_rule": RULE_NAME,
        "equivalence_rule_version": equivalence.get("rule_version"),
        "equivalence_passed": True,
        "gate_abs_tolerance": GATE_TOLERANCE,
        "max_abs_error_against_saved": error,
        "per_fold": summary.get("per_fold") or [],
        "refit_present": bundle.refit is not None,
    }


# ================================================================== the folds
def fold_map(row_ids: Sequence[str], directory: Path | str | None = None,
             checks: Optional[Checklist] = None
             ) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """``row_id -> outer fold``, read from the artefact that froze it.

    The same call the trainer and the reproduction made. Not recomputed: a
    rule that rebuilt the division would be free to hold out different rows,
    and every coefficient in this arm would stop being out-of-fold.
    """
    from .probeheads import load_bank_fold_assignment

    checks = checks or Checklist(PROTOCOL)
    directory = Path(directory) if directory is not None \
        else REPO_ROOT / FOLD_SOURCE_DIR
    assignment, source = load_bank_fold_assignment(directory)

    checks.require("fold_splits", assignment.n_splits == N_SPLITS,
                   f"{assignment.n_splits} fold(s), expected {N_SPLITS}")
    missing = [rid for rid in row_ids if rid not in assignment.fold_of_row]
    checks.require("every_row_has_a_fold", not missing,
                   f"{len(row_ids)} row(s) carry a frozen fold"
                   if not missing else f"{len(missing)} row(s) do not")
    extra = sorted(set(assignment.fold_of_row) - set(row_ids))
    checks.require(
        "the_fold_file_names_exactly_the_corpus", not extra,
        f"{len(assignment.fold_of_row)} row(s) in the assignment and "
        f"{len(row_ids)} in the corpus"
        if extra else f"{len(row_ids)} row(s), one fold each")

    of = {rid: int(assignment.fold_of(rid)) for rid in row_ids}
    sizes = {k: sum(1 for f in of.values() if f == k) for k in range(N_SPLITS)}
    checks.require("every_fold_holds_out_rows",
                   all(size > 0 for size in sizes.values()),
                   ", ".join(f"fold {k}: {sizes[k]}" for k in sorted(sizes)))
    checks.record("fold_assignment_sha256", True,
                  source.fold_assignment_sha256[:16])
    return of, {"directory": str(source.directory),
                "fold_assignment_sha256": source.fold_assignment_sha256,
                "manifest_sha256": source.manifest_sha256,
                "n_splits": int(source.n_splits),
                "held_out_rows_per_fold": sizes}


# =================================================== the reference coefficients
def reference_gates(activations: Mapping[str, np.ndarray],
                    fold_of: Mapping[str, int],
                    bundle: ControllerBundle) -> Dict[str, float]:
    """``g_R`` for every row, from the frozen state-32 activation.

    Pure and offline: reads vectors, returns numbers, loads no model, writes
    nothing. Each row is scored by ``controller_for_row``, which raises rather
    than guess, so an in-fold score cannot enter this table by accident.

    This is the **witness**, not the mechanism. What the arm applies is
    computed from the activation the generating pass itself captured.
    """
    out: Dict[str, float] = {}
    by_fold: Dict[int, List[str]] = {}
    for rid in sorted(activations):
        by_fold.setdefault(int(fold_of[rid]), []).append(rid)
    for fold in sorted(by_fold):
        checkpoint = bundle.controller_for_row(fold)
        rows = by_fold[fold]
        design = np.asarray([np.asarray(activations[rid], dtype=np.float64)
                             for rid in rows], dtype=np.float64)
        gates = gate_from_activation(design, checkpoint)
        for rid, g in zip(rows, gates):
            out[rid] = float(g)
    return out


def assert_reference_reproduces_the_run(reference: Mapping[str, float],
                                        fold_of: Mapping[str, int],
                                        summary: Mapping[str, Any],
                                        checks: Optional[Checklist] = None
                                        ) -> Dict[str, Any]:
    """The five held-out means, re-derived, against the ones v1 recorded.

    Five numbers, each an average of ninety-odd coefficients. A wrong bundle, a
    permuted fold assignment, a different activation file or a corrupted
    standardisation all move at least one of them far beyond a float32 budget,
    and all of them are caught here -- on CPU, before a model is loaded.
    """
    checks = checks or Checklist(PROTOCOL)
    recorded = {int(entry["fold"]): entry for entry in (summary.get("per_fold")
                                                        or [])}
    checks.require("recorded_folds", sorted(recorded) == list(range(N_SPLITS)),
                   f"the summary records folds {sorted(recorded)}")
    per_fold: List[Dict[str, Any]] = []
    worst = 0.0
    for fold in range(N_SPLITS):
        rows = [rid for rid in sorted(reference) if int(fold_of[rid]) == fold]
        derived = float(np.mean([reference[rid] for rid in rows]))
        expected = float(recorded[fold]["held_out_mean_g"])
        delta = abs(derived - expected)
        worst = max(worst, delta)
        checks.require(
            f"fold_{fold}_held_out_rows",
            len(rows) == int(recorded[fold]["n_held_out"]),
            f"{len(rows)} held-out row(s), the run recorded "
            f"{recorded[fold]['n_held_out']}")
        checks.require(
            f"fold_{fold}_held_out_mean_coefficient",
            delta <= MEAN_GATE_TOLERANCE,
            f"re-derived {derived:.12f}, recorded {expected:.12f}, "
            f"|delta| {delta:.3e} <= {MEAN_GATE_TOLERANCE:.3e}")
        per_fold.append({"fold": fold, "n_held_out": len(rows),
                         "rederived_held_out_mean_g": derived,
                         "recorded_held_out_mean_g": expected,
                         "abs_difference": delta})
    checks.record("coefficient_reproduction", True,
                  f"five held-out means re-derived from the frozen state-"
                  f"{READ_STATE} activations through the written checkpoints; "
                  f"worst |delta| {worst:.3e}")
    return {"per_fold": per_fold, "max_abs_difference": worst,
            "tolerance": MEAN_GATE_TOLERANCE,
            "tolerance_expression": GATE_TOLERANCE_EXPRESSION,
            "scored_out_of_fold_only": True}


def assert_gate_is_faithful(row_id: str, live: float, reference: float,
                            tolerance: float = GATE_TOLERANCE) -> float:
    """The live coefficient against the frozen-activation one. Returns |delta|.

    Raises rather than warns. A checkpoint that scores this prompt's own
    activation differently from the activation the reproduction was gated on is
    not scoring it faithfully, and an arm built on it would not be the
    controller the evidence describes.
    """
    delta = abs(float(live) - float(reference))
    if not delta <= float(tolerance):
        raise DynamicArmError(
            f"{row_id}: the controller scored the live state-{READ_STATE} "
            f"capture {live!r} and the frozen extraction {reference!r}, a "
            f"difference of {delta:.6e} against a budget of "
            f"{float(tolerance):.6e} ({GATE_TOLERANCE_EXPRESSION}). The "
            f"deployable checkpoint cannot score this activation faithfully; "
            f"the run stops rather than record a coefficient nothing "
            f"reproduces.")
    return delta


# ======================================================= the direction and bank
def assert_repair_direction_is_b_d1_minus_e(bank: Any,
                                            checks: Optional[Checklist] = None
                                            ) -> Dict[str, Any]:
    """The bank's own manifest says the repair array is ``B_d1 - E``.

    Read from the artefact rather than asserted beside it. A reversed
    subtraction produces a perfectly well-formed unit vector pointing the wrong
    way: every norm, every budget and every diagnostic would look healthy while
    the edit moved each prompt further from the behaviour it is meant to repair.
    """
    checks = checks or Checklist(PROTOCOL)
    construction = bank.manifest.get("construction") or {}
    order = (construction.get("subtraction_order") or {}).get("repair")
    if order != REPAIR_SUBTRACTION_ORDER:
        raise DynamicArmError(
            f"the direction bank's manifest records the repair subtraction "
            f"order as {order!r}; this arm applies d_R = "
            f"{REPAIR_SUBTRACTION_ORDER} and refuses to run against a "
            f"direction whose own record does not say that is what it is.")
    orthogonalisation = bank.manifest.get("orthogonalisation") or {}
    if orthogonalisation.get("applied") is not False:
        raise DynamicArmError(
            "the bank's manifest does not state that no orthogonalisation was "
            "applied; this arm applies the direct validated repair direction "
            "and the orthogonalised population bank is a different artefact.")
    checks.record("repair_direction_is_b_d1_minus_e", True,
                  f"d_R = {order}, no orthogonalisation, unit norm at states "
                  f"{list(CANDIDATE_STATES)}")
    return {"subtraction_order": order, "orthogonalisation_applied": False,
            "artifact_sha256": bank.artifact_sha256,
            "manifest_sha256": bank.manifest_sha256}


# ==================================================================== records
def controller_block(fold: int, checkpoint: Mapping[str, Any], g_r: float,
                     reference: float, bit_identical: bool) -> Dict[str, Any]:
    """The per-row controller provenance a record carries at every site.

    The *same* dict is written at both sites, so "one coefficient, applied at
    two blocks" is a property a reader can check on the record rather than a
    claim the runner makes about itself.
    """
    return {
        "fold": int(fold),
        "checkpoint": f"controller_fold_{int(fold)}.json",
        "weights_sha256": str(checkpoint["weights_sha256"]),
        "is_out_of_fold": True,
        "read_state": READ_STATE,
        "gate_input": ("the unsteered state-32 final-prompt-token activation "
                       "captured in this pass, before either site was written"),
        "g_R": float(g_r),
        "applied_alpha_R": float(ALPHA_R * g_r),
        "reference_g_R": float(reference),
        "abs_gate_difference": abs(float(g_r) - float(reference)),
        "activation_bit_identical_to_the_frozen_capture": bool(bit_identical),
    }


def locked_fields(*, direction_bank_sha256: str, dataset_sha256: str,
                  generation_config_sha256: str, model: str, revision: str,
                  tokenizer_config_sha256: str, chat_template_sha256: str,
                  intervention_type: str, budget_reference: str,
                  arm_id: str = ARM_RAW) -> Dict[str, Any]:
    """Everything every record in one output directory must agree on.

    ``alpha_R`` is deliberately absent: it is the row's own coefficient, and
    locking it would make the arm a constant-strength arm again.
    """
    if arm_id not in ARMS:
        raise DynamicArmError(f"unknown arm {arm_id!r}; this stage runs "
                              f"{list(ARMS)}")
    return {
        "protocol": PROTOCOL,
        "arm_id": arm_id,
        "alpha_S": ALPHA_S,
        "intervention_type": intervention_type,
        "budget_reference": budget_reference,
        "states": list(CANDIDATE_STATES),
        "historical_layers": list(CANDIDATE_BLOCKS),
        "model": model,
        "revision": revision,
        "generation_config_sha256": generation_config_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "chat_template_sha256": chat_template_sha256,
        "direction_bank_sha256": direction_bank_sha256,
        "dataset_sha256": dataset_sha256,
        "smoke": False,
    }


def validate_record(record: Mapping[str, Any], expected: Mapping[str, Any],
                    corpus: Mapping[str, Mapping[str, str]],
                    fold_of: Mapping[str, int],
                    reference: Mapping[str, float], where: str,
                    tolerance: float = GATE_TOLERANCE) -> None:
    """One dynamic record is what it claims to be, or it is refused.

    Checked in full rather than by arm id. The failure this exists to catch --
    a record carrying the right arm name and a coefficient from the wrong
    controller, the wrong row or the wrong site -- looks entirely well formed.
    """
    for field in RUN_LOCKED_FIELDS:
        if field not in expected:
            continue
        want, got = expected[field], record.get(field)
        if isinstance(want, (list, tuple)):
            want, got = list(want), list(got or [])
        if got != want:
            raise DynamicArmError(
                f"{where}: {field} is {got!r} and this run is {want!r}. A "
                f"record produced under different settings is not reused; it "
                f"is refused, and the row is regenerated.")
    if record.get("alpha_S") != ALPHA_S:
        raise DynamicArmError(
            f"{where}: alpha_S is {record.get('alpha_S')!r}. d_S = H - E is "
            f"retired and no coefficient scales it.")
    if list(record.get("states") or []) != list(CANDIDATE_STATES):
        raise DynamicArmError(
            f"{where}: states {record.get('states')!r}, expected "
            f"{list(CANDIDATE_STATES)} (historical blocks "
            f"{list(CANDIDATE_BLOCKS)})")
    if 34 in (record.get("states") or []) or 33 in (
            record.get("historical_layers") or []):
        raise DynamicArmError(
            f"{where}: historical block 33 / package state 34 is present. That "
            f"site failed the one-pass budget test and is not in this arm.")
    if record.get("status") != "ok":
        raise DynamicArmError(
            f"{where}: status {record.get('status')!r}. Only a completed "
            f"record counts; an error is regenerated, never inherited.")
    if record.get("error_message"):
        raise DynamicArmError(
            f"{where}: carries error_message {record['error_message']!r}")
    hooks_expected = record.get("hooks_expected")
    hooks_fired = record.get("hooks_fired")
    if hooks_expected != len(CANDIDATE_STATES) or hooks_fired != hooks_expected:
        raise DynamicArmError(
            f"{where}: {hooks_fired} hook(s) fired of {hooks_expected} "
            f"expected; this arm edits {len(CANDIDATE_STATES)} sites and a "
            f"record whose edit did not fire is not a steered record.")

    rid = record.get("row_id")
    if rid not in corpus:
        raise DynamicArmError(f"{where}: {rid!r} is not a corpus row")
    if record.get("prompt_sha256") != corpus[rid]["prompt_sha256"]:
        raise DynamicArmError(
            f"{where}: prompt digest {record.get('prompt_sha256')!r} is not "
            f"the corpus digest {corpus[rid]['prompt_sha256']!r}")
    if record.get("row_class") != corpus[rid]["final_ebh"]:
        raise DynamicArmError(
            f"{where}: class {record.get('row_class')!r}, corpus says "
            f"{corpus[rid]['final_ebh']!r}")

    per_state = record.get("per_state") or {}
    if sorted(int(s) for s in per_state) != list(CANDIDATE_STATES):
        raise DynamicArmError(
            f"{where}: per_state covers {sorted(per_state)}, expected "
            f"{list(CANDIDATE_STATES)}")
    blocks: List[Mapping[str, Any]] = []
    for state, entry in sorted(per_state.items(), key=lambda kv: int(kv[0])):
        digest = entry.get("repair_direction_sha256")
        if not digest or len(str(digest)) != 64:
            raise DynamicArmError(
                f"{where}: state {state} records no repair direction digest")
        block = entry.get("controller")
        if not isinstance(block, Mapping):
            raise DynamicArmError(
                f"{where}: state {state} records no controller block, so the "
                f"coefficient it applied cannot be traced to a controller")
        blocks.append(block)
    if blocks[0] != blocks[1]:
        raise DynamicArmError(
            f"{where}: the two sites record different controller blocks. One "
            f"g_R is computed per prompt and applied at historical blocks "
            f"{list(CANDIDATE_BLOCKS)}; two coefficients is a different "
            f"intervention.")

    block = blocks[0]
    if int(block.get("fold", -1)) != int(fold_of[rid]):
        raise DynamicArmError(
            f"{where}: scored by fold {block.get('fold')!r} and {rid} is in "
            f"fold {fold_of[rid]}. A row is scored only by the controller that "
            f"never saw it.")
    if block.get("is_out_of_fold") is not True:
        raise DynamicArmError(
            f"{where}: the controller block does not state is_out_of_fold")
    if int(block.get("read_state", -1)) != READ_STATE:
        raise DynamicArmError(
            f"{where}: read_state {block.get('read_state')!r}, expected "
            f"{READ_STATE}")

    g_r = float(block["g_R"])
    if rid in reference:
        delta = abs(g_r - float(reference[rid]))
        if not delta <= float(tolerance):
            raise DynamicArmError(
                f"{where}: the recorded coefficient {g_r!r} and the "
                f"frozen-activation coefficient {reference[rid]!r} differ by "
                f"{delta:.6e}, over the {float(tolerance):.6e} budget. The "
                f"record does not reproduce and is not counted.")
    applied = float(ALPHA_R * g_r)
    if float(record.get("alpha_R", float("nan"))) != applied:
        raise DynamicArmError(
            f"{where}: alpha_R is {record.get('alpha_R')!r} and "
            f"{ALPHA_R} * g_R is {applied!r}. The applied coefficient is "
            f"alpha_R * g_R with alpha_R frozen at {ALPHA_R}.")
    if not 0.0 < applied <= ALPHA_R:
        raise DynamicArmError(
            f"{where}: applied coefficient {applied!r} is outside "
            f"(0, {ALPHA_R}]. The ceiling is arithmetic -- g_R is a sigmoid -- "
            f"not a clamp, so a value outside it means the coefficient did not "
            f"come from this controller.")


def validate_existing_records(path: Path | str, allowed_rows: Sequence[str],
                              expected: Mapping[str, Any],
                              corpus: Mapping[str, Mapping[str, str]],
                              fold_of: Mapping[str, int],
                              reference: Mapping[str, float],
                              checks: Optional[Checklist] = None
                              ) -> Dict[str, Dict[str, Any]]:
    """Records already in this shard's file, validated before being skipped.

    A restart that trusted its own output would inherit whatever the crash
    left: a duplicate, a row from another shard, an error record, or a record
    written under settings that have since moved. Each is refused by name.
    """
    checks = checks or Checklist(PROTOCOL)
    records = read_jsonl_strict(path)
    if not records:
        checks.record("existing_records", True, "none; a fresh shard")
        return {}
    allowed = set(allowed_rows)
    seen: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        where = f"{Path(path).name}:{index}"
        rid = record.get("row_id")
        if rid in seen:
            raise DynamicArmError(
                f"{where}: {rid} appears more than once. One row is generated "
                f"once; a duplicate is refused rather than deduplicated, "
                f"because the two may differ.")
        if rid not in allowed:
            raise DynamicArmError(
                f"{where}: {rid} is not one of this shard's {len(allowed)} "
                f"row(s). A record from another shard in this file means two "
                f"processes shared it.")
        validate_record(record, expected, corpus, fold_of, reference, where)
        seen[rid] = record
    checks.record(
        "existing_records", True,
        f"{len(seen)} record(s) validated and will be skipped; "
        f"{len(allowed) - len(seen)} remain")
    return seen


def assert_shard_is_not_complete(done: Mapping[str, Any],
                                 allowed_rows: Sequence[str]) -> None:
    """A finished shard is not re-entered: a completed run is not rerun."""
    if allowed_rows and len(done) >= len(allowed_rows):
        raise DynamicArmError(
            f"this shard's {len(allowed_rows)} row(s) are already complete and "
            f"validated. A finished shard is not re-run: its records are the "
            f"result. Verify it instead, or name a new output directory.")


# ================================================================== completion
SHARD_RECORDS_GLOB = "screen_records.shard*.jsonl"


def collect_records(output_dir: Path | str, expected: Mapping[str, Any],
                    corpus: Mapping[str, Mapping[str, str]],
                    fold_of: Mapping[str, int],
                    reference: Mapping[str, float],
                    checks: Optional[Checklist] = None
                    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Every shard's records in one directory, merged and fully validated.

    Merging is the one place two shards could quietly overlap, so a row that
    appears in two files is an error rather than a deduplication. Every record
    is validated in full -- not counted -- because a directory of the right
    size holding the wrong records is exactly the outcome a completion check
    exists to refuse.
    """
    checks = checks or Checklist(PROTOCOL)
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob(SHARD_RECORDS_GLOB))
    checks.require("records_files", bool(files),
                   f"{len(files)} shard records file(s) under {output_dir}"
                   if files else f"{output_dir} holds no {SHARD_RECORDS_GLOB}")
    merged: Dict[str, Dict[str, Any]] = {}
    origin: Dict[str, str] = {}
    per_file: Dict[str, int] = {}
    for path in files:
        records = read_jsonl_strict(path)
        per_file[path.name] = len(records)
        for index, record in enumerate(records):
            where = f"{path.name}:{index}"
            rid = record.get("row_id")
            if rid in merged:
                raise DynamicArmError(
                    f"{where}: {rid} is also in {origin[rid]}. One row is "
                    f"generated once; two shards that wrote it did not "
                    f"partition the corpus.")
            validate_record(record, expected, corpus, fold_of, reference,
                            where)
            merged[rid] = record
            origin[rid] = path.name
    checks.record("records_validated", True,
                  f"{len(merged)} record(s) over {len(files)} file(s): "
                  + ", ".join(f"{name} {count}"
                              for name, count in sorted(per_file.items())))
    return merged, {"files": per_file, "records": len(merged)}


def assert_complete(records: Mapping[str, Mapping[str, Any]],
                    order: Sequence[str],
                    checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """Exactly the 480 rows, once each, in the deterministic order."""
    checks = checks or Checklist(PROTOCOL)
    checks.require("row_count", len(order) == EXPECTED_ROWS,
                   f"{len(order)} row(s) in the order, expected "
                   f"{EXPECTED_ROWS}")
    missing = [rid for rid in order if rid not in records]
    extra = sorted(set(records) - set(order))
    checks.require("completed_records", len(records) == EXPECTED_ROWS,
                   f"{len(records)} completed record(s), expected "
                   f"{EXPECTED_ROWS}")
    checks.require("no_row_is_missing", not missing,
                   f"{len(missing)} row(s) have no record"
                   if missing else f"all {EXPECTED_ROWS} rows are present")
    checks.require("no_record_is_unexpected", not extra,
                   f"{len(extra)} record(s) name rows outside the corpus order"
                   if extra else "every record names a corpus row")
    coefficients = [float(records[rid]["alpha_R"]) for rid in order]
    checks.require("every_coefficient_is_within_the_frozen_ceiling",
                   all(0.0 < c <= ALPHA_R for c in coefficients),
                   f"min {min(coefficients):.6f}, max {max(coefficients):.6f}, "
                   f"ceiling {ALPHA_R}")
    distinct = len({round(c, 12) for c in coefficients})
    checks.require(
        "the_coefficient_is_prompt_specific", distinct > 1,
        f"{distinct} distinct applied coefficient(s) over {EXPECTED_ROWS} "
        f"row(s); a single value would mean the gate was not read")
    return {"rows": len(order), "records": len(records),
            "min_applied_alpha_R": min(coefficients),
            "max_applied_alpha_R": max(coefficients),
            "mean_applied_alpha_R": float(np.mean(coefficients)),
            "distinct_applied_coefficients": distinct}


# ===================================================== the output directory
def prepare_output_dir(path: Path | str, allowed: Sequence[str],
                       resume: bool) -> Path:
    """Claim the directory, honouring ``--resume``.

    Without ``--resume`` the directory must be absent or empty. With it, a
    directory holding this shard's own files and nothing else is accepted --
    which is the case a restart is for.
    """
    path = Path(path)
    assert_not_sealed(path, "dynamic arm output directory")
    if not path.exists():
        return path
    if not path.is_dir():
        raise DynamicArmError(f"{path} exists and is not a directory")
    existing = sorted(p.name for p in path.iterdir())
    if not existing:
        return path
    if not resume:
        raise DynamicArmError(
            f"{path} already holds {existing[:5]} and --resume was not given. "
            f"A result is written once, into an empty directory. Pass "
            f"--resume to continue this shard after a crash, or name a new "
            f"directory.")
    unexpected = sorted(set(existing) - set(allowed))
    if unexpected:
        raise DynamicArmError(
            f"{path} holds {unexpected}, which this shard did not write. "
            f"Resuming into a directory another run is using would interleave "
            f"two experiments in one place.")
    return path


def shard_records_name(index: int, count: int) -> str:
    """One file per shard, so no two processes ever append to the same one."""
    return f"screen_records.shard{index:02d}of{count:02d}.jsonl"


def shard_filenames(index: int, count: int) -> Tuple[str, ...]:
    """Everything one shard is allowed to have written into its directory."""
    return (shard_records_name(index, count), COEFFICIENTS_FILENAME,
            OUTPUT_MANIFEST, OUTPUT_VALIDATION)


COEFFICIENT_COLUMNS: Tuple[str, ...] = (
    "row_id", "row_class", "fold", "controller_checkpoint",
    "controller_weights_sha256", "read_state", "g_R", "applied_alpha_R",
    "reference_g_R", "abs_gate_difference",
    "activation_bit_identical_to_the_frozen_capture",
    "apply_at_historical_blocks",
)


def coefficient_rows(records: Mapping[str, Mapping[str, Any]],
                     order: Sequence[str]) -> List[Dict[str, Any]]:
    """The applied-coefficient table, derived from the records themselves.

    Derived, never parallel: the records are the result, and this is a reading
    of them. A row is described by the controller block its own record carries.
    """
    out: List[Dict[str, Any]] = []
    for rid in order:
        if rid not in records:
            continue
        record = records[rid]
        state = str(min(CANDIDATE_STATES))
        block = record["per_state"][state]["controller"]
        out.append({
            "row_id": rid,
            "row_class": record["row_class"],
            "fold": int(block["fold"]),
            "controller_checkpoint": block["checkpoint"],
            "controller_weights_sha256": block["weights_sha256"],
            "read_state": int(block["read_state"]),
            "g_R": float(block["g_R"]),
            "applied_alpha_R": float(record["alpha_R"]),
            "reference_g_R": float(block["reference_g_R"]),
            "abs_gate_difference": float(block["abs_gate_difference"]),
            "activation_bit_identical_to_the_frozen_capture":
                bool(block["activation_bit_identical_to_the_frozen_capture"]),
            "apply_at_historical_blocks": " ".join(
                str(b) for b in CANDIDATE_BLOCKS),
        })
    return out


__all__ = [
    "ALPHA_R", "APPLY_BLOCKS", "ARMS", "ARM_RAW", "ARM_VETO", "ARM_VERSION",
    "CANDIDATE_BLOCKS", "CANDIDATE_BUDGET_STATES", "CANDIDATE_STATES",
    "CHECKPOINT_CLAUSE", "COEFFICIENTS_FILENAME", "COEFFICIENT_COLUMNS",
    "SHARD_RECORDS_GLOB", "collect_records",
    "DIAGNOSTIC_FILENAME", "DynamicArmError", "EXPECTED_ROWS",
    "FLOAT32_EPS", "FOLD_SOURCE_DIR", "GATE_TOLERANCE",
    "GATE_TOLERANCE_EXPRESSION", "MEAN_GATE_TOLERANCE", "OUTPUT_MANIFEST",
    "OUTPUT_VALIDATION", "PROTOCOL", "REFERENCE_ARM",
    "REPAIR_SUBTRACTION_ORDER", "RUN_LOCKED_FIELDS", "SCHEMA",
    "SUMMARY_FILENAME", "assert_complete", "assert_gate_is_faithful",
    "assert_reference_reproduces_the_run",
    "assert_repair_direction_is_b_d1_minus_e", "assert_shard_is_not_complete",
    "coefficient_rows", "controller_block", "fold_map", "load_reproduction",
    "locked_fields", "prepare_output_dir", "reference_gates",
    "shard_filenames", "shard_records_name", "validate_existing_records",
    "validate_record", "verify_bundle",
]
