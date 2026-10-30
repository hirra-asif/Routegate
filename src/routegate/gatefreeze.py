"""The deployment predeclaration for the safety-first RouteGate gate.

This record is written BEFORE the refit runs. It fixes what will be fitted,
from what, under which rule, to which files -- and what the resulting artifact
may and may not be used to claim.

What is being closed
--------------------
One blocker, and only one. The completed sensor run kept out-of-fold
probabilities and no weights, so the gate could only be applied to a
development row by looking that row up. This stage produces the weights, so
the same frozen gate can fire on a prompt it has never seen.

What is NOT being changed
-------------------------
The representation, the estimator, the hyperparameter grid and its selection
rule, both thresholds, the gate rule, the actuator, alpha and the intervention
sites. Every one of them is imported from the frozen artifact that already
holds it, or hash-gated against it. This stage selects nothing.

What the refit may never be used for
------------------------------------
An evaluation. A sensor fitted on all 480 development rows has seen every one
of them, so its predictions there are training-set predictions. The
development estimates remain the out-of-fold ones -- 37/89 E substantive
repairs, 0/178 material B degradations, 0/213 actionable H regressions -- and
they are unchanged by the existence of a deployment artifact.

What comes after, declared now so it cannot be chosen later
-----------------------------------------------------------
The capability suite is MMLU, GSM8K and ARC-Challenge, complete; the only two
conditions are baseline and the safety-first gate; and the final confirmatory
evaluation is the existing frozen 380-row sealed set, opened once, after
capability evaluation. No prompt of it is read here and none is reachable from
this repository.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .gatesensors import (
    ACTIVATIONS_SHA256,
    BUNDLE_FILE,
    CHECKPOINT_FILES,
    CHECKPOINT_SCHEMA,
    C_GRID,
    C_SELECTED_PER_OUTER_FOLD,
    C_SELECTION_RULE,
    CLASS_WEIGHT,
    ESTIMATOR,
    EXPECTED_ROWS,
    EXTRACTION_MANIFEST_SHA256,
    INNER_SCORING,
    INNER_SPLITS,
    KIND_REFIT,
    MAX_ITER,
    OOF_PREDICTIONS_SHA256,
    OOF_REPRODUCTION_ATOL,
    OUTER_SPLITS,
    PENALTY,
    READ_STATE,
    READ_STATE_MEANS,
    SEED,
    SENSOR_CONFIG_SHA256,
    SENSOR_E,
    SENSOR_H,
    SENSOR_NEGATIVE,
    SENSOR_POSITIVE,
    SOLVER,
    THRESHOLDS_SHA256,
    TOKEN_POSITION,
    load_thresholds,
)
from .runtime import (
    FROZEN_CHAT_TEMPLATE_SHA256,
    FROZEN_MODEL,
    FROZEN_REVISION,
    FROZEN_TOKENIZER_CONFIG_SHA256,
)
from .validation import Checklist, ValidationError, sha256_file, sha256_text

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "routegate/gate_deployment_predeclaration/1"
NAME = "routegate_gate_deployment_v1"
VERSION = 1
DIGEST_FIELD = "record_sha256"
PREDECLARATION = REPO_ROOT / "manifests" / f"{NAME}.json"

#: Where the deployment checkpoints are written. Outside every existing run
#: directory: this stage adds an artifact and edits none.
OUTPUT_DIR = "outputs/routegate_gate_deployment_v1"

#: Every input, by the digest it must still hash to.
SOURCES: Dict[str, Dict[str, str]] = {
    "residual_activations": {
        "root": "routegate-data", "paths_key": "extraction_dir",
        "path": "residual_final_prompt_token.npy",
        "sha256": ACTIVATIONS_SHA256,
        "role": "the frozen feature artifact, [480, 49, 2048]; state 45 at the "
                "final prompt token is the sensor input",
    },
    "residual_extraction_manifest": {
        "root": "routegate-data", "paths_key": "extraction_dir",
        "path": "manifest.csv", "sha256": EXTRACTION_MANIFEST_SHA256,
        "role": "the row order of the activation array",
    },
    "development_corpus": {
        "root": "routegate-data", "paths_key": "dataset",
        "path": "rq3_dataset_final.csv",
        "sha256": "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716"
                  "353e1d08",
        "role": "the 480 development rows and their frozen final_ebh labels",
    },
    "sensor_thresholds": {
        "root": "repository", "paths_key": "",
        "path": "outputs/routegate_residual_v1/thresholds.json",
        "sha256": THRESHOLDS_SHA256,
        "role": "both frozen thresholds and the frozen gate rule; read, never "
                "recomputed",
    },
    "sensor_oof_predictions": {
        "root": "repository", "paths_key": "",
        "path": "outputs/routegate_residual_v1/oof_predictions.csv",
        "sha256": OOF_PREDICTIONS_SHA256,
        "role": "the frozen out-of-fold probabilities the reconstruction must "
                "reproduce before any deployment weight may be written",
    },
    "sensor_config": {
        "root": "repository", "paths_key": "",
        "path": "outputs/routegate_residual_v1/config.json",
        "sha256": SENSOR_CONFIG_SHA256,
        "role": "the frozen protocol every constant in this stage is "
                "transcribed from",
    },
    "static_steering_master": {
        "root": "routegate-data", "paths_key": "static_steering_master",
        "path": "rq3_static_steering_ebh_master_20260811.csv",
        "sha256": "525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705"
                  "b11b9e4",
        "role": "the frozen per-row outcome table behind the reported "
                "development estimates; read by the reporting stage, not by "
                "the fit",
    },
    "static_run_config": {
        "root": "routegate-data", "paths_key": "",
        "path": "frozen/static_steering/rq3_bh_rq2_static_20260810/"
                "run_config.json",
        "sha256": "e8b1b46f0923ba1bef1a770ed1bc95d4e92c49e6457a52bcdb0ef3a0d"
                  "1e31777",
        "role": "the actuator, alpha, layers and generation settings the "
                "outcome table was produced under",
    },
}

#: The actuator, transcribed from static_run_config and NOT re-chosen. It is
#: the RQ2 primary actuator: consensus23_global_frozen, built from donors d2
#: and d3. It is NOT historical_d1_global_frozen (B_d1 - E), which is a later
#: and different direction; conflating the two would misattribute the result.
ACTUATOR: Dict[str, Any] = {
    "direction_variant": "consensus23_global_frozen",
    "role": "primary",
    "donors": ["B_d2", "B_d3"],
    "is_not": "historical_d1_global_frozen (B_d1 - E), which is a different "
              "direction used by later controller work and did not produce "
              "this outcome table",
    "site": "residual_stream",
    "historical_layers": [31, 32, 33],
    "window": 1,
    "slot": 0,
    "prefill_only": True,
    "alpha_R": 0.15,
    "alpha_S": 0.0,
    "expected_patch_hits_per_row": 3,
    "application_scaling": "accepted CPR-Min: applied_delta = unit_direction * "
                           "0.15 * current prompt layer residual norm, slot 0",
    "tensors": "routegate-data/frozen/rq2_actuator (pinned by "
               "manifests/rq2_actuator.json; not committed)",
}

TRAINING: Dict[str, Any] = {
    "what_is_refitted": "both sensors, on all 480 development rows",
    "why": "to obtain final deployment weights. The completed run kept "
           "out-of-fold probabilities and no weights, so the gate could only "
           "be applied by row-id lookup.",
    "estimator": ESTIMATOR,
    "penalty": PENALTY, "solver": SOLVER, "class_weight": CLASS_WEIGHT,
    "max_iter": MAX_ITER,
    "C_grid": list(C_GRID),
    "C_selection_rule": C_SELECTION_RULE,
    "C_selected_per_outer_fold_in_the_completed_run": {
        k: list(v) for k, v in C_SELECTED_PER_OUTER_FOLD.items()},
    "C_for_the_refit": "selected by the SAME inner rule applied to all 480 "
                       "rows. It is not copied from an outer fold and is not "
                       "chosen by hand; the value is recorded in the "
                       "checkpoint after the fit.",
    "inner_cv": f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True, "
                f"random_state={SEED}) scored by {INNER_SCORING}",
    "outer_cv": f"StratifiedKFold(n_splits={OUTER_SPLITS}, shuffle=True, "
                f"random_state={SEED}) -- the frozen fold division, "
                f"reproduced and checked, used only for the reproduction gate",
    "seed": SEED,
    "deterministic": "lbfgs on a fixed design with a fixed seed. Refitting "
                     "twice on the same inputs yields byte-identical weights, "
                     "and a test asserts it.",
    "heads": {
        SENSOR_E: {"positive": SENSOR_POSITIVE[SENSOR_E],
                   "negative": list(SENSOR_NEGATIVE[SENSOR_E]),
                   "fitted_on": "the E and B rows of the fit set"},
        SENSOR_H: {"positive": SENSOR_POSITIVE[SENSOR_H],
                   "negative": list(SENSOR_NEGATIVE[SENSOR_H]),
                   "fitted_on": "all rows of the fit set"},
    },
    "reproduction_gate": {
        "rule": "before any deployment weight is written, the five fold "
                "sensors are reconstructed under this exact specification and "
                "their out-of-fold probabilities compared against the frozen "
                "oof_predictions.csv row for row.",
        "atol": OOF_REPRODUCTION_ATOL,
        "on_failure": "nothing is written. A specification that cannot "
                      "reproduce the frozen probabilities is not the "
                      "specification that produced them, and a deployment "
                      "artifact built from it would be a different sensor.",
    },
}

CAPABILITY: Dict[str, Any] = {
    "protocol": "routegate_final_evaluation_v1",
    "protocol_manifest": "manifests/routegate_final_evaluation_v1.json",
    "note": "the capability suite, the conditions and the sealed evaluation "
            "are declared in the final-evaluation manifest and are not "
            "restated here. This record's scope is the sensor checkpoint.",
    "datasets_cached_at_freeze_time": False,
}

SEALED: Dict[str, Any] = {
    "protocol": "routegate_final_evaluation_v1",
    "opened_by_this_stage": False,
    "prompt_content_read": False,
    "commitment": "manifests/sealed_set_commitment.json",
    "expected_row_count": 380,
    "how_established": "this repository stores no sealed prompt, no sealed "
                       "prompt hash and no path to sealed material, and "
                       "configs/paths.local.json names none. Nothing here can "
                       "reach it.",
}

DOES_NOT_CLAIM: Dict[str, Any] = {
    "the_refit_is_an_evaluation": False,
    "the_development_estimates_changed": False,
    "statement": "the refit produces weights and nothing else. The reported "
                 "development estimates remain the out-of-fold 37/89 E "
                 "substantive repairs, 0/178 material B degradations and "
                 "0/213 actionable H regressions, computed from the frozen "
                 "outcome table under the frozen gate. A sensor fitted on all "
                 "480 rows has seen every one of them, so its predictions "
                 "there are training-set predictions and are never reported.",
}

DEVELOPMENT_ESTIMATES: Dict[str, Any] = {
    "standing": "out-of-fold; unchanged by this stage",
    "E_substantive_repair": [37, 89],
    "B_material_degradation": [0, 178],
    "H_actionable_regression": [0, 213],
    "steered_rows": 59,
    "source": "recomputed by routegate.thesisresults from the frozen outcome "
              "table and the frozen gate; the derived gate matches the "
              "recorded gate_full column row for row",
}


class GateFreezeError(ValidationError):
    """The deployment predeclaration could not be established."""


def canonical(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def record_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical(payload))


def build_predeclaration() -> Dict[str, Any]:
    """The committed payload. Deterministic: no timestamp, no environment."""
    thresholds = load_thresholds()
    record: Dict[str, Any] = {
        "schema": SCHEMA, "name": NAME, "version": VERSION,
        "stage": "gate_sensor_deployment_refit",
        "status": "predeclared",
        "closes_blocker": "no deployable sensor for the safety-first gate: "
                          "the completed run kept out-of-fold probabilities "
                          "and no weights, so the gate could be applied only "
                          "by row-id lookup",
        "changes_nothing_else": [
            "representation", "estimator", "hyperparameter grid",
            "C selection rule", "threshold_E", "threshold_H", "gate rule",
            "actuator direction", "alpha", "intervention layers",
        ],
        "representation": {
            "read_state": READ_STATE,
            "read_state_means": READ_STATE_MEANS,
            "warning": "package state 45 is NOT transformer layer 45",
            "token_position": TOKEN_POSITION,
            "hidden_size": 2048,
            "rows": EXPECTED_ROWS,
        },
        "gate": {
            "rule": thresholds.gate_rule,
            "threshold_E": thresholds.threshold_E,
            "threshold_H": thresholds.threshold_H,
            "thresholds_source": thresholds.source,
            "thresholds_sha256": thresholds.source_sha256,
            "unchanged": True,
            "recomputed_here": False,
        },
        "actuator": dict(ACTUATOR),
        "training": dict(TRAINING),
        "runtime": {
            "model": FROZEN_MODEL, "revision": FROZEN_REVISION,
            "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
            "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
            "decoding": "greedy", "max_new_tokens": 768,
            "design": "TWO-PASS prompt-level gate: an unsteered diagnostic "
                      "forward pass reads state 45 at the final prompt token "
                      "and yields p_E and p_H; if the gate fires the second "
                      "pass generates with the frozen actuator, otherwise it "
                      "generates normally.",
            "requires_row_id": False,
            "reads_any_per_row_table": False,
            "row_id_note": "the deployed gate takes an activation vector and "
                           "nothing else. It consults no table, and "
                           "GateBundle.decide accepts no row identifier.",
        },
        "output": {
            "directory": OUTPUT_DIR,
            "checkpoints": {k: f"{OUTPUT_DIR}/{v}"
                            for k, v in sorted(CHECKPOINT_FILES.items())},
            "bundle": f"{OUTPUT_DIR}/{BUNDLE_FILE}",
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "kind": KIND_REFIT,
            "is_new_directory": True,
            "modifies_any_existing_run": False,
        },
        "sources": {k: dict(v) for k, v in sorted(SOURCES.items())},
        "development_estimates": dict(DEVELOPMENT_ESTIMATES),
        "capability_evaluation": dict(CAPABILITY),
        "sealed_evaluation": dict(SEALED),
        "does_not_claim": dict(DOES_NOT_CLAIM),
        "contains_prompt_or_response_text": False,
        "contains_row_ids": False,
        "reads_the_sealed_set": False,
        "generates_responses": False,
        "calls_a_judge": False,
        "downloads_anything": False,
        "modifies_sigr": False,
    }
    record[DIGEST_FIELD] = record_sha256(record)
    return record


def verify_predeclaration(path: Path | str = PREDECLARATION,
                          checks: Optional[Checklist] = None
                          ) -> Dict[str, Any]:
    """The committed file is well formed, self-consistent and current."""
    checks = checks or Checklist(NAME)
    path = Path(path)
    checks.require("predeclaration_exists", path.is_file(), str(path))
    committed = json.loads(path.read_text(encoding="utf-8"))
    checks.require("schema", committed.get("schema") == SCHEMA,
                   f"{committed.get('schema')!r}")
    checks.require("name", committed.get("name") == NAME,
                   f"{committed.get('name')!r}")
    recomputed = record_sha256(committed)
    checks.require("record_digest", recomputed == committed.get(DIGEST_FIELD),
                   f"records {committed.get(DIGEST_FIELD)}, body hashes to "
                   f"{recomputed}")
    current = build_predeclaration()
    drifted = sorted(k for k in set(current) | set(committed)
                     if k != DIGEST_FIELD and current.get(k) != committed.get(k))
    checks.require("predeclaration_matches_the_package", not drifted,
                   f"{len(drifted)} field(s) differ: {drifted[:5]}" if drifted
                   else f"{len(current) - 1} field(s), every one identical")
    thresholds = load_thresholds()
    gate = committed.get("gate") or {}
    checks.require(
        "thresholds_are_the_frozen_ones",
        gate.get("threshold_E") == thresholds.threshold_E
        and gate.get("threshold_H") == thresholds.threshold_H
        and gate.get("rule") == thresholds.gate_rule,
        f"E {gate.get('threshold_E')!r} H {gate.get('threshold_H')!r}")
    cap = committed.get("capability_evaluation") or {}
    checks.require("capability_defers_to_the_final_protocol",
                   cap.get("protocol") == "routegate_final_evaluation_v1",
                   f"{cap.get('protocol')!r}")
    sealed = committed.get("sealed_evaluation") or {}
    checks.require("sealed_is_the_frozen_380",
                   sealed.get("expected_row_count") == 380
                   and sealed.get("opened_by_this_stage") is False
                   and sealed.get("prompt_content_read") is False,
                   f"{sealed.get('expected_row_count')} rows, opened="
                   f"{sealed.get('opened_by_this_stage')}")
    checks.require("development_estimates_unchanged",
                   committed["development_estimates"]["E_substantive_repair"]
                   == [37, 89]
                   and committed["development_estimates"][
                       "B_material_degradation"] == [0, 178]
                   and committed["development_estimates"][
                       "H_actionable_regression"] == [0, 213],
                   "37/89, 0/178, 0/213")
    for flag in ("reads_the_sealed_set", "generates_responses",
                 "calls_a_judge", "downloads_anything", "modifies_sigr",
                 "contains_prompt_or_response_text", "contains_row_ids"):
        checks.require(f"flag_{flag}", committed.get(flag) is False,
                       f"{flag} is {committed.get(flag)!r}")
    return {"path": str(path), "sha256": sha256_file(path),
            DIGEST_FIELD: recomputed}


__all__ = [
    "ACTUATOR", "CAPABILITY", "DEVELOPMENT_ESTIMATES", "DIGEST_FIELD",
    "DOES_NOT_CLAIM", "GateFreezeError", "NAME", "OUTPUT_DIR",
    "PREDECLARATION", "SCHEMA", "SEALED", "SOURCES", "TRAINING", "VERSION",
    "build_predeclaration", "canonical", "record_sha256",
    "verify_predeclaration",
]
