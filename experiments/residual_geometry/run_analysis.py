#!/usr/bin/env python3
"""The corrected residual-geometry analysis.

The pilot reported ROC-AUC 0.9526 at state 45. Reproduction confirmed the
arithmetic; it did not repair the design. This entry point runs the corrected
analysis declared in ``configs/residual_geometry.json``, which fixes four
things:

    state selection     nested -- chosen by an inner cross-validation inside
                        each outer training fold and scored on the outer
                        held-out fold, so the reported number is not a maximum
                        taken on the rows it is reported from
    permutation null    each permutation repeats the complete state search and
                        contributes its maximum, so the null prices the
                        selection rather than assuming it away
    magnitude confound  activation-norm, unit-L2-normalised and
                        norm-residualised probes run beside the raw probe, so
                        the directional component is separated from the
                        magnitude component that reached ROC-AUC 0.871 alone
    fold construction   StratifiedKFold repeated over many splits, with the
                        assignment written out once and reused by every state,
                        variant, control and permutation

It refuses to execute until two conditions hold:

    1. the pilot reproduction has been run and its state-45 gate passed;
    2. the operator approves the protocol explicitly on the command line.

The second condition is deliberate friction: a nested selection estimate is not
comparable to the pilot's selected maximum, and that substitution should be a
decision rather than a side effect of running a script. ``--smoke`` is exempt --
it exercises the code paths on a small subset, writes to a directory whose name
contains "smoke", and stamps every record it writes as non-reportable.

Still out of scope: the withheld class H, routing and expert features,
interventions, and any held-out benchmark material.

Usage:
    python experiments/residual_geometry/run_analysis.py --smoke
    python experiments/residual_geometry/run_analysis.py \\
        --i-have-approved-the-corrected-protocol --jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np                                                  # noqa: E402
from joblib import Parallel, delayed, parallel_backend              # noqa: E402

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.controls import (                                    # noqa: E402
    ACTIVATION_VARIANTS,
    PROMPT_LENGTH_FEATURE_NAMES,
    PROMPT_LENGTH_NOTE,
    categorical_column,
    control_catalogue,
    prompt_length_features,
    variant_description,
    variant_matrix,
    variant_transform_factory,
)
from routegate.data import (                                        # noqa: E402
    COVARIATE_COLUMNS,
    describe_state,
    load_development_inputs,
    load_input_manifest,
    load_local_paths,
)
from routegate.geometry import (                                    # noqa: E402
    centroid_separation,
    cosine_similarity_matrix,
    fit_projection,
    project,
    separation_table,
)
from routegate.metrics import (                                     # noqa: E402
    chance_levels,
    evaluate_out_of_fold,
)
from routegate.probes import (                                      # noqa: E402
    DECISION_THRESHOLD,
    cross_val_out_of_fold,
    make_categorical_probe,
    make_probe,
    probe_specification,
    raw_space_direction,
)
from routegate.runrecord import (                                   # noqa: E402
    environment_record,
    utc_now,
    write_csv,
    write_json,
    write_output_hashes,
)
from routegate.selection import (                                   # noqa: E402
    mean_cross_validated_roc_auc,
    nested_state_selection,
    permutation_null,
    permutation_search,
)
from routegate.splits import (                                      # noqa: E402
    RepeatedFoldAssignment,
    build_repeated_stratified_folds,
    folds_for_subset,
    load_repeated_fold_assignments,
    write_repeated_fold_assignments,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    safe_argmax,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "residual_geometry.json"
REPRODUCTION_VALIDATION = (REPO_ROOT / "outputs" / "residual_geometry"
                           / "pilot_reproduction" / "validation.json")

APPROVAL_FLAG = "--i-have-approved-the-corrected-protocol"
UNDEFINED = ""          # an undefined statistic is written empty, never as NaN


def prerequisites_met(validation_path: Path = REPRODUCTION_VALIDATION):
    """(ok, reason): has the pilot reproduction passed its gate?"""
    if not validation_path.exists():
        return False, (f"{validation_path} does not exist — run "
                       f"experiments/residual_geometry/reproduce_pilot.py first")
    record = json.loads(validation_path.read_text(encoding="utf-8"))
    if not record.get("all_passed"):
        failed = [e["check"] for e in record.get("checks", [])
                  if e.get("status") != "PASS"]
        return False, f"the reproduction reported failing checks: {failed}"
    summary = record.get("summary", {})
    if not summary.get("gate_passed"):
        return False, (f"the reproduction did not pass its state-"
                       f"{summary.get('gate_state')} gate")
    return True, (f"reproduction passed: state {summary.get('gate_state')} within "
                  f"{summary.get('gate_tolerance')}, largest difference over all "
                  f"states {summary.get('max_absolute_metric_difference')}")


# --------------------------------------------------------------------- workers
# Module-level so they can be dispatched to worker processes. Each is an
# independent deterministic unit of work over a shared read-only feature cube,
# so running them on one core or on many gives the same numbers.
def _state_sweep(state: int, cube: np.ndarray, y: np.ndarray,
                 assignments: Sequence, variants: Sequence[str],
                 direction_repeat: Optional[int]) -> Dict[str, Any]:
    """Every variant, every repeat, at one state."""
    X = np.asarray(cube[state], dtype=np.float64)
    metrics: List[Dict[str, Any]] = []
    scores: Dict[str, np.ndarray] = {}
    directions: Optional[np.ndarray] = None

    for variant in variants:
        factory = variant_transform_factory(variant)
        design = variant_matrix(variant, X) if factory is None else X
        for repeat, folds in enumerate(assignments):
            result = cross_val_out_of_fold(design, y, folds,
                                           probe_factory=make_probe,
                                           keep_fold_detail=True,
                                           transform_factory=factory)
            record = evaluate_out_of_fold(
                y, result.scores, result.predictions, state=state,
                degenerate=result.degenerate,
                all_folds_converged=result.all_converged,
                max_solver_iterations=result.max_iterations,
                threshold=DECISION_THRESHOLD).as_dict()
            record["variant"] = variant
            record["repeat"] = repeat
            record["mean_fold_roc_auc"] = mean_cross_validated_roc_auc(y, result)
            record["n_features"] = int(design.shape[1])
            metrics.append(record)
            scores[f"{variant}|{repeat}"] = result.scores.astype(np.float64)

            if (variant == "raw" and direction_repeat is not None
                    and repeat == direction_repeat and not result.degenerate):
                directions = np.vstack([
                    raw_space_direction(f.coef, f.scaler_scale)
                    for f in result.folds])

    return {"state": int(state), "metrics": metrics, "scores": scores,
            "directions": directions}


def _nested_task(repeat: int, cube: np.ndarray, y: np.ndarray, folds,
                 states: Sequence[int], inner_splits: int, seed: int):
    return nested_state_selection(
        lambda state: np.asarray(cube[state], dtype=np.float64), y, folds,
        states, inner_splits=inner_splits, seed=seed, repeat=repeat)


def _permutation_task(seed: int, cube: np.ndarray, y: np.ndarray,
                      repeated: RepeatedFoldAssignment,
                      states: Sequence[int]) -> Dict[str, Any]:
    return permutation_search(
        lambda state: np.asarray(cube[state], dtype=np.float64), y, repeated,
        states, seed=seed)


# ------------------------------------------------------------------- analysis
def analyse(config_path: Path = DEFAULT_CONFIG,
            output_dir: Optional[Path] = None,
            smoke: bool = False,
            jobs: int = 1,
            paths_path: Optional[Path] = None,
            permutations: Optional[int] = None,
            repeats: Optional[int] = None) -> Dict[str, Any]:
    started = time.time()
    started_utc = utc_now()

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "corrected analysis config")
    checks = Checklist("residual_geometry_corrected")

    smoke_config = config.get("smoke_test", {})
    if output_dir is None:
        output_dir = REPO_ROOT / (smoke_config["directory"] if smoke
                                  else config["outputs"]["directory"])
    output_dir = Path(output_dir)
    if smoke and "smoke" not in output_dir.name.lower():
        raise ValidationError(
            f"a smoke test must write to a directory whose name says so; "
            f"{output_dir.name!r} does not contain 'smoke'")
    if not smoke and "smoke" in output_dir.name.lower():
        raise ValidationError(
            f"{output_dir.name!r} is a smoke-test directory; a reportable run "
            f"must not write there")

    # --- inputs -------------------------------------------------------------
    input_manifest = load_input_manifest()
    local_paths = load_local_paths(paths_path) if paths_path else load_local_paths()
    print("[gate] validating the accepted development inputs ...")
    inputs = load_development_inputs(manifest=input_manifest, paths=local_paths)
    for entry in inputs.checklist.entries:
        checks.entries.append(entry)
    y = inputs.y
    print(f"[gate] {len(inputs.checklist.entries)} input checks passed; "
          f"n={inputs.n} (E={inputs.n_positive}, B={inputs.n_negative})")

    # Requirement, verified rather than assumed: a prompt-length control may use
    # a saved token count only if one exists. On this extraction none does.
    token_columns = inputs.saved_token_count_columns
    checks.record(
        "prompt_token_count_is_not_invented", not token_columns,
        f"saved token-count columns: {token_columns or 'none'}. "
        f"{PROMPT_LENGTH_NOTE}")
    for role, column in COVARIATE_COLUMNS.items():
        values = inputs.covariate(role)
        checks.require(
            f"{role}_control_uses_the_genuine_column_{column}",
            column in inputs.dataset_columns and len(set(values)) > 1,
            f"{column}: {len(set(values))} distinct value(s) over "
            f"{len(values)} analysed rows")

    checks.require("decision_threshold_is_fixed_at_one_half",
                   DECISION_THRESHOLD == 0.5,
                   "balanced accuracy and accuracy are read at a fixed "
                   "probability threshold of 0.5; no threshold is chosen on "
                   "any held-out fold")

    # --- the feature cube ---------------------------------------------------
    # [state, row, hidden unit], float64, contiguous. Built once and shared
    # read-only with every worker, so no result can depend on which core it ran
    # on, and every variant, control and permutation sees the same numbers.
    print("[data] materialising the analysed activations ...")
    block = np.asarray(inputs.activations[inputs.eb_positions], dtype=np.float64)
    cube = np.ascontiguousarray(block.transpose(1, 0, 2))
    del block
    all_states = list(range(inputs.n_states))
    checks.require(
        "the_feature_cube_matches_the_loader",
        all(np.array_equal(cube[s], inputs.state_matrix(s))
            for s in (0, inputs.n_states // 2, inputs.n_states - 1)),
        f"cube{tuple(cube.shape)} agrees with DevelopmentInputs.state_matrix "
        f"at states 0, {inputs.n_states // 2} and {inputs.n_states - 1}")

    degenerate_states = [s for s in all_states
                         if float(cube[s].std(axis=0).max()) == 0.0]
    states = ([int(s) for s in smoke_config.get("states", [0, 1, 24, 45])]
              if smoke else all_states)
    included_states = [s for s in states if s not in degenerate_states]
    checks.record(
        "degenerate_states_identified", True,
        f"{degenerate_states or 'none'} — constant across all {inputs.n} "
        f"analysed rows. {describe_state(0)} is the built-in degenerate "
        f"control; degenerate states are scored and reported, and excluded "
        f"from every maximum and from every selection")
    checks.require("candidate_states_exclude_the_degenerate_ones",
                   not set(included_states) & set(degenerate_states),
                   f"{len(included_states)} candidate state(s) of {len(states)} "
                   f"scored")

    # --- the split assignment, built once -----------------------------------
    fold_config = config["folds"]
    n_splits = int(fold_config["n_splits"])
    base_seed = int(str(fold_config["seeds"]).split("+")[0].strip())
    n_repeats = int(repeats if repeats is not None
                    else (smoke_config.get("repeats", 2) if smoke
                          else fold_config["repeats"]))
    repeated = build_repeated_stratified_folds(
        y, inputs.row_ids, n_splits=n_splits, repeats=n_repeats,
        base_seed=base_seed)
    repeated.audit(y, checks)
    print(f"[folds] {n_repeats} repeat(s) x {n_splits} stratified folds, "
          f"seeded {base_seed}..{base_seed + n_repeats - 1}")

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "cv_assignments.csv"
    write_repeated_fold_assignments(
        assignments_path, repeated, y,
        extra_columns={"array_row_index": [int(i) for i in inputs.eb_positions],
                       "source_family": inputs.source_family,
                       "topic": inputs.topic,
                       "task_frame": inputs.task_frame})
    reloaded = load_repeated_fold_assignments(assignments_path, inputs.row_ids)
    checks.require(
        "the_written_split_assignment_is_the_one_that_is_used",
        (reloaded.n_repeats == repeated.n_repeats
         and all(np.array_equal(a.fold_of_row, b.fold_of_row)
                 for a, b in zip(reloaded.assignments, repeated.assignments))),
        f"{assignments_path.name} reloads to the same "
        f"{repeated.n_repeats} x {n_splits} assignment that every state, "
        f"variant, control and permutation uses")

    variants = [v for v in config["activation_variants"]["included"]]
    for variant in variants:
        if variant not in ACTIVATION_VARIANTS:
            raise ValidationError(f"unknown activation variant {variant!r}")

    # --- the main sweep -----------------------------------------------------
    print(f"[probe] {len(states)} state(s) x {len(variants)} variant(s) x "
          f"{n_repeats} repeat(s) x {n_splits} folds, jobs={jobs}")
    sweep = _run_tasks(
        [delayed(_state_sweep)(state, cube, y, repeated.assignments, variants, 0)
         for state in states], jobs)

    state_metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    directions_by_state: Dict[int, np.ndarray] = {}
    for state, outcome in zip(states, sweep):
        for record in outcome["metrics"]:
            state_metric_rows.append({
                "state": record["state"],
                "variant": record["variant"],
                "repeat": record["repeat"],
                "n": record["n"], "n_E": record["n_E"], "n_B": record["n_B"],
                "n_features": record["n_features"],
                "roc_auc": record["roc_auc"],
                "mean_fold_roc_auc": record["mean_fold_roc_auc"],
                "pr_auc": record["pr_auc"],
                "balanced_accuracy": record["balanced_accuracy"],
                "accuracy": record["accuracy"],
                "degenerate": record["degenerate"],
                "all_folds_converged": record["all_folds_converged"],
                "max_solver_iterations": record["max_solver_iterations"],
            })
        if outcome["directions"] is not None:
            directions_by_state[state] = outcome["directions"]
        for key, scores in outcome["scores"].items():
            variant, repeat = key.split("|")
            repeat = int(repeat)
            fold_of_row = repeated.assignments[repeat].fold_of_row
            for i, rid in enumerate(inputs.row_ids):
                prediction_rows.append({
                    "row_id": rid,
                    "label": "E" if y[i] == 1 else "B",
                    "y_true": int(y[i]),
                    "feature_set": "activation",
                    "state": state,
                    "variant": variant,
                    "repeat": repeat,
                    "fold": int(fold_of_row[i]),
                    "predicted_probability_E": float(scores[i]),
                })

    checks.require(
        "every_row_is_scored_at_every_state_variant_and_repeat",
        len(prediction_rows) == len(states) * len(variants) * n_repeats * inputs.n,
        f"{len(prediction_rows)} out-of-fold predictions = {len(states)} states "
        f"x {len(variants)} variants x {n_repeats} repeats x {inputs.n} rows")
    checks.require(
        "every_probe_fit_converged",
        all(r["all_folds_converged"] for r in state_metric_rows),
        f"max solver iterations "
        f"{max(r['max_solver_iterations'] for r in state_metric_rows)} of "
        f"{probe_specification()['max_iter']}")

    # --- controls -----------------------------------------------------------
    print("[control] prompt length and the categorical covariates ...")
    control_rows: List[Dict[str, Any]] = []
    control_specs = [
        ("prompt_length",
         prompt_length_features(inputs.prompt_text), make_probe,
         list(PROMPT_LENGTH_FEATURE_NAMES)),
    ]
    for role, column in COVARIATE_COLUMNS.items():
        control_specs.append((f"{role}:{column}",
                              categorical_column(inputs.covariate(role)),
                              make_categorical_probe,
                              sorted(set(inputs.covariate(role)))))
    for name, design, factory, feature_names in control_specs:
        for repeat, folds in repeated:
            result = cross_val_out_of_fold(design, y, folds,
                                           probe_factory=factory,
                                           keep_fold_detail=True)
            record = evaluate_out_of_fold(
                y, result.scores, result.predictions, state=-1,
                degenerate=result.degenerate,
                all_folds_converged=result.all_converged,
                max_solver_iterations=result.max_iterations,
                threshold=DECISION_THRESHOLD).as_dict()
            control_rows.append({
                "control": name,
                "repeat": repeat,
                "n_categories_or_features": len(feature_names),
                "roc_auc": record["roc_auc"],
                "mean_fold_roc_auc": mean_cross_validated_roc_auc(y, result),
                "pr_auc": record["pr_auc"],
                "balanced_accuracy": record["balanced_accuracy"],
                "accuracy": record["accuracy"],
                "encoder": ("OneHotEncoder(handle_unknown='ignore'), fitted on "
                            "training rows" if factory is make_categorical_probe
                            else "none"),
            })
            for i, rid in enumerate(inputs.row_ids):
                prediction_rows.append({
                    "row_id": rid,
                    "label": "E" if y[i] == 1 else "B",
                    "y_true": int(y[i]),
                    "feature_set": "control",
                    "state": UNDEFINED,
                    "variant": name,
                    "repeat": repeat,
                    "fold": int(folds.fold_of_row[i]),
                    "predicted_probability_E": float(result.scores[i]),
                })

    # --- within-source ------------------------------------------------------
    within_config = config["within_source"]
    within_rows, within_summary, within_folds_rows = _within_source(
        inputs, cube, states, included_states, within_config, smoke,
        smoke_config, base_seed, checks, jobs)

    # --- nested selection ---------------------------------------------------
    inner_splits = int(config["nested_selection"]["inner_splits"])
    print(f"[nested] choosing the state inside each training fold "
          f"({inner_splits} inner folds, {len(included_states)} candidates) ...")
    nested = _run_tasks(
        [delayed(_nested_task)(repeat, cube, y, folds, included_states,
                               inner_splits, base_seed)
         for repeat, folds in repeated], jobs)

    nested_rows: List[Dict[str, Any]] = []
    nested_repeat_rows: List[Dict[str, Any]] = []
    for repeat, result in enumerate(nested):
        nested_rows.extend(record.as_dict() for record in result.records)
        metrics = evaluate_out_of_fold(
            y, result.scores, (result.scores >= DECISION_THRESHOLD).astype(int),
            state=-1, threshold=DECISION_THRESHOLD).as_dict()
        nested_repeat_rows.append({
            "repeat": repeat,
            "selected_states": "|".join(str(s) for s in result.selected_states),
            "distinct_selected_states": len(set(result.selected_states)),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
        })
        for i, rid in enumerate(inputs.row_ids):
            prediction_rows.append({
                "row_id": rid,
                "label": "E" if y[i] == 1 else "B",
                "y_true": int(y[i]),
                "feature_set": "nested_selection",
                "state": int(result.selected_state_of_row[i]),
                "variant": "raw",
                "repeat": repeat,
                "fold": int(result.fold_of_row[i]),
                "predicted_probability_E": float(result.scores[i]),
            })

    selected_counts = Counter(s for r in nested for s in r.selected_states)
    nested_roc = float(np.mean([r["roc_auc"] for r in nested_repeat_rows]))
    nested_roc_sd = (float(np.std([r["roc_auc"] for r in nested_repeat_rows],
                                  ddof=1)) if len(nested_repeat_rows) > 1 else 0.0)
    checks.require(
        "the_nested_estimate_never_selected_a_degenerate_state",
        not set(selected_counts) & set(degenerate_states),
        f"selected states: {dict(sorted(selected_counts.items()))}")

    # --- permutation null ---------------------------------------------------
    permutation_config = config["permutation"]
    n_permutations = int(permutations if permutations is not None
                         else (smoke_config.get("permutations", 5) if smoke
                               else permutation_config["n_permutations"]))
    permutation_repeats = min(int(permutation_config.get("repeats_used", 1)),
                              n_repeats)
    permutation_folds = RepeatedFoldAssignment(
        assignments=repeated.assignments[:permutation_repeats],
        n_splits=n_splits, base_seed=base_seed, row_ids=list(inputs.row_ids))
    permutation_seed = int(permutation_config["base_seed"])
    print(f"[null] {n_permutations} permutation(s), each searching "
          f"{len(included_states)} state(s) over {permutation_repeats} "
          f"repeat(s) ...")

    def runner(seeds: Sequence[int]) -> List[Dict[str, Any]]:
        return _run_tasks(
            [delayed(_permutation_task)(seed, cube, y, permutation_folds,
                                        included_states) for seed in seeds],
            jobs)

    null = permutation_null(
        lambda state: np.asarray(cube[state], dtype=np.float64), y,
        permutation_folds, included_states, n_permutations=n_permutations,
        base_seed=permutation_seed, runner=runner)
    checks.require(
        "the_permutation_null_prices_the_state_search",
        all(r.n_states_searched == len(included_states) for r in null["records"]),
        f"every one of {n_permutations} permutation(s) searched all "
        f"{len(included_states)} included states and contributed its maximum")
    checks.record(
        "permutation_p_value_floor", True,
        f"p >= {null['p_value_floor']:.4f} with {n_permutations} permutations "
        f"(the pilot's 20 floored it at 0.0476)")

    # --- descriptive geometry ----------------------------------------------
    print("[geometry] centroid separation and probe directions ...")
    separations = [centroid_separation(cube[state], y, state=state)
                   for state in states]
    separation_rows = separation_table(separations)
    defined = [s for s in separations
               if s.standardised_distance is not None and not s.degenerate]
    best_separation = (defined[safe_argmax(
        [s.standardised_distance for s in defined],
        "standardised centroid distance")] if defined else None)
    checks.require(
        "an_undefined_centroid_statistic_is_excluded_from_the_maximum",
        best_separation is None or not best_separation.degenerate,
        f"{len(separations) - len(defined)} state(s) have an undefined "
        f"standardised distance, written empty and excluded; the maximum is "
        f"state {best_separation.state if best_separation else UNDEFINED}")

    # the state the corrected analysis points at, chosen by the nested procedure
    headline_state = int(selected_counts.most_common(1)[0][0] if selected_counts
                         else included_states[-1])
    direction_rows, direction_matrix_rows = _direction_records(
        directions_by_state, cube, y, states, headline_state)
    projection_rows = _projection(cube[headline_state], y, inputs.row_ids,
                                  repeated.assignments[0], headline_state)

    # --- summary ------------------------------------------------------------
    summary_rows = _summarise_states(state_metric_rows, degenerate_states)
    best_raw = _variant_maximum(summary_rows, "raw")

    elapsed = time.time() - started
    summary: Dict[str, Any] = {
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "elapsed_seconds": round(elapsed, 2),
        "smoke_test": bool(smoke),
        "reportable": not smoke,
        "n": inputs.n, "n_E": inputs.n_positive, "n_B": inputs.n_negative,
        "chance_levels": chance_levels(y),
        "states_scored": len(states),
        "states_searched": len(included_states),
        "degenerate_states": degenerate_states,
        "variants": variants,
        "repeats": n_repeats,
        "n_splits": n_splits,
        "jobs": jobs,
        "selected_maximum": {
            "state": best_raw["state"],
            "mean_roc_auc": best_raw["mean_roc_auc"],
            "sd_over_repeats": best_raw["sd_over_repeats"],
            "meaning": "the best state on the rows it is scored from — the "
                       "pilot's estimand, reported here only for comparison",
        },
        "nested_estimate": {
            "mean_roc_auc_over_repeats": nested_roc,
            "sd_over_repeats": nested_roc_sd,
            "selected_state_counts": dict(sorted(selected_counts.items())),
            "modal_state": headline_state,
            "meaning": "the state is chosen inside each training fold; this is "
                       "the number that is not a maximum over the rows it is "
                       "reported from",
        },
        "permutation": {
            "n_permutations": n_permutations,
            "repeats_used": permutation_repeats,
            "observed_max_mean_roc_auc": null["observed"]["max_mean_roc_auc"],
            "observed_argmax_state": null["observed"]["argmax_state"],
            "p_value": null["p_value"],
            "p_value_floor": null["p_value_floor"],
            "null_mean": null["null_mean"],
            "null_sd": null["null_sd"],
            "null_max": null["null_max"],
            "null_quantile_95": null["null_quantile_95"],
            "statistic": null["statistic"],
        },
        "variant_maxima": {
            variant: _variant_maximum(summary_rows, variant)
            for variant in variants
        },
        "controls": _control_summary(control_rows),
        "within_source": within_summary,
        "centroid_separation_maximum": (
            {"state": best_separation.state,
             "standardised_distance": best_separation.standardised_distance}
            if best_separation else UNDEFINED),
        "headline_state": headline_state,
        "prompt_token_count_available": bool(token_columns),
        "decision_threshold": DECISION_THRESHOLD,
    }

    _write_run(output_dir, config, input_manifest, inputs, repeated, checks,
               summary, smoke,
               tables={
                   "state_metrics.csv": (list(state_metric_rows[0]), state_metric_rows),
                   "state_metrics_summary.csv": (list(summary_rows[0]), summary_rows),
                   "control_metrics.csv": (list(control_rows[0]), control_rows),
                   "oof_predictions.csv": (list(prediction_rows[0]), prediction_rows),
                   "nested_selection_folds.csv": (list(nested_rows[0]), nested_rows),
                   "nested_selection_repeats.csv": (list(nested_repeat_rows[0]),
                                                    nested_repeat_rows),
                   "permutation_null.csv": (
                       list(null["records"][0].as_dict()),
                       [r.as_dict() for r in null["records"]]),
                   "centroid_separation.csv": (list(separation_rows[0]),
                                               separation_rows),
                   "probe_direction_similarity.csv": (
                       list(direction_rows[0]) if direction_rows else ["state"],
                       direction_rows),
                   "probe_directions_headline_state.csv": (
                       list(direction_matrix_rows[0]) if direction_matrix_rows
                       else ["fold"], direction_matrix_rows),
                   "pca_projection_headline_state.csv": (
                       list(projection_rows[0]), projection_rows),
                   "within_source_metrics.csv": (
                       list(within_rows[0]) if within_rows else ["source"],
                       within_rows),
                   "within_source_folds.csv": (
                       list(within_folds_rows[0]) if within_folds_rows
                       else ["source"], within_folds_rows),
               })

    if config["outputs"].get("figures", False):
        _figures(output_dir, summary_rows, null, projection_rows, summary, smoke)

    write_output_hashes(output_dir)
    print(f"[done] {output_dir}  ({elapsed:.1f}s)")
    return {"summary": summary, "checks": checks, "output_dir": output_dir}


# ------------------------------------------------------------------ machinery
def _run_tasks(tasks: Sequence, jobs: int) -> List[Any]:
    """Run independent deterministic tasks, on one core or on many.

    BLAS is pinned to a single thread inside every worker, exactly as it is for
    a serial run, so parallelism changes the wall clock and nothing else.
    """
    if jobs == 1:
        return [function(*args, **kwargs) for function, args, kwargs in tasks]
    with parallel_backend("loky", inner_max_num_threads=1):
        return list(Parallel(n_jobs=jobs)(tasks))


def _summarise_states(rows: Sequence[Dict[str, Any]],
                      degenerate_states: Sequence[int]) -> List[Dict[str, Any]]:
    """Mean and spread over repeats, per (variant, state)."""
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["state"]), []).append(row)
    summary = []
    for (variant, state), group in sorted(grouped.items(),
                                          key=lambda kv: (kv[0][0], kv[0][1])):
        def stats(name: str):
            values = np.asarray([g[name] for g in group], dtype=float)
            return (float(values.mean()),
                    float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    float(values.min()), float(values.max()))
        roc_mean, roc_sd, roc_min, roc_max = stats("roc_auc")
        fold_mean, fold_sd, _, _ = stats("mean_fold_roc_auc")
        pr_mean, pr_sd, _, _ = stats("pr_auc")
        bal_mean, bal_sd, _, _ = stats("balanced_accuracy")
        summary.append({
            "variant": variant,
            "state": state,
            "repeats": len(group),
            "mean_roc_auc": roc_mean, "sd_roc_auc": roc_sd,
            "min_roc_auc": roc_min, "max_roc_auc": roc_max,
            "mean_of_fold_roc_auc": fold_mean, "sd_of_fold_roc_auc": fold_sd,
            "mean_pr_auc": pr_mean, "sd_pr_auc": pr_sd,
            "mean_balanced_accuracy": bal_mean, "sd_balanced_accuracy": bal_sd,
            "degenerate": state in degenerate_states,
        })
    return summary


def _variant_maximum(summary_rows: Sequence[Dict[str, Any]],
                     variant: str) -> Dict[str, Any]:
    candidates = [r for r in summary_rows
                  if r["variant"] == variant and not r["degenerate"]]
    if not candidates:
        return {"state": UNDEFINED, "mean_roc_auc": UNDEFINED}
    best = candidates[safe_argmax([r["mean_roc_auc"] for r in candidates],
                                  f"{variant} over states")]
    return {"state": best["state"], "mean_roc_auc": best["mean_roc_auc"],
            "sd_over_repeats": best["sd_roc_auc"],
            "description": variant_description(variant)}


def _control_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(row["control"], []).append(row["roc_auc"])
    return {name: {"mean_roc_auc": float(np.mean(values)),
                   "sd_roc_auc": (float(np.std(values, ddof=1))
                                  if len(values) > 1 else 0.0),
                   "repeats": len(values)}
            for name, values in sorted(grouped.items())}


def _within_source(inputs, cube, states, included_states, within_config,
                   smoke, smoke_config, base_seed, checks, jobs):
    """One analysis per source, with the folds capped by the rarer class.

    A source with fewer than two examples of either class is skipped outright:
    it cannot be cross-validated at all, and a number produced from it would be
    an artefact of which single example landed where. A source that survives
    that test can still be small, so the fold count is capped at the smaller
    class count, the exact E and B counts travel with every row of the output,
    and anything below the configured floor is flagged unstable.
    """
    column = within_config["column"]
    values = list(inputs.covariate("source"))
    repeats = int(smoke_config.get("within_source_repeats", 2) if smoke
                  else within_config["repeats"])
    max_splits = int(within_config["max_splits"])
    minimum = int(within_config["min_class_examples"])
    unstable_below = int(within_config["unstable_if_smallest_class_below"])

    rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"column": column, "sources": {}, "repeats": repeats}

    for source in sorted(set(values)):
        positions = np.array([i for i, v in enumerate(values) if v == source],
                             dtype=int)
        y_source = inputs.y[positions]
        n_e = int((y_source == 1).sum())
        n_b = int((y_source == 0).sum())
        smallest = min(n_e, n_b)
        entry: Dict[str, Any] = {"n": int(positions.size), "n_E": n_e, "n_B": n_b}

        if smallest < minimum:
            entry.update({"analysed": False,
                          "reason": f"the smaller class has {smallest} "
                                    f"example(s); at least {minimum} are needed "
                                    f"to cross-validate",
                          "n_splits": UNDEFINED, "unstable": True})
            summary["sources"][source] = entry
            checks.record(f"within_source_{source}_skipped", True,
                          f"E={n_e} B={n_b}: skipped, {entry['reason']}")
            print(f"[source] {source}: E={n_e} B={n_b} — skipped, "
                  f"{entry['reason']}")
            continue

        n_splits = int(min(max_splits, smallest))
        unstable = smallest < unstable_below
        entry.update({"analysed": True, "n_splits": n_splits,
                      "unstable": unstable,
                      "unstable_reason": (
                          f"the smaller class has {smallest} example(s), below "
                          f"{unstable_below}; estimates from this source are "
                          f"noisy and must not be read as a per-source finding"
                          if unstable else "")})
        print(f"[source] {source}: E={n_e} B={n_b}, {n_splits} fold(s) "
              f"capped by the smaller class"
              f"{'  [UNSTABLE]' if unstable else ''}")

        row_ids = [inputs.row_ids[i] for i in positions]
        assignments = []
        for repeat in range(repeats):
            folds, used = folds_for_subset(y_source, row_ids, n_splits,
                                           seed=base_seed + repeat)
            assignments.append(folds)
            for i, rid in enumerate(row_ids):
                fold_rows.append({"source": source, "repeat": repeat,
                                  "row_id": rid, "y_true": int(y_source[i]),
                                  "n_E": n_e, "n_B": n_b,
                                  "fold": int(folds.fold_of_row[i]),
                                  "n_splits": used, "unstable": unstable})

        source_cube = np.ascontiguousarray(cube[:, positions, :])
        outcomes = _run_tasks(
            [delayed(_state_sweep)(state, source_cube, y_source, assignments,
                                   ["raw"], None) for state in states], jobs)
        per_state: Dict[int, List[float]] = {}
        for state, outcome in zip(states, outcomes):
            for record in outcome["metrics"]:
                per_state.setdefault(state, []).append(record["roc_auc"])
                rows.append({
                    "source": source, "state": state,
                    "repeat": record["repeat"],
                    "n": int(positions.size), "n_E": n_e, "n_B": n_b,
                    "n_splits": n_splits, "unstable": unstable,
                    "roc_auc": record["roc_auc"],
                    "pr_auc": record["pr_auc"],
                    "balanced_accuracy": record["balanced_accuracy"],
                    "degenerate": record["degenerate"],
                })
        del source_cube

        scoreable = {s: float(np.mean(v)) for s, v in per_state.items()
                     if s in included_states}
        if scoreable:
            best = max(scoreable, key=lambda s: scoreable[s])
            entry["best_state"] = int(best)
            entry["best_mean_roc_auc"] = scoreable[best]
        summary["sources"][source] = entry
        checks.record(
            f"within_source_{source}_analysed", True,
            f"E={n_e} B={n_b}, {n_splits} fold(s) capped by the smaller class, "
            f"{repeats} repeat(s)"
            f"{'; FLAGGED UNSTABLE' if unstable else ''}")

    return rows, summary, fold_rows


def _direction_records(directions_by_state: Dict[int, np.ndarray],
                       cube: np.ndarray, y: np.ndarray, states: Sequence[int],
                       headline_state: int):
    """Fold-specific probe directions, reconstructed into raw activation units.

    Each fold keeps its own direction: the reconstruction divides by that
    fold's ``StandardScaler`` scale, and coefficients fitted against different
    scales are not on the same footing until they have been divided.
    """
    rows: List[Dict[str, Any]] = []
    for state in states:
        directions = directions_by_state.get(state)
        if directions is None:
            rows.append({"state": state,
                         "n_fold_directions": 0,
                         "mean_pairwise_cosine_between_folds": UNDEFINED,
                         "min_pairwise_cosine_between_folds": UNDEFINED,
                         "cosine_with_centroid_difference": UNDEFINED,
                         "mean_direction_norm": UNDEFINED,
                         "note": "degenerate state: no direction is defined; "
                                 "excluded from every maximum"})
            continue
        similarity = cosine_similarity_matrix(directions)
        upper = similarity[np.triu_indices(similarity.shape[0], k=1)]
        centroid = (cube[state][y == 1].mean(axis=0)
                    - cube[state][y == 0].mean(axis=0))
        mean_direction = directions.mean(axis=0)
        denominator = (np.linalg.norm(mean_direction) * np.linalg.norm(centroid))
        rows.append({
            "state": state,
            "n_fold_directions": int(directions.shape[0]),
            "mean_pairwise_cosine_between_folds": float(upper.mean()),
            "min_pairwise_cosine_between_folds": float(upper.min()),
            "cosine_with_centroid_difference": (
                float(np.dot(mean_direction, centroid) / denominator)
                if denominator > 0 else UNDEFINED),
            "mean_direction_norm": float(np.linalg.norm(directions, axis=1).mean()),
            "note": "raw-space weight = logistic coefficient / training-fold "
                    "StandardScaler scale, per fold; the cosine against the "
                    "centroid difference uses the mean of the reconstructed "
                    "fold directions, never a mean of coefficients",
        })

    # The fold-specific directions themselves, at the state the run points at.
    # One row per fold, never a fold-averaged coefficient vector.
    matrix_rows: List[Dict[str, Any]] = []
    for fold, direction in enumerate(directions_by_state.get(headline_state,
                                                             np.empty((0, 0)))):
        record: Dict[str, Any] = {"state": headline_state, "fold": fold,
                                  "repeat": 0}
        record.update({f"w{j}": float(v) for j, v in enumerate(direction)})
        matrix_rows.append(record)
    return rows, matrix_rows


def _projection(state_matrix: np.ndarray, y: np.ndarray, row_ids: Sequence[str],
                folds, state: int) -> List[Dict[str, Any]]:
    """Two principal components per row, fitted on training rows only."""
    rows: List[Dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(folds.splits()):
        scaler, projector = fit_projection(state_matrix[train_index],
                                           n_components=2)
        held_out = project(state_matrix[test_index], scaler, projector)
        for position, coordinates in zip(test_index, held_out):
            rows.append({
                "row_id": row_ids[int(position)],
                "state": state,
                "fold": fold,
                "label": "E" if y[int(position)] == 1 else "B",
                "pc1": float(coordinates[0]),
                "pc2": float(coordinates[1]),
                "basis": "standardisation and PCA fitted on this fold's "
                         "training rows; the row shown was held out",
            })
    return rows


# --------------------------------------------------------------------- output
def _write_run(output_dir: Path, config: Dict[str, Any],
               input_manifest: Dict[str, Any], inputs,
               repeated: RepeatedFoldAssignment, checks: Checklist,
               summary: Dict[str, Any], smoke: bool,
               tables: Dict[str, Any]) -> None:
    write_json(output_dir / "config.json", {
        "config": config,
        "probe": probe_specification(),
        "activation_variants": {v: variant_description(v)
                                for v in summary["variants"]},
        "controls": control_catalogue(),
        "covariate_columns": COVARIATE_COLUMNS,
        "prompt_length_note": PROMPT_LENGTH_NOTE,
        "fold_source": repeated.assignments[0].source,
        "repeats": repeated.n_repeats,
        "smoke_test": smoke,
        "reportable": not smoke,
    })
    write_json(output_dir / "environment.json", environment_record(PACKAGE_VERSION))
    write_json(output_dir / "input_hashes.json", {
        "dataset": {"path": str(inputs.dataset_path),
                    "sha256": inputs.dataset_sha256,
                    "expected": input_manifest["dataset"]["sha256"],
                    "matches": inputs.dataset_sha256
                               == input_manifest["dataset"]["sha256"]},
        "extraction_manifest": {
            "path": str(inputs.extraction_manifest_path),
            "sha256": inputs.extraction_manifest_sha256,
            "expected": input_manifest["extraction_manifest"]["sha256"],
            "matches": inputs.extraction_manifest_sha256
                       == input_manifest["extraction_manifest"]["sha256"]},
        "activation": {"path": str(inputs.activation_path),
                       "sha256": inputs.activation_sha256,
                       "expected": input_manifest["activation"]["sha256"],
                       "matches": inputs.activation_sha256
                                  == input_manifest["activation"]["sha256"],
                       "shape": list(inputs.activations.shape),
                       "dtype": inputs.activations.dtype.name,
                       "copied": False},
        "inputs_modified": False,
    })

    for name, (fieldnames, rows) in tables.items():
        write_csv(output_dir / name, fieldnames, rows)

    write_json(output_dir / "validation.json",
               {"stage": "residual_geometry_corrected",
                "smoke_test": smoke,
                "reportable": not smoke,
                **checks.as_dict(), "summary": summary})
    _write_validation_md(output_dir / "VALIDATION.md", checks, summary, smoke)


def _write_validation_md(path: Path, checks: Checklist, summary: Dict[str, Any],
                         smoke: bool) -> None:
    lines: List[str] = []
    add = lines.append

    add("# Residual geometry, corrected — validation")
    add("")
    if smoke:
        add("> **SMOKE TEST — NOT REPORTABLE.** This run exercises the code "
            "paths on a small subset of states, repeats and permutations. Its "
            "numbers are not results and must not be quoted, compared or "
            "carried into any document.")
        add("")
    add(f"Completed {summary['completed_utc']} in {summary['elapsed_seconds']} s. "
        f"Offline: no language model, no generation, no network call, no "
        f"held-out benchmark material.")
    add("")
    add("## What this corrects")
    add("")
    add("| | pilot | here |")
    add("| --- | --- | --- |")
    add("| state selection | maximum over 49 states, on the rows the score is "
        "reported from | chosen by an inner cross-validation inside each outer "
        "training fold |")
    add("| permutation null | computed at the already-selected state, 20 "
        "permutations, p floored at 0.048 | every permutation repeats the whole "
        f"state search, {summary['permutation']['n_permutations']} of them, p "
        f"floored at {summary['permutation']['p_value_floor']:.4f} |")
    add("| magnitude | unaccounted for; a norm-only control reached 0.871 | "
        "raw, unit-L2, norm-residualised and norm-only probes run side by side |")
    add("| splits | one five-fold assignment | "
        f"{summary['repeats']} repeats x {summary['n_splits']} folds, written "
        "out once and reused everywhere |")
    add("")
    add("## The two numbers, and why they differ")
    add("")
    add("| estimate | ROC-AUC | state |")
    add("| --- | --- | --- |")
    add(f"| selected maximum (the pilot's estimand) | "
        f"{summary['selected_maximum']['mean_roc_auc']:.4f} "
        f"± {summary['selected_maximum']['sd_over_repeats']:.4f} | "
        f"{summary['selected_maximum']['state']} |")
    add(f"| nested selection (not a maximum over its own rows) | "
        f"{summary['nested_estimate']['mean_roc_auc_over_repeats']:.4f} "
        f"± {summary['nested_estimate']['sd_over_repeats']:.4f} | chosen per "
        f"fold; modal {summary['nested_estimate']['modal_state']} |")
    add("")
    add("## Magnitude against direction")
    add("")
    add("| activation variant | best state | mean ROC-AUC |")
    add("| --- | --- | --- |")
    for variant, best in summary["variant_maxima"].items():
        value = (f"{best['mean_roc_auc']:.4f}"
                 if best["mean_roc_auc"] != UNDEFINED else "—")
        add(f"| {variant} | {best['state']} | {value} |")
    add("")
    add("## Controls")
    add("")
    add("| control | mean ROC-AUC | repeats |")
    add("| --- | --- | --- |")
    for name, values in summary["controls"].items():
        add(f"| {name} | {values['mean_roc_auc']:.4f} | {values['repeats']} |")
    add("")
    add(f"Prompt length is {PROMPT_LENGTH_NOTE}")
    add("")
    add("## Permutation null")
    add("")
    add(f"Observed maximum {summary['permutation']['observed_max_mean_roc_auc']:.4f} "
        f"at state {summary['permutation']['observed_argmax_state']}; null over "
        f"{summary['permutation']['n_permutations']} permutations has mean "
        f"{summary['permutation']['null_mean']:.4f}, sd "
        f"{summary['permutation']['null_sd']:.4f}, 95th percentile "
        f"{summary['permutation']['null_quantile_95']:.4f}, maximum "
        f"{summary['permutation']['null_max']:.4f}. "
        f"**p = {summary['permutation']['p_value']:.4f}** "
        f"(floor {summary['permutation']['p_value_floor']:.4f}).")
    add("")
    add(f"Statistic: {summary['permutation']['statistic']}.")
    add("")
    add("## Within source")
    add("")
    add("| source | n | E | B | folds | analysed | unstable |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for source, entry in summary["within_source"]["sources"].items():
        add(f"| {source} | {entry['n']} | {entry['n_E']} | {entry['n_B']} | "
            f"{entry['n_splits'] or '—'} | "
            f"{'yes' if entry.get('analysed') else 'no'} | "
            f"{'yes' if entry.get('unstable') else 'no'} |")
    add("")
    for source, entry in summary["within_source"]["sources"].items():
        if entry.get("unstable") and entry.get("unstable_reason"):
            add(f"- **{source}**: {entry['unstable_reason']}")
        elif not entry.get("analysed"):
            add(f"- **{source}**: skipped — {entry.get('reason', '')}")
    add("")
    add("## Degenerate states")
    add("")
    add(f"State(s) {summary['degenerate_states']} are constant across every "
        f"analysed row. State 0 is the embedding output, not layer 0: the "
        f"rendered chat prompt ends with the same token for every prompt, so it "
        f"is one identical vector and a probe there is pinned to chance by "
        f"construction. Such states are scored and reported, their undefined "
        f"statistics are written empty rather than as NaN, and they are "
        f"excluded from every maximum and from nested selection "
        f"({summary['states_searched']} of {summary['states_scored']} states "
        f"were searched).")
    add("")
    add("## Checks")
    add("")
    add("| check | status | detail |")
    add("| --- | --- | --- |")
    for entry in checks.entries:
        add(f"| {entry['check']} | {entry['status']} | {entry['detail']} |")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figures(output_dir: Path, summary_rows, null, projection_rows, summary,
             smoke: bool) -> None:
    """Four figures: the state profile, the null, the controls, the projection."""
    from routegate import plotting

    plt = plotting.configure()
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    banner = ("SMOKE TEST — not reportable. " if smoke else "")

    # 1. mean ROC-AUC by state, one line per activation variant
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    palette = [plotting.SERIES_E, plotting.SERIES_B, plotting.DIVERGING_HIGH,
               plotting.INK_MUTED]
    for index, variant in enumerate(summary["variants"]):
        rows = sorted((r for r in summary_rows if r["variant"] == variant),
                      key=lambda r: r["state"])
        states = [r["state"] for r in rows]
        means = np.array([r["mean_roc_auc"] for r in rows])
        sds = np.array([r["sd_roc_auc"] for r in rows])
        colour = palette[index % len(palette)]
        ax.plot(states, means, color=colour, linewidth=1.6, label=variant)
        ax.fill_between(states, means - sds, means + sds, color=colour,
                        alpha=0.15, linewidth=0)
    ax.axhline(0.5, color=plotting.AXIS, linewidth=0.9, linestyle="--")
    ax.set_xlabel(plotting.STATE_AXIS_LABEL)
    ax.set_ylabel("out-of-fold ROC-AUC")
    ax.legend(loc="lower right")
    plotting.tidy(ax)
    plotting.title_block(
        ax, "Where over-refusal becomes linearly readable",
        f"{banner}Mean over {summary['repeats']} repeated stratified "
        f"five-fold splits; the band is ±1 sd over repeats. Degenerate states "
        f"are drawn but excluded from every maximum.")
    fig.savefig(figures / "state_profile_by_variant.png", bbox_inches="tight")
    plt.close(fig)

    # 2. the permutation null against the observed maximum
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(null["null_values"], bins=24, color=plotting.SERIES_B, alpha=0.85)
    ax.axvline(null["observed"]["max_mean_roc_auc"], color=plotting.SERIES_E,
               linewidth=2.0)
    ax.set_xlabel("maximum mean cross-validated ROC-AUC over the state search")
    ax.set_ylabel("permutations")
    plotting.tidy(ax)
    plotting.title_block(
        ax, "A null that pays for the search",
        f"{banner}Each of {summary['permutation']['n_permutations']} label "
        f"permutations repeats the whole state search and contributes its "
        f"maximum. Observed "
        f"{null['observed']['max_mean_roc_auc']:.4f} (orange), "
        f"p = {null['p_value']:.4f}.")
    fig.savefig(figures / "permutation_null.png", bbox_inches="tight")
    plt.close(fig)

    # 3. controls and variants side by side
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    labels, values = [], []
    for variant, best in summary["variant_maxima"].items():
        if best["mean_roc_auc"] != UNDEFINED:
            labels.append(f"{variant}\n(state {best['state']})")
            values.append(best["mean_roc_auc"])
    for name, entry in summary["controls"].items():
        labels.append(name.replace(":", "\n"))
        values.append(entry["mean_roc_auc"])
    colours = ([plotting.SERIES_E] * (len(summary["variant_maxima"]))
               + [plotting.SERIES_B] * len(summary["controls"]))
    ax.bar(range(len(values)), values, color=colours[:len(values)])
    ax.axhline(0.5, color=plotting.AXIS, linewidth=0.9, linestyle="--")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("mean ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    plotting.tidy(ax)
    plotting.title_block(
        ax, "What the number is made of",
        f"{banner}Activation variants at their best state (orange) beside "
        f"controls that use no activation at all (blue). Chance is 0.5.")
    fig.savefig(figures / "variants_and_controls.png", bbox_inches="tight")
    plt.close(fig)

    # 4. held-out projection at the headline state
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for label in ("B", "E"):
        points = [(r["pc1"], r["pc2"]) for r in projection_rows
                  if r["label"] == label]
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=13,
                   color=plotting.CLASS_COLOURS[label], alpha=0.75,
                   label=plotting.CLASS_LABELS[label], linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    plotting.tidy(ax)
    plotting.title_block(
        ax, f"Held-out rows at state {summary['headline_state']}",
        f"{banner}Standardisation and PCA fitted on each fold's training rows; "
        f"every point shown was held out of the basis that placed it.")
    fig.savefig(figures / "projection_headline_state.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes; results are identical at any "
                             "value, BLAS stays pinned to one thread")
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="exercise every path on a small subset; writes to "
                             "a directory named 'smoke' and is not reportable")
    parser.add_argument(APPROVAL_FLAG, action="store_true", dest="approved",
                        help="confirm the corrected protocol is approved to run")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "corrected analysis config")

    ok, reason = prerequisites_met()
    print(f"[prerequisite] {'OK' if ok else 'NOT MET'}: {reason}")

    if not args.smoke and not args.approved:
        print()
        print("The corrected residual-geometry analysis has NOT been run.")
        print()
        print("This stage changes what the headline number means: a nested")
        print("selection estimate is not comparable to the pilot's selected")
        print("maximum, and the magnitude controls may reassign most of the")
        print("separability from geometry to vector norm. Review")
        print(f"  {args.config.relative_to(REPO_ROOT)}")
        print("and rerun with:")
        print(f"  {APPROVAL_FLAG}")
        print()
        print("To exercise the code paths without producing a reportable")
        print("number, use --smoke.")
        return 0

    if not ok:
        print(f"\nREFUSED: {reason}", file=sys.stderr)
        return 2

    if args.smoke:
        print("[smoke] NOT REPORTABLE: a subset of states, repeats and "
              "permutations, written to a directory named 'smoke'")

    outcome = analyse(config_path=args.config, output_dir=args.output_dir,
                      smoke=args.smoke, jobs=args.jobs, paths_path=args.paths,
                      permutations=args.permutations, repeats=args.repeats)
    return 0 if outcome["checks"].all_passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
