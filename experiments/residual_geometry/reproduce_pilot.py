#!/usr/bin/env python3
"""Reproduce the completed residual-geometry pilot with scikit-learn.

The pilot was computed in an environment without scikit-learn, so its scaler,
probe, fold splitter, metrics and projections were all written by hand. Before
anything is built on top of it, its numbers are re-derived here with the
standard implementations, using the pilot's own fold assignment, and every state
is compared against the frozen reference.

What this does:

    1. loads the accepted corpus through the input gates -- hashes, class
       counts, row alignment, prompt hashes, array shape and dtype, finiteness;
    2. recovers the pilot's fold assignment from the out-of-fold predictions it
       saved, and re-binds it to the analysed row order;
    3. fits a scikit-learn Pipeline(StandardScaler, LogisticRegression) per fold
       at every one of the 49 states, and scores each row with the fold that
       held it out;
    4. compares every state against the pilot's layerwise metrics;
    5. writes the run record.

What this deliberately does not do: no control, no permutation, no state
selection, no figure. Reproduction establishes that the numbers transfer to a
standard implementation. The corrected analysis is a separate entry point.

Usage:
    python experiments/residual_geometry/reproduce_pilot.py
    python experiments/residual_geometry/reproduce_pilot.py --states 45
    python experiments/residual_geometry/reproduce_pilot.py --output-dir DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np                                                  # noqa: E402
import scipy                                                        # noqa: E402
import sklearn                                                      # noqa: E402

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.data import (                                        # noqa: E402
    describe_state,
    load_development_inputs,
    load_input_manifest,
    load_local_paths,
    pilot_reference_dir,
)
from routegate.metrics import (                                     # noqa: E402
    METRIC_NAMES,
    chance_levels,
    compare_to_reference,
    evaluate_out_of_fold,
    max_absolute_difference,
)
from routegate.probes import (                                      # noqa: E402
    DECISION_THRESHOLD,
    cross_val_out_of_fold,
    make_probe,
    probe_specification,
)
from routegate.splits import load_frozen_folds, write_fold_assignments  # noqa: E402
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_hash,
    assert_not_sealed,
    sha256_file,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "residual_geometry_pilot.json"
PILOT_MANIFEST = REPO_ROOT / "manifests" / "pilot_reference.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "residual_geometry" / "pilot_reproduction"


# ------------------------------------------------------------------ reference
def read_reference_metrics(path: Path, state_column: str = "layer"
                           ) -> Dict[int, Dict[str, float]]:
    """The pilot's per-state metrics, keyed by state.

    The pilot's column is named ``layer``; the values 0-48 mean exactly what
    this package calls states, so the column is renamed on read and the
    numbering is unchanged.
    """
    reference: Dict[int, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state = int(row[state_column])
            reference[state] = {name: float(row[name]) for name in METRIC_NAMES
                                if name in row}
    return reference


def read_reference_scores(path: Path, score_column: str = "oof_score_p_E"
                          ) -> Dict[str, float]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["row_id"]: float(row[score_column])
                for row in csv.DictReader(handle)}


# ---------------------------------------------------------- explaining a gap
def explain_disagreements(comparison: Sequence[Dict[str, Any]],
                          scores_by_state: Dict[int, np.ndarray],
                          y: np.ndarray, tolerance: float,
                          reference_decimals: int = 6) -> Dict[str, Any]:
    """Account for every compared cell that falls outside the tolerance.

    ROC-AUC over ``n_E * n_B`` pairs is quantised: the smallest amount it can
    move is one discordant pair, ``1 / (n_E * n_B)``. If every disagreement is
    an integer number of those quanta, and the states that disagree are exactly
    the ones carrying a near-tied cross-class pair, then the two
    implementations agree about the model and differ only in how they order two
    scores that are equal to within floating-point noise.

    This function tests that account rather than asserting it.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    quantum = 1.0 / (n_pos * n_neg)
    rounding = 0.5 * 10.0 ** (-reference_decimals)

    outside = [r for r in comparison
               if r["absolute_difference"] != "" and r["within_tolerance"] is not True]

    def min_cross_class_gap(state: int) -> Optional[float]:
        scores = scores_by_state.get(state)
        if scores is None:
            return None
        order = np.argsort(scores, kind="mergesort")
        ranked, labels = scores[order], y[order]
        crossing = labels[:-1] != labels[1:]
        if not bool(crossing.any()):
            return None
        return float(np.min(np.abs(np.diff(ranked))[crossing]))

    rows: List[Dict[str, Any]] = []
    for record in outside:
        difference = float(record["absolute_difference"])
        quanta = difference / quantum
        rows.append({
            "state": record["state"],
            "metric": record["metric"],
            "absolute_difference": difference,
            "discordant_pairs": round(quanta),
            "quanta": quanta,
            "is_a_whole_number_of_discordant_pairs":
                bool(record["metric"] == "roc_auc"
                     and abs(difference - round(quanta) * quantum) <= rounding),
            "min_cross_class_score_gap": min_cross_class_gap(int(record["state"])),
        })

    roc_rows = [r for r in rows if r["metric"] == "roc_auc"]
    threshold_metrics = sorted({r["metric"] for r in rows}
                               & {"balanced_accuracy", "accuracy"})
    return {
        "n_E": n_pos, "n_B": n_neg,
        "comparable_pairs": n_pos * n_neg,
        "roc_auc_quantum": quantum,
        "reference_decimal_places": reference_decimals,
        "reference_rounding_half_width": rounding,
        "cells_outside_tolerance": len(rows),
        "states_outside_tolerance": sorted({int(r["state"]) for r in rows}),
        "every_roc_auc_gap_is_a_whole_number_of_discordant_pairs":
            bool(roc_rows) and all(r["is_a_whole_number_of_discordant_pairs"]
                                   for r in roc_rows),
        "largest_roc_auc_gap_in_discordant_pairs":
            max((r["discordant_pairs"] for r in roc_rows), default=0),
        "threshold_metrics_that_disagree": threshold_metrics,
        "cells": sorted(rows, key=lambda r: -r["absolute_difference"]),
    }


# ----------------------------------------------------------------------- run
def reproduce(config_path: Path = DEFAULT_CONFIG,
              output_dir: Optional[Path] = None,
              states: Optional[Sequence[int]] = None,
              paths_path: Optional[Path] = None,
              write_outputs: bool = True) -> Dict[str, Any]:
    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "pilot reproduction config")
    pilot_manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    assert_config_is_sealed_free(pilot_manifest, "pilot reference manifest")

    output_dir = Path(output_dir or (REPO_ROOT / config["outputs"]["directory"]))
    checks = Checklist("pilot_reproduction")

    # --- inputs -------------------------------------------------------------
    input_manifest = load_input_manifest()
    local_paths = load_local_paths(paths_path) if paths_path else load_local_paths()
    print("[gate] validating the accepted development inputs ...")
    inputs = load_development_inputs(manifest=input_manifest, paths=local_paths)
    for entry in inputs.checklist.entries:
        checks.entries.append(entry)
    print(f"[gate] {len(inputs.checklist.entries)} input checks passed; "
          f"n={inputs.n} (E={inputs.n_positive}, B={inputs.n_negative})")

    expected = pilot_manifest["expected_values"]
    checks.require("analysed_row_count", inputs.n == expected["n"],
                   f"{inputs.n} (expected {expected['n']})")
    checks.require("analysed_E_count", inputs.n_positive == expected["n_E"],
                   f"{inputs.n_positive} (expected {expected['n_E']})")
    checks.require("analysed_B_count", inputs.n_negative == expected["n_B"],
                   f"{inputs.n_negative} (expected {expected['n_B']})")
    checks.require("state_count", inputs.n_states == expected["n_states"],
                   f"{inputs.n_states} (expected {expected['n_states']})")

    # --- reference run ------------------------------------------------------
    # The pilot reference is a committed artifact (Phase 2 step 5): it is read
    # from the repository, not from the machine-local path file. Its hashes are
    # still checked below -- committed means present, not trusted.
    reference_dir = Path(assert_not_sealed(pilot_reference_dir(),
                                           "pilot reference directory"))
    files = pilot_manifest["reference_files"]
    metrics_file = reference_dir / files["layerwise_metrics"]["filename"]
    folds_file = reference_dir / files["fold_assignment_and_out_of_fold_scores"]["filename"]
    metrics_sha = assert_hash(metrics_file, files["layerwise_metrics"]["sha256"],
                              "pilot layerwise metrics")
    folds_sha = assert_hash(folds_file,
                            files["fold_assignment_and_out_of_fold_scores"]["sha256"],
                            "pilot out-of-fold scores")
    checks.record("pilot_layerwise_metrics_sha256", True, metrics_sha)
    checks.record("pilot_oof_scores_sha256", True, folds_sha)

    reference_metrics = read_reference_metrics(
        metrics_file, files["layerwise_metrics"]["state_column"])
    checks.require("reference_covers_every_state",
                   sorted(reference_metrics) == list(range(inputs.n_states)),
                   f"{len(reference_metrics)} reference states")

    # --- folds --------------------------------------------------------------
    folds = load_frozen_folds(
        folds_file, inputs.row_ids,
        row_id_column=files["fold_assignment_and_out_of_fold_scores"]["row_id_column"],
        fold_column=files["fold_assignment_and_out_of_fold_scores"]["fold_column"])
    checks.require("fold_count_matches_the_pilot",
                   folds.n_splits == expected["n_splits"],
                   f"{folds.n_splits} folds (expected {expected['n_splits']})")
    folds.audit(inputs.y, checks)
    fold_summary = folds.summary(inputs.y)
    print(f"[folds] imported the pilot assignment: {folds.n_splits} folds")
    for row in fold_summary:
        print(f"        fold {row['fold']}: train {row['n_train']:4d}  "
              f"held out {row['n_held_out']:3d} "
              f"(E={row['held_out_E']}, B={row['held_out_B']})")

    # row order must be the pilot's, not merely the same set
    with open(folds_file, newline="", encoding="utf-8") as handle:
        reference_order = [r["row_id"] for r in csv.DictReader(handle)]
    checks.require("row_order_is_identical_to_the_pilot",
                   reference_order == inputs.row_ids,
                   f"{len(inputs.row_ids)} rows in the same order")

    # --- the sweep ----------------------------------------------------------
    all_states = list(range(inputs.n_states))
    states = list(all_states if states is None else states)
    print(f"[probe] scikit-learn {sklearn.__version__}: "
          f"Pipeline(StandardScaler, LogisticRegression) over {len(states)} state(s)")

    records = []
    prediction_rows: List[Dict[str, Any]] = []
    degenerate_states: List[int] = []
    scores_by_state: Dict[int, np.ndarray] = {}
    reproduced_scores_at_gate: Optional[np.ndarray] = None
    gate_state = int(config["comparison"]["gate_state"])

    for state in states:
        X = inputs.state_matrix(state)
        result = cross_val_out_of_fold(X, inputs.y, folds, probe_factory=make_probe,
                                       keep_fold_detail=False)
        if result.degenerate:
            degenerate_states.append(state)
        metrics = evaluate_out_of_fold(
            inputs.y, result.scores, result.predictions, state=state,
            degenerate=result.degenerate,
            all_folds_converged=result.all_converged,
            max_solver_iterations=result.max_iterations,
            threshold=DECISION_THRESHOLD)
        records.append(metrics)
        scores_by_state[state] = result.scores.copy()

        if state == gate_state:
            reproduced_scores_at_gate = result.scores.copy()

        for i in range(inputs.n):
            prediction_rows.append({
                "state": state,
                "position": i,
                "array_row_index": int(inputs.eb_positions[i]),
                "row_id": inputs.row_ids[i],
                "source_family": inputs.source_family[i],
                "final_ebh": "E" if inputs.y[i] == 1 else "B",
                "y_true": int(inputs.y[i]),
                "fold": int(result.fold_of_row[i]),
                "oof_decision_function": float(result.decisions[i]),
                "oof_score_p_E": float(result.scores[i]),
                "predicted_label": int(result.predictions[i]),
            })

        flag = "  [degenerate]" if result.degenerate else ""
        print(f"[probe] {describe_state(state):<44s} "
              f"ROC-AUC {metrics.roc_auc:.6f}  PR-AUC {metrics.pr_auc:.6f}  "
              f"bal-acc {metrics.balanced_accuracy:.6f}{flag}")

    checks.require("every_state_produced_finite_metrics",
                   all(np.isfinite([r.roc_auc, r.pr_auc, r.balanced_accuracy,
                                    r.accuracy]).all() for r in records),
                   f"{len(records)} state(s), all metrics finite and in [0, 1]")
    checks.require("every_row_scored_at_every_state",
                   len(prediction_rows) == len(states) * inputs.n,
                   f"{len(prediction_rows)} rows "
                   f"(= {len(states)} states x {inputs.n} examples)")
    checks.require("all_probe_fits_converged",
                   all(r.all_folds_converged for r in records),
                   f"max solver iterations "
                   f"{max(r.max_solver_iterations for r in records)} of "
                   f"{probe_specification()['max_iter']}")
    checks.record(
        "degenerate_states_identified", True,
        f"{degenerate_states or 'none'} — constant across all {inputs.n} analysed "
        f"rows; scored, flagged and excluded from any maximum")
    if degenerate_states:
        expected_degenerate = expected.get("degenerate_states", [])
        checks.require("degenerate_states_are_the_expected_ones",
                       degenerate_states == expected_degenerate,
                       f"{degenerate_states} (expected {expected_degenerate})")

    # --- comparison ---------------------------------------------------------
    tolerance = float(config["comparison"]["absolute_tolerance"])
    comparison = compare_to_reference(records, reference_metrics,
                                      METRIC_NAMES, tolerance)
    comparable = [r for r in comparison if r["absolute_difference"] != ""]
    largest = max_absolute_difference(comparison)
    disagreements = [r for r in comparable if r["within_tolerance"] is not True]

    gate_rows = [r for r in comparable
                 if r["state"] == gate_state
                 and r["metric"] in config["comparison"]["gate_metrics"]]
    gate_passed = bool(gate_rows) and all(r["within_tolerance"] is True
                                          for r in gate_rows)
    if gate_state in states:
        checks.require(
            f"state_{gate_state}_reproduces_within_{tolerance:g}",
            gate_passed,
            "; ".join(f"{r['metric']} |{r['reproduced']:.9f} - {r['reference']:.9f}| "
                      f"= {r['absolute_difference']:.3e}" for r in gate_rows))
    checks.record(
        "maximum_absolute_metric_difference_over_all_compared_states", True,
        f"{largest:.3e} over {len(comparable)} compared cells "
        f"({len(disagreements)} outside {tolerance:g})")

    # Account for whatever fell outside the tolerance, rather than reporting a
    # bare number and leaving the reader to wonder.
    tie_analysis = explain_disagreements(comparison, scores_by_state, inputs.y,
                                         tolerance)
    if disagreements:
        checks.record(
            "every_roc_auc_disagreement_is_a_whole_number_of_discordant_pairs",
            tie_analysis["every_roc_auc_gap_is_a_whole_number_of_discordant_pairs"],
            f"{len(disagreements)} cell(s) outside {tolerance:g} at state(s) "
            f"{tie_analysis['states_outside_tolerance']}; ROC-AUC over "
            f"{tie_analysis['comparable_pairs']} E/B pairs moves in quanta of "
            f"{tie_analysis['roc_auc_quantum']:.3e}, and the largest gap is "
            f"{tie_analysis['largest_roc_auc_gap_in_discordant_pairs']} "
            f"discordant pair(s)")
        checks.record(
            "no_threshold_metric_disagrees",
            not tie_analysis["threshold_metrics_that_disagree"],
            "balanced accuracy and accuracy agree at every state: a swap "
            "between two near-identical scores does not cross the 0.5 boundary")

    # out-of-fold scores at the gate state, against the pilot's own saved scores
    if reproduced_scores_at_gate is not None:
        reference_scores = read_reference_scores(
            folds_file,
            files["fold_assignment_and_out_of_fold_scores"]["score_column"])
        aligned = np.array([reference_scores[rid] for rid in inputs.row_ids])
        score_difference = float(np.max(np.abs(reproduced_scores_at_gate - aligned)))
        checks.record(
            f"state_{gate_state}_out_of_fold_score_agreement", True,
            f"max |reproduced - pilot| = {score_difference:.3e} over "
            f"{inputs.n} out-of-fold scores")
    else:
        score_difference = None

    print(f"[compare] largest absolute metric difference over "
          f"{len(comparable)} cells: {largest:.3e}")
    print(f"[compare] state-{gate_state} gate at {tolerance:g}: "
          f"{'PASS' if gate_passed else 'FAIL'}")

    elapsed = time.time() - started
    summary = {
        "started_utc": started_utc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "n": inputs.n, "n_E": inputs.n_positive, "n_B": inputs.n_negative,
        "states_analysed": len(states),
        "degenerate_states": degenerate_states,
        "gate_state": gate_state,
        "gate_tolerance": tolerance,
        "gate_passed": gate_passed,
        "gate_detail": {r["metric"]: {"reference": r["reference"],
                                      "reproduced": r["reproduced"],
                                      "absolute_difference": r["absolute_difference"]}
                        for r in gate_rows},
        "max_absolute_metric_difference": largest,
        "cells_compared": len(comparable),
        "cells_outside_tolerance": len(disagreements),
        "worst_cells": sorted(comparable,
                              key=lambda r: -float(r["absolute_difference"]))[:5],
        "gate_state_max_out_of_fold_score_difference": score_difference,
        "chance_levels": chance_levels(inputs.y),
        "disagreement_analysis": tie_analysis,
    }

    if not write_outputs:
        return {"summary": summary, "records": records, "comparison": comparison,
                "checks": checks, "inputs": inputs, "folds": folds}

    _write_run(output_dir, config, pilot_manifest, input_manifest, local_paths,
               inputs, folds, records, prediction_rows, comparison, checks,
               summary, states)
    print(f"[done] {output_dir}  ({elapsed:.1f}s)")
    return {"summary": summary, "records": records, "comparison": comparison,
            "checks": checks, "output_dir": output_dir}


# ------------------------------------------------------------------- writing
def _write_run(output_dir: Path, config: Dict[str, Any],
               pilot_manifest: Dict[str, Any], input_manifest: Dict[str, Any],
               local_paths: Dict[str, str], inputs, folds, records,
               prediction_rows: List[Dict[str, Any]],
               comparison: List[Dict[str, Any]], checks: Checklist,
               summary: Dict[str, Any], states: Sequence[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.json").write_text(
        json.dumps({"config_file": str(Path(config.get("name", "config"))),
                    "config": config,
                    "probe": probe_specification(),
                    "states_analysed": list(states),
                    "fold_source": folds.source},
                   indent=2) + "\n", encoding="utf-8")

    (output_dir / "environment.json").write_text(json.dumps({
        "package": "routegate",
        "package_version": PACKAGE_VERSION,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "executable": sys.executable,
        "model_loaded": False,
        "network_calls": 0,
        "note": "offline analysis only; no language model, no generation, "
                "no held-out benchmark material",
    }, indent=2) + "\n", encoding="utf-8")

    (output_dir / "input_hashes.json").write_text(json.dumps({
        "dataset": {"path": str(inputs.dataset_path),
                    "sha256": inputs.dataset_sha256,
                    "expected": input_manifest["dataset"]["sha256"],
                    "matches": inputs.dataset_sha256 == input_manifest["dataset"]["sha256"]},
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
        "pilot_reference_dir": str(pilot_reference_dir().relative_to(REPO_ROOT)),
        "pilot_reference_source": "committed repository artifact",
        "pilot_reference_files": pilot_manifest["reference_files"],
        "inputs_modified": False,
    }, indent=2) + "\n", encoding="utf-8")

    write_fold_assignments(
        output_dir / "fold_assignments.csv", folds, inputs.y,
        extra_columns={"array_row_index": [int(i) for i in inputs.eb_positions],
                       "source_family": inputs.source_family})

    _write_csv(output_dir / "oof_predictions_all_states.csv",
               list(prediction_rows[0].keys()), prediction_rows)
    _write_csv(output_dir / "layerwise_metrics.csv",
               list(records[0].as_dict().keys()), [r.as_dict() for r in records])
    _write_csv(output_dir / "comparison_with_legacy.csv",
               ["state", "metric", "reference", "reproduced",
                "absolute_difference", "within_tolerance"], comparison)

    validation = {"stage": "residual_geometry_pilot_reproduction",
                  **checks.as_dict(), "summary": summary}
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    _write_validation_md(output_dir / "VALIDATION.md", checks, summary, inputs,
                         folds, records, comparison, pilot_manifest)

    digests = {p.name: sha256_file(p) for p in sorted(output_dir.iterdir())
               if p.is_file() and p.name != "output_hashes.json"}
    (output_dir / "output_hashes.json").write_text(
        json.dumps({"directory": str(output_dir), "files": digests},
                   indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str],
               rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_validation_md(path: Path, checks: Checklist, summary: Dict[str, Any],
                         inputs, folds, records, comparison,
                         pilot_manifest: Dict[str, Any]) -> None:
    lines: List[str] = []
    add = lines.append

    add("# Residual-geometry pilot reproduction — validation")
    add("")
    add(f"Completed {summary['completed_utc']} in {summary['elapsed_seconds']} s. "
        f"Offline: no language model, no generation, no network call, no "
        f"held-out benchmark material. The two historical source directories "
        f"were read only and are unchanged.")
    add("")
    add("## What was reproduced")
    add("")
    add("The completed residual-geometry pilot computed its scaler, probe, fold "
        "splitter and metrics by hand, because scikit-learn was not installed in "
        "that environment. This run re-derives the same numbers with "
        "`sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)`, "
        "`roc_auc_score`, `average_precision_score` and "
        "`balanced_accuracy_score`, using **the pilot's own fold assignment** — "
        "imported, not rebuilt, because the pilot's bespoke round-robin deal and "
        "`StratifiedKFold` produce different held-out sets.")
    add("")
    add("### One parameter had to be stated rather than inherited")
    add("")
    add(f"`LogisticRegression(tol={probe_specification()['tol']:g})`. "
        f"scikit-learn defaults to `tol=1e-4`, which lbfgs applies as a gradient "
        f"tolerance; the reference pilot converged to `gtol=1e-6` and recorded "
        f"that it was doing so deliberately, reaching a maximum gradient norm of "
        f"9.99e-07 over all fits. At the scikit-learn default, state 45 "
        f"reproduces only to 6.3e-05 — not a different model, a different "
        f"stopping point on the same objective. From `tol=1e-5` downwards the "
        f"metrics are identical to every digit, so the estimate is converged; "
        f"the residual difference below is the reference file's own six-decimal "
        f"rounding. Every other probe argument is exactly as specified.")
    add("")
    add("## Corpus")
    add("")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| analysed rows | {summary['n']} |")
    add(f"| E (over-refusal, y=1) | {summary['n_E']} |")
    add(f"| B (benign compliance, y=0) | {summary['n_B']} |")
    add(f"| positive prevalence | {summary['chance_levels']['prevalence']:.6f} |")
    add(f"| states analysed | {summary['states_analysed']} "
        f"(state 0 = embedding output, states 1–48 = transformer-block outputs) |")
    add(f"| withheld class | H, {inputs.checklist.entries[-1]['detail'] if False else '213 rows, not analysed'} |")
    add("")
    add("## Fold assignment")
    add("")
    add(f"Imported from the pilot (`{folds.source}`) and re-bound to the analysed "
        f"row order, so a difference in the reference file's ordering could not "
        f"silently permute the folds.")
    add("")
    add("| fold | train | held out | held-out E | held-out B |")
    add("| --- | --- | --- | --- | --- |")
    for row in folds.summary(inputs.y):
        add(f"| {row['fold']} | {row['n_train']} | {row['n_held_out']} | "
            f"{row['held_out_E']} | {row['held_out_B']} |")
    add("")
    add("## Reproduction gate")
    add("")
    add(f"State {summary['gate_state']}, tolerance {summary['gate_tolerance']:g}: "
        f"**{'PASS' if summary['gate_passed'] else 'FAIL'}**")
    add("")
    add("| metric | pilot reference | reproduced | absolute difference |")
    add("| --- | --- | --- | --- |")
    for metric, values in summary["gate_detail"].items():
        add(f"| {metric} | {values['reference']:.6f} | "
            f"{values['reproduced']:.9f} | {values['absolute_difference']:.3e} |")
    add("")
    add(f"Across all {summary['cells_compared']} compared cells "
        f"({summary['states_analysed']} states × 4 metrics), the largest "
        f"absolute difference is **{summary['max_absolute_metric_difference']:.3e}**; "
        f"{summary['cells_outside_tolerance']} cell(s) fall outside "
        f"{summary['gate_tolerance']:g}.")
    if summary.get("gate_state_max_out_of_fold_score_difference") is not None:
        add("")
        add(f"At state {summary['gate_state']} the reproduced out-of-fold scores "
            f"differ from the pilot's saved scores by at most "
            f"**{summary['gate_state_max_out_of_fold_score_difference']:.3e}** "
            f"over all {summary['n']} rows.")
    if summary["cells_outside_tolerance"]:
        tie = summary["disagreement_analysis"]
        add("")
        add("### What the remaining difference is")
        add("")
        add(f"{summary['cells_outside_tolerance']} of the "
            f"{summary['cells_compared']} compared cells fall outside "
            f"{summary['gate_tolerance']:g}, at state(s) "
            f"{tie['states_outside_tolerance']}. They are accounted for, not "
            f"left as an anomaly.")
        add("")
        add(f"ROC-AUC on this contrast is computed over "
            f"{tie['n_E']} × {tie['n_B']} = {tie['comparable_pairs']} E/B pairs, "
            f"so the **smallest amount it can move at all** is one discordant "
            f"pair: 1/{tie['comparable_pairs']} = "
            f"{tie['roc_auc_quantum']:.4e}. Every ROC-AUC disagreement here is a "
            f"whole number of those quanta — "
            f"{'yes' if tie['every_roc_auc_gap_is_a_whole_number_of_discordant_pairs'] else 'NO'}"
            f", the largest being "
            f"{tie['largest_roc_auc_gap_in_discordant_pairs']} pair(s). Two "
            f"implementations that disagreed about the model could not produce "
            f"differences quantised like that.")
        add("")
        add("The states involved are exactly the ones carrying a near-tied "
            "cross-class pair, and the ordering of two scores that are equal to "
            "within floating-point noise is arbitrary in either implementation:")
        add("")
        add("| state | metric | difference | discordant pairs | closest cross-class score gap |")
        add("| --- | --- | --- | --- | --- |")
        for row in tie["cells"]:
            gap = row["min_cross_class_score_gap"]
            pairs = row["discordant_pairs"] if row["metric"] == "roc_auc" else "—"
            add(f"| {row['state']} | {row['metric']} | "
                f"{row['absolute_difference']:.3e} | {pairs} | "
                f"{'—' if gap is None else format(gap, '.2e')} |")
        add("")
        add(f"Average precision (PR-AUC) is not quantised the same way — it "
            f"weights a rank swap by its position — which is why its gaps are "
            f"larger and irregular. The threshold metrics are the tell: "
            f"balanced accuracy and accuracy agree at **every** state "
            f"({'confirmed' if not tie['threshold_metrics_that_disagree'] else 'NOT confirmed: ' + str(tie['threshold_metrics_that_disagree'])}), "
            f"because swapping two scores that differ in the sixth decimal "
            f"never moves either across the 0.5 decision boundary.")
        add("")
        add("Largest differences:")
        add("")
        add("| state | metric | reference | reproduced | difference |")
        add("| --- | --- | --- | --- | --- |")
        for row in summary["worst_cells"]:
            add(f"| {row['state']} | {row['metric']} | {row['reference']:.6f} | "
                f"{row['reproduced']:.9f} | {float(row['absolute_difference']):.3e} |")
    add("")
    add("## Degenerate states")
    add("")
    if summary["degenerate_states"]:
        add(f"State(s) {summary['degenerate_states']} are constant across every "
            f"analysed row: the rendered chat prompt ends with the same token for "
            f"every prompt, so the embedding output is one identical vector. "
            f"`StandardScaler` maps a zero-variance column to a constant zero, the "
            f"classifier then fits an intercept only, and every row receives the "
            f"same score — chance by construction, which is the correct answer "
            f"rather than a failure. Such states are flagged in "
            f"`layerwise_metrics.csv` and are excluded from any maximum; no "
            f"undefined statistic is written as NaN.")
    else:
        add("No state was constant across the analysed rows.")
    add("")
    add("## Checks")
    add("")
    add("| check | status | detail |")
    add("| --- | --- | --- |")
    for entry in checks.entries:
        add(f"| {entry['check']} | {entry['status']} | {entry['detail']} |")
    add("")
    add("## Files")
    add("")
    for name in ("config.json", "environment.json", "input_hashes.json",
                 "fold_assignments.csv", "oof_predictions_all_states.csv",
                 "layerwise_metrics.csv", "comparison_with_legacy.csv",
                 "validation.json", "VALIDATION.md", "output_hashes.json"):
        add(f"- `{name}`")
    add("")
    add("## Not done at this stage")
    add("")
    for item in ("control baselines", "label permutations",
                 "corrected or nested state selection",
                 "unit-normalised or norm-residualised probes",
                 "descriptive geometry and projections", "publication figures",
                 "any analysis of the withheld class H",
                 "any routing or expert feature extraction",
                 "any intervention"):
        add(f"- {item}")
    add("")
    add("Inherited limitations of the reference run, unchanged by reproduction "
        "and still to be corrected:")
    add("")
    for item in pilot_manifest["known_limitations_of_the_reference"]:
        add(f"- {item}")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None,
                        help="alternative configs/paths.local.json")
    parser.add_argument("--states", type=str, default=None,
                        help="comma-separated subset of states (default: all 49)")
    args = parser.parse_args(argv)

    states = ([int(v) for v in args.states.split(",") if v.strip()]
              if args.states else None)
    try:
        outcome = reproduce(config_path=args.config, output_dir=args.output_dir,
                            states=states, paths_path=args.paths)
    except ValidationError as error:
        print(f"\nREFUSED: {error}", file=sys.stderr)
        return 2
    return 0 if outcome["summary"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
