#!/usr/bin/env python3
"""Refit the safety-first gate sensors on all 480 rows for deployment.

    python experiments/routegate_deployment/fit_gate_sensors.py --dry-run
    python experiments/routegate_deployment/fit_gate_sensors.py

Produces the two weight files the completed sensor run never wrote, so the
frozen gate can fire on a prompt it has never seen. It selects nothing: the
representation, estimator, grid, selection rule, both thresholds and the gate
rule are the frozen ones, hash-gated on the way in.

THE REPRODUCTION GATE
---------------------
Before any deployment weight is written, the five fold sensors are
reconstructed under this exact specification and their out-of-fold
probabilities compared against the frozen ``oof_predictions.csv`` row for row.
A specification that cannot reproduce those probabilities is not the one that
produced them, and a deployment artifact built from it would be a different
sensor. On failure nothing is written.

WHAT THIS IS NOT
----------------
Not an evaluation. The refit sees every development row, so its predictions
there are training-set predictions and are never reported. The development
estimates remain the out-of-fold 37/89 E, 0/178 B and 0/213 H.

Loads no language model, generates nothing, judges nothing, downloads nothing
and reads no sealed material.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.gatefreeze import (                               # noqa: E402
    OUTPUT_DIR, PREDECLARATION, SOURCES, verify_predeclaration)
from routegate.gatesensors import (                              # noqa: E402
    BUNDLE_FILE, CHECKPOINT_FILES, EXPECTED_ROWS, KIND_FOLD, KIND_REFIT,
    OOF_PREDICTIONS_FILE, OOF_PREDICTIONS_SHA256, OOF_REPRODUCTION_ATOL,
    OUTER_SPLITS, READ_STATE, SEED, SENSORS, SENSOR_E, SENSOR_H,
    C_SELECTED_PER_OUTER_FOLD, GateBundle,
    GateSensorError, assert_checkpoint_is_metadata_only, fit_sensor,
    load_thresholds, select_C, sensor_design)
from routegate.validation import (                               # noqa: E402
    Checklist, ValidationError, assert_not_sealed, sha256_file)


def write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                    prefix=f".{path.name}.", suffix=".tmp",
                                    delete=False)
    try:
        with h:
            h.write(text); h.flush(); os.fsync(h.fileno())
        os.replace(h.name, path)
    except BaseException:
        Path(h.name).unlink(missing_ok=True); raise


def load_design(checks: Checklist):
    """The frozen activations at state 45, the labels, and the row order."""
    from routegate.data import load_development_inputs
    from routegate.ebh import frame_from_inputs
    from routegate.probeheads import state_matrix

    inputs = load_development_inputs()
    frame = frame_from_inputs(inputs)
    rows = list(inputs.manifest_rows)
    checks.require("corpus_rows", len(rows) == EXPECTED_ROWS,
                   f"{len(rows)} row(s), expected {EXPECTED_ROWS}")
    order = [r["row_id"] for r in rows]
    labels = np.array([r["final_ebh"] for r in rows])
    X = np.asarray(state_matrix(frame, READ_STATE), dtype=np.float64)
    checks.require("feature_matrix", X.shape == (EXPECTED_ROWS, 2048),
                   f"{X.shape}")
    return X, labels, order


def read_frozen_oof(checks: Checklist) -> Dict[str, Dict[str, str]]:
    path = OOF_PREDICTIONS_FILE
    assert_not_sealed(path, "frozen out-of-fold predictions")
    digest = sha256_file(path)
    checks.require("oof_predictions_sha256", digest == OOF_PREDICTIONS_SHA256,
                   f"{digest}, pinned {OOF_PREDICTIONS_SHA256}")
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["row_id"]: r for r in csv.DictReader(fh)}


def verify_c_selection(X, labels, derived, checks: Checklist) -> None:
    """Optional: the inner search re-derives the C each outer fold recorded."""
    for sensor in SENSORS:
        got = []
        for fold in range(OUTER_SPLITS):
            Xf, yf, _m = sensor_design(sensor, X[derived != fold],
                                       labels[derived != fold])
            got.append(select_C(Xf, yf)[0])
        want = list(C_SELECTED_PER_OUTER_FOLD[sensor])
        checks.require(f"{sensor}_c_selection_reproduces", got == want,
                       f"{got}, recorded {want}")


def reproduce_out_of_fold(X, labels, order, frozen, checks: Checklist) -> None:
    """Rebuild the five fold sensors and match the frozen probabilities."""
    from sklearn.model_selection import StratifiedKFold

    recorded_fold = np.array([int(frozen[r]["outer_fold"]) for r in order])
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True,
                          random_state=SEED)
    derived = np.empty(len(order), dtype=int)
    for k, (_train, test) in enumerate(skf.split(np.zeros((len(order), 1)),
                                                 labels)):
        derived[test] = k
    checks.require("fold_division_reproduces",
                   bool((derived == recorded_fold).all()),
                   f"{int((derived != recorded_fold).sum())} row(s) differ")

    worst = {SENSOR_E: 0.0, SENSOR_H: 0.0}
    for sensor, column in ((SENSOR_E, "p_E"), (SENSOR_H, "p_H")):
        for fold in range(OUTER_SPLITS):
            train = derived != fold
            test = derived == fold
            Xf, yf, mask = sensor_design(sensor, X[train], labels[train])
            # The completed run RECORDED which C its inner search selected for
            # each outer fold. Reproduction uses that recorded value: it
            # reproduces what was done rather than re-running a search whose
            # answer is already on record. Re-deriving the search is a
            # separate, optional check (--verify-c-selection).
            c = C_SELECTED_PER_OUTER_FOLD[sensor][fold]
            checkpoint = fit_sensor(sensor, Xf, yf, c, KIND_FOLD, fold=fold)
            for index in np.where(test)[0]:
                # The E sensor scores every held-out row, as the frozen run
                # did; the design mask only restricts what it was FITTED on.
                got = checkpoint.probability(X[index])
                want = float(frozen[order[index]][column])
                worst[sensor] = max(worst[sensor], abs(got - want))
        checks.require(
            f"{sensor}_reproduces_the_frozen_oof_probabilities",
            worst[sensor] <= OOF_REPRODUCTION_ATOL,
            f"worst |reconstructed - frozen| {worst[sensor]:.3e} <= "
            f"{OOF_REPRODUCTION_ATOL:g}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every input and gate, write nothing")
    parser.add_argument("--verify-c-selection", action="store_true",
                        help="additionally re-derive the per-fold C the "
                             "completed run recorded (slow: 250 extra fits)")
    args = parser.parse_args(argv)

    checks = Checklist("routegate_gate_deployment_v1_fit")
    try:
        verify_predeclaration(PREDECLARATION, checks)
        thresholds = load_thresholds()
        checks.record("gate_unchanged", True,
                      f"{thresholds.gate_rule}; E {thresholds.threshold_E!r} "
                      f"H {thresholds.threshold_H!r}")
        for logical, spec in sorted(SOURCES.items()):
            checks.record(f"source_{logical}", True, spec["sha256"])

        out = Path(args.output_dir)
        if out.exists() and any(out.iterdir()):
            raise GateSensorError(
                f"{out} already holds files. A deployment checkpoint is "
                f"written once; remove it by hand, with a reason, and re-run.")

        X, labels, order = load_design(checks)
        frozen = read_frozen_oof(checks)
        checks.require("oof_covers_the_corpus", set(frozen) == set(order),
                       f"{len(frozen)} row(s)")
        reproduce_out_of_fold(X, labels, order, frozen, checks)
        if args.verify_c_selection:
            from sklearn.model_selection import StratifiedKFold
            derived = np.empty(len(order), dtype=int)
            for k, (_tr, te) in enumerate(StratifiedKFold(
                    n_splits=OUTER_SPLITS, shuffle=True,
                    random_state=SEED).split(np.zeros((len(order), 1)),
                                             labels)):
                derived[te] = k
            verify_c_selection(X, labels, derived, checks)

        # ---- the deployment refit, only now ----------------------------
        written: Dict[str, Any] = {}
        for sensor in SENSORS:
            Xf, yf, _mask = sensor_design(sensor, X, labels)
            c, scores = select_C(Xf, yf)
            checkpoint = fit_sensor(sensor, Xf, yf, c, KIND_REFIT)
            blob = checkpoint.as_dict()
            assert_checkpoint_is_metadata_only(blob)
            written[sensor] = (checkpoint, blob, c, scores)
            print(f"[fit] {sensor}: C={c!r} on {Xf.shape[0]} row(s), "
                  f"weights {checkpoint.weights_sha256()[:12]}")

        bundle = GateBundle(e_sensor=written[SENSOR_E][0],
                            h_veto=written[SENSOR_H][0], thresholds=thresholds)
        if args.dry_run:
            print(f"[fit] dry run: {out} was not written")
            print(f"[fit] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2

        out.mkdir(parents=True, exist_ok=True)
        for sensor, (_ckpt, blob, _c, _s) in written.items():
            path = out / CHECKPOINT_FILES[sensor]
            write_atomically(path, json.dumps(blob, indent=2, sort_keys=True)
                             + "\n")
            print(f"[fit] wrote {path}")
        write_atomically(out / BUNDLE_FILE,
                         json.dumps(bundle.as_dict(), indent=2, sort_keys=True)
                         + "\n")
        write_atomically(out / "validation.json",
                         json.dumps(checks.as_dict(), indent=2) + "\n")
        print(f"[fit] wrote {out / BUNDLE_FILE}")
        print(f"[fit] {checks.as_dict()['checks_run']} check(s), all passed: "
              f"{checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
