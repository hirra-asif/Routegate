#!/usr/bin/env python3
"""Fit the two frozen probe heads. CPU only: no model, no GPU, no generation.

This is a thin runner. Every scientific decision it makes was made in
``configs/dual_direction_controller.json`` and every calculation it performs
lives in ``routegate.probeheads``; what is here is argument handling, the gates
that must pass before anything is fitted, and the writing of the run record.

What it reads
    the frozen development corpus and activation array, through
    ``routegate.ebh.load_ebh_frame`` -- hash-gated, memory-mapped, never copied;
    the cross-fitted direction bank's own frozen 480-row fold assignment.

What it refuses
    an output directory that already exists with anything in it, a direction
    bank whose manifest or fold file has moved since it was frozen, a fold
    division that its stated rule does not reproduce, a configuration whose
    probe block is not the frozen specification, and any read state outside the
    three declared ones. There is no permissive mode and no option that reaches
    material outside the development corpus.

What it writes
    oof_predictions.csv, metrics.json, fold_assignments.csv, validation.json and
    run_manifest.json, and nothing else. No prompt text and no response text is
    written by this stage.

The controller is not implemented. This produces the two readings it would need
and stops there.

Usage:
    python experiments/dual_direction_controller/fit_probe_heads.py \\
        --output-dir outputs/dual_direction_controller/probe_heads_v1

    python experiments/dual_direction_controller/fit_probe_heads.py \\
        --output-dir DIR \\
        --direction-bank PATH/TO/direction_bank \\
        --expect-bank-sha256 <the bank's manifest_sha256>

``outputs/`` is gitignored, so in a worktree the direction bank may not sit at
the default repo-relative path; pass ``--direction-bank`` explicitly there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.data import (                                        # noqa: E402
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    WITHHELD_LABEL,
    load_input_manifest,
)
from routegate.ebh import EBH_LABELS, load_ebh_frame                # noqa: E402
from routegate.probeheads import (                                  # noqa: E402
    CONTROLLER_READ_STATES,
    HEAD_ORDER,
    HEAD_SPECIFICATIONS,
    HISTORICAL_LAYER_TO_STATE,
    PREDICTION_COLUMNS,
    PRIMARY_READ_LAYER,
    PRIMARY_READ_STATE,
    SENSITIVITY_STATES,
    ProbeHeadError,
    bind_fold_assignment,
    cross_check_inherited_folds,
    evaluate_state,
    fit_probe_heads,
    layer_of_state,
    load_bank_fold_assignment,
    probe_head_specification,
)
from routegate.probes import probe_specification                    # noqa: E402
from routegate.runrecord import (                                   # noqa: E402
    environment_record,
    utc_now,
    write_csv,
    write_json,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_not_sealed,
    assert_not_stale,
    sha256_file,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual_direction_controller.json"
DEFAULT_BANK = (REPO_ROOT / "outputs" / "causal_repair"
                / "orthogonalized_population_scan" / "direction_bank")

# Every source file whose contents can change a number this run reports.
CODE_FILES = (
    "src/routegate/probeheads.py",
    "src/routegate/probes.py",
    "src/routegate/ebh.py",
    "src/routegate/data.py",
    "src/routegate/splits.py",
    "src/routegate/metrics.py",
    "src/routegate/steering.py",
    "src/routegate/validation.py",
    "src/routegate/runrecord.py",
    "experiments/dual_direction_controller/fit_probe_heads.py",
)

# The estimator fields that change a fit. The configuration must agree with the
# frozen specification on all of them, and may not carry a field the frozen
# specification does not have -- that is what "no new probe hyperparameters"
# means in practice.
GATED_PROBE_FIELDS = (
    "pipeline", "penalty", "C", "class_weight", "max_iter", "solver",
    "random_state", "tol", "decision_threshold",
)

OUTPUT_PREDICTIONS = "oof_predictions.csv"
OUTPUT_METRICS = "metrics.json"
OUTPUT_FOLDS = "fold_assignments.csv"
OUTPUT_VALIDATION = "validation.json"
OUTPUT_MANIFEST = "run_manifest.json"


# ------------------------------------------------------------- the output dir
def claim_output_dir(path: Path | str) -> Path:
    """Take an empty or absent directory, or refuse.

    A run directory is a record. Writing into one that already holds a run would
    leave a directory whose files came from two executions, and no reader could
    tell which. The check is made before anything is loaded, so a refusal costs
    nothing.
    """
    path = assert_not_stale(Path(path))
    assert_not_sealed(path, "output directory")
    if path.is_symlink():
        raise ValidationError(
            f"{path} is a symlink. A run directory is written in place, and "
            f"following a link here could write outside the repository.")
    if path.exists():
        if not path.is_dir():
            raise ValidationError(f"{path} exists and is not a directory")
        entries = sorted(p.name for p in path.iterdir())
        if entries:
            raise ValidationError(
                f"{path} already holds {len(entries)} entr"
                f"{'y' if len(entries) == 1 else 'ies'} "
                f"({', '.join(entries[:5])}{'...' if len(entries) > 5 else ''}). "
                f"This runner writes a fresh record and never merges into an "
                f"existing one; give --output-dir a new path.")
    path.mkdir(parents=True, exist_ok=True)
    return path


# -------------------------------------------------------------- config gates
def check_config(config: Dict[str, Any], checks: Checklist) -> List[int]:
    """Gate the declared configuration, and return the states it declares."""
    assert_config_is_sealed_free(config, "dual direction controller config")

    checks.require("config_schema",
                   config.get("schema") == "routegate/dual_direction_controller/1",
                   str(config.get("schema")))
    checks.record("config_status", True, str(config.get("status")))

    frozen = probe_specification()
    declared = config.get("probe") or {}
    extra = sorted(set(declared) - set(frozen))
    checks.require(
        "the_config_declares_no_probe_field_the_frozen_specification_lacks",
        not extra,
        f"{len(declared)} declared field(s); no hyperparameter is introduced "
        f"at this stage")
    missing = [f for f in GATED_PROBE_FIELDS if f not in declared]
    checks.require("the_config_declares_every_estimator_field", not missing,
                   f"{list(GATED_PROBE_FIELDS)}")
    differing = [f for f in GATED_PROBE_FIELDS if declared.get(f) != frozen[f]]
    checks.require(
        "the_configured_probe_is_the_frozen_specification",
        not differing,
        f"C={frozen['C']} class_weight={frozen['class_weight']} "
        f"max_iter={frozen['max_iter']} solver={frozen['solver']} "
        f"random_state={frozen['random_state']} tol={frozen['tol']}")

    read = config.get("read_states") or {}
    checks.require(
        "the_configured_layer_to_state_mapping_is_the_packages",
        {int(k): int(v) for k, v in (read.get("layer_to_state") or {}).items()}
        == dict(HISTORICAL_LAYER_TO_STATE),
        f"historical layers {sorted(HISTORICAL_LAYER_TO_STATE)} are states "
        f"{[HISTORICAL_LAYER_TO_STATE[k] for k in sorted(HISTORICAL_LAYER_TO_STATE)]}")
    checks.require("the_primary_read_layer_is_31",
                   int(read.get("primary_read_layer", -1)) == PRIMARY_READ_LAYER,
                   f"layer {PRIMARY_READ_LAYER}")
    checks.require("the_primary_read_state_is_32",
                   int(read.get("primary_read_state", -1)) == PRIMARY_READ_STATE,
                   f"state {PRIMARY_READ_STATE} = historical layer "
                   f"{layer_of_state(PRIMARY_READ_STATE)}")
    checks.require("the_sensitivity_states_are_33_and_34",
                   tuple(int(s) for s in read.get("sensitivity_states", ()))
                   == SENSITIVITY_STATES,
                   f"states {list(SENSITIVITY_STATES)}, reported beside state "
                   f"{PRIMARY_READ_STATE} and never selected from")

    targets = config.get("targets") or {}
    for name in HEAD_ORDER:
        spec = HEAD_SPECIFICATIONS[name].as_dict()
        block = targets.get(f"{name}_head") or {}
        for field in ("positive_class", "negative_class",
                      "excluded_from_training", "trained_on", "scored_rows",
                      "label_encoding", "score_column"):
            checks.require(
                f"the_configured_{name}_head_{field}_matches_the_package",
                block.get(field) == spec[field],
                f"{field} = {spec[field]}")

    checks.require("no_calibration_model_is_declared",
                   (config.get("calibration") or {}).get("model_fitted") is False,
                   "the Brier score measures the emitted probabilities; nothing "
                   "recalibrates them")

    states = [int(s) for s in (read.get("states") or [])]
    checks.require("the_configured_states_are_the_three_read_states",
                   tuple(states) == CONTROLLER_READ_STATES,
                   f"states {list(CONTROLLER_READ_STATES)}")
    return states


def check_states(states: Sequence[int], checks: Checklist) -> List[int]:
    """Refuse any state outside the three declared ones."""
    wanted = [int(s) for s in states]
    outside = sorted(set(wanted) - set(CONTROLLER_READ_STATES))
    checks.require(
        "every_requested_state_is_a_declared_read_state",
        not outside,
        f"requested {wanted}; the declared read states are "
        f"{list(CONTROLLER_READ_STATES)}. This stage reads a predeclared region "
        f"and does not search for a state.")
    if not wanted:
        raise ValidationError("no state was requested")
    ordered = sorted(set(wanted))
    checks.require("the_primary_read_state_is_among_them",
                   PRIMARY_READ_STATE in ordered,
                   f"state {PRIMARY_READ_STATE} is the primary reading; "
                   f"{[s for s in ordered if s != PRIMARY_READ_STATE]} are "
                   f"sensitivity checks")
    return ordered


# ---------------------------------------------------------------------- run
def run(output_dir: Path, config_path: Path, bank_dir: Path,
        expect_bank_sha256: Optional[str] = None,
        states: Optional[Sequence[int]] = None,
        paths_path: Optional[Path] = None) -> Dict[str, Any]:
    checks = Checklist("dual_direction_probe_heads")
    started = utc_now()

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    declared_states = check_config(config, checks)
    requested = check_states(states if states else declared_states, checks)
    print(f"[config] {config_path.name}: status {config.get('status')}, "
          f"read states {requested} "
          f"(historical layers {[layer_of_state(s) for s in requested]})")

    output_dir = claim_output_dir(output_dir)
    print(f"[output] {output_dir}")

    print("[inputs] loading the frozen corpus and activation tensor "
          "(hash-gated, memory-mapped, never copied)")
    frame = load_ebh_frame(paths_path=paths_path)
    counts = frame.counts()
    expected = load_input_manifest()["dataset"]["counts"]
    for label in EBH_LABELS:
        checks.require(f"corpus_count_{label}", counts[label] == expected[label],
                       f"{counts[label]} (expected {expected[label]})")
    checks.require("corpus_row_count",
                   frame.n == expected["analysis_eligible"],
                   f"{frame.n} (expected {expected['analysis_eligible']})")
    print(f"[inputs] E={counts[POSITIVE_LABEL]} B={counts[NEGATIVE_LABEL]} "
          f"H={counts[WITHHELD_LABEL]} total={frame.n}")

    print(f"[folds] binding the frozen division from {bank_dir}")
    assignment, source = load_bank_fold_assignment(bank_dir, expect_bank_sha256)
    bind_fold_assignment(frame, assignment, source, checks)
    outcome = cross_check_inherited_folds(frame, assignment, source, checks)
    print(f"[folds] {source.n_splits} folds, {len(source.rows)} rows, "
          f"fold_assignment.csv sha256 {source.fold_assignment_sha256[:16]}...")
    print(f"[folds] cross-check: {outcome}")
    for row in assignment.counts():
        print(f"[folds] fold {row['fold']}: held out "
              f"E={row['held_out_E']} B={row['held_out_B']} H={row['held_out_H']}"
              f"  | training E={row['train_E']} B={row['train_B']} "
              f"H={row['train_H']}")

    prediction_rows: List[Dict[str, Any]] = []
    state_records: List[Dict[str, Any]] = []
    for state in requested:
        role = "primary" if state == PRIMARY_READ_STATE else "sensitivity check"
        print(f"[heads] state {state} (historical layer {layer_of_state(state)}, "
              f"{role}): fitting {list(HEAD_ORDER)} over {source.n_splits} folds")
        heads = fit_probe_heads(frame, assignment, state, checks)
        prediction_rows.extend(heads.prediction_rows())
        record = evaluate_state(heads)
        state_records.append(record)
        for name in HEAD_ORDER:
            head = record["heads"][name]
            print(f"[heads]   {name:6s} ROC-AUC {head['roc_auc']:.6f}  "
                  f"PR-AUC {head['pr_auc']:.6f}  Brier {head['brier']:.6f}  "
                  f"(n={head['n_scored']}, chance PR-AUC "
                  f"{head['chance']['pr_auc']:.4f})")

    checks.require(
        "every_row_carries_one_p_E_and_one_p_H_at_every_state",
        len(prediction_rows) == frame.n * len(requested),
        f"{len(prediction_rows)} rows = {frame.n} prompts x {len(requested)} "
        f"state(s)")
    checks.record(
        "no_state_was_selected", True,
        f"state {PRIMARY_READ_STATE} is the predeclared primary reading; "
        f"{[s for s in requested if s != PRIMARY_READ_STATE]} are reported "
        f"beside it as sensitivity checks and nothing is ranked")

    summary = {
        "states": requested,
        "primary_read_state": PRIMARY_READ_STATE,
        "primary_read_layer": PRIMARY_READ_LAYER,
        "sensitivity_states": [s for s in requested if s != PRIMARY_READ_STATE],
        "rows": frame.n,
        "class_counts": counts,
        "n_splits": source.n_splits,
        "fold_source": source.as_dict(),
        "calibration_model_fitted": False,
        "model_loaded": False,
        "controller_implemented": False,
    }
    written = _write_run(output_dir, config, config_path, frame, assignment,
                         source, prediction_rows, state_records, checks,
                         summary, requested, bank_dir, started)
    print(f"\n[record] {output_dir}")
    for name in written:
        print(f"[record]   {name}")
    print("\nThe controller is not implemented. This run produced the two "
          "cross-fitted readings it would need and nothing else.")
    return {"output_dir": output_dir, "summary": summary,
            "states": state_records, "checklist": checks}


def _write_run(output_dir: Path, config: Dict[str, Any], config_path: Path,
               frame, assignment, source, prediction_rows: List[Dict[str, Any]],
               state_records: List[Dict[str, Any]], checks: Checklist,
               summary: Dict[str, Any], states: Sequence[int],
               bank_dir: Path, started: str) -> List[str]:
    fold_rows = assignment.as_rows(frame)
    write_csv(output_dir / OUTPUT_FOLDS, list(fold_rows[0]), fold_rows)
    write_csv(output_dir / OUTPUT_PREDICTIONS, list(PREDICTION_COLUMNS),
              prediction_rows)
    write_json(output_dir / OUTPUT_METRICS, {
        "stage": "dual_direction_controller_probe_heads",
        "created_utc": utc_now(),
        "specification": probe_head_specification(),
        "primary_read_state": summary["primary_read_state"],
        "sensitivity_states": summary["sensitivity_states"],
        "state_selection": "none. The primary read state is predeclared and the "
                           "sensitivity states are reported beside it; no state "
                           "is ranked, selected or promoted by this stage.",
        "states": state_records,
    })
    write_json(output_dir / OUTPUT_VALIDATION, {
        "stage": "dual_direction_controller_probe_heads",
        **checks.as_dict(),
        "summary": summary,
    })

    # Written last, over the bytes the four files above actually hold.
    outputs = {name: sha256_file(output_dir / name)
               for name in (OUTPUT_FOLDS, OUTPUT_PREDICTIONS, OUTPUT_METRICS,
                            OUTPUT_VALIDATION)}
    write_json(output_dir / OUTPUT_MANIFEST, {
        "schema": "routegate/dual_direction_probe_heads/1",
        "stage": "dual_direction_controller_probe_heads",
        "started_utc": started,
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": sha256_file(config_path),
            "status": config.get("status"),
            "declared": config,
        },
        "inputs": {
            "dataset": {"path": str(frame.dataset_path),
                        "sha256": frame.dataset_sha256},
            "activation": {"path": str(frame.activation_path),
                           "sha256": frame.activation_sha256,
                           "shape": [frame.n, frame.n_states, frame.hidden_size]},
            "extraction_manifest": {
                "path": str(frame.extraction_manifest_path),
                "sha256": frame.extraction_manifest_sha256},
            "fold_source": source.as_dict(),
            "direction_bank_directory": str(Path(bank_dir).resolve()),
            "inputs_modified": False,
        },
        "code": {name: sha256_file(REPO_ROOT / name) for name in CODE_FILES},
        "outputs": outputs,
        "states": list(states),
        "environment": environment_record(PACKAGE_VERSION),
    })
    return [OUTPUT_FOLDS, OUTPUT_PREDICTIONS, OUTPUT_METRICS, OUTPUT_VALIDATION,
            OUTPUT_MANIFEST]


# ---------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="where the run record is written. Must not already hold anything.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--direction-bank", type=Path, default=DEFAULT_BANK,
        help="the frozen bank whose fold_assignment.csv this stage inherits; "
             f"default: {DEFAULT_BANK.relative_to(REPO_ROOT)}")
    parser.add_argument(
        "--expect-bank-sha256", default=None,
        help="refuse any bank whose manifest_sha256 is not this")
    parser.add_argument(
        "--states", type=int, nargs="+", default=None,
        help=f"read states; default is the configured "
             f"{list(CONTROLLER_READ_STATES)}. Any other state is refused.")
    parser.add_argument(
        "--paths", type=Path, default=None,
        help="machine-local input path file; default configs/paths.local.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(output_dir=args.output_dir, config_path=args.config,
        bank_dir=args.direction_bank,
        expect_bank_sha256=args.expect_bank_sha256,
        states=args.states, paths_path=args.paths)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProbeHeadError, ValidationError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
