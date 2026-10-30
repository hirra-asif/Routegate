#!/usr/bin/env python3
"""Freeze the cross-fitted, H-orthogonalised direction bank. No model, no GPU.

This runs first, alone, and is the only place a steering direction is estimated.
It reads the frozen development corpus and the frozen activation tensor, inherits
the E/B fold assignment the residual-geometry analysis already used, adds the H
rows to it by a rule with no random number in it, and writes -- for every fold
and all 49 stored representations -- the E−B vector, the H−B vector, the
orthogonalised vector and the unit repair direction, with their norms, their
cosines, the row identities each was and was not allowed to see, and a manifest
that hashes every file it wrote.

The scan then takes that manifest's SHA256 on the command line and refuses to
generate against anything else, so the directions cannot drift between a run and
its re-run.

What it cannot see: any generated response, any patching or steering result, and
anything belonging to the held-out benchmark split.

Writes only under
``outputs/causal_repair/orthogonalized_population_scan/direction_bank/``.

Usage:
    python experiments/causal_repair/build_orthogonalized_direction_bank.py
    python experiments/causal_repair/build_orthogonalized_direction_bank.py --dry-run
    python experiments/causal_repair/build_orthogonalized_direction_bank.py \\
        --output-dir DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import (                                        # noqa: E402
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    WITHHELD_LABEL,
    describe_state,
    load_input_manifest,
)
from routegate.ebh import (                                         # noqa: E402
    EBH_LABELS,
    FROZEN_EB_FOLD_REPEAT,
    H_ASSIGNMENT_RULE,
    N_SPLITS,
    build_ebh_fold_assignment,
    load_ebh_frame,
)
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.steering import (                                    # noqa: E402
    CAUSAL_SCAN_STATES,
    MANIFEST_FILENAME,
    N_STORED_STATES,
    STATE_EMBEDDING,
    SteeringError,
    build_direction_bank,
    write_direction_bank,
)
from routegate.validation import (                                  # noqa: E402
    ValidationError,
    assert_config_is_sealed_free,
)
from routegate import __version__ as PACKAGE_VERSION                # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "orthogonalized_population_scan.json"
SCAN_ROOT = REPO_ROOT / "outputs" / "causal_repair" / "orthogonalized_population_scan"
DEFAULT_OUTPUT = SCAN_ROOT / "direction_bank"
FROZEN_EB_FOLDS = (REPO_ROOT / "outputs" / "residual_geometry" / "corrected"
                   / "cv_assignments.csv")


def build(config_path: Path = DEFAULT_CONFIG,
          output_dir: Optional[Path] = None,
          frozen_folds: Optional[Path] = None,
          paths_path: Optional[Path] = None,
          repeat: int = FROZEN_EB_FOLD_REPEAT,
          write: bool = True) -> Dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "orthogonalized population scan config")

    input_manifest = load_input_manifest()
    expected = input_manifest["dataset"]["counts"]

    print("[inputs] loading the frozen corpus and activation tensor "
          "(hash-gated, memory-mapped, never copied)")
    frame = load_ebh_frame(paths_path=paths_path)
    counts = frame.counts()
    print(f"[inputs] E={counts[POSITIVE_LABEL]} B={counts[NEGATIVE_LABEL]} "
          f"H={counts[WITHHELD_LABEL]} total={frame.n}")

    # The counts are a gate, not a printout: a corpus that is not the one this
    # protocol was written against must stop here.
    for label in EBH_LABELS:
        if counts[label] != expected[label]:
            raise ValidationError(
                f"the corpus holds {counts[label]} {label} row(s); the accepted "
                f"dataset recorded in manifests/development_inputs.json has "
                f"{expected[label]}")
    if frame.n != expected["analysis_eligible"]:
        raise ValidationError(
            f"{frame.n} eligible rows, expected {expected['analysis_eligible']}")
    if frame.n_states != N_STORED_STATES:
        raise ValidationError(
            f"the activation tensor carries {frame.n_states} stored "
            f"representations, expected {N_STORED_STATES}")

    folds_path = Path(frozen_folds or FROZEN_EB_FOLDS)
    print(f"[folds] inheriting the E/B assignment from {folds_path} "
          f"(repeat {repeat}), unchanged")
    assignment, frozen_eb = build_ebh_fold_assignment(
        frame, folds_path, repeat=repeat, n_splits=N_SPLITS)
    checks = assignment.audit(frame, frozen_eb=frozen_eb)
    print(f"[folds] {H_ASSIGNMENT_RULE}")
    for row in assignment.counts():
        print(f"[folds] fold {row['fold']}: held out "
              f"E={row['held_out_E']} B={row['held_out_B']} H={row['held_out_H']}"
              f"  | training "
              f"E={row['train_E']} B={row['train_B']} H={row['train_H']}")
    print(f"[folds] {len(checks.entries)} check(s), all passed")

    training_positions = [assignment.training_positions(frame, fold)
                          for fold in range(N_SPLITS)]
    held_out_positions = [assignment.held_out_positions(frame, fold)
                          for fold in range(N_SPLITS)]

    # The leakage guarantee, asserted here rather than trusted: no row a fold
    # holds out may appear among the rows that fold learns from.
    for fold in range(N_SPLITS):
        for label in EBH_LABELS:
            overlap = set(training_positions[fold][label].tolist()) & set(
                held_out_positions[fold][label].tolist())
            if overlap:
                raise ValidationError(
                    f"fold {fold}, class {label}: {len(overlap)} row(s) are both "
                    f"training and held out")

    print(f"[directions] estimating {N_SPLITS} x {frame.n_states} directions "
          f"from training rows only")
    bank = build_direction_bank(frame.activations, training_positions,
                                n_states=frame.n_states,
                                causal_states=CAUSAL_SCAN_STATES)

    degenerate = [(row["fold"], row["state"]) for row in bank.rows
                  if row["status"] != "ok"]
    print(f"[directions] {len(bank.rows)} site(s); "
          f"{len(bank.rows) - len(degenerate)} usable, "
          f"{len(degenerate)} degenerate")
    for fold, state in degenerate:
        print(f"[directions] fold {fold}, {describe_state(state)}: degenerate — "
              f"{bank.row(fold, state)['status']}")
    if any(state != STATE_EMBEDDING for _fold, state in degenerate):
        raise SteeringError(
            "a transformer-block state came out degenerate; this should already "
            "have stopped the build")

    if not write:
        print("\n[dry-run] nothing was written.")
        return {"bank": bank, "assignment": assignment, "output_dir": None}

    output_dir = Path(output_dir or DEFAULT_OUTPUT)
    counts_payload = {
        "rows": frame.n,
        "by_class": counts,
        "n_splits": N_SPLITS,
        "stored_states": frame.n_states,
        "causal_scan_states": list(CAUSAL_SCAN_STATES),
        "hidden_size": frame.hidden_size,
        "per_fold": assignment.counts(),
        "targets": {
            POSITIVE_LABEL: counts[POSITIVE_LABEL],
            WITHHELD_LABEL: counts[WITHHELD_LABEL],
            "per_state": counts[POSITIVE_LABEL] + counts[WITHHELD_LABEL],
        },
        "planned_generations": {
            "intervention": ((counts[POSITIVE_LABEL] + counts[WITHHELD_LABEL])
                             * len(CAUSAL_SCAN_STATES)),
            "baseline": counts[POSITIVE_LABEL] + counts[WITHHELD_LABEL],
            "total": ((counts[POSITIVE_LABEL] + counts[WITHHELD_LABEL])
                      * (len(CAUSAL_SCAN_STATES) + 1)),
        },
        "degenerate_sites": [{"fold": fold, "state": state}
                             for fold, state in degenerate],
    }
    input_hashes = {
        "dataset": {"path": str(frame.dataset_path),
                    "sha256": frame.dataset_sha256,
                    "expected": input_manifest["dataset"]["sha256"]},
        "activation": {"path": str(frame.activation_path),
                       "sha256": frame.activation_sha256,
                       "expected": input_manifest["activation"]["sha256"],
                       "shape": [frame.n, frame.n_states, frame.hidden_size]},
        "extraction_manifest": {
            "path": str(frame.extraction_manifest_path),
            "sha256": frame.extraction_manifest_sha256,
            "expected": input_manifest["extraction_manifest"]["sha256"]},
        "inherited_fold_assignment": {
            "path": str(folds_path),
            "sha256": _sha256(folds_path),
            "repeat": int(repeat)},
        "inputs_modified": False,
    }
    bank_config = {
        "protocol": config,
        "package_version": PACKAGE_VERSION,
        "n_splits": N_SPLITS,
        "stored_states": frame.n_states,
        "causal_scan_states": list(CAUSAL_SCAN_STATES),
        "strength": config["strength"]["value"],
        "budget_reference": config["intervention"]["budget_reference"],
        "fold_source": {"E_and_B": str(folds_path), "repeat": int(repeat),
                        "H": H_ASSIGNMENT_RULE},
        "checklist": checks.as_dict(),
    }

    manifest = write_direction_bank(
        output_dir, bank,
        fold_assignment_rows=assignment.as_rows(frame),
        training_row_ids={fold: {label: assignment.training_row_ids(fold, label)
                                 for label in EBH_LABELS}
                          for fold in range(N_SPLITS)},
        held_out_row_ids={fold: {label: assignment.held_out_row_ids(fold, label)
                                 for label in EBH_LABELS}
                          for fold in range(N_SPLITS)},
        counts=counts_payload,
        input_hashes=input_hashes,
        config=bank_config,
        created_utc=utc_now(),
        environment=environment_record(PACKAGE_VERSION),
    )

    digest = manifest["manifest_sha256"]
    print(f"\n[bank] frozen: {output_dir}")
    for name in sorted(manifest["artifacts"]):
        print(f"[bank]   {name}  {manifest['artifacts'][name][:16]}...")
    print(f"[bank] {MANIFEST_FILENAME} manifest_sha256 = {digest}")
    print()
    print("Pass that hash to the scan, which refuses any other bank:")
    print(f"  --direction-bank {output_dir.relative_to(REPO_ROOT)} \\")
    print(f"  --expect-bank-sha256 {digest}")
    print()
    print("and paste it into BANK_SHA256 in")
    print("  experiments/causal_repair/submit_orthogonalized_population_scan.sh")
    return {"bank": bank, "assignment": assignment, "output_dir": output_dir,
            "manifest": manifest, "manifest_sha256": digest}


def _sha256(path: Path) -> str:
    from routegate.validation import sha256_file

    return sha256_file(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}")
    parser.add_argument("--frozen-folds", type=Path, default=None,
                        help="the E/B assignment to inherit; default is the "
                             "corrected residual-geometry run's cv_assignments.csv")
    parser.add_argument("--repeat", type=int, default=FROZEN_EB_FOLD_REPEAT,
                        help="which repeat of that file to inherit")
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate everything and print the summary without "
                             "writing the bank")
    args = parser.parse_args(argv)

    build(config_path=args.config, output_dir=args.output_dir,
          frozen_folds=args.frozen_folds, paths_path=args.paths,
          repeat=args.repeat, write=not args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
