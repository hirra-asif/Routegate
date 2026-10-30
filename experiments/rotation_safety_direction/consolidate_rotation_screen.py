#!/usr/bin/env python3
"""Consolidate the rotation screen's shards into one complete record.

The screen runs as an array of tasks, each writing its own JSONL so no two
processes ever append to one file. This puts them back together, and refuses to
call the result complete unless it is.

What it checks
--------------
Every shard must describe the same run: the model, the revision, the tokenizer
and chat-template digests, the generation-configuration digest, the direction
bank, the dataset, the intervention type, the intervention ORDER, the budget
reference, the states, the layers, the tolerance version, the predeclaration's
self-digest and the selection digest must be identical across all of them. A
single disagreement means two experiments were mixed and the consolidation
stops.

Then the arithmetic. Exactly 144 unique ``(row_id, arm_id)`` records: the 18
frozen rows at each of the eight arms. A duplicate is refused rather than
deduplicated, because the two copies may differ. A missing record is named, not
rounded down to a smaller screen. A record carrying ``baseline`` or
``repair_015`` is refused: those are inherited from phase 1 and this screen
does not produce them.

Then each record on its own terms: complete, its arm's beta and alpha_R as
predeclared, its response digest recomputed from its own response text, and its
three per-state rotation diagnostics present, at the right states, with each
layer's own H - E digest and with every predeclared invariant still holding on
the numbers it recorded.

**A smoke test is never the screen.** Filtered or limited runs mark every
record they write, and any such record here stops the consolidation.

What it does not do
-------------------
It decides no behavioural outcome. It does not judge, score, label or classify
a response, and it computes no recovery count, harm count or any other
endpoint. It concatenates, verifies and orders. What the responses mean is the
judging stage, which is separate and is not run here.

Usage
-----
    python experiments/rotation_safety_direction/consolidate_rotation_screen.py \\
        --shards-root <root holding shard_1 .. shard_N> \\
        --output-dir <a new directory> --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.directionbank import BANK_STATES, HISTORICAL_LAYERS  # noqa: E402
from routegate.dualsteer import BUDGET_REFERENCE, CLASS_H           # noqa: E402
from routegate.rotation import (                                    # noqa: E402
    ROTATION_TOLERANCES,
    TOLERANCES_VERSION,
    assert_rotation_invariants,
)
from routegate.rotationfreeze import (                              # noqa: E402
    PREDECLARATION,
    verify_predeclaration,
    verify_preserved_artifacts,
)
from routegate.rotationscreen import (                              # noqa: E402
    ANALYSIS_TABLE_RECORDS,
    ARM_BY_ID_ROTATION,
    DEFAULT_CONFIG,
    INTERVENTION_ORDER,
    INTERVENTION_TYPE,
    NEW_ARMS,
    NEW_GENERATIONS,
    N_ROWS,
    PROTOCOL,
    REFERENCE_ARM_IDS,
    REFERENCE_RECORDS,
    RECORD_FIELDS,
    RUN_LOCKED_FIELDS,
    SELECTED_ROWS,
    SELECTED_ROWS_SHA256,
    RotationScreenError,
    assert_config_matches,
    claim_output_dir,
    read_records,
    record_key,
)
from routegate.runrecord import (                                   # noqa: E402
    environment_record,
    utc_now,
    write_json,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_text,
)

SHARD_PREFIX = "shard_"
RECORDS_FILENAME = "rotation_records.jsonl"
SHARD_MANIFEST = "run_manifest.json"

CONSOLIDATED_RECORDS = "rotation_records.jsonl"
VALIDATION_FILENAME = "validation.json"
MANIFEST_FILENAME = "run_manifest.json"

#: Fields that must be present and non-empty on a finished record.
REQUIRED_ON_EVERY_RECORD: Tuple[str, ...] = (
    "row_id", "row_class", "arm_id", "prompt_sha256", "response_sha256",
    "status", "created_utc", "generation", "model", "revision", "per_state",
)

#: The keys a per-state rotation diagnostic must carry. Every one of them is a
#: quantity the predeclaration says will be recorded for every row, layer and
#: arm; a record missing one is not a record of this screen.
REQUIRED_PER_STATE: Tuple[str, ...] = (
    "state", "historical_layer", "intervention_order", "applies_repair",
    "safety_direction_sha256", "beta", "theta", "phi",
    "cosine_before", "cosine_after_float32", "cosine_after_cast",
    "norm_before_float32", "norm_after_float32", "norm_after_cast_float32",
    "relative_norm_error_float32", "relative_norm_error_after_cast",
    "relative_edit_norm_float32", "relative_edit_norm_after_cast",
    "theoretical_chord", "chord_absolute_error", "remaining_angle",
    "expected_remaining_angle", "edge_case", "tolerances_version",
    "dtype_in", "dtype_out",
)


def gate(checks: Checklist, name: str, passed: bool, detail: str) -> None:
    checks.record(name, passed, detail)
    if not passed:
        raise RotationScreenError(f"{name}: {detail}")


def shard_dirs(root: Path, count: int) -> List[Path]:
    """``shard_1`` .. ``shard_N``, all of them, in order."""
    dirs = [root / f"{SHARD_PREFIX}{index}" for index in range(1, count + 1)]
    missing = [str(path) for path in dirs if not path.is_dir()]
    if missing:
        raise RotationScreenError(
            f"{len(missing)} shard director(y/ies) absent: {missing[:3]}. A "
            f"consolidation is of the whole array or of nothing; a missing "
            f"shard is not silently skipped.")
    unexpected = sorted(
        entry.name for entry in root.iterdir()
        if entry.is_dir() and entry.name.startswith(SHARD_PREFIX)
        and entry not in dirs)
    if unexpected:
        raise RotationScreenError(
            f"{root} holds shard director(y/ies) {unexpected} outside "
            f"1..{count}. A shard nobody planned is a run from another array; "
            f"it is refused, not merged.")
    return dirs


def read_shard(directory: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = read_records(directory / RECORDS_FILENAME, strict=True)
    manifest_path = directory / SHARD_MANIFEST
    if not manifest_path.is_file():
        raise RotationScreenError(
            f"{manifest_path} is absent; a shard without its run manifest "
            f"cannot be shown to be part of this run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return records, manifest


def verify_shards_agree(manifests: Sequence[Mapping[str, Any]],
                        checks: Checklist) -> Dict[str, Any]:
    """Every shard describes the same run, or two experiments were mixed."""
    locked: Dict[str, Any] = {}
    for field_name in RUN_LOCKED_FIELDS:
        values = {json.dumps(manifest.get(field_name), sort_keys=True)
                  for manifest in manifests}
        gate(checks, f"shards_agree_on_{field_name}", len(values) == 1,
             f"{len(values)} distinct value(s) across {len(manifests)} shard(s)"
             + ("" if len(values) == 1 else f": {sorted(values)[:3]}"))
        locked[field_name] = manifests[0].get(field_name)

    for index, manifest in enumerate(manifests):
        gate(checks, f"shard_{index + 1}_is_not_a_dry_run",
             manifest.get("dry_run") is False,
             f"shard {index + 1} records dry_run="
             f"{manifest.get('dry_run')!r}; a dry run wrote no records and is "
             f"not part of a consolidation")
        gate(checks, f"shard_{index + 1}_is_not_a_smoke_test",
             manifest.get("smoke") is False,
             f"shard {index + 1} records smoke={manifest.get('smoke')!r} "
             f"({manifest.get('smoke_reasons')}); a filtered or limited run is "
             f"never the screen")

    gate(checks, "locked_protocol", locked.get("protocol") == PROTOCOL,
         f"{locked.get('protocol')!r}")
    gate(checks, "locked_intervention_type",
         locked.get("intervention_type") == INTERVENTION_TYPE,
         f"{locked.get('intervention_type')!r}")
    gate(checks, "locked_intervention_order",
         locked.get("intervention_order") == INTERVENTION_ORDER,
         f"{locked.get('intervention_order')!r}")
    gate(checks, "locked_budget_reference",
         locked.get("budget_reference") == BUDGET_REFERENCE,
         f"{locked.get('budget_reference')!r}")
    gate(checks, "locked_states", locked.get("states") == list(BANK_STATES),
         f"{locked.get('states')}")
    gate(checks, "locked_historical_layers",
         locked.get("historical_layers") == list(HISTORICAL_LAYERS),
         f"{locked.get('historical_layers')}")
    gate(checks, "locked_tolerances_version",
         locked.get("tolerances_version") == TOLERANCES_VERSION,
         f"{locked.get('tolerances_version')!r}")
    gate(checks, "locked_selected_rows_sha256",
         locked.get("selected_rows_sha256") == SELECTED_ROWS_SHA256,
         f"{locked.get('selected_rows_sha256')!r}")
    return locked


def verify_records(records: Sequence[Mapping[str, Any]],
                   locked: Mapping[str, Any], checks: Checklist
                   ) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    """Each record on its own terms, then the geometry it says it produced."""
    by_key: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    statuses: Counter = Counter()
    edge_cases: Counter = Counter()

    for index, record in enumerate(records):
        where = f"record {index}"
        unknown = sorted(set(record) - set(RECORD_FIELDS) - {"schema"})
        gate(checks, "record_fields_are_declared", not unknown,
             f"{where} carries undeclared field(s) {unknown}")
        empty = [field_name for field_name in REQUIRED_ON_EVERY_RECORD
                 if record.get(field_name) in (None, "", {}, [])]
        gate(checks, "record_is_complete", not empty,
             f"{where} is missing or empty at {empty}")

        arm_id = record.get("arm_id")
        if arm_id in REFERENCE_ARM_IDS:
            gate(checks, "no_reference_arm_was_regenerated", False,
                 f"{where} carries the inherited arm {arm_id!r}. baseline and "
                 f"repair_015 already exist in the frozen phase-1 records for "
                 f"these rows; this screen reads them and never regenerates "
                 f"them.")
        gate(checks, "record_arm_is_predeclared",
             arm_id in ARM_BY_ID_ROTATION,
             f"{where} carries arm {arm_id!r}; this screen's arms are "
             f"{sorted(ARM_BY_ID_ROTATION)}")
        arm = ARM_BY_ID_ROTATION[str(arm_id)]
        gate(checks, "record_coefficients_match_its_arm",
             (round(float(record["beta"]), 6) == round(arm.beta, 6)
              and round(float(record["alpha_R"]), 6)
              == round(arm.alpha_repair, 6)),
             f"{where} is arm {arm_id} with (beta, alpha_R) "
             f"({record['beta']}, {record['alpha_R']}); the predeclared arm is "
             f"({arm.beta}, {arm.alpha_repair})")
        gate(checks, "record_row_is_selected",
             record.get("row_id") in set(SELECTED_ROWS),
             f"{where} is row {record.get('row_id')!r}, not one of the "
             f"{N_ROWS} frozen rows")
        gate(checks, "record_row_class", record.get("row_class") == CLASS_H,
             f"{where} is class {record.get('row_class')!r}")

        for field_name in RUN_LOCKED_FIELDS:
            gate(checks, f"record_matches_locked_{field_name}",
                 record.get(field_name) == locked.get(field_name),
                 f"{where} has {field_name}={record.get(field_name)!r} and the "
                 f"run has {locked.get(field_name)!r}")

        gate(checks, "record_status_is_known",
             record.get("status") in ("ok", "aligned_no_edit"),
             f"{where} carries status {record.get('status')!r}; a record with "
             f"an error is not part of a complete screen")
        statuses[str(record.get("status"))] += 1

        digest = sha256_text(str(record.get("response_text", "")))
        gate(checks, "response_digest_recomputes",
             digest == record.get("response_sha256"),
             f"{where} records response_sha256 "
             f"{record.get('response_sha256')} and its own text hashes to "
             f"{digest}")

        per_state = record.get("per_state") or {}
        gate(checks, "record_covers_every_state",
             sorted(int(state) for state in per_state) == list(BANK_STATES),
             f"{where} carries states {sorted(per_state)}; this screen edits "
             f"{list(BANK_STATES)}")
        seen_directions = set()
        for state_key, diagnostics in sorted(per_state.items()):
            state = int(state_key)
            missing = [key for key in REQUIRED_PER_STATE
                       if key not in diagnostics]
            gate(checks, "per_state_is_complete", not missing,
                 f"{where} state {state} is missing {missing}")
            gate(checks, "per_state_layer_numbering",
                 int(diagnostics["historical_layer"]) == state - 1,
                 f"{where} state {state} claims historical layer "
                 f"{diagnostics.get('historical_layer')}")
            gate(checks, "per_state_beta_is_the_arm_beta",
                 round(float(diagnostics["beta"]), 6) == round(arm.beta, 6),
                 f"{where} state {state} rotated at beta "
                 f"{diagnostics.get('beta')} and the arm is {arm.beta}; one "
                 f"beta is shared across all three layers")
            gate(checks, "per_state_repair_matches_the_arm",
                 bool(diagnostics["applies_repair"]) == arm.applies_repair,
                 f"{where} state {state} records applies_repair="
                 f"{diagnostics.get('applies_repair')!r} and the arm is "
                 f"{arm.kind}")
            gate(checks, "per_state_order",
                 diagnostics["intervention_order"] == (
                     INTERVENTION_ORDER if arm.applies_repair
                     else "rotation_only"),
                 f"{where} state {state} records order "
                 f"{diagnostics.get('intervention_order')!r}")
            seen_directions.add(str(diagnostics["safety_direction_sha256"]))
            edge_cases[str(diagnostics.get("edge_case") or "none")] += 1
            try:
                assert_rotation_invariants(diagnostics, ROTATION_TOLERANCES,
                                           label=f"{where} state {state}")
            except ValidationError as exc:
                gate(checks, "per_state_invariants_hold", False, str(exc))
        gate(checks, "each_layer_used_its_own_direction",
             len(seen_directions) == len(BANK_STATES),
             f"{where} used {len(seen_directions)} distinct H - E vector(s) "
             f"across {len(BANK_STATES)} layers; each layer has its own and no "
             f"layer borrows another's")

        key = record_key(record)
        gate(checks, "records_are_unique", key not in by_key,
             f"{where} repeats key {key}; one (row, arm) is generated once. A "
             f"duplicate is refused rather than deduplicated, because the two "
             f"may differ.")
        by_key[key] = record

    checks.record("record_statuses", True, json.dumps(dict(statuses),
                                                      sort_keys=True))
    checks.record("rotation_edge_cases", True, json.dumps(dict(edge_cases),
                                                          sort_keys=True))
    return by_key


def verify_counts(by_key: Mapping[Tuple[Any, ...], Mapping[str, Any]],
                  checks: Checklist) -> Dict[str, Any]:
    """The inventory is complete: 18 rows at 8 arms, each exactly once."""
    records = list(by_key.values())
    gate(checks, "total_records", len(records) == NEW_GENERATIONS,
         f"{len(records)} record(s); the screen is {N_ROWS} x "
         f"{len(ARM_BY_ID_ROTATION)} = {NEW_GENERATIONS}")

    by_arm = Counter(str(record["arm_id"]) for record in records)
    short = {arm_id: by_arm.get(arm_id, 0) for arm_id in ARM_BY_ID_ROTATION
             if by_arm.get(arm_id, 0) != N_ROWS}
    gate(checks, "every_arm_is_complete", not short,
         f"arm(s) with the wrong count: {short}; each arm covers all {N_ROWS} "
         f"rows")

    present = {(str(record["row_id"]), str(record["arm_id"]))
               for record in records}
    expected = {(row_id, arm.arm_id) for row_id in SELECTED_ROWS
                for arm in NEW_ARMS}
    missing = sorted(expected - present)
    gate(checks, "no_cell_is_missing", not missing,
         f"{len(missing)} missing (row, arm) cell(s), first {missing[:3]}")
    extra = sorted(present - expected)
    gate(checks, "no_cell_is_unplanned", not extra,
         f"{len(extra)} unplanned (row, arm) cell(s), first {extra[:3]}")

    return {"records": len(records), "rows": N_ROWS,
            "arms": len(ARM_BY_ID_ROTATION),
            "by_arm": dict(sorted(by_arm.items())),
            "reference_records_not_here": REFERENCE_RECORDS,
            "analysis_table_records": ANALYSIS_TABLE_RECORDS}


def order_records(by_key: Mapping[Tuple[Any, ...], Mapping[str, Any]]
                  ) -> List[Mapping[str, Any]]:
    """One deterministic order: row id, then the arm's declared position.

    Not the order the shards happened to finish in. Two consolidations of the
    same shards must produce the same bytes, or the digest below identifies
    the scheduler rather than the data.
    """
    position = {arm.arm_id: index for index, arm in enumerate(NEW_ARMS)}
    return sorted(by_key.values(),
                  key=lambda record: (str(record["row_id"]),
                                      position[str(record["arm_id"])]))


def jsonl_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True, ensure_ascii=False)
                   + "\n" for record in records)


def consolidate(shards_root: Path, out_dir: Path, config_path: Path,
                predeclaration: Path, shard_count: int, write: bool) -> int:
    checks = Checklist("rotation_screen_consolidation")

    preserved = verify_preserved_artifacts(checks=checks)
    frozen = verify_predeclaration(predeclaration, checks=checks)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert_config_matches(config, checks=checks)

    dirs = shard_dirs(Path(shards_root), shard_count)
    shards = [read_shard(directory) for directory in dirs]
    manifests = [manifest for _, manifest in shards]
    records: List[Dict[str, Any]] = []
    for shard_records, _ in shards:
        records.extend(shard_records)
    checks.record("shards_read", True,
                  f"{len(dirs)} shard(s), {len(records)} record(s) before "
                  f"verification")

    locked = verify_shards_agree(manifests, checks)
    gate(checks, "locked_predeclaration_sha256",
         locked.get("predeclaration_sha256") == frozen["sha256"],
         f"the shards ran under predeclaration "
         f"{locked.get('predeclaration_sha256')!r} and the committed manifest "
         f"hashes to {frozen['sha256']!r}")

    by_key = verify_records(records, locked, checks)
    counts = verify_counts(by_key, checks)

    ordered = order_records(by_key)
    text = jsonl_text(ordered)
    digest = sha256_text(text)

    validation = {
        "schema": "routegate/rotation_safety_direction_consolidation/1",
        "created_utc": utc_now(),
        "protocol": PROTOCOL,
        "complete_protocol": True,
        "reportable": True,
        "is_a_candidate_not_a_result": True,
        "is_a_candidate_note": "the responses exist and are complete. Nothing "
                               "about what they MEAN is decided here: no "
                               "response is judged, scored or labelled by "
                               "this stage, and rotational H - E is a "
                               "candidate until the frozen safety rubric has "
                               "been applied and the frozen decision rule "
                               "evaluated.",
        "judges_responses": False,
        "counts": counts,
        "records_sha256": digest,
        "records_filename": CONSOLIDATED_RECORDS,
        "shards": [str(path) for path in dirs],
        "run_locked": dict(locked),
        "rotation_tolerances": ROTATION_TOLERANCES.as_dict(),
        "preserved_artifacts": preserved,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }

    print(f"[consolidate] {len(dirs)} shard(s) -> {counts['records']} record(s)")
    print(f"[consolidate] arms {json.dumps(counts['by_arm'], sort_keys=True)}")
    print(f"[consolidate] records sha256 {digest}")
    print(f"[consolidate] {checks.as_dict()['checks_run']} check(s), all "
          f"passed")

    if not write:
        print("[consolidate] dry run: nothing written")
        return 0

    claimed = claim_output_dir(out_dir, create=True, resume=False)
    (claimed / CONSOLIDATED_RECORDS).write_text(text, encoding="utf-8")
    write_json(claimed / VALIDATION_FILENAME, validation)
    write_json(claimed / MANIFEST_FILENAME, {
        "schema": "routegate/rotation_safety_direction_consolidated/1",
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "protocol": PROTOCOL,
        "predeclaration_sha256": frozen["sha256"],
        "records_sha256": digest,
        "counts": counts,
        "judges_responses": False,
        "is_a_candidate_not_a_result": True,
        **dict(locked),
    })
    print(f"[consolidate] wrote {claimed}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predeclaration", type=Path, default=PREDECLARATION)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return consolidate(args.shards_root, args.output_dir, args.config,
                           args.predeclaration, args.shard_count,
                           write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
