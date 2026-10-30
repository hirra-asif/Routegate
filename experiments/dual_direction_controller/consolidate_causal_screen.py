#!/usr/bin/env python3
"""Consolidate the causal screen's shards into one complete record.

The screen runs as eight array tasks, each writing its own JSONL so no two
processes ever append to one file. This puts them back together, and refuses to
call the result complete unless it is.

What it checks
--------------
Every shard must describe the same run. The model, the revision, the tokenizer
and chat-template digests, the generation-configuration digest, the direction
bank, the dataset, the intervention type, the budget reference, the states and
the layers must be identical across all eight; a single disagreement means two
experiments were mixed and the consolidation stops.

Then the arithmetic. Exactly 3,214 unique ``(row_id, arm_id)`` records: 445 on
the E rows and 2,769 on the H rows, with every predeclared arm present the
number of times the frozen protocol says. A duplicate is refused rather than
deduplicated, because the two copies may differ. A missing record is named, not
rounded down to a smaller screen.

Then each record on its own terms: complete, its arm's coefficients as
predeclared, and its response digest recomputed from its own response text. A
record whose digest does not match its text is refused -- it is the one failure
that would otherwise travel silently into every downstream analysis.

**A smoke test is never the screen.** Filtered or limited runs mark every
record they write, and any such record here stops the consolidation.

What it does not do
-------------------
It decides no behavioural outcome. It does not judge, score, label or classify
a response, and it computes no repair rate, refusal rate or any other endpoint.
It concatenates, verifies and orders. What the responses mean is the judging
stage, which is separate and not specified yet.

Usage
-----
    python experiments/dual_direction_controller/consolidate_causal_screen.py \\
        --shards-root <root holding shard_1 .. shard_8> \\
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
from routegate.dualsteer import (                                   # noqa: E402
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    BUDGET_REFERENCE,
    CLASS_E,
    CLASS_H,
    INTERVENTION_TYPE,
    N_E_ROWS,
    N_H_ROWS,
    RECORD_FIELDS,
    RUN_LOCKED_FIELDS,
    SCREEN_CLASSES,
    DualSteerError,
    declared_arms,
    implemented_arms,
    protocol_for,
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

DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual_direction_controller.json"
SHARD_COUNT = 8
SHARD_PREFIX = "shard_"
RECORDS_FILENAME = "screen_records.jsonl"
SHARD_MANIFEST = "run_manifest.json"

CONSOLIDATED_RECORDS = "screen_records.jsonl"
VALIDATION_FILENAME = "validation.json"
MANIFEST_FILENAME = "run_manifest.json"

#: How many records each class contributes, for the default protocol. A run
#: consolidated under another protocol uses that protocol's own split.
EXPECTED_BY_CLASS = protocol_for(DEFAULT_PROTOCOL).expected_by_class()

#: Fields that must be present and non-empty on a finished record. A record
#: missing one of these is incomplete, whatever else it holds.
REQUIRED_ON_EVERY_RECORD: Tuple[str, ...] = (
    "row_id", "row_class", "arm_id", "prompt_sha256", "response_sha256",
    "status", "created_utc", "generation", "model", "revision",
)


def gate(checks: Checklist, name: str, passed: bool, detail: str) -> None:
    checks.record(name, passed, detail)
    if not passed:
        raise DualSteerError(f"{name}: {detail}")


def shard_dirs(root: Path, count: int = SHARD_COUNT) -> List[Path]:
    """``shard_1`` .. ``shard_N``, all of them, in order."""
    dirs = [root / f"{SHARD_PREFIX}{index}" for index in range(1, count + 1)]
    missing = [str(path) for path in dirs if not path.is_dir()]
    if missing:
        raise DualSteerError(
            f"{len(missing)} shard directory(ies) are absent, first "
            f"{missing[0]}. The screen is {count} shards; consolidating a "
            f"subset would produce a short record that looks complete.")
    return dirs


def read_shard(directory: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One shard's records and its own run manifest."""
    records_path = directory / RECORDS_FILENAME
    manifest_path = directory / SHARD_MANIFEST
    for path in (records_path, manifest_path):
        if not path.is_file():
            raise DualSteerError(
                f"{path} is not a file; a shard that did not finish is not "
                f"consolidated as though it had")
    # strict: a torn line anywhere means the shard is still being written or
    # two writers shared it, and neither is a finished shard
    records = read_records(records_path, strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return records, manifest


def verify_shards_agree(manifests: Sequence[Mapping[str, Any]],
                        checks: Checklist) -> Dict[str, Any]:
    """Every shard describes one run, or there is no one run to consolidate."""
    if not manifests:
        raise DualSteerError("no shard manifests to compare")
    reference = manifests[0]
    # What a shard *is* comes before what it agrees with. "shard 2 is a smoke
    # test" says what to do; "the shards disagree on smoke" makes you go
    # looking for which one and why.
    for index, manifest in enumerate(manifests, start=1):
        if manifest.get("dry_run"):
            raise DualSteerError(
                f"shard {index} was written by a dry run; a dry run writes no "
                f"generations and cannot contribute to the screen")
        if manifest.get("smoke"):
            raise DualSteerError(
                f"shard {index} is a smoke test ("
                f"{'; '.join(manifest.get('smoke_reasons') or [])}). A smoke "
                f"output is a subset chosen by hand and is never accepted as "
                f"the complete causal screen.")

    for field in RUN_LOCKED_FIELDS:
        values = {json.dumps(manifest.get(field), sort_keys=True)
                  for manifest in manifests}
        if len(values) != 1:
            raise DualSteerError(
                f"the shards disagree on {field}: {sorted(values)[:3]}. They "
                f"are not one experiment and are not concatenated into one.")
    locked = {field: reference.get(field) for field in RUN_LOCKED_FIELDS}
    gate(checks, "shards_describe_one_run", True,
         f"{len(manifests)} shard(s) agree on {len(RUN_LOCKED_FIELDS)} "
         f"run-locked field(s)")
    return locked


def verify_protocol(locked: Mapping[str, Any], config: Mapping[str, Any],
                    checks: Checklist, protocol: Any = None) -> None:
    """The run's own settings are the frozen ones, still."""
    screen = config["causal_screen"]
    protocol = protocol or protocol_for(DEFAULT_PROTOCOL)
    gate(checks, "protocol_matches_the_records",
         locked.get("protocol") == protocol.name,
         f"records carry protocol {locked.get('protocol')!r}, this "
         f"consolidation was asked for {protocol.name!r}")
    gate(checks, "intervention_type",
         locked.get("intervention_type") == INTERVENTION_TYPE,
         f"{locked.get('intervention_type')!r}")
    gate(checks, "budget_reference",
         locked.get("budget_reference") == BUDGET_REFERENCE,
         f"{locked.get('budget_reference')!r}")
    gate(checks, "states", list(locked.get("states") or []) == list(BANK_STATES),
         f"{locked.get('states')} (historical {locked.get('historical_layers')})")
    gate(checks, "historical_layers",
         list(locked.get("historical_layers") or []) == list(HISTORICAL_LAYERS),
         f"{locked.get('historical_layers')}")
    gate(checks, "direction_bank_sha256",
         locked.get("direction_bank_sha256")
         == screen["direction_bank"]["artifact_sha256"],
         f"{locked.get('direction_bank_sha256')}")
    gate(checks, "not_a_smoke_test", locked.get("smoke") is False,
         "every record was written by an unfiltered run")

    # The arm definitions the config declares are the ones the package holds.
    # Both tables are read through dualsteer.declared_arms, which is the one
    # place that knows where a protocol's arms live: its own `arms` block, or
    # the block it names with `specification`. The exploratory protocol uses
    # the second form to point at causal_screen rather than restating thirteen
    # arms, and a configuration written before protocols existed has only
    # causal_screen -- one reader covers all three shapes.
    #
    # Checked twice, because they are two claims. First: the exploratory
    # screen's frozen table is still what the package implements, whichever
    # protocol is being consolidated. Second: the selected protocol's own
    # table is.
    exploratory = protocol_for(DEFAULT_PROTOCOL)
    screen_declared = declared_arms(config, exploratory)
    screen_implemented = implemented_arms(exploratory)
    for name in SCREEN_CLASSES:
        gate(checks, f"arm_definitions_{name}",
             screen_declared[name] == screen_implemented[name],
             f"{len(screen_implemented[name])} arm(s)")

    if protocol.name != exploratory.name:
        protocol_declared = declared_arms(config, protocol)
        protocol_implemented = implemented_arms(protocol)
        for name in SCREEN_CLASSES:
            gate(checks, f"{protocol.name}_arms_{name}",
                 protocol_declared[name] == protocol_implemented[name],
                 f"{len(protocol_implemented[name])} arm(s)")

    # The declared total, where the configuration states one. An undeclared
    # protocol never reaches here: declared_arms refuses it above.
    spec = (config.get("protocols") or {}).get(protocol.name)
    if isinstance(spec, Mapping) and "expected_generations" in spec:
        gate(checks, "protocol_total_matches_the_config",
             spec["expected_generations"] == protocol.expected_generations,
             f"config says {spec['expected_generations']}, package says "
             f"{protocol.expected_generations}")


def verify_records(records: Sequence[Mapping[str, Any]], locked: Mapping[str, Any],
                   checks: Checklist, protocol: Any = None) -> Dict[str, Any]:
    """Complete, unique, correctly armed, and each digest its own text's.

    "Correctly armed" is judged against the arms the *selected protocol*
    runs, not against every arm the package declares. Under the focused
    protocol a ``safety_010`` record is an arm belonging to another protocol,
    and it is refused here rather than accepted because some protocol
    somewhere declares it.
    """
    arm_table = (protocol or protocol_for(DEFAULT_PROTOCOL)).arm_by_id()
    by_key: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    duplicates: List[Tuple[Any, ...]] = []
    for record in records:
        key = record_key(record)
        if key in by_key:
            duplicates.append(key)
            continue
        by_key[key] = record
    gate(checks, "no_duplicate_records", not duplicates,
         f"{len(duplicates)} duplicated key(s)"
         + (f", first {duplicates[0]}" if duplicates else ""))

    incomplete: List[str] = []
    unknown_arms: List[str] = []
    wrong_alphas: List[str] = []
    bad_digest: List[str] = []
    unexpected_fields: List[str] = []
    for record in by_key.values():
        where = f"{record.get('row_id')}/{record.get('arm_id')}"
        if any(record.get(field) in (None, "") for field
               in REQUIRED_ON_EVERY_RECORD):
            incomplete.append(where)
        extra = sorted(set(record) - set(RECORD_FIELDS) - {"schema"})
        if extra:
            unexpected_fields.append(f"{where}:{extra}")
        arm = arm_table.get(record.get("arm_id"))
        if arm is None:
            unknown_arms.append(where)
            continue
        if (round(float(record["alpha_R"]), 6) != round(arm.alpha_repair, 6)
                or round(float(record["alpha_S"]), 6)
                != round(arm.alpha_safety, 6)):
            wrong_alphas.append(where)
        text = record.get("response_text")
        if text is None or sha256_text(text) != record.get("response_sha256"):
            bad_digest.append(where)

    gate(checks, "records_are_complete", not incomplete,
         f"{len(incomplete)} incomplete record(s)"
         + (f", first {incomplete[0]}" if incomplete else ""))
    gate(checks, "no_unexpected_fields", not unexpected_fields,
         f"{len(unexpected_fields)} record(s) carry fields outside the schema"
         + (f", first {unexpected_fields[0]}" if unexpected_fields else ""))
    # Said separately from "unknown arm", because it is a different mistake
    # with a different fix. A reference arm is inherited from the earlier
    # phase and is joined to the analysis table later; one appearing in a
    # generated shard means it was RE-generated, and the immutable phase-1
    # record for that cell is about to be shadowed by a fresh one.
    reference_ids = {arm.arm_id for row_class in SCREEN_CLASSES
                     for arm in (protocol
                                 or protocol_for(DEFAULT_PROTOCOL)
                                 ).reference_arms(row_class)}
    regenerated = sorted(f"{record.get('row_id')}/{record.get('arm_id')}"
                         for record in by_key.values()
                         if record.get("arm_id") in reference_ids)
    gate(checks, "no_reference_arm_was_regenerated", not regenerated,
         f"{len(regenerated)} record(s) carry an arm this protocol INHERITS "
         f"and must never generate"
         + (f", first {regenerated[0]}" if regenerated else "")
         + (f"; the inherited arms are {sorted(reference_ids)}"
            if reference_ids else "; this protocol inherits no arm"))
    gate(checks, "no_unexpected_arms", not unknown_arms,
         f"{len(unknown_arms)} record(s) carry an arm that is not predeclared"
         + (f", first {unknown_arms[0]}" if unknown_arms else ""))
    gate(checks, "coefficients_match_their_arm", not wrong_alphas,
         f"{len(wrong_alphas)} record(s) disagree with their arm's frozen "
         f"coefficients"
         + (f", first {wrong_alphas[0]}" if wrong_alphas else ""))
    gate(checks, "response_digests_recompute", not bad_digest,
         f"{len(bad_digest)} record(s) whose response_sha256 is not the digest "
         f"of their own response_text"
         + (f", first {bad_digest[0]}" if bad_digest else ""))

    mismatched_locked = [
        f"{record.get('row_id')}/{record.get('arm_id')}:{field}"
        for record in by_key.values() for field in RUN_LOCKED_FIELDS
        if record.get(field) != locked.get(field)]
    gate(checks, "records_match_the_run", not mismatched_locked,
         f"{len(mismatched_locked)} record field(s) differ from the run"
         + (f", first {mismatched_locked[0]}" if mismatched_locked else ""))

    # the per-state direction digests are one bank's, everywhere
    per_state_digests: Dict[str, set] = {}
    for record in by_key.values():
        for state, spec in (record.get("per_state") or {}).items():
            for name in ("repair_direction_sha256", "safety_direction_sha256"):
                per_state_digests.setdefault(f"{state}/{name}",
                                             set()).add(spec.get(name))
    drifted = sorted(key for key, values in per_state_digests.items()
                     if len(values) != 1)
    gate(checks, "one_direction_bank_throughout", not drifted,
         f"{len(drifted)} (state, direction) digest(s) differ between records"
         + (f", first {drifted[0]}" if drifted else ""))
    return {"unique": len(by_key), "duplicates": len(duplicates),
            "by_key": by_key}


def verify_counts(by_key: Mapping[Tuple[Any, ...], Mapping[str, Any]],
                  checks: Checklist, protocol: Any = None) -> Dict[str, Any]:
    """Exactly the protocol's size, split as the protocol says, arm by arm."""
    protocol = protocol or protocol_for(DEFAULT_PROTOCOL)
    expected_by_class = protocol.expected_by_class()
    by_class = Counter(record["row_class"] for record in by_key.values())
    by_arm = Counter(record["arm_id"] for record in by_key.values())
    rows_by_class: Dict[str, set] = {name: set() for name in SCREEN_CLASSES}
    for record in by_key.values():
        rows_by_class[record["row_class"]].add(record["row_id"])

    for name in SCREEN_CLASSES:
        gate(checks, f"{name}_records",
             by_class.get(name, 0) == expected_by_class[name],
             f"{by_class.get(name, 0)} record(s), expected "
             f"{expected_by_class[name]}")
        # The rows THIS protocol covers, not the corpus: a protocol bound to
        # a selection is complete at that selection's size.
        expected_rows = protocol.expected_rows(name)
        gate(checks, f"{name}_rows", len(rows_by_class[name]) == expected_rows,
             f"{len(rows_by_class[name])} distinct row(s), expected "
             f"{expected_rows}")

    # Counting 18 distinct rows is not the same as covering the 18 rows the
    # selection returned. A run on a different 18 would satisfy every count
    # gate above; only the identities settle it, so a bound protocol checks
    # them.
    for name in SCREEN_CLASSES:
        bound = protocol.selected_row_ids(name)
        if bound is None:
            continue
        present = rows_by_class[name]
        unexpected = sorted(present - set(bound))
        absent = sorted(set(bound) - present)
        gate(checks, f"{name}_rows_match_the_bound_selection",
             not unexpected and not absent,
             f"{len(unexpected)} row(s) outside the frozen selection and "
             f"{len(absent)} bound row(s) absent"
             + (f"; first unexpected {unexpected[0]}" if unexpected else "")
             + (f"; first absent {absent[0]}" if absent else "")
             if (unexpected or absent) else
             f"all {len(bound)} bound {name} row(s) present, none added")

    # every arm present exactly as often as the protocol says
    expected_by_arm: Counter = Counter()
    for name in SCREEN_CLASSES:
        rows = protocol.expected_rows(name)
        for arm in protocol.arms_by_class[name]:
            expected_by_arm[arm.arm_id] += rows
    wrong = {arm: (by_arm.get(arm, 0), want)
             for arm, want in expected_by_arm.items()
             if by_arm.get(arm, 0) != want}
    gate(checks, "every_arm_is_complete", not wrong,
         f"{len(wrong)} arm(s) short or over"
         + (f": {dict(sorted(wrong.items())[:3])}" if wrong else ""))

    gate(checks, "total_records",
         len(by_key) == protocol.expected_generations,
         f"{len(by_key)} unique (row_id, arm_id) record(s), expected "
         f"{protocol.expected_generations} for protocol {protocol.name}")
    return {"total": len(by_key), "by_class": dict(sorted(by_class.items())),
            "by_arm": dict(sorted(by_arm.items())),
            "rows_by_class": {name: len(rows_by_class[name])
                              for name in SCREEN_CLASSES}}


def order_records(by_key: Mapping[Tuple[Any, ...], Mapping[str, Any]],
                  protocol: Any = None) -> List[Dict[str, Any]]:
    """Class, then row id, then the arm's declared position. Deterministic.

    Not shard order and not file order: the consolidated file must be the same
    bytes whichever order the eight tasks happened to finish in.
    """
    arm_order = (protocol or protocol_for(DEFAULT_PROTOCOL)).arm_order()
    return [dict(record) for record in sorted(
        by_key.values(),
        key=lambda record: (SCREEN_CLASSES.index(record["row_class"]),
                            record["row_id"], arm_order[record["arm_id"]]))]


def jsonl_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                   for record in records)


def consolidate(shards_root: Path, out_dir: Path, config_path: Path,
                shard_count: int = SHARD_COUNT, write: bool = True,
                protocol_name: str = DEFAULT_PROTOCOL) -> Dict[str, Any]:
    checks = Checklist("dual_direction_screen_consolidation")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    protocol = protocol_for(protocol_name)

    if write and out_dir.exists() and any(out_dir.iterdir()):
        raise DualSteerError(
            f"{out_dir} already exists and is not empty. A consolidated screen "
            f"is written once; name a new directory.")

    dirs = shard_dirs(Path(shards_root), shard_count)
    records: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    per_shard: Dict[str, int] = {}
    for directory in dirs:
        shard_records, manifest = read_shard(directory)
        per_shard[directory.name] = len(shard_records)
        records.extend(shard_records)
        manifests.append(manifest)
    checks.record("shards_read", True,
                  f"{len(dirs)} shard(s), {len(records)} record(s): "
                  + ", ".join(f"{name}={count}"
                              for name, count in sorted(per_shard.items())))

    locked = verify_shards_agree(manifests, checks)
    verify_protocol(locked, config, checks, protocol)
    verified = verify_records(records, locked, checks, protocol)
    counts = verify_counts(verified["by_key"], checks, protocol)

    ordered = order_records(verified["by_key"], protocol)
    payload = jsonl_text(ordered)

    validation = {
        "schema": "routegate/dual_direction_screen_consolidation/1",
        "created_utc": utc_now(),
        "protocol": protocol.name,
        "protocol_purpose": protocol.purpose,
        "protocol_expected_generations": protocol.expected_generations,
        "protocol_tests_safety_direction": protocol.tests_safety_direction,
        "complete_protocol": True,
        "reportable": True,
        "reportable_note": f"every one of {protocol.expected_generations} "
                           f"records for {protocol.name} is present, unique "
                           f"and verified. A protocol is reportable only "
                           f"complete.",
        "complete_causal_screen": protocol.name == DEFAULT_PROTOCOL,
        "smoke": False,
        "decides_no_behavioural_outcome": True,
        "judging_note": "no response is judged, scored, labelled or classified "
                        "here, and no endpoint is computed. What these "
                        "responses mean is the judging stage.",
        "shards": per_shard,
        "counts": counts,
        "records_sha256": sha256_text(payload),
        "records_bytes": len(payload.encode("utf-8")),
        "run_locked": locked,
        "checklist": checks.as_dict(),
    }
    manifest = {
        "schema": "routegate/dual_direction_screen_consolidated/1",
        "created_utc": validation["created_utc"],
        "package_version": PACKAGE_VERSION,
        "dry_run": bool(not write),
        "shards_root": str(shards_root),
        "shard_count": shard_count,
        "shard_manifests": [{"shard": directory.name,
                             "records": per_shard[directory.name],
                             "created_utc": manifest.get("created_utc")}
                            for directory, manifest in zip(dirs, manifests)],
        "generation": manifests[0].get("generation"),
        "frozen_protocol": {
            "name": protocol.name,
            "expected_generations": protocol.expected_generations,
            "E_records": protocol.expected_by_class()[CLASS_E],
            "H_records": protocol.expected_by_class()[CLASS_H],
            "arms_per_class": {name: len(arms) for name, arms
                               in protocol.arms_by_class.items()},
            "tests_safety_direction": protocol.tests_safety_direction,
        },
        "outputs": {"records": {"name": CONSOLIDATED_RECORDS,
                                "sha256": validation["records_sha256"],
                                "bytes": validation["records_bytes"],
                                "rows": len(ordered)}},
        "ordering": "ascending by (class, row_id, declared arm position); keys "
                    "sorted within each record. The same shards give the same "
                    "bytes whatever order the tasks finished in.",
        "environment": environment_record(PACKAGE_VERSION),
        **locked,
    }

    print(f"[consolidate] protocol {protocol.name}")
    print(f"[consolidate] {len(dirs)} shard(s), {counts['total']} unique "
          f"record(s) of {protocol.expected_generations}")
    print(f"[consolidate] {CLASS_E} {counts['by_class'].get(CLASS_E)} / "
          f"{CLASS_H} {counts['by_class'].get(CLASS_H)}")
    print(f"[consolidate] records sha256 {validation['records_sha256']}")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / CONSOLIDATED_RECORDS).write_text(payload, encoding="utf-8")
        write_json(out_dir / VALIDATION_FILENAME, validation)
        write_json(out_dir / MANIFEST_FILENAME, manifest)
        print(f"[consolidate] wrote {out_dir}")
    else:
        print("[consolidate] dry run: nothing written")
    return {"validation": validation, "manifest": manifest,
            "records": ordered}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True,
                        help=f"the directory holding {SHARD_PREFIX}1 .. "
                             f"{SHARD_PREFIX}{SHARD_COUNT}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL,
                        choices=sorted(PROTOCOLS),
                        help="the named protocol these shards complete")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        consolidate(args.shards_root, args.output_dir, args.config,
                    args.shard_count, write=not args.dry_run,
                    protocol_name=args.protocol)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
