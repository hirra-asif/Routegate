#!/usr/bin/env python3
"""Consolidate the two-site arm into exactly 480 candidate records.

Three sources, combined deterministically:

    3    candidate records the interrupted run completed, pinned by digest
    477  candidate records the continuation shards generated
    480  baseline responses REFERENCED from the frozen static-steering master

The baseline stays a separate 480-row reference. It is not merged into the
candidate records and is never described as generated here: the master's own
run configuration records ``before_source.regenerated: false``, and this
consolidation carries that provenance forward rather than laundering it.

Deterministic
-------------
Records are emitted in the same ascending row-id order the shards were planned
in, so consolidating the same shards twice produces byte-identical output. The
output digest is recorded and can be recomputed by anyone holding the inputs.

Fail-closed
-----------
Every record is validated again here -- locked fields, prompt digest, class,
sites, direction digest, status, hook count -- because a record that was valid
when written can still be the wrong record to consolidate. A missing row, a
duplicate, an unknown row, a wrong arm, an error status or a truncated line
stops the consolidation rather than producing a short result.

Usage
-----
    python experiments/single_direction_controller/consolidate_two_site.py \\
        --shard-dir .../two_site_validation_v2 \\
        --output-dir .../two_site_validation_consolidated --dry-run
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

from routegate.data import (                                        # noqa: E402
    load_development_inputs,
    load_local_paths,
)
from routegate.dualsteer import BUDGET_REFERENCE                    # noqa: E402
from routegate.onepassbudget import ALPHA_R                         # noqa: E402
from routegate.runrecord import (                                   # noqa: E402
    environment_record,
    utc_now,
    write_json,
)
from routegate.runtime import (                                     # noqa: E402
    FROZEN_CHAT_TEMPLATE_SHA256,
    FROZEN_MODEL,
    FROZEN_REVISION,
    FROZEN_TOKENIZER_CONFIG_SHA256,
    GenerationConfig,
)
from routegate.twosite import (                                     # noqa: E402
    CANDIDATE_ARM_ID,
    CANDIDATE_BLOCKS,
    CANDIDATE_STATES,
    build_candidate_record,
)
from routegate.twositecontinuation import (                         # noqa: E402
    EXPECTED_NEW_CANDIDATES,
    EXPECTED_REUSABLE_CANDIDATES,
    EXPECTED_ROWS,
    PROTOCOL,
    SCHEMA,
    ContinuationError,
    continuation_counts,
    data_root,
    deterministic_row_order,
    load_reusable_candidates,
    pinned_inputs,
    read_jsonl_strict,
    read_master,
    validate_candidate_record,
    verify_pinned_inputs,
    verify_referenced_baseline,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_not_sealed,
    sha256_file,
)

PACKAGE_VERSION = "0.1.0"
OUTPUT_RECORDS = "screen_records.jsonl"
OUTPUT_BASELINE = "referenced_baseline.jsonl"
OUTPUT_MANIFEST = "run_manifest.json"
OUTPUT_VALIDATION = "validation.json"


def claim_output_dir(path: Path | str, create: bool) -> Path:
    path = Path(path)
    assert_not_sealed(path, "consolidation output directory")
    if path.exists():
        if not path.is_dir():
            raise ContinuationError(f"{path} exists and is not a directory")
        existing = sorted(p.name for p in path.iterdir())
        if existing:
            raise ContinuationError(
                f"{path} already holds {existing[:5]}. A consolidation is "
                f"written once, into an empty directory; a previous one is "
                f"never partially overwritten.")
    elif create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True,
                        help="the directory holding the continuation shards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    checks = Checklist(f"{PROTOCOL}_consolidation")

    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    assert_config_is_sealed_free(local_paths, "local path file")
    root = data_root(local_paths)
    pinned = verify_pinned_inputs(root, checks)

    inputs = load_development_inputs()
    corpus_rows = list(inputs.manifest_rows)
    corpus = {r["row_id"]: r for r in corpus_rows}
    ordered = deterministic_row_order(corpus_rows)

    master = read_master(pinned["master"]["path"])
    phase1 = read_jsonl_strict(pinned["phase1_records"]["path"])
    v1 = read_jsonl_strict(pinned["v1_records"]["path"])
    baseline = verify_referenced_baseline(master, corpus_rows, phase1, v1,
                                          checks)

    generation = GenerationConfig()
    generation.validate()
    locked = {
        "protocol": PROTOCOL, "arm_id": CANDIDATE_ARM_ID,
        "alpha_R": ALPHA_R, "alpha_S": 0.0,
        "intervention_type": "validated_two_site_repair",
        "budget_reference": BUDGET_REFERENCE,
        "states": list(CANDIDATE_STATES),
        "historical_layers": list(CANDIDATE_BLOCKS),
        "model": FROZEN_MODEL, "revision": FROZEN_REVISION,
        "generation_config_sha256": generation.fingerprint(),
        "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
        "direction_bank_sha256":
            build_candidate_record()["direction"]["artifact_sha256"],
        "dataset_sha256": inputs.dataset_sha256,
        "smoke": False,
    }

    # -- the pinned three ---------------------------------------------------
    reusable = load_reusable_candidates(pinned["v1_records"]["path"], locked,
                                        corpus, checks)

    # -- the shards ---------------------------------------------------------
    shard_dir = Path(args.shard_dir)
    if not shard_dir.is_dir():
        raise ContinuationError(f"{shard_dir} is not a directory")
    shard_files = sorted(shard_dir.glob("screen_records.shard*.jsonl"))
    checks.require("shard_files_found", bool(shard_files),
                   f"no shard records under {shard_dir}")

    generated: Dict[str, Dict[str, Any]] = {}
    provenance: Dict[str, str] = {}
    for path in shard_files:
        for index, record in enumerate(read_jsonl_strict(path)):
            where = f"{path.name}:{index}"
            rid = record.get("row_id")
            if rid in generated:
                raise ContinuationError(
                    f"{where}: {rid} was already consolidated from "
                    f"{provenance[rid]}. Two shards produced the same row, so "
                    f"they did not partition; a duplicate is refused rather "
                    f"than deduplicated.")
            if rid in reusable:
                raise ContinuationError(
                    f"{where}: {rid} is one of the {len(reusable)} rows the "
                    f"interrupted run completed and must not be regenerated")
            validate_candidate_record(record, locked, corpus, where)
            generated[rid] = record
            provenance[rid] = path.name

    checks.require(
        "generated_candidate_records", len(generated) == EXPECTED_NEW_CANDIDATES,
        f"{len(generated)} generated candidate record(s), expected "
        f"{EXPECTED_NEW_CANDIDATES}")
    checks.require(
        "reused_candidate_records", len(reusable) == EXPECTED_REUSABLE_CANDIDATES,
        f"{len(reusable)} reused record(s), expected "
        f"{EXPECTED_REUSABLE_CANDIDATES}")

    combined = {**reusable, **generated}
    missing = [rid for rid in ordered if rid not in combined]
    unknown = sorted(set(combined) - set(ordered))
    checks.require(
        "every_corpus_row_has_a_candidate", not missing,
        f"{len(missing)} row(s) have no candidate record, first "
        f"{missing[:3]}. A short arm is not a smaller result.")
    checks.require("no_unknown_row", not unknown,
                   f"{len(unknown)} record(s) are not corpus rows")
    checks.require("candidate_total", len(combined) == EXPECTED_ROWS,
                   f"{len(combined)} candidate record(s), expected "
                   f"{EXPECTED_ROWS}")

    counts = continuation_counts()
    print(f"[consolidate] reused   {len(reusable):>4} (interrupted run, pinned)")
    print(f"[consolidate] generated {len(generated):>3} "
          f"({len(shard_files)} shard file(s))")
    print(f"[consolidate] total     {len(combined):>3} candidate record(s)")
    print(f"[consolidate] baseline  {counts['referenced_baselines']:>3} "
          f"REFERENCED, {counts['generated_baselines']} generated")

    if args.dry_run:
        claim_output_dir(args.output_dir, create=False)
        print(f"[consolidate] dry run: every record validated; nothing written")
        print(f"[consolidate] output directory {args.output_dir} is claimable "
              f"and was deliberately NOT created")
        print(f"[consolidate] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2

    output_dir = claim_output_dir(args.output_dir, create=True)
    records_path = Path(output_dir) / OUTPUT_RECORDS
    with open(records_path, "w", encoding="utf-8") as handle:
        for rid in ordered:                      # deterministic order
            handle.write(json.dumps(combined[rid], sort_keys=True,
                                    ensure_ascii=False))
            handle.write("\n")

    # The referenced baseline, kept separate and labelled as referenced.
    baseline_path = Path(output_dir) / OUTPUT_BASELINE
    with open(baseline_path, "w", encoding="utf-8") as handle:
        for rid in ordered:
            handle.write(json.dumps({
                "row_id": rid,
                "row_class": corpus[rid]["final_ebh"],
                "prompt_sha256": corpus[rid]["prompt_sha256"],
                "arm_id": "baseline",
                "response_text": master[rid]["before_response_text"],
                "source": "referenced",
                "source_file": pinned["master"]["path"],
                "source_sha256": pinned["master"]["sha256"],
                "source_column": "before_response_text",
                "generated_by_this_run": False,
            }, sort_keys=True, ensure_ascii=False))
            handle.write("\n")

    manifest = {
        "schema": f"{SCHEMA}_consolidated", "protocol": PROTOCOL,
        "created_utc": utc_now(), "package_version": PACKAGE_VERSION,
        "arm": CANDIDATE_ARM_ID,
        **counts,
        "candidate_records_total": len(combined),
        "shard_files": [p.name for p in shard_files],
        "reused_rows": sorted(reusable),
        "referenced_baseline": baseline,
        "pinned_inputs": {k: v["sha256"] for k, v in pinned.items()},
        "judges_responses": False, "trains_a_controller": False,
        "controller_used": False,
        **locked,
        "outputs": {OUTPUT_RECORDS: sha256_file(records_path),
                    OUTPUT_BASELINE: sha256_file(baseline_path)},
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    write_json(Path(output_dir) / OUTPUT_MANIFEST, manifest)
    write_json(Path(output_dir) / OUTPUT_VALIDATION, checks.as_dict())
    print(f"[consolidate] wrote {output_dir}")
    for name in (OUTPUT_RECORDS, OUTPUT_BASELINE):
        print(f"[consolidate]   {name:<26} "
              f"{sha256_file(Path(output_dir) / name)}")
    return 0 if checks.all_passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
