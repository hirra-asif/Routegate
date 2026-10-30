#!/usr/bin/env python3
"""Continue the two-site causal validation: the 477 missing candidates.

The interrupted run wrote three candidate records and stopped. This runner
finishes the arm without regenerating anything that already exists and without
touching what the interrupted run left behind.

    arm            repair_015_blocks31_32
    edit           delta(i, l) = 0.15 * 1 * ||h^(l)|| * d_R(l)
    sites          historical blocks 31 and 32 = package states 32 and 33,
                   budgeted against states 31 and 32
    baselines      REFERENCED from the frozen static-steering master; none is
                   generated here
    to generate    477 candidates (480 - the 3 the interrupted run completed)

Two defects in ``run_two_site_validation.py``, found by audit and corrected
only here:

* its output-directory check ran before ``--resume`` was consulted, and a
  resumable directory is non-empty by definition, so ``--resume`` was
  unreachable;
* ``--shard-index`` and ``--shard-count`` were declared and never read, so
  every shard would have generated all 480 rows and collided.

The old runner and its records are left exactly as they are. They are the
evidence of what happened.

Usage
-----
    python experiments/single_direction_controller/run_two_site_continuation.py \\
        --direction-bank .../validated_direction_bank \\
        --output-dir .../two_site_validation_v2 \\
        --shard-index 1 --shard-count 8 --dry-run
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
from routegate.dualsteer import (                                   # noqa: E402
    BUDGET_REFERENCE,
    append_record,
    load_validated_bank,
    make_record,
)
from routegate.onepassbudget import ALPHA_R, WORST_CASE_G_R         # noqa: E402
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
    CANDIDATE_BUDGET_STATES,
    CANDIDATE_STATES,
    build_candidate_record,
    build_two_site_plans,
    verify_records,
)
from routegate.twositecontinuation import (                         # noqa: E402
    EXPECTED_NEW_CANDIDATES,
    EXPECTED_ROWS,
    PROTOCOL,
    SCHEMA,
    REUSABLE_CANDIDATE_ROWS,
    ContinuationError,
    assert_shard_is_not_complete,
    assert_shards_partition,
    continuation_counts,
    data_root,
    deterministic_row_order,
    load_reusable_candidates,
    pinned_inputs,
    read_jsonl_strict,
    read_master,
    shard_rows,
    validate_existing_shard,
    verify_pinned_inputs,
    verify_referenced_baseline,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_not_sealed,
    looks_sealed,
    sha256_file,
)

PACKAGE_VERSION = "0.1.0"
OUTPUT_MANIFEST = "run_manifest.json"
OUTPUT_VALIDATION = "validation.json"


def shard_records_name(index: int, count: int) -> str:
    """One file per shard, so no two processes ever append to the same one."""
    return f"screen_records.shard{index:02d}of{count:02d}.jsonl"


def prepare_output_dir(path: Path | str, records_name: str,
                       resume: bool) -> Path:
    """Claim the directory, honouring --resume.

    The corrected ordering. Without ``--resume`` the directory must be absent
    or empty, as before. With it, a directory holding **this shard's own
    records file and nothing else unexpected** is accepted -- which is the case
    a restart is for, and which the previous runner refused before it ever
    looked at the flag.
    """
    path = Path(path)
    assert_not_sealed(path, "two-site continuation output directory")
    if not path.exists():
        return path
    if not path.is_dir():
        raise ContinuationError(f"{path} exists and is not a directory")
    existing = sorted(p.name for p in path.iterdir())
    if not existing:
        return path
    if not resume:
        raise ContinuationError(
            f"{path} already holds {existing[:5]} and --resume was not given. "
            f"A result is written once, into an empty directory. Pass "
            f"--resume to continue this shard after a crash, or name a new "
            f"directory.")
    allowed = {records_name, OUTPUT_MANIFEST, OUTPUT_VALIDATION}
    unexpected = sorted(set(existing) - allowed)
    if unexpected:
        raise ContinuationError(
            f"{path} holds {unexpected}, which this shard did not write. "
            f"Resuming into a directory another run is using would interleave "
            f"two experiments in one place.")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=1,
                        help="1-based; with --shard-count it PARTITIONS the "
                             "deterministic 480-row order")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="continue this shard after a crash. Existing "
                             "records are validated before being skipped.")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every input, hash, row, site, shard and "
                             "reused record; load no model, write nothing")
    return parser


def run(args: argparse.Namespace) -> int:
    checks = Checklist(PROTOCOL)

    # ---- the declarations this arm exists because of ---------------------
    verify_records(checks=checks)
    candidate = build_candidate_record()
    expected_bank = candidate["direction"]["artifact_sha256"]

    # ---- the four pinned artifacts ---------------------------------------
    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    assert_config_is_sealed_free(local_paths, "local path file")
    root = data_root(local_paths)
    pinned = verify_pinned_inputs(root, checks)

    # ---- the corpus, and the master revalidated against it ---------------
    inputs = load_development_inputs()
    corpus_rows = list(inputs.manifest_rows)
    corpus = {r["row_id"]: r for r in corpus_rows}
    sealed = [r for r in corpus if looks_sealed(r)]
    checks.require("no_sealed_row_id", not sealed,
                   f"{len(sealed)} row id(s) read as held-out material")

    master = read_master(pinned["master"]["path"])
    phase1 = read_jsonl_strict(pinned["phase1_records"]["path"])
    v1 = read_jsonl_strict(pinned["v1_records"]["path"])
    baseline = verify_referenced_baseline(master, corpus_rows, phase1, v1,
                                          checks)

    # ---- the deterministic order and a genuine partition -----------------
    ordered = deterministic_row_order(corpus_rows)
    checks.require("row_order_is_the_whole_corpus",
                   len(ordered) == EXPECTED_ROWS,
                   f"{len(ordered)} row(s) in the order, expected "
                   f"{EXPECTED_ROWS}")
    assert_shards_partition(ordered, int(args.shard_count), checks)
    shard = shard_rows(ordered, int(args.shard_index), int(args.shard_count))

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
        "direction_bank_sha256": expected_bank,
        "dataset_sha256": inputs.dataset_sha256,
        "smoke": False,
    }

    # ---- the three records the interrupted run completed ------------------
    reusable = load_reusable_candidates(pinned["v1_records"]["path"], locked,
                                        corpus, checks)
    todo = [rid for rid in shard if rid not in reusable]
    skipped_reused = [rid for rid in shard if rid in reusable]
    checks.record(
        "reused_candidates_in_this_shard", True,
        f"{len(skipped_reused)} of this shard's {len(shard)} row(s) are "
        f"already complete in the interrupted run and are not regenerated")

    # ---- the output directory, AFTER --resume is known -------------------
    records_name = shard_records_name(int(args.shard_index),
                                      int(args.shard_count))
    output_dir = prepare_output_dir(args.output_dir, records_name,
                                    bool(args.resume))
    records_path = Path(output_dir) / records_name
    done = validate_existing_shard(records_path, todo, locked, corpus, checks) \
        if args.resume else {}
    if args.resume and records_path.exists():
        assert_shard_is_not_complete(done, todo)
    remaining = [rid for rid in todo if rid not in done]

    counts = continuation_counts()
    print(f"[cont] arm {CANDIDATE_ARM_ID}")
    print(f"[cont] historical blocks {list(CANDIDATE_BLOCKS)} -> package "
          f"states {list(CANDIDATE_STATES)}, budgeted against "
          f"{list(CANDIDATE_BUDGET_STATES)}")
    print(f"[cont] alpha_R={ALPHA_R} g_R={WORST_CASE_G_R} alpha_S=0")
    print(f"[cont] baselines: {counts['referenced_baselines']} REFERENCED "
          f"from the master, {counts['generated_baselines']} generated here")
    print(f"[cont]   {baseline['independently_reproduced_count']} of "
          f"{EXPECTED_ROWS} independently reproduced byte-for-byte; "
          f"{baseline['referenced_on_provenance_only']} on provenance")
    print(f"[cont] shard {args.shard_index}/{args.shard_count}: "
          f"{len(shard)} row(s)")
    print(f"[cont]   already complete in the interrupted run: "
          f"{len(skipped_reused)}")
    print(f"[cont]   already written in this shard : {len(done)}")
    print(f"[cont]   to generate now               : {len(remaining)}")
    print(f"[cont] arm total when every shard finishes: "
          f"{counts['generated_candidate_records']} generated + "
          f"{counts['reused_candidate_records']} reused = {EXPECTED_ROWS}")

    if args.dry_run:
        print(f"[cont] dry run: every input, hash, row, order, shard, site and "
              f"reused record validated; no model loaded, no token generated, "
              f"nothing written")
        print(f"[cont] output directory {args.output_dir} is claimable and was "
              f"deliberately NOT created")
        print(f"[cont] {checks.as_dict()['checks_run']} check(s), all passed: "
              f"{checks.all_passed}")
        return 0 if checks.all_passed else 2

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- generation: candidates only --------------------------------------
    from routegate.hooks import PatchController
    from routegate.runtime import (capture_prompt_states, generate_response,
                                   load_model_and_tokenizer)

    bank = load_validated_bank(Path(args.direction_bank), expected_bank, checks)
    print(f"[cont] loading {FROZEN_MODEL} @ {FROZEN_REVISION}")
    model, tokenizer = load_model_and_tokenizer(generation)
    model.eval()

    controller = PatchController(model, list(CANDIDATE_STATES),
                                 budget_reference=BUDGET_REFERENCE)
    written = errors = 0
    try:
        for position, rid in enumerate(remaining, start=1):
            row = corpus[rid]
            per_state: Dict[str, Any] = {}
            try:
                capture_prompt_states(model, tokenizer, row["prompt_text"],
                                      controller, capture_name="target",
                                      example_id=rid, config=generation)
                plans = build_two_site_plans(
                    bank, dict(controller.capture_for("target")),
                    alpha=ALPHA_R, g_r=WORST_CASE_G_R,
                    blocks=CANDIDATE_BLOCKS)
                generated = generate_response(
                    model, tokenizer, row["prompt_text"], generation,
                    controller=controller, mode="patch", plans=plans,
                    example_id=rid)
                per_state = {
                    str(state): {
                        **plans[state].diagnostics(),
                        "repair_direction_sha256": bank.direction_sha256(
                            "repair_direction", state),
                    } for state in sorted(plans)}
                status = ("ok" if all(p.status == "ok" for p in plans.values())
                          else "zero_delta")
                hooks_expected = len(plans)
                error_message = ""
            except Exception as exc:                         # noqa: BLE001
                generated = {"response_text": "", "prompt_token_count": 0,
                             "completion_token_count": 0, "eos_reached": False,
                             "hit_max_new_tokens": False,
                             "elapsed_seconds": 0.0, "applied_by_state": {}}
                status, error_message = "error", f"{type(exc).__name__}: {exc}"
                hooks_expected, per_state = 0, {}
                errors += 1
                print(f"      ERROR {rid}: {error_message}", file=sys.stderr)

            append_record(records_path, make_record(
                row_id=rid, row_class=row["final_ebh"],
                prompt_sha256=row["prompt_sha256"], status=status,
                error_message=error_message, per_state=per_state,
                hooks_expected=hooks_expected,
                hooks_fired=len(generated.get("applied_by_state") or {}),
                applied_by_state=generated.get("applied_by_state") or {},
                created_utc=utc_now(), **locked,
                **{k: generated[k] for k in
                   ("response_text", "prompt_token_count",
                    "completion_token_count", "eos_reached",
                    "hit_max_new_tokens", "elapsed_seconds")
                   if k in generated}))
            written += 1
            if position % 25 == 0 or position == len(remaining):
                print(f"[cont] {position}/{len(remaining)} generated",
                      flush=True)
    finally:
        controller.close()

    manifest = {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "continuation_of": "two_site_validation_v1",
        "created_utc": utc_now(), "package_version": PACKAGE_VERSION,
        "arm": CANDIDATE_ARM_ID,
        "shard": {"index": int(args.shard_index),
                  "count": int(args.shard_count),
                  "rows": len(shard), "generated_here": written,
                  "reused_from_the_interrupted_run": len(skipped_reused),
                  "records_file": records_name},
        **counts,
        "referenced_baseline": baseline,
        "pinned_inputs": {k: v["sha256"] for k, v in pinned.items()},
        "judges_responses": False, "trains_a_controller": False,
        "controller_used": False, "errors": errors,
        **locked,
        "environment": environment_record(PACKAGE_VERSION, model_loaded=True,
                                          generation_performed=True),
        "checklist": checks.as_dict(),
    }
    write_json(Path(output_dir) / OUTPUT_MANIFEST, manifest)
    write_json(Path(output_dir) / OUTPUT_VALIDATION, checks.as_dict())
    print(f"[cont] wrote {output_dir}")
    print(f"[cont]   {records_name} {sha256_file(records_path)}")
    print(f"[cont] {written} generated, {errors} error(s)")
    return 0 if errors == 0 and checks.all_passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
