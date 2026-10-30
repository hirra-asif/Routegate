#!/usr/bin/env python3
"""The static causal validation of the two-site candidate actuator.

One new arm, over all 480 development prompts:

    repair_015_blocks31_32
        delta(i, l) = 0.15 * 1 * ||h^(l)|| * d_R(l)   at historical blocks
        31 and 32 -- package states 32 and 33 -- at the final prompt token,
        during prefill only.

Historical block 33 is NOT edited. It failed the one-pass budget equivalence
test at the predeclared 1e-2 tolerance (7.455261e-02 maximum, 382 of 480 rows
over), and the runtime-input alternative was measured and is worse. See
``manifests/one_pass_budget_equivalence_closure_v1.json``.

This is a STATIC arm. No controller is trained, loaded or consulted; g_R is
fixed at 1, which is what makes this arm comparable to phase 1's repair_015 at
the same strength on the same corpus.

What this stage does not do
---------------------------
It does not judge. No response is scored, labelled or classified here; judging
is a separate stage through the already frozen protocol, and its rubric and
adjudication procedure are not touched. It reads no held-out material, trains
nothing, and emits no safety coefficient.

Reusing the phase-1 references
------------------------------
A baseline for these prompts already exists from phase 1. ``--reuse-baseline``
will read it **only** if every input, model, tokenizer, chat-template,
generation-configuration and direction-bank digest matches this run's; any
disagreement is refused rather than reconciled, because a baseline generated
under different settings is not a comparison, it is a confound.

Usage
-----
    python experiments/single_direction_controller/run_two_site_validation.py \\
        --direction-bank "$ROUTEGATE_DATA_DIR"/frozen/validated_direction_bank \\
        --output-dir .../two_site_validation_v1 --dry-run
"""
from __future__ import annotations

import argparse
import csv
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
    make_record,
    read_records,
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
    CANDIDATE_MANIFEST,
    CANDIDATE_STATES,
    CLOSURE_MANIFEST,
    DROPPED_BLOCKS,
    DROPPED_STATES,
    V2_SUMMARY_SHA256,
    TwoSiteError,
    build_two_site_plans,
    candidate_intervention,
    verify_records,
    verify_v2_result,
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
EXPECTED_ROWS = 480
PROTOCOL = "two_site_actuator_validation_v1"
BASELINE_ARM = "baseline"

OUTPUT_RECORDS = "screen_records.jsonl"
OUTPUT_MANIFEST = "run_manifest.json"
OUTPUT_VALIDATION = "validation.json"


def claim_output_dir(path: Path | str, create: bool) -> Path:
    """An empty or absent directory. A non-empty one is refused, never reused."""
    path = Path(path)
    assert_not_sealed(path, "two-site validation output directory")
    if path.exists():
        if not path.is_dir():
            raise TwoSiteError(f"{path} exists and is not a directory")
        existing = sorted(p.name for p in path.iterdir())
        if existing:
            raise TwoSiteError(
                f"{path} already holds {existing[:5]}. This result is written "
                f"once, into an empty directory; a previous run is never "
                f"partially overwritten. Name a new directory, or remove that "
                f"one by hand with a reason.")
    elif create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def assert_reusable_reference(reference: Path, expected: Dict[str, Any],
                              checks: Checklist) -> Dict[str, Any]:
    """A phase-1 reference is reusable only if every digest matches.

    Fail-closed and field by field. A baseline produced under a different
    model, revision, tokenizer, chat template, generation configuration,
    direction bank or corpus is not a cheaper comparison -- it is a confound
    with the same column names.
    """
    manifest_path = Path(reference) / OUTPUT_MANIFEST
    if not manifest_path.is_file():
        raise TwoSiteError(
            f"{manifest_path} does not exist; a reference run is reused only "
            f"through its own manifest, never by reading its records and "
            f"assuming where they came from")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, want in sorted(expected.items()):
        got = manifest.get(field)
        checks.require(
            f"reference_{field}_matches", got == want,
            f"the reference run records {field}={got!r} and this run is "
            f"{want!r}. A reference generated under different settings is not "
            f"reused: it is refused, and the arm is regenerated here.")
    checks.record("reference_reused", True,
                  f"{reference} matches on {len(expected)} pinned field(s)")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--reuse-baseline", type=Path, default=None,
                        help="a completed run whose baseline arm may be reused "
                             "IF every pinned digest matches; refused otherwise")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--shard-index", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every input, hash, row, state and site; "
                             "load no model and write nothing")
    return parser


def run(args: argparse.Namespace) -> int:
    checks = Checklist(PROTOCOL)

    # ---- the records this arm exists because of --------------------------
    verify_records(checks=checks)
    outcome = verify_v2_result(checks=checks)
    checks.record(
        "bound_to_v2_summary", True,
        f"{V2_SUMMARY_SHA256} -- the run whose per-block verdicts chose these "
        f"sites")
    candidate = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    checks.require(
        "candidate_is_declared_not_run",
        candidate["status"] == "DECLARED_NOT_RUN",
        f"the candidate manifest records status {candidate['status']!r}")

    intervention = candidate_intervention()
    checks.require(
        "sites_are_the_two_that_passed",
        tuple(intervention["historical_blocks"]) == CANDIDATE_BLOCKS == (31, 32),
        f"the arm edits historical blocks "
        f"{intervention['historical_blocks']}, the candidate pins "
        f"{list(CANDIDATE_BLOCKS)}")
    checks.require(
        "block_33_is_not_edited",
        33 not in intervention["historical_blocks"]
        and 34 not in intervention["package_states_edited"],
        f"historical block 33 (package state 34) is in the arm; it failed the "
        f"one-pass budget test and the closure removed it")
    checks.record(
        "numbering", True,
        f"historical blocks {list(CANDIDATE_BLOCKS)} -> package states "
        f"{list(CANDIDATE_STATES)}, budgeted against states "
        f"{list(CANDIDATE_BUDGET_STATES)}; dropped block(s) "
        f"{list(DROPPED_BLOCKS)} = state(s) {list(DROPPED_STATES)}")
    checks.require("alpha_S_is_zero", intervention["alpha_S"] == 0.0,
                   "no safety coefficient is emitted or applied")
    checks.require("alpha_R_is_the_frozen_strength",
                   intervention["alpha_R"] == ALPHA_R == 0.15,
                   f"alpha_R is {intervention['alpha_R']}, frozen at {ALPHA_R}")
    checks.require("g_R_is_one_for_the_static_arm",
                   intervention["g_R"] == WORST_CASE_G_R == 1.0,
                   "the causal validation is static; g_R is fixed at 1")
    checks.require("one_forward_pass", intervention["forward_passes"] == 1,
                   "the candidate is driven from one prefill forward pass")
    checks.require("no_controller_is_used",
                   candidate["controller"]["used_in_this_validation"] is False,
                   "no controller is trained, loaded or consulted here")

    # ---- inputs, hash-gated, before anything is loaded --------------------
    claim_output_dir(args.output_dir, create=False)
    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    assert_config_is_sealed_free(local_paths, "local path file")
    inputs = load_development_inputs()
    rows = list(inputs.manifest_rows)
    checks.require("corpus_row_count", len(rows) == EXPECTED_ROWS,
                   f"{len(rows)} row(s), expected {EXPECTED_ROWS}")
    sealed = [r["row_id"] for r in rows if looks_sealed(r["row_id"])]
    checks.require("no_sealed_row_id", not sealed,
                   f"{len(sealed)} row id(s) read as held-out material")
    checks.record("dataset_sha256", True, inputs.dataset_sha256)

    from routegate.dualsteer import load_validated_bank
    from routegate.twosite import build_candidate_record
    expected_bank = build_candidate_record()["direction"]["artifact_sha256"]
    bank_dir = assert_not_sealed(Path(args.direction_bank), "direction bank")
    bank = load_validated_bank(bank_dir, expected_bank, checks)

    generation = GenerationConfig()
    generation.validate()
    locked = {
        "model": FROZEN_MODEL, "revision": FROZEN_REVISION,
        "generation_config_sha256": generation.fingerprint(),
        "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
        "direction_bank_sha256": expected_bank,
        "dataset_sha256": inputs.dataset_sha256,
    }
    for field, value in sorted(locked.items()):
        checks.record(field, True, str(value))

    reference_manifest = None
    if args.reuse_baseline is not None:
        reference_manifest = assert_reusable_reference(
            args.reuse_baseline, locked, checks)
    generations = len(rows) * (1 if reference_manifest else 2)

    print(f"[twosite] arm {CANDIDATE_ARM_ID}")
    print(f"[twosite] historical blocks {list(CANDIDATE_BLOCKS)} -> package "
          f"states {list(CANDIDATE_STATES)}")
    print(f"[twosite] budget reference states {list(CANDIDATE_BUDGET_STATES)} "
          f"({BUDGET_REFERENCE})")
    print(f"[twosite] block 33 (state 34) is NOT edited")
    print(f"[twosite] alpha_R={ALPHA_R} g_R={WORST_CASE_G_R} alpha_S=0")
    print(f"[twosite] {len(rows)} development prompt(s)")
    print(f"[twosite] baseline: "
          f"{'reused from ' + str(args.reuse_baseline) if reference_manifest else 'generated here'}")
    print(f"[twosite] generations this run: {generations}")

    if args.dry_run:
        print(f"[twosite] dry run: every input, hash, row, site and record "
              f"validated; no model loaded, no token generated, nothing "
              f"written")
        print(f"[twosite] output directory {args.output_dir} is claimable and "
              f"was deliberately NOT created")
        print(f"[twosite] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2

    output_dir = claim_output_dir(args.output_dir, create=True)

    # ---- generation ------------------------------------------------------
    from routegate.hooks import PatchController
    from routegate.runtime import (capture_prompt_states, generate_response,
                                   load_model_and_tokenizer)

    print(f"[twosite] loading {FROZEN_MODEL} @ {FROZEN_REVISION}")
    model, tokenizer = load_model_and_tokenizer(generation)
    model.eval()

    records_path = output_dir / OUTPUT_RECORDS
    done = {(r["row_id"], r["arm_id"]) for r in read_records(records_path,
                                                             strict=False)} \
        if args.resume else set()

    arms = [CANDIDATE_ARM_ID] if reference_manifest else [BASELINE_ARM,
                                                          CANDIDATE_ARM_ID]
    controller = PatchController(model, list(CANDIDATE_STATES),
                                 budget_reference=BUDGET_REFERENCE)
    written = errors = 0
    try:
        for index, row in enumerate(rows, start=1):
            rid = row["row_id"]
            capture: Dict[int, Any] = {}
            for arm_id in arms:
                if (rid, arm_id) in done:
                    continue
                per_state: Dict[str, Any] = {}
                try:
                    if arm_id == BASELINE_ARM:
                        generated = generate_response(
                            model, tokenizer, row["prompt_text"], generation,
                            controller=controller, mode="off", example_id=rid)
                        status, hooks_expected = "baseline", 0
                    else:
                        if not capture:
                            capture_prompt_states(
                                model, tokenizer, row["prompt_text"],
                                controller, capture_name="target",
                                example_id=rid, config=generation)
                            capture = dict(controller.capture_for("target"))
                        plans = build_two_site_plans(
                            bank, capture, alpha=ALPHA_R, g_r=WORST_CASE_G_R,
                            blocks=CANDIDATE_BLOCKS)
                        generated = generate_response(
                            model, tokenizer, row["prompt_text"], generation,
                            controller=controller, mode="patch", plans=plans,
                            example_id=rid)
                        per_state = {
                            str(state): {
                                **plans[state].diagnostics(),
                                "repair_direction_sha256":
                                    bank.direction_sha256("repair_direction",
                                                          state),
                            } for state in sorted(plans)}
                        status = ("ok" if all(p.status == "ok"
                                              for p in plans.values())
                                  else "zero_delta")
                        hooks_expected = len(plans)
                    error_message = ""
                except Exception as exc:                     # noqa: BLE001
                    generated = {"response_text": "", "prompt_token_count": 0,
                                 "completion_token_count": 0,
                                 "eos_reached": False,
                                 "hit_max_new_tokens": False,
                                 "elapsed_seconds": 0.0,
                                 "applied_by_state": {}}
                    status = "error"
                    error_message = f"{type(exc).__name__}: {exc}"
                    hooks_expected, per_state = 0, {}
                    errors += 1
                    print(f"      ERROR {error_message}", file=sys.stderr)

                append_record(records_path, make_record(
                    row_id=rid, row_class=row["final_ebh"], arm_id=arm_id,
                    alpha_R=0.0 if arm_id == BASELINE_ARM else ALPHA_R,
                    alpha_S=0.0, protocol=PROTOCOL, smoke=False,
                    states=list(CANDIDATE_STATES),
                    historical_layers=list(CANDIDATE_BLOCKS),
                    budget_reference=BUDGET_REFERENCE,
                    intervention_type="validated_two_site_repair",
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
            if index % 25 == 0 or index == len(rows):
                print(f"[twosite] {index}/{len(rows)} prompts, {written} "
                      f"record(s)", flush=True)
    finally:
        controller.close()

    manifest = {
        "schema": "routegate/two_site_validation_run/1",
        "protocol": PROTOCOL, "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "arm": CANDIDATE_ARM_ID,
        "intervention": intervention,
        "bound_to_v2_summary_sha256": V2_SUMMARY_SHA256,
        "closure": CLOSURE_MANIFEST.name,
        "candidate": CANDIDATE_MANIFEST.name,
        "rows": len(rows), "records_written": written, "errors": errors,
        "expected_generations": generations,
        "baseline": ("reused: " + str(args.reuse_baseline)
                     if reference_manifest else "generated here"),
        "judges_responses": False,
        "trains_a_controller": False,
        "controller_used": False,
        **locked,
        "environment": environment_record(PACKAGE_VERSION, model_loaded=True,
                                          generation_performed=True),
        "checklist": checks.as_dict(),
    }
    write_json(output_dir / OUTPUT_MANIFEST, manifest)
    write_json(output_dir / OUTPUT_VALIDATION, checks.as_dict())
    print(f"[twosite] wrote {output_dir}")
    print(f"[twosite]   {OUTPUT_RECORDS:<22} {sha256_file(records_path)}")
    print(f"[twosite] {written} record(s), {errors} error(s)")
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
