#!/usr/bin/env python3
"""The rotational H - E development screen. Generation only.

A NEW EXPERIMENTAL CANDIDATE. Nothing this runner produces has been judged,
and nothing in this repository describes rotational H - E as validated.

What it runs
------------
The frozen ``rotation_safety_direction_v2`` inventory: the 18 H rows the phase-2
selection returned, at eight arms, for 144 generations::

    18 rows x 4 rotation-only arms   beta 0.05/0.10/0.15/0.20, repair off   72
    18 rows x 4 combined arms        alpha_R = 0.15 then the same four betas 72
                                                                     total 144

``baseline`` and ``repair_015`` for these rows already exist in the frozen
phase-1 records and are NOT regenerated. The runner refuses a record carrying
either arm id.

The edit, at historical layers 31/32/33 -- states 32/33/34 -- at the final
prompt token, prefill only. At each site, in this order::

    h'    = h + ||block_input|| * alpha_R * d_R,l        (combined arms only)
    h_rot = ||h'|| * (cos(beta*theta) x + sin(beta*theta) unit(y - <x,y> x))

with ``x = h'/||h'||``, ``y = d_S,l/||d_S,l||`` and
``theta = arccos(<x, y>)``. The rotation preserves the norm; the direction is
never rotated; each layer uses its own ``d_S,l``; one beta is shared across all
three layers. Rotation and addition do not commute, so ``repair_then_rotation``
is recorded on every record.

The repair half is built by ``routegate.dualsteer.build_plans`` at the existing
predeclared ``repair_015`` arm. Nothing here reimplements the repair edit or
re-derives its budget convention.

Order of the gates
------------------
Everything that can fail without a GPU fails without a GPU. The completed
artifacts' digests, the predeclaration's self-digest, the phase-2 selection,
the configuration, the direction bank's SHA256, the bank's own statement that
``d_S = H - E``, the corpus digest, the row ids, the prompt hashes, the whole
inventory, the rotation operator's own invariants on a synthetic vector and the
resume file's compatibility are all settled first; only then is a model loaded.

Usage
-----
    # what it would do, loading nothing and creating no output directory
    python experiments/rotation_safety_direction/run_rotation_screen.py \\
        --direction-bank <bank> --output-dir <new dir> --dry-run

    # one shard of an array job, resumable
    python experiments/rotation_safety_direction/run_rotation_screen.py \\
        --direction-bank <bank> --output-dir <dir>/shard_1 \\
        --shard-index 1 --shard-count 4 --resume
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

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.data import (                                        # noqa: E402
    SOURCE_COLUMN,
    WITHHELD_LABEL,
    load_input_manifest,
    load_local_paths,
)
from routegate.directionbank import BANK_STATES, HISTORICAL_LAYERS  # noqa: E402
from routegate.dualsteer import BUDGET_REFERENCE, CLASS_H           # noqa: E402
from routegate.phase2 import verify_phase2_selection                # noqa: E402
from routegate.rotation import (                                    # noqa: E402
    ROTATION_TOLERANCES,
    TOLERANCES_VERSION,
    assert_rotation_invariants,
    rotate_toward,
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
    NEW_GENERATIONS,
    N_ROWS,
    PROTOCOL,
    REFERENCE_ARM_IDS,
    REFERENCE_RECORDS,
    SCREEN_SCHEMA,
    SELECTED_ROWS,
    SELECTED_ROWS_SHA256,
    RotationScreenError,
    append_record,
    assert_config_matches,
    assert_inventory,
    assert_safety_direction_is_h_minus_e,
    build_inventory,
    build_site_plans,
    claim_output_dir,
    make_record,
    plan_work,
    read_records,
    record_key,
    rotation_controller,
    safety_direction_digests,
    selection_digest_holds,
    shard,
    verify_consistency,
)
from routegate.dualsteer import load_validated_bank                 # noqa: E402
from routegate.runrecord import (                                   # noqa: E402
    environment_record,
    utc_now,
    write_json,
)
from routegate.runtime import (                                     # noqa: E402
    FROZEN_CHAT_TEMPLATE_SHA256,
    FROZEN_TOKENIZER_CONFIG_SHA256,
    GenerationConfig,
    capture_prompt_states,
    generate_response,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    assert_hash,
    assert_not_sealed,
    assert_not_stale,
    sha256_text,
)

RECORDS_FILENAME = "rotation_records.jsonl"
MANIFEST_FILENAME = "run_manifest.json"


def load_selected_rows(dataset_path: Path, expected_sha256: str,
                       checks: Checklist) -> List[Dict[str, str]]:
    """The 18 bound H rows, with their prompt digests.

    Read from the accepted corpus and checked against its recorded digest
    before anything else looks at a row. The row set is the phase-2 selection
    and nothing widens it: a corpus row that is not in the selection is not
    read into the inventory at all.
    """
    path = assert_not_stale(Path(dataset_path))
    assert_not_sealed(path, "rotation screen dataset")
    assert_hash(path, expected_sha256, "dataset")
    with open(path, newline="", encoding="utf-8") as handle:
        corpus = list(csv.DictReader(handle))

    bound = set(SELECTED_ROWS)
    found: Dict[str, Dict[str, str]] = {}
    for row in corpus:
        if row["row_id"] not in bound:
            continue
        if row["final_ebh"] != WITHHELD_LABEL:
            raise RotationScreenError(
                f"{row['row_id']} is labelled {row['final_ebh']!r} in the "
                f"corpus and the selection calls it an H row. A row whose "
                f"class has moved is not steered on the strength of a stale "
                f"selection.")
        if row["row_id"] in found:
            raise RotationScreenError(
                f"duplicate row id {row['row_id']!r} in the corpus")
        prompt = row["prompt_text"]
        found[row["row_id"]] = {
            "row_id": row["row_id"], "row_class": CLASS_H,
            "prompt_text": prompt, "prompt_sha256": sha256_text(prompt),
            "source_family": row[SOURCE_COLUMN],
        }

    missing = sorted(bound - set(found))
    if missing:
        raise RotationScreenError(
            f"{len(missing)} selected row(s) are not in the accepted corpus "
            f"(first {missing[0]!r}). The selection is bound to that corpus; a "
            f"short inventory is refused rather than run.")
    rows = [found[row_id] for row_id in SELECTED_ROWS]
    checks.record("corpus_rows", True,
                  f"{len(rows)} of {len(corpus)} corpus row(s), all "
                  f"{WITHHELD_LABEL}, matching the frozen selection")
    return rows


def parse_selection(spec: str, flag: str) -> set:
    """A comma-separated filter value as a set, refusing an empty element."""
    if not spec:
        return set()
    parts = [part.strip() for part in spec.split(",")]
    if any(not part for part in parts):
        raise RotationScreenError(
            f"{flag}={spec!r} has an empty element; give a comma-separated "
            f"list with no stray commas")
    return set(parts)


def operator_selftest(checks: Checklist) -> Dict[str, Any]:
    """The rotation operator's own claims, on a synthetic vector, before a GPU.

    Not a substitute for the test suite: it is the same arithmetic the run is
    about to perform, executed once on this machine, in this container, with
    this torch, so a numerical stack that cannot hold the invariants is caught
    here rather than 40 generations into an array job.
    """
    import torch

    generator = torch.Generator().manual_seed(20260830)
    h = torch.randn(2048, generator=generator, dtype=torch.float32) * 7.5
    direction = torch.randn(2048, generator=generator, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction)

    worst = 0.0
    for beta in (0.05, 0.10, 0.15, 0.20):
        outcome = rotate_toward(h, direction, beta)
        assert_rotation_invariants(outcome.diagnostics, ROTATION_TOLERANCES,
                                   label=f"selftest beta={beta}")
        worst = max(worst,
                    float(outcome.diagnostics["relative_norm_error_float32"]))
    identity = rotate_toward(h, direction, 0.0)
    if not bool(torch.equal(identity.tensor, h)):
        raise RotationScreenError(
            "beta = 0 did not return the original tensor bitwise")
    checks.record(
        "rotation_operator_selftest", True,
        f"4 beta value(s) on a synthetic 2048-vector; worst relative float32 "
        f"norm error {worst:.3e}, tolerance "
        f"{ROTATION_TOLERANCES.norm_preservation_rtol_float32:.3e}; beta = 0 "
        f"is bitwise identity")
    return {"worst_relative_norm_error_float32": worst,
            "betas": [0.05, 0.10, 0.15, 0.20]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True,
                        help="the validated direction bank directory")
    parser.add_argument("--expect-bank-sha256", type=str, default=None,
                        help="the pinned validated_direction_bank.npz digest; "
                             "defaults to the one in the frozen configuration")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predeclaration", type=Path, default=PREDECLARATION)
    parser.add_argument("--only-arm", dest="only_arm", type=str, default="",
                        help="restrict to these arm ids, comma separated. "
                             "Narrowing makes the run a smoke test.")
    parser.add_argument("--only-row-id", dest="only_row_id", type=str,
                        default="",
                        help="restrict to these row ids, comma separated. "
                             "Narrowing makes the run a smoke test.")
    parser.add_argument("--shard-index", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many generations; 0 means all")
    parser.add_argument("--device", type=str, default="auto",
                        help="device_map for the pinned model, e.g. auto or "
                             "cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and validate everything; load no model, "
                             "write nothing, and create no output directory")
    return parser


def run(args: argparse.Namespace) -> int:
    checks = Checklist("rotation_safety_direction_screen")

    # ---- the completed record is untouched -------------------------------
    #
    # Before the configuration is opened and before the bank is hashed: this
    # screen adds work beside the completed phase-1, phase-2, probe,
    # controller, static two-site and judging artifacts and never edits them.
    preserved = verify_preserved_artifacts(checks=checks)

    # ---- the predeclaration, and what it is bound to ----------------------
    frozen = verify_predeclaration(args.predeclaration, checks=checks)
    predeclaration = frozen["committed"]
    predeclaration_digest = str(frozen["sha256"])
    verify_phase2_selection(checks=checks)
    selection_digest_holds()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert_config_matches(config, checks=checks)

    expected_bank = (args.expect_bank_sha256
                     or config["direction_bank"]["artifact_sha256"])

    # ---- inputs, before anything is loaded -------------------------------
    input_manifest = load_input_manifest()
    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    dataset_path = Path(local_paths["dataset"])
    dataset_sha = input_manifest["dataset"]["sha256"]

    bank = load_validated_bank(args.direction_bank, expected_bank, checks)
    if bank.manifest_sha256 != config["direction_bank"]["manifest_sha256"]:
        raise RotationScreenError(
            f"the bank's manifest hashes to {bank.manifest_sha256} and this "
            f"screen is pinned to "
            f"{config['direction_bank']['manifest_sha256']}. The manifest is "
            f"where the H - E subtraction order is read from; a different one "
            f"is refused, not adopted.")
    direction_metadata = assert_safety_direction_is_h_minus_e(bank, checks)
    safety_digests = safety_direction_digests(bank)
    pinned_safety = (predeclaration["frozen_inputs"]["direction_bank"]
                     ["safety_direction_sha256"])
    if safety_digests != pinned_safety:
        raise RotationScreenError(
            f"the per-layer H - E digests are {safety_digests} and the "
            f"predeclaration pins {pinned_safety}. A bank that hashes "
            f"correctly and carries a different vector at one layer is still "
            f"a different experiment.")
    checks.record("safety_direction_digests", True,
                  ", ".join(f"state {state} {digest[:12]}"
                            for state, digest in sorted(safety_digests.items())))

    rows = load_selected_rows(dataset_path, dataset_sha, checks)
    inventory = build_inventory(rows)
    counts = assert_inventory(inventory, checks)

    # ---- the operator's own claims, on this numerical stack ---------------
    selftest = operator_selftest(checks)

    # Every filter narrows the frozen inventory, and all of them are applied
    # before the shard split and before --limit. --limit is a truncation, not
    # a selection.
    only_arm = parse_selection(args.only_arm, "--only-arm")
    only_row = parse_selection(args.only_row_id, "--only-row-id")
    reference_named = sorted(only_arm & set(REFERENCE_ARM_IDS))
    if reference_named:
        raise RotationScreenError(
            f"--only-arm names {reference_named}, which are INHERITED "
            f"reference arms. They already exist in the frozen phase-1 records "
            f"for these rows and this screen reads them; it does not "
            f"regenerate them.")
    unknown_arms = sorted(only_arm - set(ARM_BY_ID_ROTATION))
    if unknown_arms:
        raise RotationScreenError(
            f"--only-arm names {unknown_arms}, which are not arms of this "
            f"screen. Its arms are {sorted(ARM_BY_ID_ROTATION)}.")
    unknown_rows = sorted(only_row - set(SELECTED_ROWS))
    if unknown_rows:
        raise RotationScreenError(
            f"--only-row-id names {len(unknown_rows)} row(s) that are not in "
            f"the frozen {N_ROWS}-row selection (first {unknown_rows[0]!r})")

    selected = list(inventory)
    if only_arm:
        selected = [item for item in selected if item["arm_id"] in only_arm]
    if only_row:
        selected = [item for item in selected if item["row_id"] in only_row]
    if not selected:
        raise RotationScreenError(
            "the filters select no generation; nothing would be run and an "
            "empty output is not a result")
    filtered = len(selected)
    selected = shard(selected, args.shard_index, args.shard_count)
    checks.record("filters", True,
                  f"arms {sorted(only_arm) or 'all'}, rows "
                  f"{sorted(only_row) or 'all'}: {filtered} of "
                  f"{counts['total']} generation(s)")
    checks.record("shard", True,
                  f"{args.shard_index}/{args.shard_count}: {len(selected)} of "
                  f"{filtered} filtered generation(s)")

    smoke_reasons: List[str] = []
    if only_arm:
        smoke_reasons.append(f"--only-arm {sorted(only_arm)}")
    if only_row:
        smoke_reasons.append(f"--only-row-id {sorted(only_row)}")
    if args.limit:
        smoke_reasons.append(f"--limit {args.limit}")
    smoke = bool(smoke_reasons)

    generation = GenerationConfig(device_map=args.device)
    generation.validate()
    generation_sha = generation.fingerprint()

    run_locked = {
        "direction_bank_sha256": bank.artifact_sha256,
        "generation_config_sha256": generation_sha,
        "model": generation.model, "revision": generation.revision,
        "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
        "intervention_type": INTERVENTION_TYPE,
        "intervention_order": INTERVENTION_ORDER,
        "budget_reference": BUDGET_REFERENCE,
        "states": list(BANK_STATES),
        "historical_layers": list(HISTORICAL_LAYERS),
        "dataset_sha256": dataset_sha,
        "smoke": smoke,
        "protocol": PROTOCOL,
        "tolerances_version": TOLERANCES_VERSION,
        "predeclaration_sha256": predeclaration_digest,
        "selected_rows_sha256": SELECTED_ROWS_SHA256,
    }

    # ---- the resume file --------------------------------------------------
    #
    # --resume is consulted by claim_output_dir itself, so a resumable
    # directory is not refused before the flag is ever read.
    out_dir = claim_output_dir(args.output_dir, create=False,
                               resume=args.resume)
    records_path = out_dir / RECORDS_FILENAME
    existing = read_records(records_path, strict=not args.resume)
    if existing and not args.resume:
        raise RotationScreenError(
            f"{records_path} already holds {len(existing)} record(s) and "
            f"--resume was not given. An existing screen is never appended to "
            f"by accident; pass --resume or name a new directory.")
    verify_consistency(existing, run_locked)
    done = {record_key(record) for record in existing}
    checks.record("resume", True,
                  f"{len(done)} completed generation(s) already recorded")

    todo = plan_work(selected, done, bank.artifact_sha256, generation_sha)
    if args.limit:
        todo = todo[:args.limit]

    by_arm: Dict[str, int] = {}
    for item in todo:
        by_arm[item["arm_id"]] = by_arm.get(item["arm_id"], 0) + 1

    print(f"[rotation] CANDIDATE screen {PROTOCOL} -- not validated, not run")
    print(f"[rotation] predeclaration {predeclaration_digest[:16]}  bank "
          f"{bank.artifact_sha256[:16]}  generation {generation_sha[:16]}")
    print(f"[rotation] {preserved['verified']} completed artifact(s) verified "
          f"unchanged")
    print(f"[rotation] d_S = {direction_metadata['subtraction_order']}; "
          f"positive beta rotates TOWARD it")
    print(f"[rotation] inventory {counts['total']} "
          f"({counts['rows']} rows x {counts['arms']} arms); "
          f"{REFERENCE_RECORDS} inherited; table {ANALYSIS_TABLE_RECORDS}")
    print(f"[rotation] shard {args.shard_index}/{args.shard_count}: "
          f"{len(selected)} selected, {len(done)} done, {len(todo)} to run")
    print(f"[rotation] states {list(BANK_STATES)} "
          f"(historical layers {list(HISTORICAL_LAYERS)}), final prompt "
          f"token, prefill only, order {INTERVENTION_ORDER}")
    print(f"[rotation] operator selftest worst relative norm error "
          f"{selftest['worst_relative_norm_error_float32']:.3e}")
    if smoke:
        print(f"[rotation] SMOKE TEST -- {'; '.join(smoke_reasons)}. This is a "
              f"subset of the {NEW_GENERATIONS}-generation {PROTOCOL} "
              f"protocol and is never the protocol itself.")

    environment = environment_record(
        PACKAGE_VERSION,
        model_loaded=not args.dry_run,
        generation_performed=not args.dry_run)

    manifest = {
        "schema": SCREEN_SCHEMA,
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "dry_run": bool(args.dry_run),
        "smoke": smoke,
        "smoke_reasons": smoke_reasons,
        "complete_screen": (not smoke and not args.dry_run
                            and args.shard_count == 1),
        "complete_screen_note":
            "true only for an unfiltered, unsharded, unlimited real run. A "
            "sharded run is a part of the complete screen and is completed by "
            "consolidate_rotation_screen.py; a filtered or limited run is a "
            "smoke test and is never the screen.",
        "reportable": not args.dry_run and not smoke,
        "is_a_candidate_not_a_result": True,
        "filters": {"only_arm": sorted(only_arm),
                    "only_row_id": sorted(only_row),
                    "limit": int(args.limit)},
        "frozen_protocol": {
            "protocol": PROTOCOL,
            "expected_generations": NEW_GENERATIONS,
            "rows": N_ROWS,
            "arms": len(ARM_BY_ID_ROTATION),
            "reference_arms": list(REFERENCE_ARM_IDS),
            "reference_arms_are_inherited_not_generated": True,
            "reference_records": REFERENCE_RECORDS,
            "analysis_table_records": ANALYSIS_TABLE_RECORDS,
            "selected_this_run": len(selected),
        },
        "judges_responses": False,
        "trains_a_controller": False,
        "orthogonalisation_applied": False,
        "direction_bank": str(args.direction_bank),
        "direction_bank_manifest_sha256": bank.manifest_sha256,
        "direction_metadata": direction_metadata,
        "safety_direction_sha256": safety_digests,
        "repair_direction_sha256": {
            str(state): bank.direction_sha256("repair_direction", state)
            for state in bank.states},
        "dataset": str(dataset_path),
        "generation": generation.as_dict(),
        "rotation_tolerances": ROTATION_TOLERANCES.as_dict(),
        "rotation_operator_selftest": selftest,
        "arms": [ARM_BY_ID_ROTATION[arm_id].as_dict()
                 for arm_id in sorted(ARM_BY_ID_ROTATION)],
        "inventory": counts,
        "shard": {"index": args.shard_index, "count": args.shard_count,
                  "selected": len(selected)},
        "planned_this_run": len(todo),
        "planned_by_arm": dict(sorted(by_arm.items())),
        "already_done": len(done),
        "preserved_artifacts": preserved,
        "environment": environment,
        "checklist": checks.as_dict(),
        **run_locked,
    }

    if args.dry_run:
        print("[rotation] dry run: no model loaded, nothing written, no "
              "output directory created")
        summary = {
            "planned": len(todo),
            "by_arm": dict(sorted(by_arm.items())),
            "bound_rows": N_ROWS,
            "arms_per_row": len(ARM_BY_ID_ROTATION),
            "reference_arms_not_scheduled": list(REFERENCE_ARM_IDS),
            "reference_arms_in_this_plan": sorted(
                set(by_arm) & set(REFERENCE_ARM_IDS)),
            "reference_records_inherited": REFERENCE_RECORDS,
            "analysis_table_records": ANALYSIS_TABLE_RECORDS,
            "output_dir_exists": Path(args.output_dir).exists(),
        }
        print(json.dumps(summary, indent=2))
        return 0

    out_dir = claim_output_dir(args.output_dir, create=True,
                               resume=args.resume)

    # ---- only now is a model loaded --------------------------------------
    from routegate.runtime import load_model_and_tokenizer

    print(f"[load] {generation.model} @ {generation.revision} "
          f"(device_map={args.device})")
    model, tokenizer = load_model_and_tokenizer(generation)
    controller = rotation_controller(model, states=list(BANK_STATES),
                                     budget_reference=BUDGET_REFERENCE,
                                     tolerances=ROTATION_TOLERANCES)

    # Written after the load, not before it: the manifest states that a model
    # was loaded, and it should not be able to say so about a run where the
    # load then failed.
    write_json(out_dir / MANIFEST_FILENAME, manifest)

    by_row = {row["row_id"]: row for row in rows}
    captured_row: Optional[str] = None
    capture: Dict[int, Any] = {}
    written = errors = 0

    try:
        for position, item in enumerate(todo, start=1):
            row = by_row[item["row_id"]]
            arm = ARM_BY_ID_ROTATION[item["arm_id"]]
            print(f"[{position}/{len(todo)}] {item['row_id']} {arm.arm_id} "
                  f"(beta={arm.beta}, aR={arm.alpha_repair}, "
                  f"{arm.kind})", flush=True)

            per_state: Dict[str, Any] = {}
            try:
                if captured_row != item["row_id"]:
                    capture_prompt_states(
                        model, tokenizer, row["prompt_text"], controller,
                        capture_name="target", example_id=item["row_id"],
                        config=generation)
                    capture = dict(controller.capture_for("target"))
                    captured_row = item["row_id"]
                plans = build_site_plans(bank, arm, capture,
                                         tolerances=ROTATION_TOLERANCES,
                                         budget_reference=BUDGET_REFERENCE)
                generated = generate_response(
                    model, tokenizer, row["prompt_text"], generation,
                    controller=controller, mode="patch", plans=plans,
                    example_id=item["row_id"])
                per_state = {
                    str(state): dict(diagnostics) for state, diagnostics
                    in sorted(controller.rotation_diagnostics.items())}
                hooks_expected = sum(1 for plan in plans.values()
                                     if plan.applies_anything)
                edge = sorted({str(entry.get("edge_case") or "")
                               for entry in per_state.values()} - {""})
                status = "aligned_no_edit" if edge else "ok"
                error_message = ""
            except Exception as exc:                     # noqa: BLE001
                generated = {"response_text": "", "prompt_token_count": 0,
                             "completion_token_count": 0, "eos_reached": False,
                             "hit_max_new_tokens": False,
                             "elapsed_seconds": 0.0, "applied_by_state": {}}
                status, error_message = "error", f"{type(exc).__name__}: {exc}"
                hooks_expected = 0
                per_state = {}
                captured_row = None
                errors += 1
                print(f"      ERROR {error_message}", file=sys.stderr)

            hits = dict(getattr(controller.run, "hits", {}) or {})
            record = make_record(
                row_id=item["row_id"], row_class=CLASS_H, arm_id=arm.arm_id,
                beta=arm.beta, alpha_R=arm.alpha_repair,
                prompt_sha256=item["prompt_sha256"],
                response_text=generated["response_text"],
                response_sha256=sha256_text(generated["response_text"]),
                prompt_token_count=generated["prompt_token_count"],
                completion_token_count=generated["completion_token_count"],
                eos_reached=generated["eos_reached"],
                hit_max_new_tokens=generated["hit_max_new_tokens"],
                elapsed_seconds=generated["elapsed_seconds"],
                status=status, error_message=error_message,
                applied_by_state={str(state): int(count)
                                  for state, count in sorted(hits.items())},
                hooks_fired=sum(int(count) for count in hits.values()),
                hooks_expected=hooks_expected, per_state=per_state,
                generation=generation.as_dict(),
                created_utc=utc_now(), environment=environment,
                **run_locked)
            append_record(records_path, record)
            written += 1
    finally:
        controller.close()

    print(f"[rotation] wrote {written} record(s) to {records_path}"
          + (f"; {errors} error(s)" if errors else ""))
    return 1 if errors else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
