#!/usr/bin/env python3
"""Preflight: does the one-pass actuator budget reproduce the validated one?

The controller specification wants a single forward pass. The validated
actuator reads its budget from an unsteered capture taken in a *separate* pass.
Blocks 31 and 32 are mathematically identical under both conventions, because
their reference is snapshotted before anything upstream of it is edited. Block
33 is not: by the time it is reached, block 32 has already consumed an edited
state 32, and no closed form says how far that moves the norm. This runner
measures it on the frozen model.

Two things are checked, in two different currencies, and version 1 of this
runner conflated them:

* **exactly**, within one forward pass -- that the budget is read before the
  activation is edited. Same pass, same device, same dtype, so no reduction
  order can reach it;
* **to the predeclared tolerance**, across the two forward passes -- every
  block's budget, blocks 31 and 32 included. Their two numbers are produced by
  separate forwards with their norms reduced on different devices, so they
  agree to float32 rounding and not bit for bit. v1 demanded bit-identity there
  and refused a complete run over a 3.527095e-07 disagreement, roughly three
  float32 epsilons.

    delta(i, l) = 0.15 * g_R * r(i, l) * d_R(l)      l = 31, 32, 33
    g_R = 1                                          the worst case
    alpha_S = 0                                      permanently

What it does
------------
Two prefill forward passes per prompt, over all 480 development prompts:

    pass A   unsteered, capturing states 31/32/33 -- the validated two-pass
             budgets, through the package's own PatchController
    pass B   steered, recording what each budget would have been under the
             one-pass hook ordering

then subtracts them, per block, per row.

What it does not do
-------------------
It generates no token -- prefill only, ``use_cache=False``, and the generated
token count is recorded and asserted to be zero. It trains nothing, judges
nothing, selects nothing and touches no held-out material. It writes one
immutable output directory and refuses a non-empty one.

Usage
-----
    python experiments/single_direction_controller/budget_equivalence_check.py \\
        --direction-bank "$ROUTEGATE_DATA_DIR"/frozen/validated_direction_bank \\
        --output-dir outputs/single_direction_controller/budget_equivalence_v1 \\
        --dry-run

Drop --dry-run to load the frozen model and run the real check. Exit status is
0 only when every measured block is inside the frozen tolerance.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import (                                        # noqa: E402
    load_development_inputs,
    load_input_manifest,
    load_local_paths,
)
from routegate.dualsteer import load_validated_bank                 # noqa: E402
from routegate.onepassbudget import (                               # noqa: E402
    ACTUATOR_BLOCKS,
    ALPHA_R,
    FLOAT32_REDUCTION_SCALE,
    MATHEMATICALLY_IDENTICAL_BLOCKS,
    MEASURED_BLOCK,
    SAME_FORWARD_INVARIANT_BLOCKS,
    SCHEMA,
    TOLERANCE,
    WORST_CASE_G_R,
    BudgetEquivalenceError,
    RowBudgets,
    block_input_state,
    block_output_state,
    budget_norm,
    budget_reference_states,
    decide,
    summarise,
    verify_read_before_write,
)
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

#: The frozen 480-row division. Read for its row ORDER only: no fold is used by
#: this check, and no probe or controller is fitted here.
DEFAULT_FOLD_DIR = (REPO_ROOT / "outputs" / "causal_repair"
                    / "orthogonalized_population_scan" / "direction_bank")

OUTPUT_MANIFEST = "run_manifest.json"
OUTPUT_ROWS = "per_row_budgets.csv"
OUTPUT_SUMMARY = "summary.json"
OUTPUT_INPUT_HASHES = "input_hashes.json"
OUTPUT_HASHES = "output_hashes.json"


def default_bank_sha256() -> str:
    """The pinned artifact digest, read from the committed configuration.

    Never transcribed by hand: the configuration and the phase-2
    predeclaration both carry it and are required to agree, so a digest that
    has moved in one of them stops this runner rather than being adopted.
    """
    config = json.loads(
        (REPO_ROOT / "configs" / "dual_direction_controller.json")
        .read_text(encoding="utf-8"))
    pinned = config["causal_screen"]["direction_bank"]["artifact_sha256"]
    predeclaration = json.loads(
        (REPO_ROOT / "manifests"
         / "focused_direction_validation_v1_phase2.json")
        .read_text(encoding="utf-8"))
    inherited = predeclaration["inherited_frozen"]["direction_bank_sha256"]
    if pinned != inherited:
        raise BudgetEquivalenceError(
            f"configs/dual_direction_controller.json pins the direction bank "
            f"at {pinned} and the phase-2 predeclaration at {inherited}. Two "
            f"committed records disagree about which artefact the validated "
            f"actuator used; that is resolved deliberately, not by preferring "
            f"one of them here.")
    return str(pinned)


def check_output_dir(path: Path | str) -> Path:
    """The output directory is absent or empty. Nothing is created here.

    Separated from creating it so ``--dry-run`` can prove the directory is
    claimable without occupying it. A dry run that wrote into the final
    directory would leave it non-empty and make the real run it was meant to
    clear the way for impossible.
    """
    path = Path(path)
    assert_not_sealed(path, "budget equivalence output directory")
    if path.exists():
        if not path.is_dir():
            raise BudgetEquivalenceError(f"{path} exists and is not a directory")
        existing = sorted(p.name for p in path.iterdir())
        if existing:
            raise BudgetEquivalenceError(
                f"{path} already holds {existing[:5]}. This result is "
                f"immutable: it is written once, into an empty directory, and "
                f"a previous run is never partially overwritten. Name a new "
                f"directory, or remove that one by hand with a reason.")
    return path


def claim_output_dir(path: Path | str) -> Path:
    """Check it, then create it. Only a real run reaches this."""
    path = check_output_dir(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_row_order(rows: Sequence[Dict[str, str]], fold_dir: Path,
                     checks: Checklist) -> Dict[str, str]:
    """The corpus rows are the frozen division's rows, in its order.

    The fold file's ``array_row_index`` is the activation row position, so its
    order *is* the order every frozen artefact was built against. A row that is
    absent, added or moved would still produce 480 well-formed comparisons --
    against the wrong prompts -- so the order is checked rather than the count.
    """
    fold_path = fold_dir / "fold_assignment.csv"
    if not fold_path.is_file():
        raise BudgetEquivalenceError(
            f"{fold_path} does not exist; it is the frozen 480-row division "
            f"this corpus is checked against, and a row order taken on trust "
            f"is not one")
    assert_not_sealed(fold_path, "fold assignment")
    with open(fold_path, newline="", encoding="utf-8") as handle:
        frozen = list(csv.DictReader(handle))

    positions = [int(r["array_row_index"]) for r in frozen]
    checks.require(
        "fold_file_is_in_activation_row_order",
        positions == list(range(len(frozen))),
        f"{fold_path.name} does not run 0..{len(frozen) - 1} in file order")

    expected_ids = [r["row_id"] for r in frozen]
    actual_ids = [r["row_id"] for r in rows]
    checks.require(
        "no_row_is_missing_or_added",
        sorted(expected_ids) == sorted(actual_ids),
        f"{len(set(expected_ids) ^ set(actual_ids))} row id(s) differ between "
        f"the corpus and {fold_path.name}")
    checks.require(
        "no_row_is_reordered", expected_ids == actual_ids,
        f"the corpus row order differs from {fold_path.name} at "
        f"{sum(1 for a, b in zip(expected_ids, actual_ids) if a != b)} "
        f"position(s); the fold file's order is the activation row order and "
        f"every frozen artefact was built against it")
    digest = sha256_file(fold_path)
    checks.record("fold_assignment_sha256", True, digest)
    return {"fold_assignment": str(fold_path), "fold_assignment_sha256": digest}


def load_rows(checks: Checklist) -> Any:
    """Every development prompt, hash-gated, with no held-out row among them."""
    inputs = load_development_inputs()
    rows = list(inputs.manifest_rows)
    checks.require("corpus_row_count", len(rows) == EXPECTED_ROWS,
                   f"{len(rows)} row(s), expected {EXPECTED_ROWS}")
    checks.record("dataset_sha256", True, inputs.dataset_sha256)
    checks.record("activation_sha256", True, inputs.activation_sha256)
    checks.record("extraction_manifest_sha256", True,
                  inputs.extraction_manifest_sha256)

    sealed = [row["row_id"] for row in rows if looks_sealed(row["row_id"])]
    checks.require(
        "no_sealed_row_id", not sealed,
        f"{len(sealed)} row id(s) read as held-out material, first "
        f"{sealed[:3]}; this stage runs on development prompts only")
    missing = [row["row_id"] for row in rows if not row.get("prompt_text")]
    checks.require("every_row_carries_its_prompt", not missing,
                   f"{len(missing)} row(s) have no prompt text")
    duplicated = sorted({r["row_id"] for r in rows
                         if [x["row_id"] for x in rows].count(r["row_id"]) > 1})
    checks.require("row_ids_are_unique", not duplicated,
                   f"{len(duplicated)} duplicated row id(s) {duplicated[:3]}")
    return inputs, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True,
                        help="the validated direction bank directory")
    parser.add_argument("--expect-bank-sha256", type=str, default=None,
                        help="the pinned validated_direction_bank.npz digest; "
                             "defaults to the committed one")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--fold-dir", type=Path, default=DEFAULT_FOLD_DIR,
                        help="the frozen 480-row division whose row ORDER the "
                             "corpus is checked against")
    parser.add_argument("--blocks", type=str,
                        default=",".join(str(b) for b in ACTUATOR_BLOCKS),
                        help="historical actuator blocks, comma separated")
    parser.add_argument("--g-r", type=float, default=WORST_CASE_G_R,
                        help="the gate value; this check runs the worst case 1")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help="the frozen maximum relative difference")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prefill-only", action="store_true", default=True,
                        help="always true; no token is ever generated")
    parser.add_argument("--token-position", type=str,
                        default="final prompt token")
    parser.add_argument("--fail-closed", action="store_true", default=True,
                        help="always true; a failing block exits non-zero")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every input and claim; load no model")
    return parser


def run(args: argparse.Namespace) -> int:
    checks = Checklist("one_pass_budget_equivalence")
    blocks = tuple(int(b) for b in str(args.blocks).split(",") if b.strip())

    if args.token_position != "final prompt token":
        raise BudgetEquivalenceError(
            f"--token-position is {args.token_position!r}; the validated "
            f"actuator edits the final prompt token and this check measures "
            f"that actuator")
    if not 0.0 < float(args.g_r) <= 1.0:
        raise BudgetEquivalenceError(f"--g-r is {args.g_r}; the gate is in (0, 1]")
    if float(args.tolerance) != TOLERANCE:
        raise BudgetEquivalenceError(
            f"--tolerance is {args.tolerance}; the frozen tolerance is "
            f"{TOLERANCE} and is not relaxed to make a run pass")
    checks.require("actuator_blocks", blocks == ACTUATOR_BLOCKS,
                   f"{list(blocks)}, expected {list(ACTUATOR_BLOCKS)}")
    checks.record("alpha_S", True,
                  "0 permanently; d_S = H - E is retired as an actuator")
    checks.record("g_R", True, f"{args.g_r} (worst case)")

    # ---- the layer numbering, written down rather than assumed -------------
    for block in blocks:
        checks.record(
            f"block_{block}_mapping", True,
            f"historical block {block}: input = state "
            f"{block_input_state(block)}, output = state "
            f"{block_output_state(block)}")

    # ---- inputs, all hash-gated, before anything is loaded -----------------
    # Checked, not claimed: a dry run must not occupy the directory the real
    # run needs to find empty.
    check_output_dir(args.output_dir)
    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    assert_config_is_sealed_free(local_paths, "local path file")
    inputs, rows = load_rows(checks)
    fold_source = assert_row_order(rows, Path(args.fold_dir), checks)

    expected_bank = args.expect_bank_sha256 or default_bank_sha256()
    bank_dir = assert_not_sealed(Path(args.direction_bank), "direction bank")
    bank = load_validated_bank(bank_dir, expected_bank, checks)
    checks.record("direction_bank_manifest_sha256", True, bank.manifest_sha256)
    checks.require(
        "direction_bank_is_not_orthogonalised",
        bank.manifest.get("orthogonalisation", {}).get("applied") is False,
        "the validated bank states orthogonalisation.applied: false")

    generation = GenerationConfig()
    generation.validate()
    checks.record("model", True, f"{FROZEN_MODEL} @ {FROZEN_REVISION}")
    checks.record("generation_config_sha256", True, generation.fingerprint())
    checks.record("tokenizer_config_sha256", True,
                  FROZEN_TOKENIZER_CONFIG_SHA256)
    checks.record("chat_template_sha256", True, FROZEN_CHAT_TEMPLATE_SHA256)

    directions = {block: bank.directions_for_state(block_output_state(block))[0]
                  for block in blocks}
    for block in blocks:
        checks.record(
            f"direction_block_{block}_sha256", True,
            bank.direction_sha256("repair_direction",
                                  block_output_state(block)))

    print(f"[budget] {len(rows)} development prompt(s)")
    print(f"[budget] blocks {list(blocks)} -> states "
          f"{[block_output_state(b) for b in blocks]}")
    print(f"[budget] budget reference states "
          f"{list(budget_reference_states(blocks))}")
    print(f"[budget] alpha_R={ALPHA_R} g_R={args.g_r} "
          f"strength={ALPHA_R * float(args.g_r)}")
    print(f"[budget] direction bank {expected_bank[:16]}... verified")
    print(f"[budget] tolerance {args.tolerance:g} on the maximum relative "
          f"difference")

    input_hashes = {
        "dataset_sha256": inputs.dataset_sha256,
        "activation_sha256": inputs.activation_sha256,
        "extraction_manifest_sha256": inputs.extraction_manifest_sha256,
        "direction_bank_sha256": expected_bank,
        "direction_bank_manifest_sha256": bank.manifest_sha256,
        "direction_sha256_by_block": {
            str(b): bank.direction_sha256("repair_direction",
                                          block_output_state(b))
            for b in blocks},
        "model": FROZEN_MODEL,
        "revision": FROZEN_REVISION,
        "generation_config_sha256": generation.fingerprint(),
        "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
        "rows": len(rows),
        **fold_source,
    }

    if args.dry_run:
        print(f"[budget] dry run: every input, hash, row, order, state and "
              f"direction validated; no model loaded, no token generated, "
              f"nothing written")
        print(f"[budget] output directory {args.output_dir} is claimable "
              f"(absent or empty) and was deliberately NOT created")
        print(f"[budget] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        for key, value in sorted(input_hashes.items()):
            if isinstance(value, str) and len(value) == 64:
                print(f"[budget]   {key:<34} {value}")
        return 0 if checks.all_passed else 2

    output_dir = claim_output_dir(args.output_dir)

    # ---- the real check ----------------------------------------------------
    from routegate.hooks import PatchController
    from routegate.onepassbudget import OnePassBudgetRecorder
    from routegate.runtime import capture_prompt_states, load_model_and_tokenizer

    print(f"[budget] loading {FROZEN_MODEL} @ {FROZEN_REVISION} "
          f"(device_map={args.device})")
    model, tokenizer = load_model_and_tokenizer(generation)
    model.eval()

    states = [block_output_state(b) for b in blocks]
    measured: List[RowBudgets] = []
    import torch

    from routegate.runtime import tokenize_prompt

    # Both sets of hooks are installed exactly once, so the order in which they
    # fire is fixed for the whole run. The recorder is armed only for pass B;
    # the reference pass must see the model as the validated convention saw it.
    controller = PatchController(model, states, budget_reference="block_input")
    recorder = OnePassBudgetRecorder(model, directions, blocks,
                                     alpha=ALPHA_R, g_r=float(args.g_r))
    cache_deltas: List[float] = []
    try:
        for index, row in enumerate(rows, start=1):
            rid = row["row_id"]

            # -- pass A: the validated two-pass reference, unsteered ---------
            recorder.disarm()
            capture_name = f"reference:{rid}"
            capture_prompt_states(model, tokenizer, row["prompt_text"],
                                  controller, capture_name, rid, generation)
            capture = controller.captures.pop(capture_name)
            missing = [block_input_state(b) for b in blocks
                       if block_input_state(b) not in capture]
            if missing:
                raise BudgetEquivalenceError(
                    f"{rid}: the reference pass captured no state(s) {missing}")
            # budget_norm on both sides: a difference between the conventions
            # is then a difference in the activation, never in the arithmetic.
            two_pass = {b: budget_norm(capture[block_input_state(b)])
                        for b in blocks}

            # -- pass B: steered, one-pass hook ordering ---------------------
            controller.reset_run(mode="off", example_id=rid)
            recorder.arm()
            batch = tokenize_prompt(tokenizer, row["prompt_text"], model,
                                    generation)
            with torch.inference_mode():
                model(**batch, use_cache=False)
            recorder.disarm()

            absent = [b for b in blocks if b not in recorder.one_pass]
            if absent:
                raise BudgetEquivalenceError(
                    f"{rid}: no one-pass budget recorded for block(s) {absent}; "
                    f"the hook on the preceding block did not fire, so the "
                    f"ordering this check rests on does not hold")
            if recorder.decode_forwards:
                raise BudgetEquivalenceError(
                    f"{rid}: {recorder.decode_forwards} forward pass(es) beyond "
                    f"the prefill; this stage is prefill only and decodes "
                    f"nothing")

            budgets = RowBudgets(
                row_id=rid, two_pass=two_pass,
                one_pass=dict(recorder.one_pass),
                one_pass_runtime_input={b: v for b, v
                                        in recorder.runtime_input.items()
                                        if b in blocks},
                read_before_write=dict(recorder.read_before_write),
                prefill_tokens=int(recorder.prefill_tokens),
                generated_tokens=0)
            measured.append(budgets)

            # Diagnostic, not a gate: the live unsteered reference against the
            # frozen capture every probe was fitted on. A large drift here
            # would mean this session's model is not the one behind the frozen
            # artefacts, which is worth seeing even though it gates nothing.
            cached = inputs.activations[index - 1]
            for b in blocks:
                ref = float(np.linalg.norm(
                    np.asarray(cached[block_input_state(b)], dtype=np.float64)))
                if ref > 0.0:
                    cache_deltas.append(abs(two_pass[b] - ref) / ref)

            if index % 25 == 0 or index == len(rows):
                print(f"[budget] {index}/{len(rows)} prompts", flush=True)
    finally:
        controller.close()
        recorder.close()

    return _report(measured, blocks, args, checks, output_dir, input_hashes,
                   bank, generation, cache_deltas)


def _report(measured: Sequence[RowBudgets], blocks: Sequence[int],
            args: argparse.Namespace, checks: Checklist, output_dir: Path,
            input_hashes: Dict[str, Any], bank: Any,
            generation: GenerationConfig,
            cache_deltas: Sequence[float]) -> int:
    generated = sum(r.generated_tokens for r in measured)
    # Recorded, not required. Every gate below is evaluated without raising and
    # the artifacts are written before any of them decides the exit status --
    # v1 raised here and discarded 960 completed forward passes, which is how a
    # run that had already answered the question came to leave no answer.
    checks.record("no_token_was_generated", generated == 0,
                  f"{generated} generated token(s); this stage is prefill "
                  f"only and generating one would make it a different stage")
    checks.record("every_row_measured", len(measured) == EXPECTED_ROWS,
                  f"{len(measured)} row(s), expected {EXPECTED_ROWS}")

    # The one invariant that is genuinely exact. It raises on a real
    # read-after-write bug; caught here so the evidence is still written, then
    # re-reported as a fatal exit below.
    invariant_error: Optional[str] = None
    try:
        invariant = verify_read_before_write(
            measured, SAME_FORWARD_INVARIANT_BLOCKS, checks)
    except ValidationError as exc:
        invariant, invariant_error = {}, str(exc)

    summaries = {b: summarise(measured, b, float(args.tolerance))
                 for b in blocks}
    runtime_variant = summarise(measured, MEASURED_BLOCK,
                                float(args.tolerance), runtime_input=True)
    verdict = decide(summaries, float(args.tolerance))

    print()
    print(f"{'block':>5} {'max':>12} {'mean':>12} {'median':>12} "
          f"{'p95':>12} {'p99':>12} {'>tol':>6}")
    for b in blocks:
        s = summaries[b]
        print(f"{b:>5} {s['max']:>12.6e} {s['mean']:>12.6e} "
              f"{s['median']:>12.6e} {s['p95']:>12.6e} {s['p99']:>12.6e} "
              f"{s['n_exceeding_tolerance']:>6}")
    if runtime_variant.get("measured"):
        s = runtime_variant
        print(f"{'33*':>5} {s['max']:>12.6e} {s['mean']:>12.6e} "
              f"{s['median']:>12.6e} {s['p95']:>12.6e} {s['p99']:>12.6e} "
              f"{s['n_exceeding_tolerance']:>6}   (* runtime-input variant)")
    for b in blocks:
        failing = summaries[b]["row_ids_exceeding_tolerance"]
        if failing:
            print(f"\n[budget] block {b}: {len(failing)} row(s) exceed "
                  f"{args.tolerance:g}")
            for rid in failing:
                print(f"[budget]   {rid}")

    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "dry_run": False,
        "stage": "single_direction_controller_budget_equivalence",
        "trains_a_controller": False,
        "judges_responses": False,
        "generates_tokens": False,
        "generated_tokens": generated,
        "prefill_only": True,
        "token_position": args.token_position,
        "rows": len(measured),
        "blocks": list(blocks),
        "states_edited": [block_output_state(b) for b in blocks],
        "budget_reference_states": list(budget_reference_states(blocks)),
        "alpha_R": ALPHA_R,
        "g_R": float(args.g_r),
        "alpha_S": 0.0,
        "tolerance": float(args.tolerance),
        "direction_bank": str(Path(args.direction_bank).resolve()),
        "supersedes": {
            "run": "budget_equivalence_v1",
            "outcome": "BLOCKED (exit 2)",
            "block_31_max_relative_difference": 3.527095e-07,
            "why_it_was_refused":
                "v1 required blocks 31 and 32 to be bit-identical across two "
                "separate forward passes. Their norms are reduced on different "
                "devices, so they agree to float32 rounding and not bit for "
                "bit; 3.527095e-07 is about three float32 epsilons and some "
                "four orders of magnitude below the bfloat16 resolution the "
                "activations are stored at. The gate was wrong, not the "
                "actuator.",
            "wrote_no_artifacts":
                "the gate raised inside the report before anything was "
                "written, so 960 completed forward passes were discarded and "
                "no v1 block-33 result exists. This run writes every artifact "
                "before any gate decides the exit status.",
            "v1_directory_preserved": True,
            "recorded_in": "docs/decision_log.md, 2026-08-26",
        },
        "inputs": dict(input_hashes),
        "environment": environment_record(PACKAGE_VERSION, model_loaded=True,
                                          generation_performed=False),
        "checklist": checks.as_dict(),
    }
    cache = ({"measured": True,
              "max_relative_difference": float(max(cache_deltas)),
              "mean_relative_difference":
                  float(sum(cache_deltas) / len(cache_deltas)),
              "comparisons": len(cache_deltas),
              "is_a_gate": False,
              "note": "the live unsteered reference budgets against the frozen "
                      "activation capture every probe was fitted on. Reported "
                      "so a session whose model differs from the one behind "
                      "the frozen artefacts is visible; it gates nothing here."}
             if cache_deltas else {"measured": False})
    summary = {
        "schema": SCHEMA,
        "created_utc": manifest["created_utc"],
        "rows": len(measured),
        "generated_tokens": generated,
        "same_forward_read_before_write": {
            "blocks": list(SAME_FORWARD_INVARIANT_BLOCKS),
            "holds": invariant_error is None,
            "evidence": {str(b): invariant[b] for b in invariant},
            "error": invariant_error,
            "statement": "the exact invariant: within one forward pass, on one "
                         "device, the budget is read before the activation is "
                         "edited. Reduction order cannot reach it.",
        },
        "mathematically_identical_blocks": {
            "blocks": list(MATHEMATICALLY_IDENTICAL_BLOCKS),
            "observed_max": {str(b): summaries[b]["max"]
                             for b in MATHEMATICALLY_IDENTICAL_BLOCKS
                             if b in summaries},
            "float32_reduction_scale": FLOAT32_REDUCTION_SCALE,
            "gated_at": float(args.tolerance),
            "note": "these blocks are identical in exact arithmetic. The two "
                    "numbers come from two separate forward passes with their "
                    "norms reduced on different devices, so they agree to "
                    "float32 rounding rather than bit for bit. They are gated "
                    "at the predeclared tolerance like every other "
                    "cross-forward comparison; the scale above is context for "
                    "a reader, not a threshold.",
        },
        "per_block": {str(b): summaries[b] for b in summaries},
        "block_33_runtime_input_variant": runtime_variant,
        "frozen_capture_cross_check": cache,
        "decision": verdict,
    }

    fields = sorted({k for r in measured for k in r.as_dict()})
    fields = ["row_id"] + [f for f in fields if f != "row_id"]
    with open(output_dir / OUTPUT_ROWS, "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in measured:
            writer.writerow(r.as_dict())
    write_json(output_dir / OUTPUT_INPUT_HASHES, input_hashes)
    write_json(output_dir / OUTPUT_MANIFEST, manifest)
    write_json(output_dir / OUTPUT_SUMMARY, summary)
    write_json(output_dir / OUTPUT_HASHES,
               {name: sha256_file(output_dir / name)
                for name in (OUTPUT_ROWS, OUTPUT_INPUT_HASHES,
                             OUTPUT_MANIFEST, OUTPUT_SUMMARY)})

    if cache.get("measured"):
        print(f"[budget] frozen-capture cross-check (diagnostic, not a gate): "
              f"max {cache['max_relative_difference']:.6e}")
    print(f"\n[budget] wrote {output_dir}")
    for name in (OUTPUT_ROWS, OUTPUT_INPUT_HASHES, OUTPUT_MANIFEST,
                 OUTPUT_SUMMARY, OUTPUT_HASHES):
        print(f"[budget]   {name:<22} {sha256_file(output_dir / name)}")

    # Every artifact is on disk. Only now does anything decide an exit status,
    # so a refused run still leaves the evidence it was refused on.
    if invariant_error is not None:
        print(f"\n[FATAL] {invariant_error}", file=sys.stderr)
        print("ONE_PASS_EQUIVALENCE: BLOCKED", file=sys.stderr)
        return 2
    if not checks.all_passed:
        for entry in checks.failures:
            print(f"\n[FATAL] {entry['check']}: {entry['detail']}",
                  file=sys.stderr)
        print("ONE_PASS_EQUIVALENCE: BLOCKED", file=sys.stderr)
        return 2

    print(f"[budget] same-forward read-before-write invariant: holds on "
          f"{len(measured)} row(s) at blocks "
          f"{list(SAME_FORWARD_INVARIANT_BLOCKS)}")
    print(f"[budget] decision: {verdict['decision']}")
    print(f"[budget] {verdict['statement']}")
    print(f"ONE_PASS_EQUIVALENCE: {verdict['decision']}")
    return 0 if verdict["decision"] == "PASS" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        print("ONE_PASS_EQUIVALENCE: BLOCKED", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
