#!/usr/bin/env python3
"""Generate the two dynamic-controller arms over the 480 development rows.

    python experiments/single_direction_controller/run_dynamic_arms.py \\
        --direction-bank .../validated_direction_bank \\
        --controllers .../gate_oof_v1_repro \\
        --output-dir .../dynamic_arms_v1 \\
        --arm dynamic_raw --shard-index 1 --shard-count 1 --dry-run

THE TWO ARMS, ONE CONTROLLER
----------------------------
``dynamic_raw``        applied coefficient 0.15 * g_R
``dynamic_hard_veto``  the same g_R, zeroed only where the frozen safety veto
                       fires

They share one controller. No second training run exists and none is permitted:
the veto is an inference-time brake around the same gate, it is not trained,
and it can only lower a coefficient.

HOW MANY RESPONSES ARE ACTUALLY NEW
-----------------------------------
Fewer than the arm count suggests, and the arithmetic is exact rather than
hopeful:

* ``dynamic_raw`` needs all 480. Nothing existing was generated at a per-row
  coefficient, so none of it can be reused.
* ``dynamic_hard_veto`` needs **none**. On the 192 rows where the veto fires,
  ``g_R`` becomes exactly 0.0, so the applied coefficient is exactly 0.0, so
  the edit is the addition of a zero vector and the response is the unsteered
  one -- the referenced baseline, already generated. On the other 288 rows the
  veto changes nothing, so the response is ``dynamic_raw``'s, generated above.

So the second arm is a **deterministic recombination**, not a generation run,
and this runner refuses to generate for it.

HOW A COEFFICIENT IS OBTAINED, AND HOW IT IS NOT
-------------------------------------------------
Not by row-id lookup. ``gate_oof_v1``'s coefficient column is never opened at
generation time: it cannot score a prompt it has not seen, and reading it would
put every development row's answer inside the arm that used it.

Per row, in the pass that generates the response:

1. capture the unsteered state-32 final-prompt-token activation, **before**
   either site is written;
2. ``livecontroller.plan_live_edit`` computes one ``g_R`` from that vector,
   through the fold checkpoint that never trained on the row;
3. apply ``0.15 * g_R * ||h^(l)|| * d_R,l`` at historical blocks 31 and 32 --
   the same coefficient at both. Historical block 33 is not a site.

The same checkpoints also score the frozen extraction's own state-32
activations, offline and on CPU. That gives a reference coefficient per row,
which is what the live one is checked against: the five per-fold held-out means
must reproduce the ones ``gate_oof_v1`` recorded before a model is loaded, and
every row's live coefficient must agree with its reference to within the
reproduction's own float32 budget or the run stops.

MODES
-----
``--dry-run``  validates every input, hash, fold, row, site, coefficient and
               shard. Loads no model, generates nothing, writes nothing.
(default)      generates this shard's rows. Resumable: a row already recorded
               is never regenerated, and an errored row is left unwritten so a
               restart retries it.
``--verify``   re-validates a finished output directory against the same gates
               and requires exactly 480 completed records. Writes nothing.

Nothing here trains. The language model, both probes, the direction bank and
the controller weights are all frozen; no gradient is taken anywhere in this
file, and neither the sealed set nor any judge is reachable from it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np                                                # noqa: E402

from routegate.dualsteer import (                                 # noqa: E402
    BUDGET_REFERENCE,
    append_record,
    load_validated_bank,
    make_record,
)
from routegate.dynamicarms import (                               # noqa: E402
    ARM_RAW,
    ARM_VETO,
    ARMS,
    COEFFICIENTS_FILENAME,
    COEFFICIENT_COLUMNS,
    EXPECTED_ROWS,
    GATE_TOLERANCE,
    GATE_TOLERANCE_EXPRESSION,
    OUTPUT_MANIFEST,
    OUTPUT_VALIDATION,
    PROTOCOL,
    REFERENCE_ARM,
    SCHEMA,
    DynamicArmError,
    assert_complete,
    assert_gate_is_faithful,
    assert_reference_reproduces_the_run,
    assert_repair_direction_is_b_d1_minus_e,
    assert_shard_is_not_complete,
    coefficient_rows,
    collect_records,
    controller_block,
    fold_map,
    load_reproduction,
    locked_fields,
    prepare_output_dir,
    reference_gates,
    shard_filenames,
    shard_records_name,
    validate_existing_records,
    verify_bundle,
)
from routegate.gatecontroller import (                            # noqa: E402
    ALPHA_R,
    ALPHA_S,
    N_SPLITS,
    READ_STATE,
)
from routegate.livecontroller import (                            # noqa: E402
    APPLY_BLOCKS,
    apply_hard_veto,
    gate_from_activation,
    load_bundle,
    plan_live_edit,
)
from routegate.runrecord import (                                 # noqa: E402
    environment_record,
    utc_now,
    write_csv,
    write_json,
)
from routegate.runtime import (                                   # noqa: E402
    FROZEN_CHAT_TEMPLATE_SHA256,
    FROZEN_MODEL,
    FROZEN_REVISION,
    FROZEN_TOKENIZER_CONFIG_SHA256,
    GenerationConfig,
)
from routegate.twosite import (                                   # noqa: E402
    CANDIDATE_BLOCKS,
    CANDIDATE_BUDGET_STATES,
    CANDIDATE_STATES,
    build_candidate_record,
    build_two_site_plans,
    verify_records,
)
from routegate.twositecontinuation import (                       # noqa: E402
    assert_shards_partition,
    deterministic_row_order,
    shard_rows,
)
from routegate.validation import (                                # noqa: E402
    Checklist,
    ValidationError,
    assert_not_sealed,
    looks_sealed,
    sha256_file,
)

PACKAGE_VERSION = "0.1.0"
INTERVENTION_TYPE = "dynamic_two_site_repair"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True,
                        help="the reproduction run's checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=list(ARMS), default=ARM_RAW)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=1,
                        help="1-based; 1 with --shard-count 1 is sequential")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="skip rows already recorded; never regenerate")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and gate only; writes nothing, generates "
                             "nothing, loads no model")
    parser.add_argument("--verify", action="store_true",
                        help="re-validate a finished output directory and "
                             "require exactly 480 completed records; loads no "
                             "model and writes nothing")
    return parser


def plan_rows(bundle, activations: Mapping[str, np.ndarray],
              fold_of: Mapping[str, int],
              z_safety: Mapping[str, float],
              arm: str,
              gates: Optional[Mapping[str, float]] = None
              ) -> List[Dict[str, Any]]:
    """One coefficient per row, from the controller that never saw it.

    Pure: reads activations and returns numbers. No model is loaded, no
    response is generated and nothing is written.

    ``gates`` is the same computation done in one batched pass per fold; when
    it is given the arithmetic is not repeated here, only read.
    """
    plan: List[Dict[str, Any]] = []
    for row_id in sorted(activations):
        fold = int(fold_of[row_id])
        checkpoint = bundle.controller_for_row(fold)
        if gates is not None and row_id in gates:
            g = float(gates[row_id])
        else:
            g = float(gate_from_activation(
                np.asarray(activations[row_id]).reshape(1, -1), checkpoint)[0])
        vetoed = False
        if arm == ARM_VETO:
            gated = float(apply_hard_veto(
                np.array([g]), np.array([z_safety[row_id]]),
                float(checkpoint["tau_hi"]))[0])
            vetoed = gated != g
            g = gated
        plan.append({
            "row_id": row_id, "fold": fold, "arm_id": arm,
            "g_R": g, "applied_alpha_R": ALPHA_R * g, "vetoed": vetoed,
            "apply_at_historical_blocks": list(APPLY_BLOCKS),
            "reuse_from": (REFERENCE_ARM if (arm == ARM_VETO and vetoed)
                           else (ARM_RAW if arm == ARM_VETO else None)),
        })
    return plan


def reuse_arithmetic(plan: Sequence[Mapping[str, Any]], arm: str
                     ) -> Dict[str, Any]:
    """How many rows are genuinely new, and why the rest are not.

    A reused row must be provably identical, not merely similar: a vetoed row
    has coefficient exactly 0.0, and an unvetoed row under the veto arm has the
    coefficient ``dynamic_raw`` already used. Anything else is generated.
    """
    if arm == ARM_RAW:
        return {"arm": arm, "rows": len(plan), "new": len(plan),
                "reused": 0, "reused_from": {},
                "why": "no existing arm was generated at a per-row "
                       "coefficient, so nothing can be reused"}
    zero = [r for r in plan if r["vetoed"]]
    non_zero = [r for r in plan if not r["vetoed"]]
    bad = [r["row_id"] for r in zero if r["applied_alpha_R"] != 0.0]
    if bad:
        raise DynamicArmError(
            f"{len(bad)} vetoed row(s) do not carry an exactly zero "
            f"coefficient, first {bad[:3]}. A reused baseline response is "
            f"only correct when the edit is exactly the zero vector.")
    return {
        "arm": arm, "rows": len(plan), "new": 0,
        "reused": len(plan),
        "reused_from": {REFERENCE_ARM: len(zero), ARM_RAW: len(non_zero)},
        "why": "a vetoed row applies exactly 0.0 and is the unsteered "
               "response; an unvetoed row applies the coefficient "
               "dynamic_raw already used. Neither needs a new forward pass.",
    }


def prepare(args: argparse.Namespace, checks: Checklist) -> Dict[str, Any]:
    """Every frozen input, gate and derived quantity, before any model exists.

    Ordered so the cheapest refusal comes first and nothing expensive is
    reached until the controller, the folds, the direction and the corpus have
    all agreed with each other.
    """
    assert_not_sealed(args.output_dir, "dynamic arm output directory")
    assert_not_sealed(args.direction_bank, "direction bank")
    assert_not_sealed(args.controllers, "controller bundle")

    # ---- the declarations this arm inherits -------------------------------
    verify_records(checks=checks)
    candidate = build_candidate_record()
    expected_bank = candidate["direction"]["artifact_sha256"]

    checks.require("apply_sites", tuple(APPLY_BLOCKS) == (31, 32),
                   "historical blocks 31 and 32; block 33 is excluded because "
                   "an edit at 31 changes the activation entering it, so its "
                   "unsteered budget does not exist in the same pass")
    checks.require("block_33_is_not_a_site",
                   33 not in CANDIDATE_BLOCKS and 34 not in CANDIDATE_STATES,
                   f"sites are historical {list(CANDIDATE_BLOCKS)} = package "
                   f"states {list(CANDIDATE_STATES)}, budgeted against "
                   f"{list(CANDIDATE_BUDGET_STATES)}")
    checks.require("alpha_R", ALPHA_R == 0.15, f"alpha_R = {ALPHA_R}")
    checks.require("alpha_S", ALPHA_S == 0.0,
                   "d_S = H - E is retired; no coefficient scales it")

    # ---- the controller bundle -------------------------------------------
    bundle = load_bundle(args.controllers)
    checks.require("five_fold_controllers", len(bundle.folds) == N_SPLITS,
                   f"{len(bundle.folds)} fold controller(s), expected "
                   f"{N_SPLITS}")
    checks.require(
        "every_fold_controller_is_out_of_fold",
        all(c["is_out_of_fold"] for c in bundle.folds.values()),
        "a row is scored only by the controller that never saw its fold")
    checks.record(
        "deployment_refit_present", bundle.refit is not None,
        "the all-development refit exists but never scores a development row"
        if bundle.refit is not None else "no refit in this bundle")
    provenance = verify_bundle(args.controllers, bundle, checks)
    summary, _diagnostic = load_reproduction(args.controllers)

    # ---- the direction ----------------------------------------------------
    bank = load_validated_bank(Path(args.direction_bank), expected_bank, checks)
    direction = assert_repair_direction_is_b_d1_minus_e(bank, checks)

    # ---- the corpus, its order and its frozen folds -----------------------
    from routegate.data import load_development_inputs
    from routegate.ebh import frame_from_inputs
    from routegate.probeheads import state_matrix

    inputs = load_development_inputs(**({"paths_path": args.paths}
                                        if args.paths else {}))
    frame = frame_from_inputs(inputs)
    checks.require("corpus_rows", frame.n == EXPECTED_ROWS,
                   f"{frame.n} row(s), expected {EXPECTED_ROWS}")

    # ``deterministic_row_order`` takes the corpus *records* and reads
    # ``row_id`` off each one, which is how both two-site stages call it.
    # ``inputs.manifest_rows`` is the same 480 rows the frame is built from,
    # so the sorted order is unchanged -- and it is deliberately not
    # ``inputs.row_ids``, which is the 267-row E/B projection.
    corpus_rows = list(inputs.manifest_rows)
    corpus = {r["row_id"]: r for r in corpus_rows}
    sealed = [r for r in corpus if looks_sealed(r)]
    checks.require("no_sealed_row_id", not sealed,
                   f"{len(sealed)} row id(s) read as held-out material")

    order = deterministic_row_order(corpus_rows)
    checks.require("row_order_is_the_whole_corpus",
                   len(order) == EXPECTED_ROWS,
                   f"{len(order)} row(s) in the order, expected "
                   f"{EXPECTED_ROWS}")
    checks.require(
        "row_order_matches_the_activation_frame",
        set(order) == set(frame.row_ids),
        "the sharded order and the activation frame name the same rows; "
        "otherwise a coefficient would be planned for a row the frame does "
        "not hold, or a row the frame holds would never be planned")
    fold_of, fold_source = fold_map(order, checks=checks)

    # ---- the reference coefficients, and what they must reproduce ---------
    design = state_matrix(frame, READ_STATE)
    position = frame.position_of
    frozen = {rid: design[position[rid]] for rid in order}
    reference = reference_gates(frozen, fold_of, bundle)
    reproduction = assert_reference_reproduces_the_run(
        reference, fold_of, summary, checks)

    # ---- the shard --------------------------------------------------------
    assert_shards_partition(order, int(args.shard_count), checks)
    mine = shard_rows(order, int(args.shard_index), int(args.shard_count))
    checks.record("shard", True,
                  f"shard {args.shard_index}/{args.shard_count}: "
                  f"{len(mine)} row(s)"
                  + (" (sequential)" if args.shard_count == 1 else ""))

    generation = GenerationConfig()
    generation.validate()
    locked = locked_fields(
        direction_bank_sha256=bank.artifact_sha256,
        dataset_sha256=inputs.dataset_sha256,
        generation_config_sha256=generation.fingerprint(),
        model=FROZEN_MODEL, revision=FROZEN_REVISION,
        tokenizer_config_sha256=FROZEN_TOKENIZER_CONFIG_SHA256,
        chat_template_sha256=FROZEN_CHAT_TEMPLATE_SHA256,
        intervention_type=INTERVENTION_TYPE,
        budget_reference=BUDGET_REFERENCE, arm_id=args.arm)

    return {"bundle": bundle, "bank": bank, "corpus": corpus, "order": order,
            "fold_of": fold_of, "fold_source": fold_source,
            "reference": reference, "frozen": frozen, "shard": mine,
            "generation": generation, "locked": locked,
            "provenance": provenance, "direction": direction,
            "reproduction": reproduction,
            "dataset_sha256": inputs.dataset_sha256}


def announce(args: argparse.Namespace, prepared: Mapping[str, Any]) -> None:
    reference = prepared["reference"]
    values = sorted(reference.values())
    print(f"[dynamic] arm {args.arm}")
    print(f"[dynamic] sites historical blocks {list(CANDIDATE_BLOCKS)} -> "
          f"package states {list(CANDIDATE_STATES)}, budgeted against "
          f"{list(CANDIDATE_BUDGET_STATES)}; block 33 is not a site")
    print(f"[dynamic] alpha_R {ALPHA_R}, alpha_S {ALPHA_S}, one g_R per "
          f"prompt reused at both sites")
    print(f"[dynamic] d_R = {prepared['direction']['subtraction_order']}, bank "
          f"{prepared['bank'].artifact_sha256[:16]}")
    print(f"[dynamic] {len(prepared['bundle'].folds)} fold controller(s); a "
          f"row is scored by the one that never saw it")
    print(f"[dynamic] reference coefficients from the frozen state-"
          f"{READ_STATE} activations: min {values[0]:.6e}, median "
          f"{values[len(values) // 2]:.6f}, max {values[-1]:.6f} "
          f"(every one strictly positive)")
    print(f"[dynamic] per-fold held-out means reproduce gate_oof_v1's own to "
          f"{prepared['reproduction']['max_abs_difference']:.3e} "
          f"(budget {GATE_TOLERANCE:.3e} = {GATE_TOLERANCE_EXPRESSION})")
    print(f"[dynamic] shard {args.shard_index}/{args.shard_count}: "
          f"{len(prepared['shard'])} row(s)")


def generate(args: argparse.Namespace, prepared: Mapping[str, Any],
             checks: Checklist) -> int:
    """This shard's rows, one generation forward pass each."""
    bundle, bank = prepared["bundle"], prepared["bank"]
    corpus, fold_of = prepared["corpus"], prepared["fold_of"]
    reference, frozen = prepared["reference"], prepared["frozen"]
    generation, locked = prepared["generation"], prepared["locked"]

    records_name = shard_records_name(int(args.shard_index),
                                      int(args.shard_count))
    output_dir = prepare_output_dir(
        args.output_dir,
        shard_filenames(int(args.shard_index), int(args.shard_count)),
        bool(args.resume))
    records_path = Path(output_dir) / records_name
    done = (validate_existing_records(records_path, prepared["shard"], locked,
                                      corpus, fold_of, reference, checks)
            if args.resume else {})
    if args.resume and records_path.exists():
        assert_shard_is_not_complete(done, prepared["shard"])
    remaining = [rid for rid in prepared["shard"] if rid not in done]

    print(f"[dynamic]   already written in this shard: {len(done)}")
    print(f"[dynamic]   to generate now             : {len(remaining)}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from routegate.hooks import PatchController
    from routegate.runtime import (capture_prompt_states, generate_response,
                                   load_model_and_tokenizer)

    print(f"[dynamic] loading {FROZEN_MODEL} @ {FROZEN_REVISION}")
    model, tokenizer = load_model_and_tokenizer(generation)
    model.eval()

    from routegate.livecontroller import assert_no_parameter_requires_grad
    assert_no_parameter_requires_grad([model])
    checks.record("language_model_is_frozen", True,
                  "no parameter of the loaded model can accumulate a gradient")

    controller = PatchController(model, list(CANDIDATE_STATES),
                                 budget_reference=BUDGET_REFERENCE)
    written = 0
    bit_identical = 0
    worst_gate_difference = 0.0
    errors: List[Dict[str, str]] = []
    try:
        for position, rid in enumerate(remaining, start=1):
            row = corpus[rid]
            fold = int(fold_of[rid])
            checkpoint = bundle.controller_for_row(fold)
            try:
                # Read before write: the controller's input is the unsteered
                # state, captured in this pass, and not something an earlier
                # edit produced.
                capture_prompt_states(model, tokenizer, row["prompt_text"],
                                      controller, capture_name="target",
                                      example_id=rid, config=generation)
                capture = dict(controller.capture_for("target"))
                activation = np.asarray(capture[READ_STATE], dtype=np.float64)
                live = plan_live_edit(activation, fold, bundle)
            except DynamicArmError:
                raise
            except Exception as exc:                         # noqa: BLE001
                errors.append({"row_id": rid,
                               "error": f"{type(exc).__name__}: {exc}"})
                print(f"      ERROR {rid}: {errors[-1]['error']}",
                      file=sys.stderr)
                continue

            # Faithfulness is a gate, not a diagnostic: it is evaluated
            # outside the per-row error handler so a controller that cannot
            # score this activation stops the run instead of being recorded.
            delta = assert_gate_is_faithful(rid, live.g_r, reference[rid])
            worst_gate_difference = max(worst_gate_difference, delta)
            same = bool(np.array_equal(
                np.asarray(capture[READ_STATE], dtype=np.float32),
                np.asarray(frozen[rid], dtype=np.float32)))
            bit_identical += int(same)
            block = controller_block(fold, checkpoint, live.g_r,
                                     reference[rid], same)

            try:
                plans = build_two_site_plans(
                    bank, capture, alpha=ALPHA_R, g_r=live.g_r,
                    blocks=CANDIDATE_BLOCKS)
                generated = generate_response(
                    model, tokenizer, row["prompt_text"], generation,
                    controller=controller, mode="patch", plans=plans,
                    example_id=rid)
            except Exception as exc:                         # noqa: BLE001
                errors.append({"row_id": rid,
                               "error": f"{type(exc).__name__}: {exc}"})
                print(f"      ERROR {rid}: {errors[-1]['error']}",
                      file=sys.stderr)
                continue

            per_state = {
                str(state): {
                    **plans[state].diagnostics(),
                    "repair_direction_sha256": bank.direction_sha256(
                        "repair_direction", state),
                    "controller": dict(block),
                } for state in sorted(plans)}
            status = ("ok" if all(p.status == "ok" for p in plans.values())
                      else "zero_delta")

            append_record(records_path, make_record(
                row_id=rid, row_class=row["final_ebh"],
                prompt_sha256=row["prompt_sha256"],
                alpha_R=float(live.applied_alpha),
                status=status, error_message="", per_state=per_state,
                hooks_expected=len(plans),
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
                print(f"[dynamic] {position}/{len(remaining)} generated",
                      flush=True)
    finally:
        controller.close()

    checks.record(
        "live_coefficients_reproduce_the_frozen_ones", True,
        f"{written} row(s) generated; worst |live g_R - frozen g_R| "
        f"{worst_gate_difference:.3e} against {GATE_TOLERANCE:.3e}; "
        f"{bit_identical}/{written} live captures bit-identical to the frozen "
        f"extraction" if written else "no row generated in this process")

    complete = dict(done)
    complete.update({r["row_id"]: r for r in
                     validate_existing_records(records_path, prepared["shard"],
                                               locked, corpus, fold_of,
                                               reference).values()})
    rows = coefficient_rows(complete, prepared["shard"])
    write_csv(Path(output_dir) / COEFFICIENTS_FILENAME,
              list(COEFFICIENT_COLUMNS), rows)

    manifest = {
        "schema": SCHEMA, "protocol": PROTOCOL, "version": 1,
        "created_utc": utc_now(), "package_version": PACKAGE_VERSION,
        "arm": args.arm,
        "shard": {"index": int(args.shard_index),
                  "count": int(args.shard_count),
                  "rows": len(prepared["shard"]),
                  "generated_here": written,
                  "already_present": len(done),
                  "complete": len(complete),
                  "records_file": records_name},
        "expected_rows_when_every_shard_finishes": EXPECTED_ROWS,
        "coefficient": {
            "mechanism": "one g_R per prompt from the live unsteered state-"
                         f"{READ_STATE} final-prompt-token activation, scored "
                         "by the fold controller that never trained on the "
                         "row; no row-id coefficient lookup",
            "alpha_R": ALPHA_R,
            "applied_at_historical_blocks": list(CANDIDATE_BLOCKS),
            "same_coefficient_at_every_site": True,
            "worst_abs_gate_difference": worst_gate_difference,
            "gate_tolerance": GATE_TOLERANCE,
            "gate_tolerance_expression": GATE_TOLERANCE_EXPRESSION,
            "live_captures_bit_identical_to_the_frozen_extraction":
                bit_identical,
        },
        "controller": prepared["provenance"],
        "folds": prepared["fold_source"],
        "reproduction": prepared["reproduction"],
        "direction": prepared["direction"],
        "generation_forward_passes_per_prompt": 1,
        "forward_passes_per_prompt": {
            "generation": 1,
            "prompt_prefill_capture": 1,
            "note": "one generation per prompt. The capture is a prefill-only "
                    "forward that emits no token and exists so the controller "
                    "reads the unsteered state before either site is written; "
                    "it is the same read-before-write shape the validated "
                    "static two-site arm used.",
        },
        "judges_responses": False, "trains_anything": False,
        "reads_the_sealed_set": False,
        "errors": errors,
        **prepared["locked"],
        "environment": environment_record(PACKAGE_VERSION, model_loaded=True,
                                          generation_performed=True),
        "checklist": checks.as_dict(),
    }
    write_json(Path(output_dir) / OUTPUT_MANIFEST, manifest)
    write_json(Path(output_dir) / OUTPUT_VALIDATION, checks.as_dict())
    print(f"[dynamic] wrote {output_dir}")
    print(f"[dynamic]   {records_name} {sha256_file(records_path)}")
    print(f"[dynamic] {written} generated, {len(errors)} error(s), "
          f"{len(complete)}/{len(prepared['shard'])} complete in this shard")
    return 0 if not errors and checks.all_passed else 1


def verify(args: argparse.Namespace, prepared: Mapping[str, Any],
           checks: Checklist) -> int:
    """A finished directory, re-validated. Loads no model, writes nothing."""
    records, files = collect_records(
        args.output_dir, prepared["locked"], prepared["corpus"],
        prepared["fold_of"], prepared["reference"], checks)
    reading = assert_complete(records, prepared["order"], checks)
    print(f"[dynamic] verified {args.output_dir}")
    for name, count in sorted(files["files"].items()):
        print(f"[dynamic]   {name}: {count} record(s)")
    print(f"[dynamic] {reading['records']}/{reading['rows']} completed "
          f"record(s), every one out-of-fold, at historical blocks "
          f"{list(CANDIDATE_BLOCKS)}")
    print(f"[dynamic] applied coefficient: min "
          f"{reading['min_applied_alpha_R']:.6f}, mean "
          f"{reading['mean_applied_alpha_R']:.6f}, max "
          f"{reading['max_applied_alpha_R']:.6f}, "
          f"{reading['distinct_applied_coefficients']} distinct value(s)")
    print(f"[dynamic] {checks.as_dict()['checks_run']} check(s), all passed: "
          f"{checks.all_passed}")
    print(f"[dynamic] verification wrote nothing")
    return 0 if checks.all_passed else 2


def run(args: argparse.Namespace) -> int:
    if args.dry_run and args.verify:
        raise DynamicArmError(
            "--dry-run plans a run and --verify reads a finished one; they "
            "are not the same question and are not combined")
    if args.arm == ARM_VETO and not args.dry_run:
        raise DynamicArmError(
            f"{ARM_VETO} generates nothing. On a vetoed row the applied "
            f"coefficient is exactly 0.0, so the edit is the zero vector and "
            f"the response is the referenced {REFERENCE_ARM}; on every other "
            f"row it is {ARM_RAW}'s own response. The arm is a deterministic "
            f"recombination and is assembled from records that already exist.")

    checks = Checklist(f"dynamic_arms_{args.arm}")
    prepared = prepare(args, checks)
    announce(args, prepared)

    plan = plan_rows(prepared["bundle"], prepared["frozen"],
                     prepared["fold_of"], {}, args.arm,
                     gates=prepared["reference"])
    arithmetic = reuse_arithmetic(plan, args.arm)
    print(f"[dynamic] {arithmetic['new']} row(s) genuinely new, "
          f"{arithmetic['reused']} reused: {arithmetic['why']}")

    if args.verify:
        return verify(args, prepared, checks)

    if args.dry_run:
        print(f"[dynamic] dry run: no model loaded, no response generated, "
              f"nothing written")
        print(f"[dynamic] output directory {args.output_dir} is claimable and "
              f"was deliberately NOT created")
        print(f"[dynamic] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2

    return generate(args, prepared, checks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
