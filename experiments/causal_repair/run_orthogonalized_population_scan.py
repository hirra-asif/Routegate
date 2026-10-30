#!/usr/bin/env python3
"""Layer-wise causal activation steering over all 48 transformer blocks.

For each held-out target this runs two conditions:

    baseline                          the model untouched -- generated once per
                                      target, because it does not depend on the
                                      state
    orthogonalized_population_repair  the cross-fitted, H-orthogonalised repair
                                      direction, ADDED at one transformer-block
                                      output at the final prompt token

The edit, for a target held out in fold f at block-output state l:

    applied_delta = repair_direction[f, l] * strength * ||target block-input activation||

The direction is not estimated here. It is read from a bank frozen and hashed by
`build_orthogonalized_direction_bank.py` before any model was loaded, and this
script refuses a bank whose manifest hash is not the one it was told to expect.
It also re-checks, per generation, that the target is recorded as held out in the
fold whose direction it is about to receive.

This is steering, not patching: nothing is copied from another prompt, nothing is
replaced, and no donor is involved. Exact activation patching with semantically
matched E/B pairs is a later confirmatory experiment at whichever state(s) this
scan selects.

Intervention sites are the 48 transformer-block outputs, states 1-48. State 0 is
the embedding output before block 0; it is stored in the bank for descriptive
analysis and is refused here.

Usage:
    # what the full scan would do, without loading the model
    python experiments/causal_repair/run_orthogonalized_population_scan.py \\
        --dry-run --all-targets --all-block-states \\
        --conditions baseline,orthogonalized_population_repair \\
        --direction-bank outputs/causal_repair/orthogonalized_population_scan/direction_bank

    # one array task's slice, resumable
    python experiments/causal_repair/run_orthogonalized_population_scan.py \\
        --all-targets --states 1-7 \\
        --conditions baseline,orthogonalized_population_repair \\
        --direction-bank outputs/causal_repair/orthogonalized_population_scan/direction_bank \\
        --expect-bank-sha256 <hash> \\
        --output-dir outputs/causal_repair/orthogonalized_population_scan/scan/task_1 \\
        --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import (                                        # noqa: E402
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SOURCE_COLUMN,
    WITHHELD_LABEL,
    describe_state,
    load_input_manifest,
    load_local_paths,
)
from routegate.intervention import (                                # noqa: E402
    BUDGET_REFERENCES,
    budget_reference_state,
)
from routegate.runrecord import (                                   # noqa: E402
    environment_record,
    utc_now,
    write_json,
)
from routegate.runtime import (                                     # noqa: E402
    EXPECTED_NUM_LAYERS,
    GenerationConfig,
    capture_prompt_states,
    generate_response,
)
from routegate.steering import (                                    # noqa: E402
    CAUSAL_SCAN_STATES,
    FROZEN_SCAN_STRENGTH,
    INTERVENTION_TYPE,
    STATE_EMBEDDING,
    SteeringError,
    build_steering_plan,
    load_direction_bank,
)
from routegate.steerrecord import (                                 # noqa: E402
    CONDITION_BASELINE,
    CONDITION_REPAIR,
    CONDITIONS,
    append_record,
    completed_keys,
    make_record,
    plan_work,
    read_records,
    transformer_block_of_state,
    verify_consistency,
)
from routegate.validation import (                                  # noqa: E402
    ValidationError,
    assert_config_is_sealed_free,
    assert_hash,
    assert_not_sealed,
    assert_not_stale,
)
from routegate import __version__ as PACKAGE_VERSION                # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "orthogonalized_population_scan.json"
SCAN_ROOT = REPO_ROOT / "outputs" / "causal_repair" / "orthogonalized_population_scan"
DEFAULT_BANK = SCAN_ROOT / "direction_bank"
DEFAULT_OUTPUT = SCAN_ROOT / "scan"
RECORDS_FILENAME = "steer_records.jsonl"

# The classes that receive an intervention. B builds the directions and is not a
# target during this discovery scan; its preservation is checked later, at the
# selected state(s), where a control costs a few hundred generations rather than
# fifteen thousand.
TARGET_LABELS: Tuple[str, str] = (POSITIVE_LABEL, WITHHELD_LABEL)

# One --states chunk: a signed integer, optionally with an inclusive upper bound.
_STATE_CHUNK = re.compile(r"^(?P<low>[+-]?\d+)(?:-(?P<high>[+-]?\d+))?$")


def parse_states(spec: str, n_states: int) -> List[int]:
    """'1,8-14' -> [1, 8, ..., 14], validated against the model and the protocol.

    State 0 is refused rather than silently dropped. It is the embedding output
    before transformer block 0, it is not a transformer layer, and it is not part
    of the primary causal block scan -- a command line that asks for it has
    misread the protocol, and quietly ignoring it would put a run's state list
    out of step with what was requested.
    """
    bounds: List[int] = []
    states: List[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _STATE_CHUNK.match(chunk)
        if match is None:
            raise ValidationError(
                f"cannot read {chunk!r} as a state: write a single state ('7'), "
                f"an inclusive range ('8-14'), or a comma-separated list of "
                f"either, within 1..{n_states - 1}")
        low = int(match.group("low"))
        high = int(match.group("high")) if match.group("high") is not None else low
        if high < low:
            raise ValidationError(
                f"the range {chunk!r} ends before it starts; write it as "
                f"{high}-{low} if that is what was meant")
        bounds.extend((low, high))
        states.extend(range(low, high + 1))

    outside = sorted({b for b in bounds if not 0 <= b < n_states})
    if outside:
        raise ValidationError(
            f"state(s) {outside} are outside 0..{n_states - 1}")
    if STATE_EMBEDDING in states:
        raise ValidationError(
            "state 0 is the embedding output before transformer block 0, not a "
            "transformer layer. The primary causal scan intervenes at states "
            f"1-{n_states - 1}, the outputs of the {n_states - 1} transformer "
            "blocks, and state 0 is excluded from it and from the layer ranking. "
            "It is retained in the direction bank for descriptive analysis only.")
    return sorted(set(states))


def parse_list(spec: Optional[str]) -> List[str]:
    return [v.strip() for v in str(spec or "").split(",") if v.strip()]


def load_target_table(dataset_path: Path, expected_sha256: str
                      ) -> Dict[str, Dict[str, str]]:
    """row_id -> prompt, class, source family and the recorded original response.

    Every eligible row is read, including B, so that a B row offered as a target
    is refused for being B rather than for being unknown.
    """
    path = assert_not_stale(Path(dataset_path))
    assert_not_sealed(path, "population steering dataset")
    assert_hash(path, expected_sha256, "dataset")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row["final_ebh"] not in (POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL):
            continue
        table[row["row_id"]] = {
            "prompt_text": row["prompt_text"],
            "final_ebh": row["final_ebh"],
            "source_family": row[SOURCE_COLUMN],
            "original_response": row.get("response_text", ""),
        }
    if not table:
        raise ValidationError(f"{path} carries no eligible rows")
    return table


def resolve_targets(bank: Any, table: Dict[str, Dict[str, str]],
                    requested: Sequence[str], all_targets: bool,
                    target_labels: Sequence[str]) -> List[str]:
    """The targets this run will steer, checked against the frozen assignment."""
    known = {rid: label for rid, label in bank.label_of_row.items()}
    if all_targets:
        targets = sorted(rid for rid, label in known.items()
                         if label in target_labels)
    else:
        targets = list(requested)
    if not targets:
        raise ValidationError(
            "no target selected; use --targets or --all-targets")

    unknown = [t for t in targets if t not in known]
    if unknown:
        raise ValidationError(
            f"{len(unknown)} requested target(s) are not in the frozen fold "
            f"assignment (first: {unknown[0]}). Targets are not invented at run "
            f"time.")
    missing = [t for t in targets if t not in table]
    if missing:
        raise ValidationError(
            f"{len(missing)} requested target(s) are not in the accepted corpus "
            f"(first: {missing[0]})")

    disagree = [t for t in targets if table[t]["final_ebh"] != known[t]]
    if disagree:
        raise ValidationError(
            f"{len(disagree)} target(s) are labelled differently by the corpus "
            f"and by the frozen fold assignment (first: {disagree[0]}); the bank "
            f"was built against a different dataset")

    benign = [t for t in targets if known[t] == NEGATIVE_LABEL]
    if benign:
        raise ValidationError(
            f"{len(benign)} requested target(s) are class {NEGATIVE_LABEL} "
            f"(first: {benign[0]}). B rows construct the directions and are not "
            f"intervention targets in this discovery scan; B preservation is a "
            f"later control at the selected state(s).")
    wrong = [t for t in targets if known[t] not in target_labels]
    if wrong:
        raise ValidationError(
            f"{len(wrong)} requested target(s) are not of class "
            f"{list(target_labels)} (first: {wrong[0]})")
    return sorted(set(targets))


def build_work_list(targets: Sequence[str], states: Sequence[int],
                    conditions: Sequence[str], strength: float
                    ) -> List[Dict[str, Any]]:
    """Every generation this run intends, target-major then state-ascending.

    Target-major is not cosmetic: one capture forward per target supplies the
    budget reference for every state that target is steered at, so the ordering
    is what keeps the scan to one extra prefill per target rather than one per
    generation.

    The baseline is one generation per target, not one per state. Its record
    carries a blank state, which is how this package writes "not applicable"
    without writing a number that could be sorted next to a real state.
    """
    work: List[Dict[str, Any]] = []
    for target in targets:
        if CONDITION_BASELINE in conditions:
            work.append({"target_row_id": target, "state": "",
                         "condition": CONDITION_BASELINE,
                         "intervention_type": "none",
                         "intervention_strength": 0.0})
        if CONDITION_REPAIR in conditions:
            for state in states:
                work.append({"target_row_id": target, "state": int(state),
                             "condition": CONDITION_REPAIR,
                             "intervention_type": INTERVENTION_TYPE,
                             "intervention_strength": float(strength)})
    return work


def work_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (item["target_row_id"], item["state"], item["condition"],
            item["intervention_type"], item["intervention_strength"])


def count_work(work: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    intervention = sum(1 for item in work if item["condition"] == CONDITION_REPAIR)
    baseline = sum(1 for item in work if item["condition"] == CONDITION_BASELINE)
    return {"intervention": intervention, "baseline": baseline,
            "total": intervention + baseline}


def run(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "orthogonalized population scan config")

    generation = GenerationConfig(
        max_new_tokens=int(args.max_new_tokens
                           or config["generation"]["max_new_tokens"]))
    generation.validate()

    if args.budget_reference not in BUDGET_REFERENCES:
        raise ValidationError(
            f"unknown budget reference {args.budget_reference!r}; valid: "
            f"{list(BUDGET_REFERENCES)}")

    strength = float(args.strength)
    if strength != FROZEN_SCAN_STRENGTH:
        raise ValidationError(
            f"strength {strength} is not the frozen scan strength "
            f"{FROZEN_SCAN_STRENGTH}. That value is a pre-existing pilot "
            f"operating point, not a measured optimum; it is held fixed here so "
            f"that block depth and edit magnitude are not tuned jointly across "
            f"48 blocks. A strength-sensitivity analysis is a separate run at "
            f"the state(s) this scan selects.")

    conditions = parse_list(args.conditions)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValidationError(
            f"unknown condition(s) {unknown}; valid: {list(CONDITIONS)}")
    if not conditions:
        raise ValidationError("no condition selected; nothing to run")

    n_states = EXPECTED_NUM_LAYERS + 1
    states = (list(CAUSAL_SCAN_STATES) if args.all_block_states
              else parse_states(args.states or "", n_states))
    if not states and CONDITION_REPAIR in conditions:
        raise ValidationError(
            "an intervention condition was requested but no state was selected; "
            "use --states or --all-block-states")

    # --- the frozen bank, and the corpus it was built against -----------------
    input_manifest = load_input_manifest()
    local_paths = load_local_paths(args.paths) if args.paths else load_local_paths()
    dataset_path = Path(local_paths["dataset"])
    dataset_sha = input_manifest["dataset"]["sha256"]

    bank = load_direction_bank(args.direction_bank, args.expect_bank_sha256)
    if bank.dataset_sha256 != dataset_sha:
        raise SteeringError(
            f"the direction bank was built against dataset "
            f"{bank.dataset_sha256} but this run reads {dataset_sha}")
    if float(bank.strength) != strength:
        raise SteeringError(
            f"the bank declares strength {bank.strength} and this run was asked "
            f"for {strength}")
    table = load_target_table(dataset_path, dataset_sha)
    targets = resolve_targets(bank, table, parse_list(args.targets),
                              args.all_targets, TARGET_LABELS)
    if args.limit:
        targets = targets[:int(args.limit)]

    # Every direction this run will reach for must exist, be usable, and hash to
    # what the manifest says -- checked before the model is loaded, not on the
    # generation that needs it.
    for state in states:
        for fold in range(bank.n_folds):
            bank.repair_direction(fold, state)

    # --- the work list, and what of it is already done ----------------------
    output_dir = Path(args.output_dir or DEFAULT_OUTPUT)
    if args.smoke and "smoke" not in output_dir.name.lower():
        raise ValidationError(
            f"a smoke run must write to a directory whose name says so; "
            f"{output_dir.name!r} does not contain 'smoke'")
    records_path = output_dir / RECORDS_FILENAME

    work = build_work_list(targets, states, conditions, strength)
    counts = count_work(work)
    existing = read_records(records_path) if records_path.exists() else []
    if existing and not args.resume:
        raise ValidationError(
            f"{records_path} already holds {len(existing)} record(s). Pass "
            f"--resume to continue it, or choose another --output-dir; this "
            f"script will not append to a run it was not told to continue.")
    summary = verify_consistency(
        existing,
        generation_config_sha256=generation.fingerprint(),
        direction_bank_sha256=bank.manifest_sha256,
        dataset_sha256=dataset_sha)
    done = completed_keys(existing)
    todo, skipped = plan_work([work_key(item) for item in work], done)
    by_key = {work_key(item): item for item in work}

    by_label = {label: sum(1 for t in targets if table[t]["final_ebh"] == label)
                for label in TARGET_LABELS}
    print(f"[plan] {len(targets)} target(s): "
          + ", ".join(f"{label}={by_label[label]}" for label in TARGET_LABELS))
    print(f"[plan] {len(states)} transformer-block state(s): "
          f"{states[0] if states else '—'}"
          f"{'..' + str(states[-1]) if len(states) > 1 else ''} "
          f"(block outputs; state 0 is excluded)")
    print(f"[plan] conditions {conditions}, strength {strength}, "
          f"budget reference {args.budget_reference}")
    print(f"[plan] direction bank {Path(args.direction_bank)} "
          f"@ {bank.manifest_sha256}")
    print(f"[plan] model {generation.model} @ {generation.revision}, greedy, "
          f"max_new_tokens={generation.max_new_tokens}, thinking disabled")
    print()
    print(f"[counts] intervention generations : {counts['intervention']}")
    print(f"[counts] baseline generations     : {counts['baseline']}")
    print(f"[counts] total generations        : {counts['total']}")
    print()
    print(f"[plan] {len(skipped)} already done, {len(todo)} to run")
    if existing:
        print(f"[plan] resuming into {records_path} ({summary})")

    if args.dry_run:
        print("\n[dry-run] no model was loaded and nothing was generated.")
        for key in todo[:10]:
            item = by_key[key]
            target = item["target_row_id"]
            state = item["state"]
            print(f"          {item['condition']:<32s} {target} "
                  f"[{table[target]['final_ebh']}, fold {bank.fold_of(target)}]  "
                  f"{describe_state(state) if state != '' else 'no state'}")
        if len(todo) > 10:
            print(f"          ... and {len(todo) - 10} more")
        return 0

    if not todo:
        print("[done] nothing to do; every requested generation is already "
              "recorded")
        return 0

    # --- execution ----------------------------------------------------------
    from routegate.hooks import PatchController          # torch only from here
    from routegate.runtime import load_model_and_tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or utc_now().replace(":", "").replace("-", "")
    write_json(output_dir / "run_config.json", {
        "run_id": run_id,
        "created_utc": utc_now(),
        "package_version": PACKAGE_VERSION,
        "config": config,
        "generation": generation.as_dict(),
        "generation_config_sha256": generation.fingerprint(),
        "direction_bank": str(args.direction_bank),
        "direction_bank_sha256": bank.manifest_sha256,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "states": states,
        "state_0_excluded": True,
        "conditions": conditions,
        "strength": strength,
        "budget_reference": args.budget_reference,
        "intervention_type": INTERVENTION_TYPE,
        "targets": targets,
        "requested_generations": counts,
        "smoke": bool(args.smoke),
        "reportable": not args.smoke,
        "environment": environment_record(PACKAGE_VERSION),
    })

    print(f"[load] {generation.model} @ {generation.revision}")
    model, tokenizer = load_model_and_tokenizer(generation)
    controller = PatchController(model, states=list(states),
                                 budget_reference=args.budget_reference)
    captured_target: Optional[str] = None
    capture: Dict[int, Any] = {}
    written = errors = 0

    try:
        for position, key in enumerate(todo, start=1):
            item = by_key[key]
            target = item["target_row_id"]
            state = item["state"]
            condition = item["condition"]
            label = table[target]["final_ebh"]
            fold = bank.fold_of(target)
            print(f"[{position}/{len(todo)}] {condition} {target} [{label}, "
                  f"fold {fold}] "
                  f"{'' if state == '' else describe_state(state)}", flush=True)

            plan = None
            direction_hash = ""
            direction_status = ""
            try:
                if condition == CONDITION_BASELINE:
                    generated = generate_response(
                        model, tokenizer, table[target]["prompt_text"],
                        generation, controller=controller, mode="off",
                        example_id=target)
                    status = "baseline"
                else:
                    # The cross-fitting guarantee, re-asserted per generation
                    # against the bank's own written record of which rows each
                    # fold trained on.
                    bank.assert_held_out(target, fold)
                    direction = bank.repair_direction(fold, state)
                    direction_hash = bank.direction_sha256(fold, state)
                    direction_status = bank.status_by_site[(fold, state)]

                    if captured_target != target:
                        capture_prompt_states(
                            model, tokenizer, table[target]["prompt_text"],
                            controller, capture_name="target", example_id=target,
                            config=generation)
                        capture = dict(controller.capture_for("target"))
                        captured_target = target

                    reference_state, _ = budget_reference_state(
                        state, args.budget_reference)
                    plan = build_steering_plan(
                        state=state, direction=direction, strength=strength,
                        reference_vector=capture[reference_state],
                        budget_reference=args.budget_reference)
                    generated = generate_response(
                        model, tokenizer, table[target]["prompt_text"],
                        generation, controller=controller, mode="patch",
                        plans={state: plan}, example_id=target)
                    status = "ok" if plan.status == "ok" else "zero_delta"
                error_message = ""
            except Exception as exc:                     # noqa: BLE001
                generated = {"response_text": "", "prompt_token_count": 0,
                             "completion_token_count": 0, "eos_reached": False,
                             "hit_max_new_tokens": False, "elapsed_seconds": 0.0}
                plan = None
                status = "error"
                error_message = f"{type(exc).__name__}: {exc}"
                errors += 1
                captured_target = None
                print(f"      ERROR {error_message}", file=sys.stderr)

            record = make_record(
                run_id=run_id,
                created_utc=utc_now(),
                target_row_id=target,
                target_label=label,
                target_fold=fold,
                target_source_family=table[target]["source_family"],
                condition=condition,
                state=state,
                transformer_block_index=transformer_block_of_state(state),
                intervention_type=item["intervention_type"],
                intervention_strength=float(item["intervention_strength"]),
                direction_sha256=direction_hash,
                direction_fold=(fold if condition == CONDITION_REPAIR else ""),
                direction_status=direction_status,
                budget_reference=args.budget_reference,
                budget_reference_state=(plan.budget_reference_state
                                        if plan is not None else ""),
                budget=(plan.budget if plan is not None else 0.0),
                reference_norm=(plan.reference_norm if plan is not None else 0.0),
                raw_delta_norm=(plan.raw_delta_norm if plan is not None else 0.0),
                applied_delta_norm=(plan.applied_delta_norm
                                    if plan is not None else 0.0),
                model=generation.model,
                model_revision=generation.revision,
                generation_settings=generation.as_dict(),
                generation_config_sha256=generation.fingerprint(),
                prompt_token_count=generated["prompt_token_count"],
                completion_token_count=generated["completion_token_count"],
                eos_reached=generated["eos_reached"],
                hit_max_new_tokens=generated["hit_max_new_tokens"],
                elapsed_seconds=generated["elapsed_seconds"],
                original_response=table[target]["original_response"],
                steered_response=generated["response_text"],
                output_status=status,
                error_message=error_message,
                reportable=bool(not args.smoke and status != "error"),
                reportable_reason=(
                    "smoke run: not reportable" if args.smoke
                    else "generation failed" if status == "error"
                    else "full run under the frozen configuration and bank"),
                dataset_sha256=dataset_sha,
                direction_bank_sha256=bank.manifest_sha256,
            )
            append_record(records_path, record)
            written += 1
    finally:
        controller.close()

    print(f"[done] {written} record(s) written to {records_path} "
          f"({errors} error(s))")
    return 1 if errors else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None)

    parser.add_argument("--direction-bank", type=Path, default=DEFAULT_BANK,
                        help="the frozen direction-bank directory")
    parser.add_argument("--expect-bank-sha256", default=None,
                        help="refuse any direction bank but this one")

    parser.add_argument("--targets", default=None,
                        help="comma-separated target row_ids")
    parser.add_argument("--all-targets", action="store_true",
                        help=f"every {POSITIVE_LABEL} and {WITHHELD_LABEL} row of "
                             f"the frozen fold assignment")
    parser.add_argument("--states", default=None,
                        help="comma-separated block-output states, ranges "
                             "allowed: 1,8-14. State 0 is refused.")
    parser.add_argument("--all-block-states", action="store_true",
                        help=f"every transformer-block output, states "
                             f"{CAUSAL_SCAN_STATES[0]}-{CAUSAL_SCAN_STATES[-1]}")
    parser.add_argument("--conditions",
                        default=f"{CONDITION_BASELINE},{CONDITION_REPAIR}",
                        help=f"comma-separated, from {list(CONDITIONS)}")
    parser.add_argument("--strength", type=float, default=FROZEN_SCAN_STRENGTH,
                        help="frozen at the pilot operating value for this scan")
    parser.add_argument("--budget-reference", default="block_input",
                        choices=list(BUDGET_REFERENCES))
    parser.add_argument("--max-new-tokens", type=int, default=None)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N targets only")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="plan the work and stop; loads no model")
    parser.add_argument("--smoke", action="store_true",
                        help="mark every record non-reportable; requires an "
                             "output directory named 'smoke'")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
