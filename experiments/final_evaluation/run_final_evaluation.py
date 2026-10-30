#!/usr/bin/env python3
"""One resumable runner for all nine final-evaluation arms.

    # validate an arm without loading a model or writing anything
    python experiments/final_evaluation/run_final_evaluation.py \
        --arm tinymmlu__baseline --dry-run

    # generate (foreground; needs a GPU session)
    python experiments/final_evaluation/run_final_evaluation.py \
        --arm tinymmlu__baseline

    # generate exactly ONE official row, into the normal arm output
    python experiments/final_evaluation/run_final_evaluation.py \
        --arm tinymmlu__baseline --item-index 0

    # read-only progress across every arm
    python experiments/final_evaluation/run_final_evaluation.py --progress

Nine arms: three datasets x three conditions, declared in
``manifests/routegate_final_evaluation_v5.json`` and read from it rather than
restated here. The two rejected hybrids are not conditions and cannot be
selected: ``--arm`` accepts only the nine frozen ids.

RESUMABLE
---------
Each arm appends to its own ``records.jsonl``. On restart the runner reads the
item ids already present, keeps them, and generates only what is missing. The
identical command resumes; nothing is overwritten and nothing is regenerated.

ONE ITEM AT A TIME
------------------
``--item-index N`` restricts the SAME loop to the one official row whose
``item_index`` is ``N``. It is a filter, not a second execution path: the
prompt, model, controller, hooks, generation call, record schema and output
root are the ordinary ones, and ``item_id``, ``item_index`` and the prompt
digest are unchanged. A later command WITHOUT ``--item-index`` reads the normal
resume set, skips that completed row and generates the remaining ones.

An arm is marked complete only when every official row is present, unique and
validates. A one-row invocation therefore never writes a completion artifact,
and every invocation appends a line to ``invocations.jsonl`` recording what it
found and what it wrote.

THE SEALED SPLIT
----------------
Its path is not stored in this repository and is not recorded in any output.
It is supplied at launch with ``--sealed-manifest``, and a sealed arm refuses
to start until every capability arm is complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.finaleval import (                                # noqa: E402
    ARMS, COMPLETION_FILE, CONDITIONS, COND_BASELINE, COND_DYNAMIC,
    COND_ROUTEGATE, DATASET_SPEC, DATA_ROOT_NAME, DS_SEALED, INVOCATIONS_FILE,
    MANIFEST_FILE, PREDECLARATION, RECORDS_FILE, RECORD_SCHEMA,
    VALIDATION_FILE, FinalEvaluationError,
    arm_id, arm_output_dir, expected_rows, verify_protocol)
from routegate.finalrunner import (                              # noqa: E402
    COND_BASELINE, COND_DYNAMIC, COND_ROUTEGATE, DYNAMIC_BLOCKS,
    DYNAMIC_HOOKS, DYNAMIC_READ_STATE, DYNAMIC_STATES, GATE_READ_STATE,
    STATIC_BLOCKS, STATIC_HOOKS, STATIC_STATES, TELEMETRY_FIELDS,
    ConditionRuntime, DispatchError, assert_record_is_complete, execute_item)
from routegate.gatesensors import load_bundle, load_thresholds  # noqa: E402
from routegate.runtime import (                                  # noqa: E402
    FROZEN_CHAT_TEMPLATE_SHA256, FROZEN_MODEL, FROZEN_REVISION,
    FROZEN_TOKENIZER_CONFIG_SHA256)
from routegate.triplets import sha256_text                       # noqa: E402
from routegate.validation import (                               # noqa: E402
    Checklist, ValidationError, assert_not_sealed, looks_sealed, sha256_file)

ARM_IDS = [arm_id(d, c) for d, c in ARMS]
SENSOR_DIR = REPO_ROOT / "outputs" / "routegate_gate_deployment_v1"
CONTROLLER_DIR = Path.home() / "routegate-data" / "gate_oof_v1_repro"
DEFAULT_DATA_ROOT = Path.home() / "routegate-data"


def split_arm(value: str) -> tuple:
    if value not in ARM_IDS:
        raise FinalEvaluationError(
            f"{value!r} is not one of the nine frozen arms {ARM_IDS}")
    dataset, condition = value.split("__", 1)
    return dataset, condition


def load_items(dataset: str, sealed_manifest: Optional[Path],
               checks: Checklist) -> List[Dict[str, str]]:
    """The prompts for one dataset. Hash-gated for capability."""
    spec = DATASET_SPEC[dataset]
    if dataset == DS_SEALED:
        if sealed_manifest is None:
            raise FinalEvaluationError(
                "the sealed split is supplied at launch with "
                "--sealed-manifest. Its path is not stored in this "
                "repository and cannot be inferred.")
        path = Path(sealed_manifest)
        if not path.is_file():
            raise FinalEvaluationError(f"{path} is not a file")
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        checks.require("sealed_row_count", len(rows) == spec["rows"],
                       f"{len(rows)} row(s), expected {spec['rows']}")
        items = [{"item_index": i,
                  "item_id": str(r.get("row_id") or r.get("id") or i),
                  "prompt": str(r["prompt_text"])}
                 for i, r in enumerate(rows)]
        checks.record("sealed_path_is_not_recorded", True,
                      "the sealed path is used and never written into any "
                      "record, manifest or validation file")
        return items
    import pandas as pd
    local = Path.home() / spec["local"]
    if not local.is_file():
        raise FinalEvaluationError(
            f"{local} is absent. Acquire it at the pinned revision first.")
    assert_not_sealed(local, f"{dataset} parquet")
    digest = sha256_file(local)
    checks.require(f"{dataset}_sha256", digest == spec["sha256"],
                   f"{digest}, pinned {spec['sha256']}")
    frame = pd.read_parquet(local)
    checks.require(f"{dataset}_rows", len(frame) == spec["rows"],
                   f"{len(frame)} row(s), expected {spec['rows']}")
    column = spec["prompt_column"]
    checks.require(f"{dataset}_prompt_column", column in frame.columns,
                   f"{column!r} in {list(frame.columns)}")
    return [{"item_index": i, "item_id": f"{dataset}:{i:04d}",
             "prompt": str(v)}
            for i, v in enumerate(frame[column].tolist())]


def completed_items(records_path: Path) -> Dict[str, int]:
    """Item ids already recorded, so a restart regenerates none of them."""
    done: Dict[str, int] = {}
    if not records_path.is_file():
        return done
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[str(row["item_id"])] = int(row.get("item_index", -1))
    return done


def select_items(items: List[Dict[str, Any]], item_index: Optional[int],
                 want: int, checks: Checklist) -> List[Dict[str, Any]]:
    """The official rows this invocation will consider. A filter, nothing more.

    Without ``--item-index`` this is every row, in order, exactly as before.
    With it, it is the single row whose ORIGINAL ``item_index`` is that value:
    the id, the index, the prompt and the ordering are untouched.
    """
    if item_index is None:
        checks.record("item_selection", True,
                      f"all {len(items)} official row(s), in order")
        return items
    index = int(item_index)
    if not 0 <= index < want:
        raise FinalEvaluationError(
            f"--item-index {index} is outside [0, {want}) for this dataset. "
            f"The index is checked before anything is loaded.")
    chosen = [i for i in items if int(i["item_index"]) == index]
    if len(chosen) != 1:
        raise FinalEvaluationError(
            f"--item-index {index} selected {len(chosen)} row(s); the "
            f"official order carries each index exactly once")
    checks.record("item_selection", True,
                  f"one official row: item_index {index}, id "
                  f"{chosen[0]['item_id']}")
    return chosen


def append_invocation(path: Path, entry: Mapping[str, Any]) -> None:
    """Append-only audit trail: what this invocation found and wrote."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def finalise_if_complete(out_dir: Path, records_path: Path, want: int,
                         checks: Checklist) -> bool:
    """Mark the arm complete ONLY when every official row is present and unique.

    A partial run writes nothing here. Completion is a claim about all ``want``
    rows and is never made on the strength of one.
    """
    done = completed_items(records_path)
    if len(done) != want:
        checks.record("arm_is_incomplete", True,
                      f"{len(done)}/{want} row(s) present; no completion "
                      f"artifact is written")
        return False
    indexes = sorted(done.values())
    if indexes != list(range(want)):
        raise FinalEvaluationError(
            f"{records_path} holds {len(done)} unique id(s) but their "
            f"item_index set is not 0..{want - 1}; the arm is not complete")
    payload = {"complete": True, "rows": want,
               "records_file": records_path.name,
               "records_sha256": sha256_file(records_path),
               "protocol": PREDECLARATION.stem,
               "record_schema": RECORD_SCHEMA,
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime())}
    (out_dir / COMPLETION_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks.record("arm_is_complete", True,
                  f"{want}/{want} unique row(s); {COMPLETION_FILE} written")
    return True


def progress(data_root: Path) -> int:
    """Read-only: what every arm has, and what it still needs."""
    print(f"[progress] root {data_root / DATA_ROOT_NAME}")
    total_done = total_want = 0
    for dataset, condition in ARMS:
        name = arm_id(dataset, condition)
        want = expected_rows(dataset)
        path = data_root / arm_output_dir(dataset, condition) / RECORDS_FILE
        have = len(completed_items(path))
        total_done += have
        total_want += want
        state = ("complete" if have == want
                 else ("not started" if have == 0 else "partial"))
        print(f"[progress]   {name:<38} {have:>4}/{want:<4} {state}")
    print(f"[progress] TOTAL {total_done}/{total_want} "
          f"({100.0 * total_done / total_want:.1f}%)")
    return 0


def prepare_condition(condition: str, checks: Checklist) -> ConditionRuntime:
    """Everything a condition needs, verified, before any model is loaded.

    The object returned is REQUIRED by execute_item. It is not advisory and it
    is not discarded: v1's defect was exactly that.
    """
    if condition == COND_BASELINE:
        checks.record("condition", True,
                      "baseline: no controller, no hook, mode off")
        return ConditionRuntime(condition=condition, steers=False,
                                hooks_expected=0)

    if condition == COND_ROUTEGATE:
        from routegate.actuator import PRIMARY_VARIANT, load_frozen_actuator
        from routegate.finaleval import SENSOR_WEIGHTS_SHA256
        thresholds = load_thresholds()
        bundle = load_bundle(SENSOR_DIR, thresholds)
        checks.require("sensor_bundle_is_the_deployment_refit",
                       bundle.e_sensor.kind == "all_development_refit"
                       and bundle.h_veto.kind == "all_development_refit",
                       f"{bundle.e_sensor.kind}")
        checks.require(
            "sensor_weights_match_the_frozen_protocol",
            bundle.e_sensor.weights_sha256()
            == SENSOR_WEIGHTS_SHA256["E_sensor"]
            and bundle.h_veto.weights_sha256()
            == SENSOR_WEIGHTS_SHA256["H_veto"],
            f"E {bundle.e_sensor.weights_sha256()[:12]} "
            f"H {bundle.h_veto.weights_sha256()[:12]}")
        actuator = load_frozen_actuator(PRIMARY_VARIANT)
        checks.record("static_actuator", True,
                      f"{PRIMARY_VARIANT} at historical blocks "
                      f"{list(STATIC_BLOCKS)}")
        checks.record("gate_rule", True, thresholds.gate_rule)
        checks.record("no_row_id_lookup", True,
                      "GateBundle.decide takes an activation and nothing else")
        return ConditionRuntime(
            condition=condition, steers=True,
            hook_states=tuple(sorted(set(STATIC_STATES) | {GATE_READ_STATE})),
            capture_state=GATE_READ_STATE, apply_blocks=STATIC_BLOCKS,
            hooks_expected=STATIC_HOOKS, bundle=bundle,
            thresholds=thresholds, actuator=actuator)

    if condition == COND_DYNAMIC:
        from routegate.livecontroller import load_bundle as load_controllers
        from routegate.dualsteer import load_validated_bank
        from routegate.twosite import build_candidate_record
        from routegate.finaleval import CONTROLLER_WEIGHTS_SHA256
        controllers = load_controllers(CONTROLLER_DIR)
        if controllers.refit is None:
            raise FinalEvaluationError(
                "the all-development refit controller is absent; dynamic_raw "
                "cannot run on an unseen prompt without it")
        refit = controllers.deployment_controller()
        checks.require("controller_is_the_deployment_refit",
                       refit["kind"] == "all_development_refit"
                       and refit["is_out_of_fold"] is False,
                       f"{refit['kind']}")
        checks.require(
            "controller_weights_match_the_frozen_protocol",
            refit["weights_sha256"] == CONTROLLER_WEIGHTS_SHA256,
            refit["weights_sha256"][:12])
        bank = load_validated_bank(
            Path.home() / "routegate-data/frozen/validated_direction_bank",
            build_candidate_record()["direction"]["artifact_sha256"], checks)
        checks.record("no_row_id_lookup", True,
                      "g_R is computed from the activation by "
                      "livecontroller.gate_from_activation")
        return ConditionRuntime(
            condition=condition, steers=True, hook_states=DYNAMIC_STATES,
            capture_state=DYNAMIC_READ_STATE, apply_blocks=DYNAMIC_BLOCKS,
            hooks_expected=DYNAMIC_HOOKS, controller_checkpoint=refit,
            direction_bank=bank)

    raise FinalEvaluationError(f"unknown condition {condition!r}")


def assert_capability_complete(data_root: Path, checks: Checklist) -> None:
    """A sealed arm does not start until every capability arm is complete."""
    pending = []
    for dataset, condition in ARMS:
        if dataset == DS_SEALED:
            continue
        path = data_root / arm_output_dir(dataset, condition) / RECORDS_FILE
        if len(completed_items(path)) != expected_rows(dataset):
            pending.append(arm_id(dataset, condition))
    checks.require("capability_completes_before_the_sealed_split_is_opened",
                   not pending,
                   f"{len(pending)} capability arm(s) incomplete: "
                   f"{pending}" if pending else "all six capability arms "
                   "complete")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_IDS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sealed-manifest", type=Path, default=None,
                        help="the sealed split, supplied at launch; never "
                             "stored in this repository")
    parser.add_argument("--item-index", type=int, default=None,
                        help="generate exactly the one official row with this "
                             "item_index, into the normal arm output. A "
                             "filter on the ordinary loop, not a second "
                             "execution path; a later run without it resumes "
                             "and skips this row.")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate everything, load no model, write "
                             "nothing")
    parser.add_argument("--progress", action="store_true",
                        help="read-only: what every arm has")
    args = parser.parse_args(argv)

    if args.progress:
        return progress(args.data_root)
    if not args.arm:
        parser.error("--arm is required unless --progress is given")

    dataset, condition = split_arm(args.arm)
    checks = Checklist(f"final_evaluation_{args.arm}")
    try:
        verify_protocol(PREDECLARATION, checks)
        spec = DATASET_SPEC[dataset]
        out_dir = args.data_root / arm_output_dir(dataset, condition)
        records_path = out_dir / RECORDS_FILE
        prepared = prepare_condition(condition, checks)

        if dataset == DS_SEALED and not args.dry_run:
            assert_capability_complete(args.data_root, checks)

        want = expected_rows(dataset)
        if args.item_index is not None and not 0 <= int(args.item_index) < want:
            # Checked before the dataset, the checkpoints or the model.
            raise FinalEvaluationError(
                f"--item-index {args.item_index} is outside [0, {want}) for "
                f"{dataset}. The index is validated before anything is "
                f"loaded.")
        done = completed_items(records_path)
        checks.record("resume", True,
                      f"{len(done)} record(s) already present, "
                      f"{want - len(done)} to generate")

        if dataset == DS_SEALED and args.sealed_manifest is None:
            checks.record(
                "sealed_input_pending", True,
                "the sealed split is supplied at launch with "
                "--sealed-manifest; its path is not stored here, so a dry run "
                "validates the protocol, the condition and the output root "
                "and stops short of the prompts")
            items: List[Dict[str, str]] = []
        else:
            items = load_items(dataset, args.sealed_manifest, checks)
            checks.require("item_count", len(items) == want,
                           f"{len(items)} item(s), expected {want}")
            items = select_items(items, args.item_index, want, checks)

        prompt_mode = str(spec["prompt_mode"])
        stop_strings = list(spec.get("stop_strings") or [])
        checks.record("stop_strings", True,
                      f"{dataset}: {stop_strings or 'none'}")
        checks.record("prompt_mode", True,
                      f"{dataset} uses {prompt_mode}: "
                      f"{spec['prompt_mode_note'][:90]}")
        checks.require(
            "prepared_condition_is_used_not_discarded",
            isinstance(prepared, ConditionRuntime)
            and prepared.condition == condition,
            f"{type(prepared).__name__} for {condition}")
        checks.require(
            "steered_conditions_declare_hooks",
            (prepared.hooks_expected == 0) == (condition == COND_BASELINE),
            f"{condition}: hooks_expected {prepared.hooks_expected}")
        if prepared.needs_controller:
            checks.require(
                "capture_state_is_hooked",
                prepared.capture_state in prepared.hook_states,
                f"state {prepared.capture_state} in {list(prepared.hook_states)}")

        print(f"[run] arm       {args.arm}")
        print(f"[run] dataset   {dataset} ({want} rows, "
              f"max_new_tokens {spec['max_new_tokens']}, {spec['decoding']})")
        print(f"[run] condition {condition}")
        print(f"[run] model     {FROZEN_MODEL} @ {FROZEN_REVISION[:12]}")
        print(f"[run] output    {out_dir}")
        scheduled = [i for i in items if i["item_id"] not in done]
        print(f"[run] resume    {len(done)}/{want} present, "
              f"{want - len(done)} remaining")
        print(f"[run] scheduled {len(scheduled)} row(s) this invocation"
              + (f" (item_index {args.item_index})"
                 if args.item_index is not None else ""))
        checks.record("scheduled_this_invocation", True,
                      f"{len(scheduled)} of {len(items)} selected row(s); "
                      f"{len(done)} already present are skipped, never "
                      f"regenerated")

        print(f"[run] prompt    {prompt_mode}; stop "
              f"{stop_strings or 'none'}")
        print(f"[run] dispatch  {condition}: hooks_expected "
              f"{prepared.hooks_expected}, controller "
              f"{'yes' if prepared.needs_controller else 'no'}, states "
              f"{list(prepared.hook_states) or '-'}")
        if args.dry_run:
            print(f"[run] dry run: no model loaded, no response generated, "
                  f"nothing written")
            print(f"[run] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2

        # ---- real generation -------------------------------------------
        from routegate.hooks import PatchController
        from routegate.livecontroller import assert_no_parameter_requires_grad
        from routegate.runtime import (GenerationConfig,
                                       load_model_and_tokenizer)
        generation = GenerationConfig(max_new_tokens=int(spec["max_new_tokens"]))
        generation.validate()
        model, tokenizer = load_model_and_tokenizer(generation)
        model.eval()
        assert_no_parameter_requires_grad([model])

        # The prepared condition decides whether a controller exists at all.
        controller = None
        if prepared.needs_controller:
            controller = PatchController(model, list(prepared.hook_states),
                                         budget_reference="block_input")
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            for item in items:
                index = int(item["item_index"])
                if item["item_id"] in done:
                    continue
                started = time.time()
                produced = execute_item(
                    prepared, model=model, tokenizer=tokenizer,
                    controller=controller, prompt=item["prompt"],
                    item_id=item["item_id"], generation=generation,
                    prompt_mode=prompt_mode, stop_strings=stop_strings)
                record = {
                    "schema": RECORD_SCHEMA, "protocol": PREDECLARATION.stem,
                    "arm_id": args.arm, "dataset": dataset,
                    "condition": condition, "item_index": index,
                    "item_id": item["item_id"],
                    "prompt_sha256": sha256_text(item["prompt"]),
                    "prompt_mode": prompt_mode,
                    "stop_strings": list(stop_strings),
                    "stop_fired": bool(produced.get("stop_fired")),
                    "stop_reason": produced.get("stop_reason"),
                    "status": "ok", "error_message": "",
                    "model": FROZEN_MODEL, "revision": FROZEN_REVISION,
                    "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
                    "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
                    "decoding": spec["decoding"],
                    "max_new_tokens": int(spec["max_new_tokens"]),
                    "do_sample": False, "sampling_seed": None,
                    "elapsed_seconds": time.time() - started,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime()),
                    **{k: produced[k] for k in TELEMETRY_FIELDS},
                    "mode": produced["mode"],
                    "noop_reason": produced.get("noop_reason"),
                    **{k: produced[k] for k in
                       ("response_text", "prompt_token_count",
                        "completion_token_count", "eos_reached",
                        "hit_max_new_tokens") if k in produced},
                }
                # A record that cannot evidence what it claims is not written.
                assert_record_is_complete(record)
                with open(records_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
                written += 1
                if written % 25 == 0:
                    print(f"[run] {len(done) + written}/{want}", flush=True)
        finally:
            if controller is not None:
                controller.close()

        present_after = len(completed_items(records_path))
        append_invocation(out_dir / INVOCATIONS_FILE, {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arm_id": args.arm, "protocol": PREDECLARATION.stem,
            "item_index": args.item_index,
            "selected_rows": len(items),
            "present_before": len(done),
            "generated_here": written,
            "present_after": present_after,
            "expected_rows": want,
            "arm_complete": present_after == want,
        })
        complete = finalise_if_complete(out_dir, records_path, want, checks)
        print(f"[run] arm       {present_after}/{want} present; complete: "
              f"{complete}")
        print(f"[run] wrote {written} record(s); {records_path}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
