#!/usr/bin/env python3
"""Causal activation patching, one state at a time, over the development E set.

For each E target this can run three conditions:

    baseline      the model untouched -- generated once per target, because it
                  does not depend on the state
    self_patch    the target as its own donor, so the delta is exactly zero and
                  nothing is applied; the control that proves an inert run is
                  inert and that the harness itself changes nothing
    donor_patch   the B-donor direction, added at one state, at the final prompt
                  token

The intervention is the historical one (see configs/causal_repair.json for what
was recovered and from where):

    applied = unit(donor_state - target_state) * strength * ||reference||

added to the hooked module's output at the final prompt token of the *prefill*
forward, with the reference norm read from the patched block's input -- the
preceding state -- as RQ1 and the RQ2 frozen actuator both did.

Donors are not chosen here. They are read from a manifest frozen by
`build_donor_manifest.py` before any model was loaded, and this script refuses a
manifest whose hash is not the one it was told to expect.

Usage:
    # what would run, without loading the model
    python experiments/causal_repair/run_patching.py --dry-run \\
        --all-development-targets --all-states \\
        --donor-manifest outputs/causal_repair/donor_manifest.json

    # a single target at three states
    python experiments/causal_repair/run_patching.py \\
        --targets orbench_hard:00042:abc123 --states 32,33,34 \\
        --conditions baseline,donor_patch \\
        --donor-manifest outputs/causal_repair/donor_manifest.json \\
        --expect-manifest-sha256 <hash>

    # the full scan, resumable
    python experiments/causal_repair/run_patching.py \\
        --all-development-targets --all-states --resume \\
        --conditions baseline,self_patch,donor_patch \\
        --donor-manifest outputs/causal_repair/donor_manifest.json \\
        --expect-manifest-sha256 <hash>
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
    WITHHELD_LABEL,
    describe_state,
    load_input_manifest,
    load_local_paths,
)
from routegate.donors import (                                      # noqa: E402
    DonorError,
    load_manifest,
    pairs_from_manifest,
)
from routegate.intervention import (                                # noqa: E402
    BUDGET_REFERENCES,
    STATE_EMBEDDING,
    budget_reference_state,
    build_plan,
)
from routegate.patchrecord import (                                 # noqa: E402
    CONDITIONS,
    append_record,
    completed_keys,
    make_record,
    plan_work,
    read_records,
    verify_consistency,
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
from routegate.validation import (                                  # noqa: E402
    ValidationError,
    assert_config_is_sealed_free,
    assert_hash,
    assert_not_sealed,
    assert_not_stale,
)
from routegate import __version__ as PACKAGE_VERSION                # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "causal_repair.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "causal_repair"
RECORDS_FILENAME = "patch_records.jsonl"

# One --states chunk: a signed integer, optionally followed by an inclusive
# upper bound. The sign is part of the number, which is what keeps "-1" a state
# rather than a half-written range.
_STATE_CHUNK = re.compile(r"^(?P<low>[+-]?\d+)(?:-(?P<high>[+-]?\d+))?$")

# The condition -> intervention type map. Nothing else may set an intervention.
CONDITION_INTERVENTION = {
    "baseline": "none",
    "self_patch": "self_patch",
    "donor_patch": "donor_direction",
}


def parse_states(spec: str, n_states: int) -> List[int]:
    """'0,3,31-34' -> [0, 3, 31, 32, 33, 34], validated against the model.

    A leading minus is a negative number, not an empty lower bound: ``-1`` is
    state -1, which is refused for being outside the model. Splitting on the
    first ``-`` instead would make it an unparseable range and raise a bare
    ``ValueError``, and ``-1`` is exactly what someone types when they mean "the
    last one" -- the answer they need is which states actually exist.

    Every rejection is a ``ValidationError`` carrying the valid range, so a
    mistyped ``--states`` stops the command line rather than surfacing as a
    traceback from inside the parser.
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
                f"cannot read {chunk!r} as a state: write a single state "
                f"('32'), an inclusive range ('32-34'), or a comma-separated "
                f"list of either, within 0..{n_states - 1}")
        low = int(match.group("low"))
        high = int(match.group("high")) if match.group("high") is not None else low
        if high < low:
            raise ValidationError(
                f"the range {chunk!r} ends before it starts; write it as "
                f"{high}-{low} if that is what was meant")
        # The endpoints, not the expansion: an inclusive range is inside the
        # model exactly when both of its ends are, and checking the ends keeps
        # the refusal naming what was typed.
        bounds.extend((low, high))
        states.extend(range(low, high + 1))

    outside = sorted({bound for bound in bounds if not 0 <= bound < n_states})
    if outside:
        raise ValidationError(
            f"state(s) {outside} are outside 0..{n_states - 1}; state 0 is the "
            f"embedding output and states 1-{n_states - 1} are transformer-block "
            f"outputs")
    return sorted(set(states))


def parse_list(spec: Optional[str]) -> List[str]:
    return [v.strip() for v in str(spec or "").split(",") if v.strip()]


def load_prompts_and_responses(dataset_path: Path, expected_sha256: str
                               ) -> Dict[str, Dict[str, str]]:
    """row_id -> prompt, label and the recorded original response.

    Read separately from the pairing loader, which is projected to the metadata
    columns precisely so that donor selection cannot see a response. Recording
    one is a different act from choosing on one: the original response is what a
    patched response is compared against, and it comes from the corpus rather
    than from a fresh baseline, so it is the same text the adjudication saw.
    """
    path = assert_not_stale(Path(dataset_path))
    assert_not_sealed(path, "causal repair dataset")
    assert_hash(path, expected_sha256, "dataset")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row["final_ebh"] not in (POSITIVE_LABEL, NEGATIVE_LABEL):
            continue
        table[row["row_id"]] = {
            "prompt_text": row["prompt_text"],
            "final_ebh": row["final_ebh"],
            "original_response": row.get("response_text", ""),
        }
    if not table:
        raise ValidationError(f"{path} carries no {POSITIVE_LABEL}/"
                              f"{NEGATIVE_LABEL} rows")
    return table


def assert_no_withheld(row_ids: Sequence[str], table: Dict[str, Dict[str, str]],
                       where: str) -> None:
    """This stage touches E and B only."""
    bad = [r for r in row_ids
           if table.get(r, {}).get("final_ebh") == WITHHELD_LABEL]
    if bad:
        raise ValidationError(
            f"{where}: {len(bad)} row(s) of the withheld class {WITHHELD_LABEL} "
            f"were requested (first: {bad[0]}). This stage runs E targets "
            f"against B donors only.")


def build_work_list(targets: Sequence[str], states: Sequence[int],
                    conditions: Sequence[str], self_patch_states: Sequence[int],
                    pairs: Dict[str, Dict[str, Any]], strength: float
                    ) -> List[Dict[str, Any]]:
    """Every generation this run intends to perform, in a deterministic order.

    The baseline is one generation per target, not one per state: it does not
    depend on the state, and 49 identical baselines per target would be 48
    wasted decodes each. Its record carries an empty state, which is this
    package's way of writing "not applicable" without writing a number.
    """
    work: List[Dict[str, Any]] = []
    for target in targets:
        donor = pairs[target]["donor_row_id"]
        if "baseline" in conditions:
            work.append({"target_row_id": target, "donor_row_id": "",
                         "state": "", "condition": "baseline",
                         "intervention_type": "none",
                         "intervention_strength": 0.0})
        if "self_patch" in conditions:
            for state in self_patch_states:
                work.append({"target_row_id": target, "donor_row_id": target,
                             "state": state, "condition": "self_patch",
                             "intervention_type": "self_patch",
                             "intervention_strength": strength})
        if "donor_patch" in conditions:
            for state in states:
                work.append({"target_row_id": target, "donor_row_id": donor,
                             "state": state, "condition": "donor_patch",
                             "intervention_type": "donor_direction",
                             "intervention_strength": strength})
    return work


def work_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (item["target_row_id"], item["donor_row_id"], item["state"],
            item["intervention_type"], item["intervention_strength"],
            item["condition"])


def hooked_states(states: Sequence[int], self_patch_states: Sequence[int],
                  conditions: Sequence[str]) -> List[int]:
    """Every state the controller must install a hook for.

    Not simply ``--states``. The zero-delta control has its own state list, and
    ``--self-patch-states`` is not required to be a subset -- the array job runs
    seven intervention states per task and the control at one of them, but a
    single-state debugging run might do the reverse. A plan for a state the
    controller never hooked is refused at ``reset_run``, which would surface as
    a per-generation error long after the model had loaded, so the union is
    resolved here instead.

    A baseline-only run hooks nothing at all: it needs no capture and applies no
    delta, so there is no reason for it to carry 49 forward hooks.
    """
    wanted = set()
    if "donor_patch" in conditions:
        wanted.update(int(state) for state in states)
    if "self_patch" in conditions:
        wanted.update(int(state) for state in self_patch_states)
    return sorted(wanted)


def run(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "causal repair config")

    generation = GenerationConfig(
        max_new_tokens=int(args.max_new_tokens
                           or config["generation"]["max_new_tokens"]))
    generation.validate()

    if args.budget_reference not in BUDGET_REFERENCES:
        raise ValidationError(
            f"unknown budget reference {args.budget_reference!r}; valid: "
            f"{list(BUDGET_REFERENCES)}")
    conditions = parse_list(args.conditions)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValidationError(
            f"unknown condition(s) {unknown}; valid: {list(CONDITIONS)}")
    if not conditions:
        raise ValidationError("no condition selected; nothing to run")

    n_states = EXPECTED_NUM_LAYERS + 1
    states = (list(range(n_states)) if args.all_states
              else parse_states(args.states or "", n_states))
    if not states and {"self_patch", "donor_patch"} & set(conditions):
        raise ValidationError(
            "an intervention condition was requested but no state was selected; "
            "use --states or --all-states")
    self_patch_states = (parse_states(args.self_patch_states, n_states)
                         if args.self_patch_states else list(states))

    # --- inputs and the frozen pairing --------------------------------------
    input_manifest = load_input_manifest()
    local_paths = load_local_paths(args.paths) if args.paths else load_local_paths()
    dataset_path = Path(local_paths["dataset"])
    dataset_sha = input_manifest["dataset"]["sha256"]
    table = load_prompts_and_responses(dataset_path, dataset_sha)

    manifest = load_manifest(args.donor_manifest, args.expect_manifest_sha256)
    if manifest["dataset"]["sha256"] != dataset_sha:
        raise DonorError(
            f"the donor manifest was built against dataset "
            f"{manifest['dataset']['sha256']} but this run reads {dataset_sha}")
    pairs = pairs_from_manifest(manifest)

    targets = (sorted(pairs) if args.all_development_targets
               else parse_list(args.targets))
    if not targets:
        raise ValidationError(
            "no target selected; use --targets or --all-development-targets")
    unknown_targets = [t for t in targets if t not in pairs]
    if unknown_targets:
        raise ValidationError(
            f"{len(unknown_targets)} requested target(s) are not in the frozen "
            f"manifest (first: {unknown_targets[0]}). Targets are not invented "
            f"at run time.")
    if args.donors:
        wanted = set(parse_list(args.donors))
        targets = [t for t in targets if pairs[t]["donor_row_id"] in wanted]
        if not targets:
            raise ValidationError(
                "no target in the frozen manifest uses any of the requested "
                "donors; donors are not reassigned at run time")
    if args.limit:
        targets = targets[:int(args.limit)]

    assert_no_withheld(targets, table, "targets")
    assert_no_withheld([pairs[t]["donor_row_id"] for t in targets], table,
                       "donors")
    for target in targets:
        if table[target]["final_ebh"] != POSITIVE_LABEL:
            raise ValidationError(
                f"target {target} is labelled {table[target]['final_ebh']}, not "
                f"{POSITIVE_LABEL}")
        donor = pairs[target]["donor_row_id"]
        if table[donor]["final_ebh"] != NEGATIVE_LABEL:
            raise ValidationError(
                f"donor {donor} is labelled {table[donor]['final_ebh']}, not "
                f"{NEGATIVE_LABEL}")

    # --- the work list, and what of it is already done ----------------------
    output_dir = Path(args.output_dir or DEFAULT_OUTPUT)
    if args.smoke and "smoke" not in output_dir.name.lower():
        raise ValidationError(
            f"a smoke run must write to a directory whose name says so; "
            f"{output_dir.name!r} does not contain 'smoke'")
    records_path = output_dir / RECORDS_FILENAME

    work = build_work_list(targets, states, conditions, self_patch_states,
                           pairs, float(args.strength))
    existing = read_records(records_path) if records_path.exists() else []
    if existing and not args.resume:
        raise ValidationError(
            f"{records_path} already holds {len(existing)} record(s). Pass "
            f"--resume to continue it, or choose another --output-dir; this "
            f"script will not append to a run it was not told to continue.")
    summary = verify_consistency(
        existing,
        generation_config_sha256=generation.fingerprint(),
        donor_manifest_sha256=manifest["manifest_sha256"],
        dataset_sha256=dataset_sha)
    done = completed_keys(existing)
    todo, skipped = plan_work([work_key(item) for item in work], done)
    by_key = {work_key(item): item for item in work}

    print(f"[plan] {len(targets)} target(s) x {len(states)} state(s), "
          f"conditions {conditions}")
    print(f"[plan] {len(work)} generation(s) requested, {len(skipped)} already "
          f"done, {len(todo)} to run")
    if existing:
        print(f"[plan] resuming into {records_path} ({summary})")
    print(f"[plan] intervention {CONDITION_INTERVENTION} at strength "
          f"{args.strength}, budget reference {args.budget_reference}")
    print(f"[plan] model {generation.model} @ {generation.revision}, "
          f"greedy, max_new_tokens={generation.max_new_tokens}, "
          f"thinking disabled")

    if args.dry_run:
        print("\n[dry-run] no model was loaded and nothing was generated.")
        for key in todo[:10]:
            item = by_key[key]
            state = item["state"]
            print(f"          {item['condition']:<12s} "
                  f"{item['target_row_id']} <- {item['donor_row_id'] or '—'}  "
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
        "donor_manifest": str(args.donor_manifest),
        "donor_manifest_sha256": manifest["manifest_sha256"],
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "states": states,
        "self_patch_states": self_patch_states,
        "hooked_states": hooked_states(states, self_patch_states, conditions),
        "conditions": conditions,
        "strength": float(args.strength),
        "budget_reference": args.budget_reference,
        "targets": targets,
        "smoke": bool(args.smoke),
        "reportable": not args.smoke,
        "environment": environment_record(PACKAGE_VERSION),
    })

    print(f"[load] {generation.model} @ {generation.revision}")
    model, tokenizer = load_model_and_tokenizer(generation)
    controller = PatchController(
        model, states=hooked_states(states, self_patch_states, conditions),
        budget_reference=args.budget_reference)
    donor_cache: Dict[str, Dict[int, Any]] = {}
    written = errors = 0

    try:
        for position, key in enumerate(todo, start=1):
            item = by_key[key]
            target = item["target_row_id"]
            donor = item["donor_row_id"]
            state = item["state"]
            condition = item["condition"]
            print(f"[{position}/{len(todo)}] {condition} {target} "
                  f"{'' if state == '' else describe_state(state)}", flush=True)

            try:
                plan = None
                if condition == "baseline":
                    generated = generate_response(
                        model, tokenizer, table[target]["prompt_text"],
                        generation, controller=controller, mode="off",
                        example_id=target)
                    status = "baseline"
                else:
                    capture_prompt_states(model, tokenizer,
                                          table[target]["prompt_text"],
                                          controller, capture_name="target",
                                          example_id=target, config=generation)
                    target_capture = dict(controller.capture_for("target"))

                    if condition == "self_patch":
                        donor_capture = target_capture
                    else:
                        if donor not in donor_cache:
                            capture_prompt_states(
                                model, tokenizer, table[donor]["prompt_text"],
                                controller, capture_name=f"donor:{donor}",
                                example_id=donor, config=generation)
                            donor_cache[donor] = dict(
                                controller.capture_for(f"donor:{donor}"))
                            controller.drop_capture(f"donor:{donor}")
                        donor_capture = donor_cache[donor]

                    reference_state, _ = budget_reference_state(
                        state, args.budget_reference)
                    plan = build_plan(
                        state=state,
                        intervention_type=item["intervention_type"],
                        strength=float(item["intervention_strength"]),
                        target_state_vector=target_capture[state],
                        donor_state_vector=donor_capture[state],
                        reference_vector=target_capture[reference_state],
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
                print(f"      ERROR {error_message}", file=sys.stderr)

            # The match tier describes the frozen B donor, so it belongs only to
            # the condition that actually uses it. A self-patch donor is the
            # target itself and was matched to nothing.
            pair = pairs[target] if condition == "donor_patch" else {}
            record = make_record(
                run_id=run_id,
                created_utc=utc_now(),
                target_row_id=target,
                target_label=table[target]["final_ebh"],
                donor_row_id=donor,
                donor_label=(table[donor]["final_ebh"] if donor else ""),
                donor_match_tier=pair.get("tier", ""),
                donor_match_exact=bool(pair.get("exact_match", False)),
                condition=condition,
                state=state,
                historical_block_index=("" if state in ("", STATE_EMBEDDING)
                                        else int(state) - 1),
                intervention_type=item["intervention_type"],
                intervention_strength=float(item["intervention_strength"]),
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
                patched_response=generated["response_text"],
                output_status=status,
                error_message=error_message,
                reportable=bool(not args.smoke and status != "error"),
                reportable_reason=(
                    "smoke run: not reportable" if args.smoke
                    else "generation failed" if status == "error"
                    else "full run under the frozen configuration and manifest"),
                dataset_sha256=dataset_sha,
                donor_manifest_sha256=manifest["manifest_sha256"],
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

    parser.add_argument("--donor-manifest", type=Path,
                        default=DEFAULT_OUTPUT / "donor_manifest.json")
    parser.add_argument("--expect-manifest-sha256", default=None,
                        help="refuse any donor manifest but this one")

    parser.add_argument("--targets", default=None,
                        help="comma-separated target row_ids")
    parser.add_argument("--all-development-targets", action="store_true",
                        help="every E target in the frozen manifest")
    parser.add_argument("--donors", default=None,
                        help="restrict to targets whose frozen donor is in this "
                             "comma-separated list; donors are never reassigned")
    parser.add_argument("--states", default=None,
                        help="comma-separated states, ranges allowed: 0,32-34")
    parser.add_argument("--all-states", action="store_true",
                        help="every state, 0 (embedding output) to 48")
    parser.add_argument("--self-patch-states", default=None,
                        help="states for the zero-delta control (default: the "
                             "same states as the intervention)")
    parser.add_argument("--conditions", default="baseline,donor_patch",
                        help=f"comma-separated, from {list(CONDITIONS)}")
    parser.add_argument("--strength", type=float, default=0.2)
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
