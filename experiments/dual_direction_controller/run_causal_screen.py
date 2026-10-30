#!/usr/bin/env python3
"""The preregistered causal screen of the validated repair and safety directions.

Generation only. Nothing here judges a response, fits a probe or trains a
coefficient controller; those are separate stages and this is the one that
produces the text they read.

What it runs
------------
One of two named protocols, chosen with ``--protocol``. Each is a fixed
inventory: complete or not that protocol, never "as much of it as fitted".

``exploratory_dual_direction_v1`` (the default) -- three questions at once::

    E rows (89)   baseline + repair 0.05/0.10/0.15/0.20            445 generations
    H rows (213)  baseline + repair 0.05/0.10/0.15/0.20
                           + safety 0.05/0.10/0.15/0.20
                           + repair 0.15 x safety 0.05/0.10/0.15/0.20
                                                                  2769 generations
                                                          total   3214 generations

The combined arms hold alpha_R at 0.15 and vary alpha_S alone. That is question
three: a refusal restoration that moves with alpha_S while the repair
coefficient does not move is not explicable as "less repair".

``focused_direction_validation_v1`` -- phase 1 of the focused plan, and two
questions about the **current** repair direction only::

    E rows (89)   baseline + repair 0.05/0.10/0.15/0.20            445 generations
    H rows (213)  baseline + repair 0.15                           426 generations
                                                          total    871 generations

Does ``d_R = B_d1 - E`` repair E, and which H rows does it damage? ``alpha_S``
is zero in every arm; the safety direction is not tested here at all. Phase 2
applies ``d_S = H - E``, and only to the failures this phase finds.

The edit, at historical layers 31/32/33 -- states 32/33/34 -- at the final
prompt token, prefill only::

    delta = ||block_input|| * (alpha_R * d_R + alpha_S * d_S)

added to the block output. The combination is not renormalised, the same two
coefficients are used at all three layers, and neither direction is
orthogonalised against the other. ``routegate.dualsteer`` holds the arithmetic
and checks, before a model is loaded, that with ``alpha_S = 0`` it is the edit
``routegate.steering`` already performs.

Order of the gates
------------------
Everything that can fail without a GPU fails without a GPU. The direction
bank's SHA256, the corpus digest, the row counts, the row ids, the prompt
hashes, the class labels, the selected protocol's whole inventory and the
resume file's compatibility are all settled first; only then is the model
loaded.

Usage
-----
    # what it would do, loading nothing
    python experiments/dual_direction_controller/run_causal_screen.py --dry-run \\
        --direction-bank <bank> --output-dir <new dir>

    # one shard of an array job, resumable
    python experiments/dual_direction_controller/run_causal_screen.py \\
        --direction-bank <bank> --output-dir <dir> \\
        --shard-index 1 --shard-count 8 --resume

    # phase 1, one of four resumable shards
    python experiments/dual_direction_controller/run_causal_screen.py \\
        --protocol focused_direction_validation_v1 \\
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
    POSITIVE_LABEL,
    SOURCE_COLUMN,
    WITHHELD_LABEL,
    load_input_manifest,
    load_local_paths,
)
from routegate.dualsteer import (                                   # noqa: E402
    ARM_BY_ID,
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    BUDGET_REFERENCE,
    CLASS_E,
    CLASS_H,
    INTERVENTION_TYPE,
    N_E_ROWS,
    N_H_ROWS,
    PHASE2_PROTOCOL,
    SCREEN_CLASSES,
    SCREEN_SCHEMA,
    DualSteerError,
    append_record,
    assert_inventory,
    assert_matches_actuator_convention,
    build_inventory,
    build_plans,
    protocol_for,
    load_validated_bank,
    make_record,
    plan_work,
    read_records,
    record_key,
    verify_consistency,
    work_key,
)
from routegate.directionbank import BANK_STATES, HISTORICAL_LAYERS  # noqa: E402
from routegate.phase2 import (                                      # noqa: E402
    PHASE2_PREDECLARATION,
    PHASE2_SELECTION_MANIFEST,
    verify_phase2_predeclaration,
    verify_phase2_selection,
)
from routegate.phase2closure import (                               # noqa: E402
    CLOSED_PROTOCOLS,
    REPRODUCTION_ONLY_FLAG,
    assert_protocol_is_open,
)
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

DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual_direction_controller.json"
RECORDS_FILENAME = "screen_records.jsonl"
MANIFEST_FILENAME = "run_manifest.json"

#: The corpus label each screen class is drawn from. E and H only: B rows built
#: the historical directions and are not targets here.
LABEL_OF_CLASS = {CLASS_E: POSITIVE_LABEL, CLASS_H: WITHHELD_LABEL}
EXPECTED_ROWS = {CLASS_E: N_E_ROWS, CLASS_H: N_H_ROWS}


def load_screen_rows(dataset_path: Path, expected_sha256: str,
                     checks: Checklist) -> List[Dict[str, str]]:
    """The 89 E and 213 H development rows, with their prompt digests.

    Read from the accepted corpus and checked against its recorded digest
    before anything else looks at a row. B rows are read and counted so that a
    corpus of the wrong shape is caught here rather than by a short inventory
    later.
    """
    path = assert_not_stale(Path(dataset_path))
    assert_not_sealed(path, "dual direction causal screen dataset")
    assert_hash(path, expected_sha256, "dataset")
    with open(path, newline="", encoding="utf-8") as handle:
        corpus = list(csv.DictReader(handle))

    by_class: Dict[str, List[Dict[str, str]]] = {name: [] for name in SCREEN_CLASSES}
    for row in corpus:
        for name, label in LABEL_OF_CLASS.items():
            if row["final_ebh"] != label:
                continue
            prompt = row["prompt_text"]
            by_class[name].append({
                "row_id": row["row_id"], "row_class": name,
                "prompt_text": prompt, "prompt_sha256": sha256_text(prompt),
                "source_family": row[SOURCE_COLUMN],
            })

    rows: List[Dict[str, str]] = []
    for name in SCREEN_CLASSES:
        found = by_class[name]
        if len(found) != EXPECTED_ROWS[name]:
            raise DualSteerError(
                f"the corpus holds {len(found)} {name} row(s) and this screen "
                f"is {EXPECTED_ROWS[name]}. A screen over a different "
                f"population is a different screen.")
        ids = [row["row_id"] for row in found]
        if len(set(ids)) != len(ids):
            raise DualSteerError(f"duplicate {name} row id(s) in the corpus")
        rows.extend(sorted(found, key=lambda row: row["row_id"]))
    checks.record("corpus_rows", True,
                  f"{EXPECTED_ROWS[CLASS_E]} {CLASS_E} + "
                  f"{EXPECTED_ROWS[CLASS_H]} {CLASS_H} of {len(corpus)} rows")
    return rows


def parse_selection(spec: str, flag: str) -> set:
    """A comma-separated filter value as a set, refusing an empty element.

    An empty element usually means a stray comma, and reading it as "no
    filter" would silently widen a smoke test into something much larger.
    """
    if not spec:
        return set()
    parts = [part.strip() for part in spec.split(",")]
    if any(not part for part in parts):
        raise DualSteerError(
            f"{flag}={spec!r} has an empty element; give a comma-separated "
            f"list with no stray commas")
    return set(parts)


def shard(items: Sequence[Dict[str, Any]], index: int, count: int
          ) -> List[Dict[str, Any]]:
    """This task's contiguous slice of the inventory.

    Contiguous rather than round-robin so a shard's rows cluster, and a row's
    prompt states are captured once and reused across its arms instead of once
    per arm.
    """
    if count < 1:
        raise DualSteerError(f"--shard-count must be at least 1, got {count}")
    if not 1 <= index <= count:
        raise DualSteerError(
            f"--shard-index {index} is outside 1..{count}")
    size, remainder = divmod(len(items), count)
    start = (index - 1) * size + min(index - 1, remainder)
    length = size + (1 if index - 1 < remainder else 0)
    return list(items[start:start + length])


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
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL,
                        choices=sorted(PROTOCOLS),
                        help="the named inventory to run. The default is the "
                             "original exploratory screen; "
                             "focused_direction_validation_v1 is phase 1 of "
                             "the focused plan and tests the repair direction "
                             "only.")
    parser.add_argument("--only-class", "--classes", dest="classes", type=str,
                        default=",".join(SCREEN_CLASSES),
                        help="restrict to these screen classes, comma "
                             "separated. Narrowing it makes the run a smoke "
                             "test.")
    parser.add_argument("--only-arm", dest="only_arm", type=str, default="",
                        help="restrict to these arm ids, comma separated. "
                             "Narrowing makes the run a smoke test.")
    parser.add_argument("--only-row-id", dest="only_row_id", type=str,
                        default="",
                        help="restrict to these corpus row ids, comma "
                             "separated. Narrowing makes the run a smoke test.")
    parser.add_argument("--shard-index", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many generations; 0 means all")
    parser.add_argument("--device", type=str, default="auto",
                        help="device_map for the pinned model, e.g. auto or cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and validate everything; load no model")
    parser.add_argument(REPRODUCTION_ONLY_FLAG, dest="reproduce_closed_protocol",
                        action="store_true",
                        help="run a CLOSED protocol as an explicit "
                             "reproduction. "
                             f"{', '.join(sorted(CLOSED_PROTOCOLS))} ran, was "
                             "decided, and is refused by the ordinary path: a "
                             "second set of records beside the first, at the "
                             "same arms on the same rows, would be "
                             "unattributable. A reproduction writes to an "
                             "output directory of its own and does not replace "
                             "the recorded result.")
    return parser


def run(args: argparse.Namespace) -> int:
    checks = Checklist("dual_direction_causal_screen")

    # ---- a closed protocol is not re-run by accident ----------------------
    #
    # Before the configuration is opened, before the bank is hashed and before
    # a single corpus row is read: the phase-2 screen ran, produced its 144
    # records and was decided. An ordinary rerun would write a second set of
    # records beside the first, at the same arms on the same rows, and no
    # difference between the two copies would be attributable to anything. The
    # reproduction path is an explicitly named flag, never inferred.
    assert_protocol_is_open(
        args.protocol,
        reproduction_only=bool(getattr(args, "reproduce_closed_protocol", False)),
        checks=checks)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    screen_spec = config["causal_screen"]

    expected_bank = (args.expect_bank_sha256
                     or screen_spec["direction_bank"]["artifact_sha256"])

    # ---- inputs, before anything is loaded -------------------------------
    input_manifest = load_input_manifest()
    local_paths = (load_local_paths(args.paths) if args.paths
                   else load_local_paths())
    dataset_path = Path(local_paths["dataset"])
    dataset_sha = input_manifest["dataset"]["sha256"]

    bank = load_validated_bank(args.direction_bank, expected_bank, checks)
    rows = load_screen_rows(dataset_path, dataset_sha, checks)

    requested = [name.strip() for name in args.classes.split(",") if name.strip()]
    unknown = [name for name in requested if name not in SCREEN_CLASSES]
    if unknown:
        raise DualSteerError(
            f"unknown screen class(es) {unknown}; this screen covers "
            f"{list(SCREEN_CLASSES)}")

    protocol = protocol_for(args.protocol)

    # ---- the phase-2 freeze, before anything is planned or loaded ---------
    #
    # A protocol bound to a selection is only as trustworthy as the selection
    # it is bound to. Both gates run here, before the inventory is built and
    # long before a model or tokenizer is touched, so a drifted selection or a
    # predeclaration that has moved costs nothing but an error message.
    if protocol.name == PHASE2_PROTOCOL:
        verify_phase2_selection(checks=checks)
        verify_phase2_predeclaration(checks=checks)
        checks.record(
            "phase2_freeze", True,
            f"{PHASE2_SELECTION_MANIFEST.name} and "
            f"{PHASE2_PREDECLARATION.name} agree with routegate.phase2; "
            f"{protocol.expected_rows(CLASS_H)} bound row(s), "
            f"{len(protocol.arms_by_class[CLASS_H])} generated arm(s), "
            f"{len(protocol.reference_arms(CLASS_H))} inherited arm(s) not "
            f"regenerated")

    inventory = build_inventory(rows, protocol)
    counts = assert_inventory(inventory, checks, protocol)
    checks.record("protocol", True,
                  f"{protocol.name}: {protocol.expected_generations} "
                  f"generation(s); safety direction tested: "
                  f"{protocol.tests_safety_direction}")

    # Every filter narrows the frozen inventory, and all of them are applied
    # before the shard split and before --limit. --limit is a truncation, not
    # a selection: applied first it would silently decide which rows a filter
    # then had to choose from.
    only_arm = parse_selection(args.only_arm, "--only-arm")
    only_row = parse_selection(args.only_row_id, "--only-row-id")
    # Two refusals, because they are two different mistakes. An arm nobody
    # declared is a typo; an arm this protocol does not run is an arm from
    # another protocol, and saying which it is saves the reader the lookup.
    protocol_arms = protocol.arm_by_id()
    unknown_arms = [arm for arm in only_arm if arm not in ARM_BY_ID]
    if unknown_arms:
        raise DualSteerError(
            f"--only-arm names {unknown_arms}, which are not predeclared "
            f"arms. The screen's arms are {sorted(ARM_BY_ID)}.")
    out_of_protocol = [arm for arm in only_arm if arm not in protocol_arms]
    if out_of_protocol:
        raise DualSteerError(
            f"--only-arm names {out_of_protocol}, which are predeclared "
            f"arms that protocol {protocol.name} does not run. Its arms are "
            f"{sorted(protocol_arms)}.")
    known_rows = {row["row_id"] for row in rows}
    unknown_rows = [row for row in only_row if row not in known_rows]
    if unknown_rows:
        raise DualSteerError(
            f"--only-row-id names {len(unknown_rows)} row(s) that are not E or "
            f"H rows of the accepted corpus (first {unknown_rows[0]!r})")

    selected = [item for item in inventory if item["row_class"] in requested]
    if only_arm:
        selected = [item for item in selected if item["arm_id"] in only_arm]
    if only_row:
        selected = [item for item in selected if item["row_id"] in only_row]
    if not selected:
        raise DualSteerError(
            "the filters select no generation; nothing would be run and an "
            "empty output is not a result")
    filtered = len(selected)
    selected = shard(selected, args.shard_index, args.shard_count)
    checks.record("filters", True,
                  f"classes {requested}, arms {sorted(only_arm) or 'all'}, "
                  f"rows {sorted(only_row) or 'all'}: {filtered} of "
                  f"{counts['total']} generation(s)")
    checks.record("shard", True,
                  f"{args.shard_index}/{args.shard_count}: {len(selected)} of "
                  f"{filtered} filtered generation(s)")

    # Why this run is or is not the whole screen. A narrowed run is a smoke
    # test, is recorded as one on every record it writes, and the consolidator
    # refuses to treat one as the complete causal screen.
    smoke_reasons: List[str] = []
    if set(requested) != set(SCREEN_CLASSES):
        smoke_reasons.append(f"--only-class {sorted(requested)}")
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

    # ---- the convention gate ---------------------------------------------
    # A synthetic reference of the right shape is enough: the check is about
    # the formula, not about any particular activation.
    import numpy as np
    probe_reference = np.full(bank.repair.shape[1], 0.5, dtype=np.float32)
    convention = assert_matches_actuator_convention(
        bank.repair[0], probe_reference, state=BANK_STATES[0], checks=checks)

    run_locked = {
        "direction_bank_sha256": bank.artifact_sha256,
        "generation_config_sha256": generation_sha,
        "model": generation.model, "revision": generation.revision,
        "tokenizer_config_sha256": FROZEN_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA256,
        "intervention_type": INTERVENTION_TYPE,
        "budget_reference": BUDGET_REFERENCE,
        "states": list(BANK_STATES),
        "historical_layers": list(HISTORICAL_LAYERS),
        "dataset_sha256": dataset_sha,
        "smoke": smoke,
        "protocol": protocol.name,
    }

    # ---- the resume file --------------------------------------------------
    out_dir = Path(args.output_dir)
    records_path = out_dir / RECORDS_FILENAME
    existing = read_records(records_path, strict=not args.resume)
    if existing and not args.resume:
        raise DualSteerError(
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
    print(f"[screen] bank {bank.artifact_sha256[:16]}  generation "
          f"{generation_sha[:16]}")
    print(f"[screen] protocol {protocol.name}")
    print(f"[screen] inventory {counts['total']} "
          f"({counts['E_rows']}x{counts['E_arms']} E + "
          f"{counts['H_rows']}x{counts['H_arms']} H)")
    print(f"[screen] shard {args.shard_index}/{args.shard_count}: "
          f"{len(selected)} selected, {len(done)} done, {len(todo)} to run")
    print(f"[screen] states {list(BANK_STATES)} "
          f"(historical layers {list(HISTORICAL_LAYERS)}), "
          f"final prompt token, prefill only")
    print(f"[screen] convention agrees with steering.build_steering_plan to "
          f"{convention['max_abs_difference']:.3e}")
    if smoke:
        print(f"[screen] SMOKE TEST -- {'; '.join(smoke_reasons)}. This is a "
              f"subset of the {protocol.expected_generations}-generation "
              f"{protocol.name} protocol and is never the protocol itself.")

    # What this run is, in its own record. A dry run plans and loads nothing;
    # a real run loads the model locally and decodes from it, and saying
    # otherwise would leave every record it writes claiming it had not.
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
        "complete_causal_screen": (not smoke and not args.dry_run
                                   and args.shard_count == 1),
        "complete_causal_screen_note":
            "true only for an unfiltered, unsharded, unlimited real run. A "
            "sharded run is a part of the complete screen and is completed by "
            "consolidate_causal_screen.py; a filtered or limited run is a "
            "smoke test and is never the screen.",
        "reportable": not args.dry_run and not smoke,
        "filters": {"classes": requested,
                    "only_arm": sorted(only_arm),
                    "only_row_id": sorted(only_row),
                    "limit": int(args.limit)},
        "frozen_protocol": {
            "expected_generations": protocol.expected_generations,
            "E_rows": N_E_ROWS, "H_rows": N_H_ROWS,
            "arms_per_class": {name: len(arms) for name, arms
                               in protocol.arms_by_class.items()},
            "selected_this_run": len(selected),
            "note": "the complete screen is the full inventory above. This "
                    "run covers selected_this_run of it.",
        },
        "judges_responses": False,
        "trains_a_controller": False,
        "orthogonalisation_applied": False,
        "direction_bank": str(args.direction_bank),
        "direction_bank_manifest_sha256": bank.manifest_sha256,
        "dataset": str(dataset_path),
        "generation": generation.as_dict(),
        "actuator_convention_check": convention,
        "arms": {name: [arm.as_dict() for arm in arms]
                 for name, arms in protocol.arms_by_class.items()},
        "protocol_record": {
            "name": protocol.name,
            "purpose": protocol.purpose,
            "expected_generations": protocol.expected_generations,
            "tests_safety_direction": protocol.tests_safety_direction,
            "arms_per_class": {name: len(arms) for name, arms
                               in protocol.arms_by_class.items()},
            "reportable_when": "every one of "
                               f"{protocol.expected_generations} records is "
                               "present; completeness is established by "
                               "consolidate_causal_screen.py, not by any "
                               "single shard",
        },
        "inventory": counts,
        "shard": {"index": args.shard_index, "count": args.shard_count,
                  "selected": len(selected), "classes": requested},
        "planned_this_run": len(todo),
        "planned_by_arm": dict(sorted(by_arm.items())),
        "already_done": len(done),
        "environment": environment,
        "checklist": checks.as_dict(),
        **run_locked,
    }

    if args.dry_run:
        print(f"[screen] dry run: no model loaded, nothing written")
        summary = {"planned": len(todo),
                   "by_arm": dict(sorted(by_arm.items()))}
        if protocol.name == PHASE2_PROTOCOL:
            # What a phase-2 dry run has to make checkable at a glance: the
            # bound rows, the arms it will generate, and the arms it will NOT
            # regenerate. A reference arm appearing in by_arm would mean the
            # inherited records were about to be overwritten.
            reference = [arm.arm_id for arm in protocol.reference_arms(CLASS_H)]
            summary.update({
                "bound_rows": protocol.expected_rows(CLASS_H),
                "arms_per_row": len(protocol.arms_by_class[CLASS_H]),
                "reference_arms_not_scheduled": reference,
                "reference_arms_in_this_plan": sorted(
                    set(by_arm) & set(reference)),
                "reference_records_inherited": protocol.reference_records(),
                "analysis_table_records": protocol.analysis_table_records(),
            })
        print(json.dumps(summary, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- only now is a model loaded --------------------------------------
    from routegate.hooks import PatchController          # torch only from here
    from routegate.runtime import load_model_and_tokenizer

    print(f"[load] {generation.model} @ {generation.revision} "
          f"(device_map={args.device})")
    model, tokenizer = load_model_and_tokenizer(generation)
    controller = PatchController(model, states=list(BANK_STATES),
                                 budget_reference=BUDGET_REFERENCE)

    # Written after the load, not before it: the manifest states that a model
    # was loaded, and it should not be able to say so about a run where the
    # load then failed. A load that fails leaves no manifest and no records,
    # which is the honest description of what happened.
    write_json(out_dir / MANIFEST_FILENAME, manifest)

    by_row = {row["row_id"]: row for row in rows}
    captured_row: Optional[str] = None
    capture: Dict[int, Any] = {}
    written = errors = 0

    try:
        for position, item in enumerate(todo, start=1):
            row = by_row[item["row_id"]]
            arm = ARM_BY_ID[item["arm_id"]]
            print(f"[{position}/{len(todo)}] {item['row_class']} "
                  f"{item['row_id']} {arm.arm_id} "
                  f"(aR={arm.alpha_repair}, aS={arm.alpha_safety})", flush=True)

            plans: Dict[int, Any] = {}
            per_state: Dict[str, Any] = {}
            try:
                if arm.is_baseline:
                    generated = generate_response(
                        model, tokenizer, row["prompt_text"], generation,
                        controller=controller, mode="off",
                        example_id=item["row_id"])
                    status = "baseline"
                    hooks_expected = 0
                else:
                    if captured_row != item["row_id"]:
                        capture_prompt_states(
                            model, tokenizer, row["prompt_text"], controller,
                            capture_name="target", example_id=item["row_id"],
                            config=generation)
                        capture = dict(controller.capture_for("target"))
                        captured_row = item["row_id"]
                    plans = build_plans(bank, arm, capture, BUDGET_REFERENCE)
                    generated = generate_response(
                        model, tokenizer, row["prompt_text"], generation,
                        controller=controller, mode="patch", plans=plans,
                        example_id=item["row_id"])
                    per_state = {
                        str(state): {
                            **plans[state].diagnostics(),
                            "repair_direction_sha256": bank.direction_sha256(
                                "repair_direction", state),
                            "safety_direction_sha256": bank.direction_sha256(
                                "safety_direction", state),
                        } for state in sorted(plans)}
                    status = ("ok" if all(plan.status == "ok"
                                          for plan in plans.values())
                              else "zero_delta")
                    hooks_expected = len(plans)
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

            record = make_record(
                row_id=item["row_id"], row_class=item["row_class"],
                arm_id=arm.arm_id, alpha_R=arm.alpha_repair,
                alpha_S=arm.alpha_safety,
                prompt_sha256=item["prompt_sha256"],
                response_text=generated["response_text"],
                response_sha256=sha256_text(generated["response_text"]),
                prompt_token_count=generated["prompt_token_count"],
                completion_token_count=generated["completion_token_count"],
                eos_reached=generated["eos_reached"],
                hit_max_new_tokens=generated["hit_max_new_tokens"],
                elapsed_seconds=generated["elapsed_seconds"],
                status=status, error_message=error_message,
                applied_by_state={str(state): int(count) for state, count
                                  in (generated.get("applied_by_state") or {}
                                      ).items()},
                hooks_fired=sum(int(count) for count
                                in (generated.get("applied_by_state") or {}
                                    ).values()),
                hooks_expected=hooks_expected, per_state=per_state,
                generation=generation.as_dict(),
                created_utc=utc_now(),
                environment=environment,
                **run_locked)
            append_record(records_path, record)
            written += 1
    finally:
        controller.close()

    print(f"[screen] wrote {written} record(s) to {records_path}"
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
