#!/usr/bin/env python3
"""Build the blind phase-2 judging packet. Calls nothing, generates nothing.

No model is loaded, no GPU is touched, no API is called and **no response is
generated**. Every prompt and every response already exists, frozen, and is
reconciled by SHA256 before it is written into a row.

What it builds
--------------
One judged row per phase-2 screen record, 144 of them, all on one rubric::

    H records (144)  ->  safety_judge_input.jsonl   the frozen actionable-
                                                    harmful-help safety rubric

Axis 1 and Axis 2 are **not** run on these cells. Axis 1 is prompt-only and was
settled when the corpus was built; Axis 2 is the E-side response-behaviour
question phase 1 asked, and phase 2 steers no E row.

The two reference arms are **not** here
---------------------------------------
``baseline`` and ``repair_015`` for these 18 rows were judged in phase 1. This
packet holds none of them: the build refuses a record carrying one, and the
inherited cells are reconciled against the completed phase-1 adjudication
before anything is written, so a run that could not be joined to its references
fails here rather than after 288 billed calls.

Blindness
---------
A judge row carries exactly ``row_id``, ``prompt_text`` and ``response_text``,
constructed by whitelist projection from three named values. The arm, the two
coefficients, the direction, the layers, the grouping of a row's arms, the
expected result -- and, in this phase, the fact that every row in the packet
was SELECTED for having regressed -- are all absent, and the build refuses a
row that carries any of them. The opaque ids are freshly minted, independently
randomised, and prefixed so they cannot collide with the phase-1 packet's ids,
the reviewer ids or either retrospective packet's ids.

Rows are written in ascending opaque-id order. Because the ids are
cryptographically random and are minted before the sort, that order is
independent of arm and of corpus position, so a judge reading the file in order
sees no grouping. The scientific identity survives only in
``private/row_map.jsonl``, which is written under the machine-local data root
and is never committed.

Usage
-----
    python experiments/focused_direction_judging/build_phase2_judge_packet.py --dry-run
    python experiments/focused_direction_judging/build_phase2_judge_packet.py
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

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.focusedjudging import (                              # noqa: E402
    CAPPED_RESPONSE_POLICY,
    ROW_MAP_FILE,
    assert_no_corpus_row_id,
    assert_projection_is_exact,
    assert_rendered_request_is_projection,
    declared_adjudication_procedure,
    judging_root,
    load_prompt_texts,
    project_judge_row,
)
from routegate.phase2 import (                                      # noqa: E402
    PHASE2_DECISION_RULE,
    PHASE2_PRIMARY_ENDPOINT,
    verify_phase2_predeclaration,
    verify_phase2_selection,
)
from routegate.phase2judging import (                               # noqa: E402
    CONSOLIDATED_SUBDIR,
    DEDUPLICATION,
    FROZEN_COVERAGE,
    FROZEN_PHASE2_INPUT,
    INHERITED_GATES,
    JUDGING_SUBDIR,
    PHASE2_ALPHAS,
    PHASE2_ARM_IDS,
    PHASE2_AXIS,
    PHASE2_FOREIGN_ROW_PREFIXES,
    PHASE2_REFERENCE_ARM_IDS,
    PHASE2_ROW_PREFIX,
    PHASE2_ROWS,
    PLANNED_CALLS_BOTH_JUDGES,
    PLANNED_CALLS_PER_JUDGE,
    PROTOCOL_NAME,
    REUSED_UNCHANGED,
    WITHHELD,
    Phase2JudgingError,
    assert_inherited_configuration,
    assert_phase2_row_has_no_forbidden_field,
    load_inherited_references,
    load_phase2_consolidation_claims,
    load_phase2_screen_records,
    phase2_root,
)
from routegate.triplets import (                                    # noqa: E402
    JUDGE_ROW_FIELDS,
    JUDGES,
    MAX_TOKENS,
    TEMPERATURE,
    assert_ids_unique,
    assert_no_text_fields,
    assert_row_is_blind,
    load_protocol,
    load_protocol_manifest,
    new_opaque_ids,
    working_dir,
    write_jsonl,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
    sha256_text,
)

PACKET_NAME = "focused_direction_validation_v1_phase2_packet"


def collect_items(records: Sequence[Dict[str, Any]],
                  prompts: Dict[str, Dict[str, str]],
                  checks: Checklist) -> List[Dict[str, Any]]:
    """One judged item per phase-2 record, with its text reconciled by digest."""
    items: List[Dict[str, Any]] = []
    for number, record in enumerate(records, start=1):
        origin = f"phase-2 screen record {number}"
        screen_row_id = record["row_id"]
        corpus = prompts.get(screen_row_id)
        if corpus is None:
            raise Phase2JudgingError(
                f"{origin}: {screen_row_id!r} is not a row of the pinned "
                f"corpus. The prompt text is resolved from the corpus and is "
                f"never reconstructed from anywhere else.")
        if corpus["row_class"] != record["row_class"]:
            raise Phase2JudgingError(
                f"{origin}: {screen_row_id!r} is class {record['row_class']!r} "
                f"in the screen and {corpus['row_class']!r} in the corpus")
        if corpus["prompt_sha256"] != record["prompt_sha256"]:
            raise Phase2JudgingError(
                f"{origin}: the corpus prompt for {screen_row_id!r} hashes to "
                f"{corpus['prompt_sha256']} and the record recorded "
                f"{record['prompt_sha256']}. The prompt behind this response "
                f"has moved; refused rather than judged against the wrong one.")
        items.append({
            "screen_row_id": screen_row_id,
            "row_class": record["row_class"],
            "arm_id": record["arm_id"],
            "axis": PHASE2_AXIS,
            "prompt_text": corpus["prompt_text"],
            "prompt_sha256": record["prompt_sha256"],
            "response_text": record["response_text"],
            "response_sha256": record["response_sha256"],
            "hit_max_new_tokens": bool(record["hit_max_new_tokens"]),
            "completion_token_count": int(record["completion_token_count"]),
        })
    checks.record("items_collected", True,
                  f"{len(items)} item(s) over {PHASE2_ROWS} bound row(s)")
    return items


def build(consolidated: Path, out_dir: Path, protocol_dir: Path,
          dataset: Path, phase1_adjudicated: Path,
          write: bool = True) -> Dict[str, Any]:
    checks = Checklist(PACKET_NAME)
    manifest = load_protocol_manifest()

    protocol = load_protocol(PHASE2_AXIS, directory=protocol_dir,
                             manifest=manifest)
    for key, value in protocol.hashes().items():
        checks.record(key, True, value)

    # The freeze this phase is bound to, and the configuration it inherits.
    # Checked before a row is rendered: a packet built against a selection or a
    # predeclaration that has moved is a packet for a different experiment.
    selection = verify_phase2_selection(checks=checks)
    verify_phase2_predeclaration(checks=checks)
    assert_inherited_configuration(checks)
    procedure = declared_adjudication_procedure()
    checks.record("adjudication_procedure", True, procedure)

    records = load_phase2_screen_records(consolidated, checks)
    claims = load_phase2_consolidation_claims(consolidated, checks)
    prompts = load_prompt_texts(dataset, checks)
    items = collect_items(records, prompts, checks)

    # The 36 inherited references, reconciled BEFORE any billed call. A phase-2
    # run whose references cannot be reconciled produces 144 verdicts that
    # nothing can be compared with.
    prompt_by_row = {item["screen_row_id"]: item["prompt_sha256"]
                     for item in items}
    inherited = load_inherited_references(phase1_adjudicated, prompt_by_row,
                                          procedure, checks)

    screen_row_ids = sorted(prompts)
    ids = new_opaque_ids(PHASE2_ROW_PREFIX[PHASE2_AXIS], len(items))
    assert_ids_unique(ids)

    minted: List[Dict[str, Any]] = []
    private_rows: List[Dict[str, Any]] = []
    for row_id, item in zip(ids, items):
        label = f"{PHASE2_AXIS} row {row_id}"
        # Whitelist projection. The row is CONSTRUCTED from three named values;
        # the screen record is never copied, filtered or stripped, so no
        # treatment field can reach the judge by omission.
        row = project_judge_row(item, PHASE2_AXIS, row_id)
        assert_projection_is_exact(row, PHASE2_AXIS, item, label,
                                   prefix=PHASE2_ROW_PREFIX[PHASE2_AXIS])
        assert_row_is_blind(row, PHASE2_AXIS)
        assert_phase2_row_has_no_forbidden_field(row, label)
        system, user = protocol.render(row)
        # The request is the frozen instructions plus the frozen template
        # filled with exactly those three values, byte for byte. Authentic
        # prompt and response text is never inspected for what it says.
        assert_rendered_request_is_projection(system, user, row, protocol,
                                              label)
        assert_no_corpus_row_id(user, screen_row_ids, label)
        minted.append(row)
        private_rows.append({
            "row_id": row_id,
            "axis": PHASE2_AXIS,
            "screen_row_id": item["screen_row_id"],
            "row_class": item["row_class"],
            "arm_id": item["arm_id"],
            "prompt_sha256": item["prompt_sha256"],
            "response_sha256": item["response_sha256"],
            "hit_max_new_tokens": item["hit_max_new_tokens"],
            "completion_token_count": item["completion_token_count"],
            "rendered_user_sha256": sha256_text(user),
            "rendered_system_sha256": sha256_text(system),
        })

    # ascending opaque-id order: random with respect to arm and corpus
    # position, so the file itself carries no grouping
    judge_rows = sorted(minted, key=lambda r: r["row_id"])

    all_ids = [row["row_id"] for row in judge_rows]
    assert_ids_unique(all_ids)
    checks.require("opaque_ids_unique", len(all_ids) == len(set(all_ids)),
                   f"{len(all_ids)} id(s), {len(set(all_ids))} unique")
    checks.require(
        "ids_do_not_collide_with_another_packet",
        not any(i.startswith(PHASE2_FOREIGN_ROW_PREFIXES) for i in all_ids),
        f"phase-2 ids use {PHASE2_ROW_PREFIX[PHASE2_AXIS]!r} and are "
        f"independently randomised from phase 1's and from both retrospective "
        f"packets'")
    checks.require("rows_total", len(all_ids) == FROZEN_PHASE2_INPUT["rows"],
                   f"{len(all_ids)} row(s), expected "
                   f"{FROZEN_PHASE2_INPUT['rows']}")

    by_arm: Dict[str, int] = {}
    for entry in private_rows:
        by_arm[entry["arm_id"]] = by_arm.get(entry["arm_id"], 0) + 1
    checks.require("rows_by_arm", by_arm == FROZEN_COVERAGE["cells_by_arm"],
                   f"{dict(sorted(by_arm.items()))}, expected "
                   f"{FROZEN_COVERAGE['cells_by_arm']}")
    checks.require(
        "no_inherited_reference_arm_is_in_this_packet",
        not (set(by_arm) & set(PHASE2_REFERENCE_ARM_IDS)),
        f"{list(PHASE2_REFERENCE_ARM_IDS)} are read from the completed phase-1 "
        f"adjudication and are never re-judged")

    private_by_cell = {(r["screen_row_id"], r["arm_id"], r["axis"])
                       for r in private_rows}
    checks.require("one_row_per_cell",
                   len(private_by_cell) == len(private_rows),
                   f"{len(private_rows)} row(s) over {len(private_by_cell)} "
                   f"distinct (screen_row_id, arm_id, axis) cell(s)")

    summary: Dict[str, Any] = {
        "packet": PACKET_NAME,
        "protocol": PROTOCOL_NAME,
        "phase": 2,
        "created_utc": utc_now(),
        "purpose": "blind phase-2 judging: the frozen actionable-harmful-help "
                   "safety rubric on all 144 phase-2 H responses, over the 18 "
                   "rows the frozen phase-1 regression rule returned",
        "generates_responses": False,
        "runs_axis1": False,
        "runs_axis2": False,
        "runs_axis1_or_axis2_note":
            "phase 2 judges the safety axis only. Axis 1 is prompt-only and "
            "was settled when the corpus was built; Axis 2 is the E-side "
            "response-behaviour question phase 1 asked, and phase 2 steers no "
            "E row.",
        "rubric": PHASE2_AXIS,
        "arms_judged": list(PHASE2_ARM_IDS),
        "arms_inherited_not_judged": list(PHASE2_REFERENCE_ARM_IDS),
        "rows": PHASE2_ROWS,
        "rows_by_axis": {PHASE2_AXIS: len(judge_rows)},
        "rows_total": len(all_ids),
        "rows_by_arm": dict(sorted(by_arm.items())),
        "alpha_s_grid": list(PHASE2_ALPHAS),
        "calls_per_judge": PLANNED_CALLS_PER_JUDGE,
        "calls_both_judges": PLANNED_CALLS_BOTH_JUDGES,
        "judges": list(JUDGES),
        "judge_settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        "adjudication_procedure": procedure,
        "protocol_hashes": protocol.hashes(),
        "safety_version": manifest["safety_version"]["name"],
        "deduplication": dict(DEDUPLICATION),
        "capped_responses": dict(CAPPED_RESPONSE_POLICY),
        "input": {
            "consolidated_dir": str(consolidated),
            "records_sha256": FROZEN_PHASE2_INPUT["records_sha256"],
            "records_rows": FROZEN_PHASE2_INPUT["rows"],
            "run_manifest_sha256": claims["run_manifest_sha256"],
            "validation_sha256": claims["validation_sha256"],
            "dataset_sha256": FROZEN_PHASE2_INPUT["dataset_sha256"],
            "direction_bank_sha256":
                FROZEN_PHASE2_INPUT["direction_bank_sha256"],
            "generation_config_sha256":
                FROZEN_PHASE2_INPUT["generation_config_sha256"],
        },
        "selection": {
            "manifest": "focused_direction_validation_v1_phase2_selection.json",
            "manifest_sha256": selection["manifest_sha256"],
            "selected_rows_sha256": selection["selected_rows_sha256"],
            "rows": PHASE2_ROWS,
        },
        "inherited_references": {
            "source": str(phase1_adjudicated),
            "cells": len(inherited),
            "arms": list(PHASE2_REFERENCE_ARM_IDS),
            "gates": list(INHERITED_GATES),
            "regenerated": False,
            "rejudged": False,
            "readjudicated": False,
            "note": "reconciled here so a run that could not be joined to its "
                    "references fails before the first billed call, not after "
                    "288 of them",
        },
        "coverage_expected": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in FROZEN_COVERAGE.items()},
        "blindness": {
            "judge_row_fields": {PHASE2_AXIS: list(JUDGE_ROW_FIELDS[PHASE2_AXIS])},
            "row_order": "ascending opaque id; the ids are cryptographically "
                         "random and are minted before the sort, so the order "
                         "is independent of arm and corpus position",
            "private_map": f"private/{ROW_MAP_FILE}, machine-local, never "
                           f"committed",
            "prompt_or_response_text_in_git": False,
            "selection_status_withheld": True,
            "selection_status_withheld_note":
                "every row in this packet is one of the 18 that regressed. A "
                "judge told so would be told what to find, so no row, request "
                "or file name says it.",
        },
        "decision_rule": PHASE2_DECISION_RULE["name"],
        "decision_rule_lives_in": "routegate.phase2.evaluate_phase2_decision",
        "primary_endpoint": PHASE2_PRIMARY_ENDPOINT["name"],
        "reused_unchanged": dict(REUSED_UNCHANGED),
        "withheld": dict(WITHHELD),
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the phase-2 packet manifest")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        write_jsonl(out_dir / f"{PHASE2_AXIS}_judge_input.jsonl", judge_rows)
        write_jsonl(out_dir / "private" / ROW_MAP_FILE,
                    sorted(private_rows, key=lambda r: r["row_id"]))
        (out_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        sums = []
        for path in sorted(p for p in out_dir.rglob("*")
                           if p.is_file() and p.name != "SHA256SUMS.txt"):
            sums.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
        (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n",
                                                encoding="utf-8")
        print(f"[packet] wrote {out_dir}")

    print(f"[packet] {PHASE2_AXIS:<8} {len(judge_rows)} row(s) over "
          f"{PHASE2_ROWS} row(s) x {len(PHASE2_ARM_IDS)} arm(s)")
    print(f"[packet] inherited {len(inherited)} reference cell(s) reconciled, "
          f"none regenerated or re-judged")
    print(f"[packet] {PLANNED_CALLS_PER_JUDGE} call(s) per judge, "
          f"{PLANNED_CALLS_BOTH_JUDGES} across both")
    print("[packet] no model was loaded, no response was generated, "
          "no API was called")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated", type=Path, default=None,
                        help="the phase-2 consolidated package directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None,
                        help="the frozen judge-protocol text directory")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="the accepted corpus; defaults to paths.local.json")
    parser.add_argument("--phase1-adjudicated", type=Path, default=None,
                        help="the completed phase-1 adjudication the 36 "
                             "reference cells are read from")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every gate and write nothing")
    args = parser.parse_args(argv)

    try:
        root = phase2_root()
        consolidated = args.consolidated or (root / CONSOLIDATED_SUBDIR)
        out_dir = args.output_dir or (root / JUDGING_SUBDIR / "packet")
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        phase1 = args.phase1_adjudicated or (
            judging_root() / "judging" / "adjudicated")
        dataset = args.dataset
        if dataset is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            dataset = Path(paths["dataset"])
        build(consolidated, out_dir, protocol_dir, dataset, phase1,
              write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                       # noqa: BLE001 - fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
