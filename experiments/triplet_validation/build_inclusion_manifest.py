#!/usr/bin/env python3
"""Produce manifests/rq1_triplet_inclusion.json. Metadata only, or nothing.

The last step. It reads the adjudicated cells, applies the frozen inclusion
rules, and writes one committed manifest carrying no prompt text and no
response text -- only digests, verdicts, record hashes and statuses.

It refuses to write at all until every required judgement exists. A manifest
produced from partial evidence would be read as a validation record, and it
would not be one.

Three statuses, kept apart
--------------------------
``validated``   every cell the rule needs is present and passes
``invalid``     every cell is present and at least one fails
``unresolved``  a cell the rule needs has no adjudicated verdict

"we do not know" and "we know it fails" are different findings and only the
second is a reason to exclude a case. A failing case is reported as failing,
with its reason; it is never replaced by a substitute, and this script has no
code path that could substitute one.

Naming the evidence
-------------------
``--adjudicated-cells`` names the record to build from, and names it exactly:
the directory holding ``adjudicated_cells.jsonl``, or that file. This matters
once more than one adjudication exists on disk. The experiment was reviewed in
two passes -- 50 supplemental cells, then the 60 historical ones -- and each
pass leaves a record. Only the merged 110-cell record is the complete evidence,
and either half of it would build a manifest that reads like a validation and
is not one. So the count is checked: anything other than the full grid of 110
required cells is refused unless ``--allow-incomplete`` says, explicitly, that
a REPORT-ONLY draft is what was wanted.

Usage
-----
    python experiments/triplet_validation/build_inclusion_manifest.py --dry-run
    python experiments/triplet_validation/build_inclusion_manifest.py \\
        --adjudicated-cells $TV/adjudicated_complete \\
        --output manifests/rq1_triplet_inclusion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.runrecord import utc_now                              # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ADJUDICATED_CELLS_FILE,
    ARMS,
    EXPECTED_CASE_IDS,
    INCLUSION_MANIFEST,
    INCLUSION_RULES,
    RETAINED_SEPARATELY,
    STATUS_UNRESOLVED,
    STATUS_VALIDATED,
    TripletError,
    assert_no_text_fields,
    evaluate_rule,
    index_cells,
    load_protocol_manifest,
    read_jsonl,
    required_cell_count,
    required_cells,
    working_dir,
)
from routegate.validation import Checklist                           # noqa: E402


def resolve_cells(adjudicated: Path) -> Path:
    """The cell file named by ``--adjudicated-cells``, directory or file.

    Resolved rather than assumed: with two adjudication records on disk -- the
    first pass and the merged one -- the difference between them is the
    difference between a draft and the validation record.
    """
    path = (adjudicated / ADJUDICATED_CELLS_FILE if adjudicated.is_dir()
            else adjudicated)
    if not path.is_file():
        raise TripletError(
            f"{path} is not a file. --adjudicated-cells names the completed "
            f"evidence record: the directory holding {ADJUDICATED_CELLS_FILE} "
            f"or that file itself.")
    return path


def build(adjudicated: Path, out_path: Path, allow_incomplete: bool,
          write: bool = True) -> Dict[str, Any]:
    checks = Checklist("rq1_triplet_inclusion")
    protocol = load_protocol_manifest()
    cells_path = resolve_cells(adjudicated)
    rows = read_jsonl(cells_path)
    if not rows:
        raise TripletError(f"{cells_path} holds no adjudicated cells")

    # Keyed, so a cell recorded twice is refused rather than resolved by
    # whichever line was read last.
    indexed = index_cells(rows, str(cells_path))
    by_case: Dict[str, Dict[tuple, Dict[str, Any]]] = defaultdict(dict)
    for (case_id, arm, axis), row in indexed.items():
        by_case[case_id][(arm, axis)] = row

    needed = required_cells()

    # A case that produced no cell at all does not appear in by_case, so
    # iterating by_case alone would report it as complete by never mentioning
    # it. The ten operative case ids are frozen; the check is against that
    # list, not against whatever happens to have arrived.
    present_ids = set(by_case)
    expected_ids = set(EXPECTED_CASE_IDS)
    absent = sorted(expected_ids - present_ids)
    unexpected = sorted(present_ids - expected_ids)
    checks.record("all_ten_cases_present", not absent,
                  f"{len(present_ids)}/{len(EXPECTED_CASE_IDS)} case(s)"
                  + (f"; entirely absent: {absent}" if absent else ""))
    checks.record("no_unexpected_cases", not unexpected,
                  f"{len(unexpected)} unexpected"
                  + (f": {unexpected}" if unexpected else ""))
    if unexpected:
        raise TripletError(
            f"adjudicated cells carry case id(s) that are not among the ten "
            f"frozen operative cases: {unexpected}. The ten are fixed before "
            f"any verdict exists; an eleventh case is a data error, and "
            f"silently including it would be a substitution.")

    gaps: List[str] = []
    for case_id in sorted(expected_ids):
        cells = by_case.get(case_id, {})
        if not cells:
            # Named explicitly, once per required cell, so the gap count is
            # the true number of missing judgements rather than one line that
            # hides eleven.
            gaps.extend(f"{case_id}/{arm}/{axis}" for arm, axis in needed)
            continue
        for arm, axis in needed:
            cell = cells.get((arm, axis))
            if cell is None or cell.get("verdict") is None:
                gaps.append(f"{case_id}/{arm}/{axis}")

    expected_cell_total = required_cell_count()
    resolved = expected_cell_total - len(gaps)
    checks.record("required_cells_per_case", True,
                  f"{len(needed)} cell(s): {needed}")
    checks.record("required_cell_total",
                  expected_cell_total == len(needed) * len(EXPECTED_CASE_IDS),
                  f"{len(needed)} x {len(EXPECTED_CASE_IDS)} = "
                  f"{expected_cell_total}")
    checks.record("all_required_cells_resolved", not gaps,
                  f"{resolved}/{expected_cell_total} resolved, "
                  f"{len(gaps)} gap(s)" + (f": {gaps[:8]}" if gaps else ""))
    checks.record("missing_or_unresolved_cells", not gaps,
                  f"{len(gaps)} gap(s)" + (f": {gaps[:8]}" if gaps else ""))

    # The record's own size, kept apart from the gap count. A half-record --
    # the 50 supplemental cells, or the 60 historical ones -- shows up as
    # gaps; a record of the right size that carries cells no rule asks for
    # does not, and is equally not the evidence record.
    complete = len(indexed) == expected_cell_total
    checks.record("evidence_record_is_the_complete_grid", complete,
                  f"{len(indexed)} adjudicated cell(s) in {cells_path.name}, "
                  f"expected {expected_cell_total}")

    if gaps and not allow_incomplete:
        raise TripletError(
            f"{len(gaps)} required judgement(s) are missing or unresolved, so "
            f"no inclusion manifest is written. The record holds "
            f"{len(indexed)} of {expected_cell_total} cell(s); the first few "
            f"gaps are {gaps[:8]}. A manifest built from partial evidence "
            f"reads as a validation record and is not one. Merge the passes "
            f"into one complete record, or pass --allow-incomplete to emit a "
            f"REPORT-ONLY draft that is written beside the manifest and never "
            f"to it.")
    if not complete and not allow_incomplete:
        raise TripletError(
            f"{cells_path} holds {len(indexed)} adjudicated cell(s) and the "
            f"complete evidence record is {expected_cell_total}. Every "
            f"required judgement is present, so this record carries "
            f"{len(indexed) - expected_cell_total:+d} cell(s) beyond the "
            f"frozen grid. It is refused rather than trimmed; pass "
            f"--allow-incomplete for a REPORT-ONLY draft.")

    cases: Dict[str, Any] = {}
    totals: Dict[str, Dict[str, int]] = {
        name: defaultdict(int) for name in INCLUSION_RULES}
    # Over the frozen ten, not over what arrived. A case with no cells at all
    # must still appear, and must appear as unresolved on every rule -- an
    # absent case is "we do not know", and dropping it from the manifest would
    # silently turn that into "nothing to report".
    for case_id in sorted(EXPECTED_CASE_IDS):
        cells = by_case.get(case_id, {})
        verdicts: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
        arm_records: Dict[str, Any] = {}
        for (arm, axis), cell in sorted(cells.items()):
            verdicts[arm][axis] = cell.get("verdict")
            record = arm_records.setdefault(arm, {
                "prompt_sha256": cell.get("prompt_sha256"),
                "response_sha256": cell.get("response_sha256"),
                # Preserved so a reader can see that a response was capped, and
                # see that the cap changed nothing: the verdict beside it comes
                # from the frozen rubric applied to the observed response.
                "reached_max_tokens": cell.get("reached_max_tokens"),
                "output_token_count": cell.get("output_token_count"),
                "axes": {},
            })
            record["axes"][axis] = {
                "verdict": cell.get("verdict"),
                "evidence_source": cell.get("label_source"),
                "adjudication_rule": cell.get("adjudication_rule"),
                "judges_agree": cell.get("judges_agree"),
                "sonnet_verdict": cell.get("sonnet_verdict"),
                "opus_verdict": cell.get("opus_verdict"),
                "sonnet_record_sha256": cell.get("sonnet_record_sha256"),
                "opus_record_sha256": cell.get("opus_record_sha256"),
                "human_record_sha256": cell.get("human_record_sha256"),
                "retained_separately": cell.get("verdict") in RETAINED_SEPARATELY,
            }

        rules: Dict[str, Any] = {}
        for name in INCLUSION_RULES:
            status, reasons = evaluate_rule(name, verdicts)
            totals[name][status] += 1
            rules[name] = {
                "inclusion_status": status,
                "exclusion_reason": reasons or None,
                "purpose": INCLUSION_RULES[name]["purpose"],
            }
        cases[case_id] = {"arms": arm_records, "rules": rules,
                          "cells_present": len(cells),
                          "cells_expected": len(needed),
                          "entirely_absent": not cells}

    manifest = {
        "schema": "routegate/rq1_triplet_inclusion/1",
        "description": "Inclusion status of the ten operative E/B/H cases, from "
                       "retrospective two-axis judging plus blinded human "
                       "adjudication. Metadata only: no prompt text and no "
                       "response text appears anywhere in this file.",
        "created_utc": utc_now(),
        "retrospective": True,
        "retrospective_note": "these judgements were made after the steering "
                              "outcomes were known. Every judge and the human "
                              "reviewer were blind to those outcomes, to the "
                              "E/B/H role and to each other. No case was "
                              "replaced after validation.",
        "expected_case_ids": list(EXPECTED_CASE_IDS),
        "expected_cells_per_case": len(needed),
        "expected_cells_total": expected_cell_total,
        "adjudicated_cells_read": len(indexed),
        "adjudicated_cells_source": cells_path.name,
        "resolved_cells": resolved,
        "cases_entirely_absent": absent,
        "patched_responses_used_for_inclusion": False,
        "capped_response_policy": (
            "a response that reached max_tokens is used exactly as historically "
            "stored. It is never regenerated or extended, and the cap neither "
            "passes nor fails it. The frozen Axis-2 rubric classifies the "
            "observed response; only SA satisfies donor inclusion. "
            "reached_max_tokens is preserved beside every arm as evidence."),
        "protocol_hashes": protocol["protocol_hashes"],
        "axis1_version": protocol["axis1_version"]["name"],
        "axis1_version_caveat": protocol["axis1_version"]["recalibration"][
            "acceptance_criterion_2"],
        "axis2_version": protocol["axis2_version"]["name"],
        "inclusion_rules": {
            name: {"purpose": spec["purpose"],
                   "requires": {arm: {axis: list(values)
                                      for axis, values in axes.items()}
                                for arm, axes in spec["requires"].items()}}
            for name, spec in INCLUSION_RULES.items()},
        "retained_separately": list(RETAINED_SEPARATELY),
        "retained_separately_note": "these behaviour codes satisfy no rule and "
                                    "are never coerced into a passing label",
        "totals": {name: dict(sorted(counts.items()))
                   for name, counts in totals.items()},
        "cases": cases,
        "checklist": checks.as_dict(),
    }
    if gaps or not complete:
        manifest["REPORT_ONLY_DRAFT"] = True
        manifest["gaps"] = gaps

    assert_no_text_fields(manifest, "rq1_triplet_inclusion.json")
    checks.record("manifest_is_metadata_only", True,
                  "no prompt or response text field present")

    if write:
        target = (out_path if not gaps and complete
                  else out_path.with_name(out_path.stem + ".DRAFT.json"))
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                          encoding="utf-8")
        print(f"[inclusion] wrote {target}")
    for name, counts in manifest["totals"].items():
        print(f"[inclusion] {name:<28} {dict(counts)}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudicated-cells", "--adjudicated",
                        dest="adjudicated_cells", type=Path, default=None,
                        help="the complete evidence record: the directory "
                             "holding " + ADJUDICATED_CELLS_FILE + ", or that "
                             "file. Named explicitly, because more than one "
                             "adjudication record exists on disk.")
    parser.add_argument("--output", type=Path, default=INCLUSION_MANIFEST)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="emit a REPORT-ONLY draft beside the manifest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = working_dir()
    adjudicated = args.adjudicated_cells or (root / "adjudicated")
    try:
        build(adjudicated, args.output, args.allow_incomplete,
              write=not args.dry_run)
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
