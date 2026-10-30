#!/usr/bin/env python3
"""The phase-2 readings, produced from the completed phase-2 adjudication.

Offline and deterministic. Calls nothing, judges nothing, decides no controller
setting and reads no evaluation material.

**It does not implement the decision.** The phase-2 decision rule -- the
recovery threshold, the safety-only veto, the tie-break, the undetermined
branch and the no-qualifier branch -- was frozen in ``routegate.phase2`` before
any phase-2 output existed. This script produces the four counts per alpha_S
that the rule reads and calls
``routegate.phase2.evaluate_phase2_decision`` with them. There is no second
copy of the rule here, and a reading that disagreed with it would be a bug in
this file rather than a different answer.

What it produces
----------------
``analysis_table.jsonl``     the 180 records: 144 adjudicated in phase 2 and 36
                             inherited from the completed phase-1 adjudication,
                             projected onto one shape, each naming its source
                             and carrying the digest of the record it came
                             from. Nothing is rewritten.
``a_safety_counts_by_arm``   YES/NO/UNCLEAR/UNRESOLVED at each of the ten arms.
``b_per_row_transitions``    each of the 18 rows at every arm. 18 is small
                             enough that the per-row table is the honest
                             presentation.
``c_counts_by_strength``     the four predeclared counts at each alpha_S.
``d_decision``               whatever ``evaluate_phase2_decision`` returns:
                             ``STRENGTH_SELECTED``, ``NO_STRENGTH_QUALIFIES``
                             or ``UNDETERMINED``.

What it refuses
---------------
*An inherited reference that cannot be reconciled.* The 36 phase-1 cells must
be exactly 36, over exactly the 18 bound rows and the two reference arms, on
the safety rubric, under the declared procedure, decided, and judged over the
same prompt digest the phase-2 records carry -- and their verdicts must be the
selection rule that produced these rows: NO at baseline, YES at repair_015. A
mismatch stops the reading rather than being resolved in favour of either copy.

*A predeclaration or a selection that has moved.* Both are verified against the
package before a count is taken.

*A population rate.* These 18 rows were selected because they regressed. No
count here is divided by any corpus denominator.

Usage
-----
    python experiments/focused_direction_judging/report_phase2.py --dry-run
    python experiments/focused_direction_judging/report_phase2.py \\
        --adjudicated $FDV/phase2/judging/adjudicated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.focusedjudging import (                              # noqa: E402
    ADJUDICATED_CELLS_FILE,
    BASELINE_ARM,
    H_TEST_ARM,
    JUDGING_MANIFEST,
    declared_adjudication_procedure,
    judging_root,
)
from routegate.dualsteer import PHASE2_SELECTED_ROWS                # noqa: E402
from routegate.phase2 import (                                      # noqa: E402
    PHASE2_DECISION_RULE,
    PHASE2_PREDECLARATION,
    PHASE2_PRIMARY_ENDPOINT,
    PHASE2_SECONDARY_RESULTS,
    PHASE2_WITHHELD,
    evaluate_phase2_decision,
    verify_phase2_predeclaration,
    verify_phase2_selection,
)
from routegate.phase2judging import (                               # noqa: E402
    FROZEN_COVERAGE,
    FROZEN_PHASE2_INPUT,
    INHERITED_GATES,
    JUDGING_SUBDIR,
    PHASE2_ALPHAS,
    PHASE2_ARM_IDS,
    PHASE2_AXIS,
    PHASE2_REFERENCE_ARM_IDS,
    PHASE2_ROWS,
    PROTOCOL_NAME,
    REUSED_UNCHANGED,
    WITHHELD,
    Phase2JudgingError,
    analysis_table,
    assert_inherited_configuration,
    counts_by_strength,
    index_phase2_cells,
    load_inherited_references,
    per_row_transitions,
    phase2_root,
    verdicts_by_arm,
)
from routegate.triplets import (                                    # noqa: E402
    AXIS_VALUES,
    SOURCE_UNRESOLVED,
    assert_no_text_fields,
    read_jsonl,
    write_jsonl,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
)

UNRESOLVED = "UNRESOLVED"
ALL_ARMS = PHASE2_REFERENCE_ARM_IDS + PHASE2_ARM_IDS


def load_phase2_adjudicated(path: Path) -> Dict[Any, Dict[str, Any]]:
    file = path / ADJUDICATED_CELLS_FILE if path.is_dir() else path
    if not file.is_file():
        raise Phase2JudgingError(
            f"{file} is not a file. --adjudicated names the directory holding "
            f"phase 2's {ADJUDICATED_CELLS_FILE} or that file itself.")
    cells = index_phase2_cells(read_jsonl(file), str(file))
    if not cells:
        raise Phase2JudgingError(f"{file} holds no adjudicated cell")
    return cells


def safety_counts_by_arm(verdicts: Mapping[Any, Optional[str]],
                         rows: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """The label counts per arm, with UNRESOLVED as its own key."""
    table: Dict[str, Dict[str, int]] = {}
    for arm in ALL_ARMS:
        counts = {label: 0 for label in AXIS_VALUES[PHASE2_AXIS]}
        counts[UNRESOLVED] = 0
        for row in rows:
            value = verdicts.get((row, arm))
            counts[value if value is not None else UNRESOLVED] += 1
        table[arm] = counts
    return table


def report(adjudicated: Path, phase1_adjudicated: Path, out_dir: Path,
           write: bool) -> Dict[str, Any]:
    checks = Checklist("focused_phase2_report")

    # ---- the freeze, before a single count is taken ------------------------
    selection = verify_phase2_selection(checks=checks)
    verify_phase2_predeclaration(checks=checks)
    assert_inherited_configuration(checks)
    procedure = declared_adjudication_procedure()
    checks.record("adjudication_procedure_declared", True, procedure)

    file = (adjudicated / ADJUDICATED_CELLS_FILE if adjudicated.is_dir()
            else adjudicated)
    cells = load_phase2_adjudicated(adjudicated)
    checks.require("phase2_adjudicated_cells",
                   len(cells) == FROZEN_COVERAGE["cells"],
                   f"{len(cells)} cell(s), expected {FROZEN_COVERAGE['cells']}")

    procedures = sorted({r.get("adjudication_procedure")
                         for r in cells.values()})
    checks.require("one_adjudication_procedure", len(procedures) == 1,
                   f"{procedures}")
    checks.require("procedure_matches_the_declaration",
                   procedures[0] == procedure,
                   f"records say {procedures[0]!r}, "
                   f"{JUDGING_MANIFEST.name} declares {procedure!r}")

    # ---- the 36 inherited references, reconciled and read only -------------
    # One prompt per row, across all eight arms. Taking the first and hoping
    # would make the inherited-reference reconciliation below check a digest
    # that only some of the row's cells actually carry.
    prompt_by_row: Dict[str, str] = {}
    for (row_id, arm_id, _axis), record in sorted(cells.items()):
        digest = record["prompt_sha256"]
        if prompt_by_row.setdefault(row_id, digest) != digest:
            raise Phase2JudgingError(
                f"{row_id}: arm {arm_id} was judged over prompt {digest} and "
                f"another arm of the same row over "
                f"{prompt_by_row[row_id]}. One row is one prompt; two would "
                f"make the transition table compare different questions.")
    checks.record("one_prompt_digest_per_row", True,
                  f"{len(prompt_by_row)} row(s), one prompt digest each")
    inherited = load_inherited_references(phase1_adjudicated, prompt_by_row,
                                          procedure, checks)

    table = analysis_table(cells, inherited, checks)
    verdicts = verdicts_by_arm(table)
    rows = list(PHASE2_SELECTED_ROWS)

    unresolved = sum(1 for row in table
                     if row["label_source"] == SOURCE_UNRESOLVED)
    checks.record("unresolved_cells", True, str(unresolved))

    counts = safety_counts_by_arm(verdicts, rows)
    transitions = per_row_transitions(verdicts, rows)
    by_strength = counts_by_strength(verdicts, rows)

    # ---- the frozen rule, called and not restated --------------------------
    decision = evaluate_phase2_decision(
        {alpha: entry for alpha, entry in by_strength.items()})

    payload: Dict[str, Any] = {
        "schema": "routegate/focused_direction_phase2_readings/1",
        "protocol": PROTOCOL_NAME,
        "phase": 2,
        "created_utc": utc_now(),
        "adjudication_procedure": procedure,
        "rows": PHASE2_ROWS,
        "cells_judged_in_phase2": len(cells),
        "cells_inherited_from_phase1": len(inherited),
        "analysis_table_records": len(table),
        "unresolved_cells": unresolved,
        "complete_adjudication": unresolved == 0,
        "a_safety_counts_by_arm": counts,
        "b_per_row_transitions": transitions,
        "c_counts_by_strength": {
            f"{alpha:.2f}": dict(entry)
            for alpha, entry in sorted(by_strength.items())},
        "d_decision": decision,
        "decision_rule": PHASE2_DECISION_RULE["name"],
        "decision_rule_lives_in": "routegate.phase2.evaluate_phase2_decision",
        "decision_rule_is_not_reimplemented_here": True,
        "primary_endpoint": dict(PHASE2_PRIMARY_ENDPOINT),
        "secondary_results": [dict(entry) for entry in
                              PHASE2_SECONDARY_RESULTS],
        "inherited_references": {
            "source": str(phase1_adjudicated),
            "cells": len(inherited),
            "arms": list(PHASE2_REFERENCE_ARM_IDS),
            "gates": list(INHERITED_GATES),
            "regenerated": False,
            "rejudged": False,
            "readjudicated": False,
            "rewritten": False,
            "note": "baseline and repair_015 for these 18 rows were generated, "
                    "judged and adjudicated in phase 1 at the same model, "
                    "revision, direction bank, generation configuration, "
                    "tokenizer and chat template, under this same procedure and "
                    "the same frozen safety rubric. They are read.",
        },
        "selection": {
            "manifest": "focused_direction_validation_v1_phase2_selection.json",
            "manifest_sha256": selection["manifest_sha256"],
            "selected_rows_sha256": selection["selected_rows_sha256"],
            "rows": PHASE2_ROWS,
        },
        "reportable": True,
        "reportable_note":
            "the frozen decision rule covers the undecided case itself: a "
            "strength holding an undecided row is UNDETERMINED and no strength "
            "is selected. The outcome above is that rule's answer to these "
            "counts, whatever it is."
            if unresolved else
            "every one of the 180 records carries a decided verdict",
        "does_not_claim": {
            "a_population_rate": False,
            "a_controller_setting": False,
            "a_deployment_strength": False,
            "evidence_about_an_alpha_s_off_this_grid": False,
            "an_axis1_or_axis2_verdict_for_these_cells": False,
            "that_any_reference_arm_was_regenerated": False,
        },
        "withheld": {**dict(PHASE2_WITHHELD), **dict(WITHHELD)},
        "reused_unchanged": dict(REUSED_UNCHANGED),
        "inputs": {
            "phase2_adjudicated_cells": str(file),
            "phase2_adjudicated_cells_sha256": sha256_file(file),
            "phase1_adjudicated_cells": str(
                phase1_adjudicated / ADJUDICATED_CELLS_FILE
                if phase1_adjudicated.is_dir() else phase1_adjudicated),
            "records_sha256": FROZEN_PHASE2_INPUT["records_sha256"],
            "predeclaration": PHASE2_PREDECLARATION.name,
            "judging_predeclaration": JUDGING_MANIFEST.name,
        },
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(payload, "the phase-2 readings")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "analysis_table.jsonl", table)
        (out_dir / "phase2_readings.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[report] wrote {out_dir / 'phase2_readings.json'}")
        print(f"[report] wrote {out_dir / 'analysis_table.jsonl'} "
              f"({len(table)} records)")

    print(f"[report] procedure        {procedure}")
    print(f"[report] {'judged in phase 2':<21} {len(cells)}")
    print(f"[report] {'inherited from phase 1':<21} {len(inherited)}")
    print(f"[report] {'analysis table':<21} {len(table)}")
    print(f"[report] unresolved cells {unresolved}")
    print(f"[report] baseline/{H_TEST_ARM} (inherited): "
          f"{counts[BASELINE_ARM]} / {counts[H_TEST_ARM]}")
    for alpha in PHASE2_ALPHAS:
        entry = by_strength[alpha]
        print(f"[report] alpha_S={alpha:<5} recovered "
              f"{entry['combined_recovered']}/{PHASE2_ROWS} "
              f"(undet {entry['combined_undetermined']}), harmful alone "
              f"{entry['safety_only_harmful']}/{PHASE2_ROWS} "
              f"(undet {entry['safety_only_undetermined']})")
    print(f"[report] outcome          {decision['outcome']}")
    print(f"[report] {decision['statement']}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudicated", type=Path, default=None,
                        help="the completed phase-2 adjudication")
    parser.add_argument("--phase1-adjudicated", type=Path, default=None,
                        help="the completed phase-1 adjudication the 36 "
                             "reference cells are read from")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = phase2_root() / JUDGING_SUBDIR
        adjudicated = args.adjudicated or (root / "adjudicated")
        phase1 = args.phase1_adjudicated or (
            judging_root() / "judging" / "adjudicated")
        out_dir = args.output_dir or (root / "readings")
        report(adjudicated, phase1, out_dir, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
