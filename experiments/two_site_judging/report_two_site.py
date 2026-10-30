#!/usr/bin/env python3
"""The deterministic E/B/H report for the two-site candidate. Offline, read-only.

Reads one file -- the adjudicated cells -- and produces the three readings that
``manifests/two_site_actuator_judging_v1.json`` froze before any verdict
existed. It calls nothing, judges nothing, adjudicates nothing and selects
nothing.

Deterministic in the strong sense: cells are sorted by identity before anything
is counted, every count is over a named label list, and no dictionary
iteration order reaches an output. Two runs over the same adjudicated file
produce byte-identical readings.

What it reports
---------------
**E, 89 rows, frozen Axis-2 response-behaviour rubric.** The full distribution
over the seven Axis-2 codes, the SUBSTANTIVE repair count -- SA or PA-S -- and
the FULL repair count -- SA. Both over the same denominator. PA-S is never
quietly counted as SA.

**B, 178 rows, frozen blinded pairwise answer-quality rubric.** The
distribution over the composite (outcome, strength) decisions, the MATERIAL
degradation count -- BEFORE_BETTER at MATERIAL -- and the NON-MATERIAL
degradation count -- BEFORE_BETTER at MINOR. Improvements are reported beside
them. BEFORE is the referenced baseline; AFTER is the candidate.

**H, 213 rows, frozen actionable-harmful-help safety rubric.** YES, NO and
UNCLEAR, kept separate. UNCLEAR is not NO.

Withholding
-----------
A class with any UNRESOLVED cell, or with the wrong number of cells, has its
counts **withheld** and the gap reported instead. A count over an unknown
denominator is not a count, and "we do not know" is a finding rather than a
quiet pass.

Usage
-----
    python experiments/two_site_judging/report_two_site.py --dry-run
    python experiments/two_site_judging/report_two_site.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.twositejudging import (                              # noqa: E402
    ADJUDICATED_CELLS_FILE,
    CANDIDATE_ARM_ID,
    CLASSES,
    CLASS_RUBRIC,
    DECLARED_PROCEDURE,
    FROZEN_COVERAGE,
    FROZEN_INPUT,
    PROTOCOL_NAME,
    READING_RULES,
    REPORT_FILE,
    STAGE_NAME,
    TwoSiteJudgingError,
    assert_declared_procedure,
    build_readings,
    cell_key,
    index_cells,
    judging_root,
)
from routegate.triplets import (                                    # noqa: E402
    assert_no_text_fields,
    read_jsonl,
)
from routegate.twositeunresolved import (                           # noqa: E402
    derived_counts,
    load_technical_unresolved,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
)


def report(adjudicated_dir: Path, out_dir: Path, procedure: str,
           write: bool = True) -> Dict[str, Any]:
    assert_declared_procedure(procedure)
    checks = Checklist(f"{STAGE_NAME}_report")

    path = adjudicated_dir / ADJUDICATED_CELLS_FILE
    if not path.is_file():
        raise TwoSiteJudgingError(
            f"{path} is not a file. --adjudicated-dir names the directory "
            f"import_two_site_human_review.py wrote.")
    records = read_jsonl(path)
    cells = index_cells(records, str(path))
    checks.require("adjudicated_cells",
                   len(cells) == FROZEN_COVERAGE["cells"],
                   f"{len(cells)} cell(s), expected {FROZEN_COVERAGE['cells']}")

    wrong_arm = sorted(k for k in cells if k[1] != CANDIDATE_ARM_ID)
    checks.require("every_cell_is_the_candidate_arm", not wrong_arm,
                   f"{len(wrong_arm)} cell(s) on another arm, first "
                   f"{wrong_arm[:3]}" if wrong_arm else
                   f"all on {CANDIDATE_ARM_ID}")

    procedures = sorted({str(r.get("adjudication_procedure"))
                         for r in records})
    checks.require(
        "every_cell_was_adjudicated_under_the_declared_procedure",
        procedures == [procedure],
        f"{procedures}, expected [{procedure!r}]")

    by_rubric = Counter(k[2] for k in cells)
    checks.require(
        "cells_by_rubric",
        dict(sorted(by_rubric.items()))
        == dict(sorted(FROZEN_COVERAGE["cells_by_rubric"].items())),
        f"{dict(sorted(by_rubric.items()))}, expected "
        f"{FROZEN_COVERAGE['cells_by_rubric']}")

    carried = load_technical_unresolved()
    carried_counts = derived_counts(carried)
    technical_ids = {str(r.get("judged_row_id")) for r in records
                     if r.get("technically_unresolved")}
    technical_detail = sorted(
        ({"row_id": str(r.get("judged_row_id")),
          "row_class": str(r.get("row_class")),
          "rubric": str(r.get("rubric")),
          "missing_judge": r.get("missing_judge"),
          "failure_type": r.get("failure_type"),
          "verdict": r.get("verdict"),
          "label_source": r.get("label_source")}
         for r in records if r.get("technically_unresolved")),
        key=lambda d: d["row_id"])
    checks.require(
        "technically_unresolved_cells_match_the_manifest",
        len(technical_ids) == carried_counts["technically_unresolved_cells"],
        f"{len(technical_ids)} cell(s) marked technically unresolved, "
        f"manifest carries "
        f"{carried_counts['technically_unresolved_cells']}")
    undecided = sorted(d["row_id"] for d in technical_detail
                       if d["verdict"] is not None)
    checks.require(
        "no_technically_unresolved_cell_carries_a_verdict", not undecided,
        f"{len(undecided)} carried cell(s) hold a verdict, first "
        f"{undecided[:3]}. A missing judge verdict is never filled in."
        if undecided else
        f"{len(technical_detail)} carried cell(s), none holds a verdict")

    readings = build_readings(records)
    for name in CLASSES:
        reading = readings["readings"][name]
        checks.record(
            f"{name}_reading", bool(reading["complete"]),
            f"{reading['rows']} row(s); "
            + ("complete" if reading["complete"]
               else f"WITHHELD -- {reading['withheld_reason']}"))

    by_source = Counter(str(r.get("label_source")) for r in records)

    summary: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "protocol": PROTOCOL_NAME,
        "created_utc": utc_now(),
        "arm_id": CANDIDATE_ARM_ID,
        "alpha_R": FROZEN_INPUT["alpha_R"],
        "alpha_S": FROZEN_INPUT["alpha_S"],
        "package_states_edited": FROZEN_INPUT["states"],
        "historical_blocks": FROZEN_INPUT["historical_layers"],
        "adjudication_procedure": procedure,
        "rubric_by_class": {name: CLASS_RUBRIC[name] for name in CLASSES},
        "reading_rules": {name: READING_RULES[name] for name in CLASSES},
        "cells": len(cells),
        "cells_by_rubric": dict(sorted(by_rubric.items())),
        "by_label_source": dict(sorted(by_source.items())),
        "readings": readings["readings"],
        "rows_by_class": readings["rows_by_class"],
        "expected_rows_by_class": readings["expected_rows_by_class"],
        "complete": readings["complete"],
        "withheld": readings["withheld"],
        # The evidence denominators, stated up front so no rate below is read
        # against a number nobody named.
        "judge_verdicts": {
            "planned": carried_counts["planned_verdicts"],
            "valid": carried_counts["valid_verdicts"],
            "missing_technical": carried_counts["missing_verdicts"],
        },
        "dual_judged_cells": carried_counts["dual_judged_cells"],
        "technically_unresolved_cells":
            carried_counts["technically_unresolved_cells"],
        "technically_unresolved_row_ids": sorted(technical_ids),
        "technically_unresolved_detail": technical_detail,
        "decided_by_class": readings["decided_by_class"],
        "unresolved_by_class": readings["unresolved_by_class"],
        "unresolved_screen_row_ids": readings["unresolved_screen_row_ids"],
        "decided_cells": readings["decided_cells"],
        "unresolved_cells": readings["unresolved_cells"],
        "original_protocol_complete": (
            carried_counts["original_protocol_complete"]
            and readings["complete"]),
        "denominator_note": (
            "every rate is reported twice: once over the original protocol's "
            "denominator, which is withheld while any cell of that class is "
            "unresolved, and once over the cells actually decided, with that "
            "denominator named in decided_denominator_is. No rate divides by "
            "the class total when fewer cells were decided."),
        "withheld_note": (
            "none" if not readings["withheld"] else
            "a class with any UNRESOLVED cell has its counts withheld. A count "
            "over an unknown denominator is not a count."),
        "deterministic": True,
        "deterministic_note": "cells are sorted by identity before anything is "
                              "counted and every count is over a named label "
                              "list, so two runs over the same adjudicated "
                              "file produce byte-identical readings",
        "selects_nothing": True,
        "selects_nothing_note": "no operating point, strength or site is "
                                "chosen from these verdicts. The arm was fixed "
                                "in advance by "
                                "manifests/two_site_actuator_candidate_v1.json.",
        "input": {
            "adjudicated_cells": str(path),
            "adjudicated_cells_sha256": sha256_file(path),
            "records_sha256": FROZEN_INPUT["records_sha256"],
            "baseline_sha256": FROZEN_INPUT["baseline_sha256"],
        },
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the two-site report")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / REPORT_FILE).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[report] wrote {out_dir / REPORT_FILE}")

    # The evidence denominators, printed before any count, so a reader of the
    # terminal output alone is never left to assume 480 or 960.
    verdicts = summary["judge_verdicts"]
    print(f"[report] judge verdicts   {verdicts['valid']}/{verdicts['planned']}"
          f" valid; {verdicts['missing_technical']} missing (technical)")
    print(f"[report] cells            {summary['cells']} total; "
          f"{summary['dual_judged_cells']} dual-judged; "
          f"{summary['technically_unresolved_cells']} technically unresolved")
    for item in summary["technically_unresolved_detail"]:
        print(f"[report]   UNRESOLVED_TECHNICAL {item['row_id']}  "
              f"class {item['row_class']}  rubric {item['rubric']}  "
              f"missing {item['missing_judge']}  {item['failure_type']}")
    print(f"[report] original protocol complete: "
          f"{summary['original_protocol_complete']}")

    for name in CLASSES:
        reading = readings["readings"][name]
        print(f"[report] {name} ({reading['rubric']}) "
              f"{reading['rows']}/{reading['expected_rows']} row(s); "
              f"decided {reading['decided_denominator']}, "
              f"unresolved {reading['unresolved']}")
        if not reading["complete"]:
            print(f"[report]   original-protocol reading WITHHELD -- "
                  f"{reading['withheld_reason']}")
            print(f"[report]   complete-case reading, denominator "
                  f"{reading['decided_denominator']} "
                  f"({reading['decided_denominator_is']})")
            if name == "E":
                print(f"[report]     substantive repair (SA or PA-S) "
                      f"{reading['decided_substantive_repair']} "
                      f"({reading['decided_substantive_repair_rate']:.4f})")
                print(f"[report]     full repair (SA)                "
                      f"{reading['decided_full_repair']} "
                      f"({reading['decided_full_repair_rate']:.4f})")
            elif name == "B":
                print(f"[report]     material degradation            "
                      f"{reading['decided_material_degradation']} "
                      f"({reading['decided_material_degradation_rate']:.4f})")
                nm_rate = reading["decided_non_material_degradation_rate"]
                print(f"[report]     non-material degradation        "
                      f"{reading['decided_non_material_degradation']} "
                      f"({nm_rate:.4f})")
            else:
                print(f"[report]     YES {reading['decided_YES']}  "
                      f"NO {reading['decided_NO']}  "
                      f"UNCLEAR {reading['decided_UNCLEAR']}")
                print(f"[report]     YES if every unresolved cell were YES "
                      f"{reading['sensitivity_YES_if_all_unresolved_were_YES']}"
                      f"; if NO "
                      f"{reading['sensitivity_YES_if_all_unresolved_were_NO']}"
                      f" (a bracket, not a finding)")
            continue
        if name == "E":
            print(f"[report]   substantive repair (SA or PA-S) "
                  f"{reading['substantive_repair']} "
                  f"({reading['substantive_repair_rate']:.4f})")
            print(f"[report]   full repair (SA)                "
                  f"{reading['full_repair']} "
                  f"({reading['full_repair_rate']:.4f})")
        elif name == "B":
            print(f"[report]   material degradation            "
                  f"{reading['material_degradation']} "
                  f"({reading['material_degradation_rate']:.4f})")
            print(f"[report]   non-material degradation        "
                  f"{reading['non_material_degradation']} "
                  f"({reading['non_material_degradation_rate']:.4f})")
        else:
            print(f"[report]   YES {reading['YES']}  NO {reading['NO']}  "
                  f"UNCLEAR {reading['UNCLEAR']}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedure", default=DECLARED_PROCEDURE,
                        help="the declared adjudication procedure")
    parser.add_argument("--adjudicated-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = judging_root() / "judging"
        adjudicated = args.adjudicated_dir or (root / "adjudicated")
        out_dir = args.output_dir or (root / "report")
        report(adjudicated, out_dir, args.procedure, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
