#!/usr/bin/env python3
"""Join two judges' verdicts back onto phase-2 cells. Deterministic, offline.

Reads only run records and the private row map. Calls nothing, generates
nothing, and decides nothing: adjudication is ``import_phase2_human_review.py``
and the readings are ``report_phase2.py``.

What joining does
-----------------
Each judge produced one verdict per opaque row. This script maps every row back
through the private map to the cell it stands for -- ``(screen_row_id, arm_id,
safety)`` -- and records, per cell, the Sonnet verdict, the Opus verdict and
whether they agree. The packet is not deduplicated, so the map is one row to
one cell, and a row that fans out to two cells, or a cell claimed by two rows,
is a defect and is refused.

Counting the result
-------------------
A join that silently reads nothing looks, in the summary alone, exactly like a
run nobody has done yet. So the run counts itself against
``routegate.phase2judging.FROZEN_COVERAGE``: 144 cells, 18 per arm across the
eight generated arms, over exactly the 18 bound rows, from 144 successful
records per judge, with every cell terminal in ``agree`` or ``disagree``. Any
other shape is a failure.

No cell here carries one judge's verdict
----------------------------------------
Phase 1 carries a pinned, committed technical-exclusion manifest for two cells
Opus never answered. **Phase 2 has no such manifest and this script reads
none.** A phase-2 cell with fewer than two verdicts is a failure, not an
exclusion: an exclusion is a deliberate, committed, evidence-backed act, and
inventing one at join time is exactly what a fail-closed join must refuse.

Disagreement is **not** resolved here, and is **not** exposed to the human
reviewer. It is written to a private file that the review export never reads,
so a reviewer cannot be anchored by knowing the judges split.

Usage
-----
    python experiments/focused_direction_judging/join_phase2_verdicts.py --dry-run
    python experiments/focused_direction_judging/join_phase2_verdicts.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.focusedjudging import (                              # noqa: E402
    CARRIED_CELL_FIELDS,
    JOINED_CELLS_FILE,
    JOINED_ROWS_FILE,
    PROCEDURE_CORPUS,
    ROW_MAP_FILE,
)
from routegate.phase2judging import (                               # noqa: E402
    AXES_USED,
    FROZEN_COVERAGE,
    JUDGING_SUBDIR,
    PHASE2_AXIS,
    PHASE2_REFERENCE_ARM_IDS,
    PHASE2_ROW_PREFIX,
    PROTOCOL_NAME,
    Phase2JudgingError,
    check_phase2_coverage,
    phase2_cell_key,
    phase2_root,
)
from routegate.triplets import (                                    # noqa: E402
    HISTORICAL_TEXT_FIELDS,
    JUDGES,
    RESULT_FILES,
    assert_no_text_fields,
    assert_opaque,
    detect_result_format,
    read_jsonl,
    result_is_success,
    result_join_key,
    result_record_hash,
    result_verdict,
    write_jsonl,
)
from routegate.validation import Checklist, ValidationError         # noqa: E402


def load_row_map(packet: Path) -> Dict[str, Dict[str, Any]]:
    """``opaque row_id -> private map entry``, validated on the way in."""
    path = packet / "private" / ROW_MAP_FILE
    if not path.is_file():
        raise Phase2JudgingError(
            f"{path} is not a file. --packet names the directory the phase-2 "
            f"packet was built into.")
    out: Dict[str, Dict[str, Any]] = {}
    for entry in read_jsonl(path):
        row_id = entry.get("row_id")
        axis = entry.get("axis")
        if axis not in AXES_USED:
            raise Phase2JudgingError(
                f"{path}: row {row_id!r} carries axis {axis!r}, and phase 2 "
                f"uses {list(AXES_USED)}")
        # The axis a row belongs to is checked against the opaque prefix the
        # packet minted for it, not against the file it was read from.
        assert_opaque(str(row_id), PHASE2_ROW_PREFIX[axis])
        if row_id in out:
            raise Phase2JudgingError(
                f"{path}: {row_id!r} appears twice in the private map")
        phase2_cell_key(entry, str(path))
        if entry.get("arm_id") in PHASE2_REFERENCE_ARM_IDS:
            raise Phase2JudgingError(
                f"{path}: row {row_id!r} stands for the inherited arm "
                f"{entry['arm_id']!r}. Those cells are read from phase 1 and "
                f"are never re-judged.")
        out[str(row_id)] = dict(entry)
    return out


def load_run(directory: Path) -> Dict[str, Dict[str, Any]]:
    """``row_id -> record`` for one judge's run. Successful rows only.

    The record format is **named** per record rather than inferred from a field
    that happens to be missing, so a schema drift is refused instead of read as
    an empty run.
    """
    path = directory / "results.jsonl"
    if not path.is_file():
        raise Phase2JudgingError(
            f"{path} is absent. A judging run directory with no results file "
            f"is refused rather than read as an empty run, which is what a "
            f"silent layout failure looks like.")
    out: Dict[str, Dict[str, Any]] = {}
    for record in read_jsonl(path):
        fmt = detect_result_format(record)
        if not result_is_success(record):
            continue
        row_id = result_join_key(record)
        if row_id in out:
            raise Phase2JudgingError(
                f"{directory}: two successful records for {row_id!r}. A "
                f"duplicate successful call means the resume key is not "
                f"unique; refusing to continue.")
        record["_result_format"] = fmt
        out[row_id] = record
    if not out:
        raise Phase2JudgingError(
            f"{path} holds no successful record. An empty run is refused "
            f"rather than joined to nothing.")
    return out


def failed_rows(directory: Path) -> Dict[str, int]:
    """``row_id -> how many times it failed`` in one run directory."""
    out: Dict[str, int] = {}
    for names in RESULT_FILES.values():
        path = directory / names["failures"]
        if path.is_file():
            for record in read_jsonl(path):
                row_id = str(record.get("row_id", ""))
                out[row_id] = out.get(row_id, 0) + 1
    return out


def unresolved_failed_rows(failures: Mapping[str, int],
                           successful: Mapping[str, Any]) -> List[str]:
    """The rows that failed and STILL have no successful record."""
    return sorted(row_id for row_id in failures if row_id not in successful)


def join(packet: Path, judging: Path, out_dir: Path, write: bool = True,
         expect: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    checks = Checklist("focused_phase2_join")
    row_map = load_row_map(packet)

    runs: Dict[str, Dict[str, Dict[str, Any]]] = {j: {} for j in JUDGES}
    failures: Dict[str, Dict[str, int]] = {j: {} for j in JUDGES}
    directories: Dict[str, List[str]] = {j: [] for j in JUDGES}
    if not judging.is_dir():
        raise Phase2JudgingError(f"{judging} is not a directory")
    for directory in sorted(p for p in judging.iterdir() if p.is_dir()):
        name = directory.name
        for judge in JUDGES:
            if not name.endswith(f"__{judge}"):
                continue
            directories[judge].append(name)
            for row_id, record in load_run(directory).items():
                if row_id in runs[judge]:
                    raise Phase2JudgingError(
                        f"{judge} has two records for {row_id}, in "
                        f"{runs[judge][row_id]['_dir']} and {name}")
                record["_dir"] = name
                runs[judge][row_id] = record
            for row_id, attempts in failed_rows(directory).items():
                failures[judge][row_id] = (failures[judge].get(row_id, 0)
                                           + attempts)

    for judge in JUDGES:
        unresolved = unresolved_failed_rows(failures[judge], runs[judge])
        resolved = sorted(set(failures[judge]) - set(unresolved))
        checks.record(f"{judge}_rows", True,
                      f"{len(runs[judge])} successful row(s) from "
                      f"{directories[judge]}")
        checks.record(
            f"{judge}_failed_rows_since_answered", True,
            f"{len(resolved)} row(s) failed and were later answered; their "
            f"failure history is kept and is not an open failure")
        checks.require(
            f"{judge}_no_unresolved_failure", not unresolved,
            f"{len(unresolved)} failed row(s) with no verdict"
            + (f": {unresolved[:3]}. A failure is not a verdict; re-run the "
               f"judge to retry those rows, and note that a failed call is "
               f"never retried automatically." if unresolved else
               f"; every row that failed has since been answered"))

    cells: Dict[Any, Dict[str, Any]] = {}
    private: List[Dict[str, Any]] = []
    for row_id, entry in sorted(row_map.items()):
        axis = entry["axis"]
        verdicts: Dict[str, Optional[str]] = {}
        record_hashes: Dict[str, Optional[str]] = {}
        formats_used: Dict[str, str] = {}
        for judge in JUDGES:
            record = runs[judge].get(row_id)
            verdicts[judge] = (result_verdict(record, axis)
                               if record is not None else None)
            record_hashes[judge] = (result_record_hash(record)
                                    if record is not None else None)
            if record is not None:
                formats_used[judge] = record["_result_format"]

        present = [j for j in JUDGES if verdicts[j] is not None]
        if not present:
            state, agree = "missing", None
        elif len(present) < len(JUDGES):
            state, agree = "incomplete", None
        else:
            agree = verdicts[JUDGES[0]] == verdicts[JUDGES[1]]
            state = "agree" if agree else "disagree"

        key = phase2_cell_key(entry, "the private row map")
        if key in cells:
            raise Phase2JudgingError(
                f"cell {key} is claimed by row {cells[key]['row_id']} and by "
                f"row {row_id}. This packet is not deduplicated, so one cell "
                f"is one row; two rows over one cell means the map is wrong, "
                f"and agreeing verdicts would only hide that.")
        cell = {
            "screen_row_id": entry["screen_row_id"],
            "arm_id": entry["arm_id"],
            "axis": axis,
            "row_class": entry["row_class"],
            "row_id": row_id,
            "state": state,
            "verdicts": verdicts,
            "record_sha256": record_hashes,
            "prompt_sha256": entry["prompt_sha256"],
            "response_sha256": entry["response_sha256"],
            "result_formats": dict(formats_used),
        }
        for field_name in CARRIED_CELL_FIELDS:
            cell[field_name] = entry.get(field_name)
        cells[key] = cell
        private.append({"row_id": row_id, "axis": axis, "state": state,
                        "verdicts": verdicts, "agree": agree,
                        "result_formats": dict(formats_used),
                        "screen_row_id": entry["screen_row_id"],
                        "arm_id": entry["arm_id"]})

    by_state: Dict[str, int] = defaultdict(int)
    for cell in cells.values():
        by_state[cell["state"]] += 1

    coverage = check_phase2_coverage(cells, runs, checks, expect)

    summary = {
        "protocol": PROTOCOL_NAME,
        "phase": 2,
        "created_utc": utc_now(),
        "cells": len(cells),
        "by_state": dict(sorted(by_state.items())),
        "rows": len(row_map),
        "records_by_judge": {j: len(runs[j]) for j in JUDGES},
        "failed_rows_by_judge": {j: sorted(failures[j]) for j in JUDGES},
        "failure_attempts_by_judge": {j: dict(sorted(failures[j].items()))
                                      for j in JUDGES},
        "unresolved_failed_rows_by_judge": {
            j: unresolved_failed_rows(failures[j], runs[j]) for j in JUDGES},
        "technical_exclusion": {
            "manifest": None,
            "excluded_cells": 0,
            "note": "phase 2 has no technical-exclusion manifest and this join "
                    "reads none. A cell with fewer than two judge verdicts is "
                    "a failure here, not an exclusion: an exclusion is a "
                    "committed, evidence-backed act and is never invented at "
                    "join time.",
        },
        "adjudication_procedure_unchanged": PROCEDURE_CORPUS,
        "inherited_reference_arms": list(PHASE2_REFERENCE_ARM_IDS),
        "inherited_reference_arms_note":
            "not in this join. They were judged in phase 1 and are read from "
            "the completed phase-1 adjudication by report_phase2.py.",
        "judges": list(JUDGES),
        "axis": PHASE2_AXIS,
        "coverage_expected": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in (expect or FROZEN_COVERAGE).items()},
        "coverage_observed": coverage,
        "disagreement_is_private": True,
        "disagreement_note": "per-cell judge verdicts and agreement live in "
                             f"private/{JOINED_CELLS_FILE} and are not read by "
                             "export_phase2_human_review.py. A reviewer who "
                             "knew the judges had split would not be blind.",
        "resolves_nothing": True,
        "resolves_nothing_note": "this stage produces no adjudicated verdict, "
                                 "no outcome count and no selected strength.",
        "result_formats_seen": sorted({fmt for cell in cells.values()
                                       for fmt in cell["result_formats"].values()}),
        "text_never_copied": sorted(HISTORICAL_TEXT_FIELDS),
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the phase-2 join summary")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        write_jsonl(out_dir / "private" / JOINED_CELLS_FILE,
                    [cells[k] for k in sorted(cells)])
        write_jsonl(out_dir / "private" / JOINED_ROWS_FILE, private)
        (out_dir / "join_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[join] wrote {out_dir}")
    for judge in JUDGES:
        print(f"[join] {judge + ' records':<22} {len(runs[judge])}")
    print(f"[join] {'cells':<22} {len(cells)}")
    for state, count in sorted(by_state.items()):
        print(f"[join] {state:<22} {count}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--judging", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = phase2_root() / JUDGING_SUBDIR
        packet = args.packet or (root / "packet")
        judging = args.judging or (root / "runs")
        out_dir = args.output_dir or (root / "joined")
        join(packet, judging, out_dir, write=not args.dry_run, expect=None)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
