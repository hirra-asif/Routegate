#!/usr/bin/env python3
"""Join two judges' verdicts back onto phase-1 cells. Deterministic, offline.

Reads only run records and the private row map. Calls nothing, generates
nothing, and decides nothing: adjudication is ``import_human_review.py`` and
the readings are ``report_phase1.py``.

What joining does
-----------------
Each judge produced one verdict per opaque row. This script maps every row back
through the private map to the cell it stands for -- ``(screen_row_id, arm_id,
axis)`` -- and records, per cell, the Sonnet verdict, the Opus verdict and
whether they agree. The packet is not deduplicated, so the map is one row to
one cell and a row that fans out to two cells, or a cell claimed by two rows,
is a defect and is refused.

Counting the result
-------------------
A join that silently reads nothing looks, in the summary alone, exactly like a
run nobody has done yet. So the run counts itself against
``routegate.focusedjudging.FROZEN_COVERAGE``: 871 cells, 445 on the Axis-2
rubric and 426 on the safety rubric, over 89 E rows and 213 H rows, from 871
successful records per judge, with every cell terminal in ``agree`` or
``disagree``. Any other shape is a failure.

Disagreement is **not** resolved here, and is **not** exposed to the human
reviewer. It is written to a private file that the review export never reads,
so a reviewer cannot be anchored by knowing the judges split. The adjudication
importer reads it afterwards.

Incomplete runs
---------------
A cell with a verdict from only one judge is ``incomplete`` and carries no
agreement value. A cell with no verdict is ``missing``. Neither is silently
treated as agreement, and neither becomes an input to any reading.

Usage
-----
    python experiments/focused_direction_judging/join_verdicts.py --dry-run
    python experiments/focused_direction_judging/join_verdicts.py
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
    AXES_USED,
    AXIS_CLASS,
    CARRIED_CELL_FIELDS,
    FROZEN_COVERAGE,
    JOINED_CELLS_FILE,
    JOINED_ROWS_FILE,
    PROCEDURE_CORPUS,
    PROTOCOL_NAME,
    ROW_MAP_FILE,
    ROW_PREFIX,
    TECHNICAL_EXCLUSIONS_MANIFEST,
    FocusedJudgingError,
    cell_key,
    coverage_after_technical_exclusions,
    excluded_row_ids,
    judging_root,
    load_technical_exclusions,
    verify_technical_exclusions,
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
from routegate.validation import Checklist, ValidationError        # noqa: E402


def load_row_map(packet: Path) -> Dict[str, Dict[str, Any]]:
    """``opaque row_id -> private map entry``, validated on the way in."""
    path = packet / "private" / ROW_MAP_FILE
    if not path.is_file():
        raise FocusedJudgingError(
            f"{path} is not a file. --packet names the directory the phase-1 "
            f"packet was built into.")
    out: Dict[str, Dict[str, Any]] = {}
    for entry in read_jsonl(path):
        row_id = entry.get("row_id")
        axis = entry.get("axis")
        if axis not in AXES_USED:
            raise FocusedJudgingError(
                f"{path}: row {row_id!r} carries axis {axis!r}, and this phase "
                f"uses {list(AXES_USED)}")
        # The axis a row belongs to is checked against the opaque prefix the
        # packet minted for that axis, not against the file it was read from.
        assert_opaque(str(row_id), ROW_PREFIX[axis])
        if row_id in out:
            raise FocusedJudgingError(
                f"{path}: {row_id!r} appears twice in the private map")
        cell_key(entry, str(path))
        out[str(row_id)] = dict(entry)
    return out


def load_run(directory: Path) -> Dict[str, Dict[str, Any]]:
    """``row_id -> record`` for one judge/axis run. Successful rows only.

    The record format is **named** per record rather than inferred from a field
    that happens to be missing, so a schema drift is refused instead of read as
    an empty run. Failed rows are read too, and are counted -- a failure is not
    a verdict, but a run with failures is not a finished run either.
    """
    path = directory / "results.jsonl"
    if not path.is_file():
        raise FocusedJudgingError(
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
            raise FocusedJudgingError(
                f"{directory}: two successful records for {row_id!r}. A "
                f"duplicate successful call means the resume key is not "
                f"unique; refusing to continue.")
        record["_result_format"] = fmt
        out[row_id] = record
    if not out:
        raise FocusedJudgingError(
            f"{path} holds no successful record. An empty run is refused "
            f"rather than joined to nothing.")
    return out


def failed_rows(directory: Path) -> Dict[str, int]:
    """``row_id -> how many times it failed`` in one run directory.

    Rows, not entries. A retried row has one failure record per attempt, and
    ``recover_failed_rows.py`` keeps that history forever: raw evidence is
    append-only and ``failures.jsonl`` is never rewritten. Counting entries
    would therefore make a run whose failures were all resolved look
    permanently broken -- and would go on saying so after every one of its rows
    had a verdict.
    """
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
    """The rows that failed and STILL have no successful record.

    This is the gap the join must refuse on. A row that failed and was later
    answered -- by a re-run, or by an offline recovery from its own
    provenance -- is not a gap; its failure history is evidence of what it took
    to answer it, and is kept for exactly that reason.
    """
    return sorted(row_id for row_id in failures if row_id not in successful)


def check_coverage(cells: Mapping[Any, Mapping[str, Any]],
                   runs: Mapping[str, Mapping[str, Any]],
                   expect: Mapping[str, Any], checks: Checklist
                   ) -> Dict[str, Any]:
    """The arithmetic a complete phase-1 join must satisfy."""
    def gate(name: str, passed: bool, detail: str) -> None:
        checks.record(name, passed, detail)
        if not passed:
            raise FocusedJudgingError(f"{name}: {detail}")

    for judge in sorted(runs):
        gate(f"{judge}_record_count",
             len(runs[judge]) == expect["records_per_judge"],
             f"{judge} has {len(runs[judge])} successful record(s), and this "
             f"phase has {expect['records_per_judge']}")

    gate("cells_total", len(cells) == expect["cells"],
         f"{len(cells)} unique cell(s), expected {expect['cells']}")

    by_axis: Dict[str, int] = defaultdict(int)
    by_arm: Dict[str, int] = defaultdict(int)
    by_class: Dict[str, int] = defaultdict(int)
    rows_by_class: Dict[str, set] = defaultdict(set)
    for (screen_row_id, arm_id, axis) in cells:
        by_axis[axis] += 1
        by_arm[arm_id] += 1
        by_class[AXIS_CLASS[axis]] += 1
        rows_by_class[AXIS_CLASS[axis]].add(screen_row_id)

    for axis, want in sorted(expect["cells_by_axis"].items()):
        gate(f"cells_on_{axis}", by_axis.get(axis, 0) == want,
             f"{by_axis.get(axis, 0)} cell(s), expected {want}")
    gate("cells_by_arm", dict(sorted(by_arm.items())) == dict(
        sorted(expect["cells_by_arm"].items())),
         f"{dict(sorted(by_arm.items()))}, expected {expect['cells_by_arm']}")
    gate("cells_by_class", dict(sorted(by_class.items())) == dict(
        sorted(expect["cells_by_class"].items())),
         f"{dict(sorted(by_class.items()))}, expected "
         f"{expect['cells_by_class']}")
    counted = {name: len(ids) for name, ids in sorted(rows_by_class.items())}
    gate("rows_by_class", counted == dict(sorted(expect["rows_by_class"].items())),
         f"{counted}, expected {expect['rows_by_class']}")

    stray = sorted({cell["state"] for cell in cells.values()}
                   - set(expect["terminal_states"]))
    gate("every_cell_is_agree_or_disagree", not stray,
         f"the join holds the non-terminal state(s) {stray}; a cell judged by "
         f"one judge or by neither is not a result")

    return {"cells": len(cells), "cells_by_axis": dict(sorted(by_axis.items())),
            "cells_by_arm": dict(sorted(by_arm.items())),
            "cells_by_class": dict(sorted(by_class.items())),
            "rows_by_class": counted,
            "records_by_judge": {j: len(runs[j]) for j in sorted(runs)}}


def join(packet: Path, judging: Path, out_dir: Path, write: bool = True,
         expect: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    checks = Checklist("focused_phase1_join")
    row_map = load_row_map(packet)

    runs: Dict[str, Dict[str, Dict[str, Any]]] = {j: {} for j in JUDGES}
    failures: Dict[str, Dict[str, int]] = {j: {} for j in JUDGES}
    directories: Dict[str, List[str]] = {j: [] for j in JUDGES}
    run_dirs: Dict[str, Path] = {}
    if not judging.is_dir():
        raise FocusedJudgingError(f"{judging} is not a directory")
    for directory in sorted(p for p in judging.iterdir() if p.is_dir()):
        name = directory.name
        for judge in JUDGES:
            if not name.endswith(f"__{judge}"):
                continue
            directories[judge].append(name)
            run_dirs[name] = directory
            for row_id, record in load_run(directory).items():
                if row_id in runs[judge]:
                    raise FocusedJudgingError(
                        f"{judge} has two records for {row_id}, in "
                        f"{runs[judge][row_id]['_dir']} and {name}")
                record["_dir"] = name
                runs[judge][row_id] = record
            for row_id, attempts in failed_rows(directory).items():
                failures[judge][row_id] = failures[judge].get(row_id, 0) + attempts
    # ---- the paired technical exclusion, verified before anything is joined --
    #
    # The excluded cells are removed from BOTH judges and from the row map, so
    # no cell is ever created for them. That is what keeps the comparison
    # symmetric: every cell that reaches the join needs two judge verdicts, and
    # an excluded cell is not a cell with one.
    full_row_map = dict(row_map)
    exclusions = load_technical_exclusions()
    excluded = set(excluded_row_ids(exclusions))
    exclusion_report = verify_technical_exclusions(
        exclusions, row_map, runs, run_dirs)
    if excluded:
        missing_from_map = sorted(excluded - set(row_map))
        if missing_from_map:
            raise FocusedJudgingError(
                f"excluded cell(s) {missing_from_map} are not in this packet")
        dropped = {judge: sorted(set(runs[judge]) & excluded) for judge in JUDGES}
        for judge in JUDGES:
            for row_id in dropped[judge]:
                del runs[judge][row_id]
            for row_id in list(failures[judge]):
                if row_id in excluded:
                    del failures[judge][row_id]
        for row_id in excluded:
            row_map.pop(row_id, None)
        checks.record(
            "technical_exclusion_applied", True,
            f"{len(excluded)} cell(s) over "
            f"{exclusion_report['entries']} paired exclusion(s); removed from "
            f"{ {j: len(v) for j, v in dropped.items()} }")

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

        key = cell_key(entry, "the private row map")
        if key in cells:
            raise FocusedJudgingError(
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

    # The coverage a complete join must satisfy: the frozen protocol less
    # exactly the pinned exclusions. Derived and cross-checked against the
    # manifest, never widened to fit what the run happened to produce.
    if expect is None:
        expect = coverage_after_technical_exclusions(exclusions, full_row_map)
    coverage: Optional[Dict[str, Any]] = None
    if expect is not None:
        coverage = check_coverage(cells, runs, expect, checks)

    summary = {
        "protocol": PROTOCOL_NAME,
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
            "manifest": TECHNICAL_EXCLUSIONS_MANIFEST.name,
            "is_outcome_based_exclusion": False,
            "is_post_run_technical_deviation": True,
            "adjudication_procedure_unchanged": PROCEDURE_CORPUS,
            "excluded_row_ids": sorted(excluded),
            "excluded_cells": exclusion_report["cells"],
            "excluded_rows": len({c["screen_row_id"]
                                  for c in exclusion_report.get("checked", [])}),
            "gates": exclusion_report["gates"],
            "checked": exclusion_report.get("checked", []),
            "note": "both judges' records for these cells are removed together "
                    "so the paired comparison stays symmetric; no cell in this "
                    "join carries fewer than two judge verdicts",
        },
        "judges": list(JUDGES),
        "coverage_expected": ({k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in expect.items()}
                              if expect is not None else None),
        "coverage_observed": coverage,
        "disagreement_is_private": True,
        "disagreement_note": "per-cell judge verdicts and agreement live in "
                             f"private/{JOINED_CELLS_FILE} and are not read by "
                             "export_human_review.py. A reviewer who knew the "
                             "judges had split would not be blind.",
        "resolves_nothing": True,
        "resolves_nothing_note": "this stage produces no adjudicated verdict, "
                                 "no outcome count and no selected strength.",
        "result_formats_seen": sorted({fmt for cell in cells.values()
                                       for fmt in cell["result_formats"].values()}),
        "text_never_copied": sorted(HISTORICAL_TEXT_FIELDS),
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the join summary")

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
        root = judging_root() / "judging"
        packet = args.packet or (root / "packet")
        judging = args.judging or (root / "runs")
        out_dir = args.output_dir or (root / "joined")
        join(packet, judging, out_dir, write=not args.dry_run, expect=None)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
