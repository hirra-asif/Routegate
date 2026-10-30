#!/usr/bin/env python3
"""Join two judges' verdicts back onto the 480 two-site cells. Offline.

Reads only run records and the private row map. Calls nothing, generates
nothing, and decides nothing: adjudication is
``import_two_site_human_review.py`` and the readings are
``report_two_site.py``.

What joining does
-----------------
Each judge produced one verdict per opaque row. This script maps every row back
through the private map to the cell it stands for -- ``(screen_row_id, arm_id,
rubric)`` -- and records, per cell, the Sonnet verdict, the Opus verdict and
whether they agree. The packet is not deduplicated, so the map is one row to
one cell, and a row that fans out to two cells, or a cell claimed by two rows,
is a defect and is refused.

Decoding the quality rows
-------------------------
A quality judge answers about presentation positions A and B. The private map
says which position held the candidate, so this is where -- and only where --
an ``A_BETTER`` becomes a ``BEFORE_BETTER`` or an ``AFTER_BETTER``. Decoding is
a per-row bijection applied identically to both judges, so it preserves
agreement exactly: two judges agree after decoding if and only if they agreed
before it.

The outcome and the strength are joined as **one composite verdict**. Two
judges agree about a B cell only when they name the same winner AND the same
strength; naming the same winner at different strengths is not agreement, and
that cell goes to a human under the declared procedure.

Counting the result
-------------------
A join that silently reads nothing looks, in the summary alone, exactly like a
run nobody has done yet. So the run counts itself against
``routegate.twositejudging.FROZEN_COVERAGE``: 480 cells, 89 on Axis-2, 178 on
quality and 213 on safety, over 89 E, 178 B and 213 H rows, from 480 successful
records per judge, with every cell terminal in ``agree`` or ``disagree``. Any
other shape is a failure.

Disagreement is **not** resolved here, and is **not** exposed to the human
reviewer. It is written to a private file that the review export never reads,
so a reviewer cannot be anchored by knowing the judges split.

Usage
-----
    python experiments/two_site_judging/join_two_site_verdicts.py --dry-run
    python experiments/two_site_judging/join_two_site_verdicts.py
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
from routegate.twositejudging import (                              # noqa: E402
    CARRIED_CELL_FIELDS,
    CELL_DIGEST_FIELDS,
    DECLARED_PROCEDURE,
    FROZEN_COVERAGE,
    JOINED_CELLS_FILE,
    JOINED_ROWS_FILE,
    PAIRWISE_RUBRICS,
    POSITION_A,
    POSITION_B,
    PROTOCOL_NAME,
    ROW_MAP_FILE,
    ROW_PREFIX,
    RUBRICS_USED,
    RUBRIC_CLASS,
    STAGE_NAME,
    TwoSiteJudgingError,
    assert_orientation,
    assert_output_dir_is_new,
    cell_key,
    compose_quality_verdict,
    decode_preference,
    judging_root,
    quality_pairing_is_consistent,
)
from routegate.triplets import (                                    # noqa: E402
    HISTORICAL_TEXT_FIELDS,
    JUDGES,
    QUALITY,
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
from routegate.twositeunresolved import (                           # noqa: E402
    STATE_UNRESOLVED_TECHNICAL,
    TECHNICAL_UNRESOLVED_MANIFEST,
    derived_counts,
    load_technical_unresolved,
    unresolved_index,
    unresolved_row_ids,
    verify_technical_unresolved,
)
from routegate.validation import Checklist, ValidationError         # noqa: E402


def load_row_map(packet: Path) -> Dict[str, Dict[str, Any]]:
    """``opaque row_id -> private map entry``, validated on the way in."""
    path = packet / "private" / ROW_MAP_FILE
    if not path.is_file():
        raise TwoSiteJudgingError(
            f"{path} is not a file. --packet names the directory the two-site "
            f"packet was built into.")
    out: Dict[str, Dict[str, Any]] = {}
    for entry in read_jsonl(path):
        row_id = entry.get("row_id")
        rubric = entry.get("rubric")
        if rubric not in RUBRICS_USED:
            raise TwoSiteJudgingError(
                f"{path}: row {row_id!r} carries rubric {rubric!r}, and this "
                f"stage uses {list(RUBRICS_USED)}")
        # The rubric a row belongs to is checked against the opaque prefix the
        # packet minted for that rubric, not against the file it was read from.
        assert_opaque(str(row_id), ROW_PREFIX[rubric])
        if row_id in out:
            raise TwoSiteJudgingError(
                f"{path}: {row_id!r} appears twice in the private map")
        cell_key(entry, str(path))
        if rubric in PAIRWISE_RUBRICS:
            assert_orientation(
                {POSITION_A: entry["response_a_origin"],
                 POSITION_B: entry["response_b_origin"]},
                f"{path}: row {row_id!r}")
        for field_name in CELL_DIGEST_FIELDS[rubric]:
            if not entry.get(field_name):
                raise TwoSiteJudgingError(
                    f"{path}: row {row_id!r} carries no {field_name}. A cell "
                    f"whose text cannot be pinned is not joinable.")
        out[str(row_id)] = dict(entry)
    return out


def load_run(directory: Path) -> Dict[str, Dict[str, Any]]:
    """``row_id -> record`` for one judge/rubric run. Successful rows only.

    The record format is **named** per record rather than inferred from a field
    that happens to be missing, so a schema drift is refused instead of read as
    an empty run.
    """
    path = directory / "results.jsonl"
    if not path.is_file():
        raise TwoSiteJudgingError(
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
            raise TwoSiteJudgingError(
                f"{directory}: two successful records for {row_id!r}. A "
                f"duplicate successful call means the resume key is not "
                f"unique; refusing to continue.")
        record["_result_format"] = fmt
        out[row_id] = record
    if not out:
        raise TwoSiteJudgingError(
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


def failure_records(directory: Path) -> Dict[str, List[Dict[str, Any]]]:
    """``row_id -> [failure record, ...]`` for one run directory.

    The records are read for attempt count and preserved-body digests only.
    Their ``error`` strings are never read here: the first Sonnet pass wrote
    credential material into that field, and nothing in the scientific path
    should depend on parsing it.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for names in RESULT_FILES.values():
        path = directory / names["failures"]
        if path.is_file():
            for record in read_jsonl(path):
                out.setdefault(str(record.get("row_id", "")), []).append(
                    dict(record))
    return out


def unresolved_failed_rows(failures: Mapping[str, int],
                           successful: Mapping[str, Any]) -> List[str]:
    """The rows that failed and STILL have no successful record."""
    return sorted(row_id for row_id in failures if row_id not in successful)


def cell_verdict(record: Optional[Mapping[str, Any]], rubric: str,
                 entry: Mapping[str, Any], row_id: str
                 ) -> tuple[Optional[str], Optional[bool]]:
    """One judge's verdict for one cell, decoded where the rubric needs it.

    Returns ``(verdict, pairing_consistent)``. ``pairing_consistent`` is
    ``None`` for the two single-response rubrics, which have no pairing rule.
    """
    if record is None:
        return None, None
    if rubric != QUALITY:
        return result_verdict(record, rubric), None
    parsed = record.get("parsed_result")
    if not isinstance(parsed, Mapping):
        return None, None
    orientation = assert_orientation(
        {POSITION_A: entry["response_a_origin"],
         POSITION_B: entry["response_b_origin"]}, f"row {row_id}")
    outcome = decode_preference(str(parsed["preference"]), orientation,
                                f"row {row_id}")
    verdict = compose_quality_verdict(outcome, str(parsed["difference"]),
                                      f"row {row_id}")
    return verdict, quality_pairing_is_consistent(verdict)


def check_coverage(cells: Mapping[Any, Mapping[str, Any]],
                   runs: Mapping[str, Mapping[str, Any]],
                   expect: Mapping[str, Any], checks: Checklist,
                   missing_by_judge: Optional[Mapping[str, int]] = None,
                   carried_rows: Optional[Sequence[str]] = None
                   ) -> Dict[str, Any]:
    """The arithmetic a two-site join must satisfy.

    ``missing_by_judge`` is how many verdicts the verified technical manifest
    says a judge never produced. It is subtracted from that judge's expected
    record count and from nothing else: the cell total, the per-rubric totals
    and the per-class totals are unchanged, because a technically unresolved
    cell is still a cell and is still counted.
    """
    missing_by_judge = missing_by_judge or {}
    carried = set(carried_rows or ())

    def gate(name: str, passed: bool, detail: str) -> None:
        checks.record(name, passed, detail)
        if not passed:
            raise TwoSiteJudgingError(f"{name}: {detail}")

    for judge in sorted(runs):
        want = expect["records_per_judge"] - int(missing_by_judge.get(judge, 0))
        gate(f"{judge}_record_count",
             len(runs[judge]) == want,
             f"{judge} has {len(runs[judge])} successful record(s), and this "
             f"stage has {expect['records_per_judge']}"
             + (f" less {missing_by_judge[judge]} carried as technically "
                f"unresolved = {want}" if missing_by_judge.get(judge) else ""))

    gate("cells_total", len(cells) == expect["cells"],
         f"{len(cells)} unique cell(s), expected {expect['cells']}")

    by_rubric: Dict[str, int] = defaultdict(int)
    by_arm: Dict[str, int] = defaultdict(int)
    by_class: Dict[str, int] = defaultdict(int)
    rows_by_class: Dict[str, set] = defaultdict(set)
    for (screen_row_id, arm_id, rubric) in cells:
        by_rubric[rubric] += 1
        by_arm[arm_id] += 1
        by_class[RUBRIC_CLASS[rubric]] += 1
        rows_by_class[RUBRIC_CLASS[rubric]].add(screen_row_id)

    for rubric, want in sorted(expect["cells_by_rubric"].items()):
        gate(f"cells_on_{rubric}", by_rubric.get(rubric, 0) == want,
             f"{by_rubric.get(rubric, 0)} cell(s), expected {want}")
    gate("cells_by_arm",
         dict(sorted(by_arm.items())) == dict(sorted(
             expect["cells_by_arm"].items())),
         f"{dict(sorted(by_arm.items()))}, expected {expect['cells_by_arm']}")
    gate("cells_by_class",
         dict(sorted(by_class.items())) == dict(sorted(
             expect["cells_by_class"].items())),
         f"{dict(sorted(by_class.items()))}, expected "
         f"{expect['cells_by_class']}")
    counted = {name: len(ids) for name, ids in sorted(rows_by_class.items())}
    gate("rows_by_class",
         counted == dict(sorted(expect["rows_by_class"].items())),
         f"{counted}, expected {expect['rows_by_class']}")

    allowed_states = set(expect["terminal_states"])
    if carried:
        allowed_states.add(STATE_UNRESOLVED_TECHNICAL)
    stray = sorted({cell["state"] for cell in cells.values()}
                   - allowed_states)
    gate("every_cell_is_agree_disagree_or_carried", not stray,
         f"the join holds the non-terminal state(s) {stray}; a cell judged by "
         f"one judge or by neither is not a result unless the verified "
         f"technical manifest carries it")

    off_manifest = sorted(
        cell["row_id"] for cell in cells.values()
        if cell["state"] == STATE_UNRESOLVED_TECHNICAL
        and cell["row_id"] not in carried)
    gate("only_manifest_cells_are_carried", not off_manifest,
         f"{len(off_manifest)} cell(s) hold the carried state without being "
         f"named in the manifest, first {off_manifest[:3]}" if off_manifest
         else f"{len(carried)} carried cell(s), all named in the manifest")

    unmet = sorted(
        row_id for row_id in carried
        if row_id not in {c["row_id"] for c in cells.values()
                          if c["state"] == STATE_UNRESOLVED_TECHNICAL})
    gate("every_manifest_cell_is_carried", not unmet,
         f"the manifest names {unmet[:3]} but the join did not carry them; a "
         f"judge may since have answered, and the manifest must then be "
         f"revised by hand" if unmet else
         f"every manifest cell is carried")

    return {"cells": len(cells),
            "cells_by_rubric": dict(sorted(by_rubric.items())),
            "cells_by_arm": dict(sorted(by_arm.items())),
            "cells_by_class": dict(sorted(by_class.items())),
            "rows_by_class": counted,
            "records_by_judge": {j: len(runs[j]) for j in sorted(runs)}}


def join(packet: Path, judging: Path, out_dir: Path, write: bool = True,
         expect: Optional[Mapping[str, Any]] = None,
         unresolved_manifest: Optional[Path] = None) -> Dict[str, Any]:
    """Join both judges onto the cells.

    ``unresolved_manifest`` names the committed technical-unresolved manifest.
    It defaults to the real one; a path that does not exist yields a manifest
    carrying nothing, which is exactly the behaviour this stage had before the
    carry-forward existed.
    """
    checks = Checklist(f"{STAGE_NAME}_join")
    if write:
        assert_output_dir_is_new(out_dir, (packet, judging), "the join")
    row_map = load_row_map(packet)

    runs: Dict[str, Dict[str, Dict[str, Any]]] = {j: {} for j in JUDGES}
    failures: Dict[str, Dict[str, int]] = {j: {} for j in JUDGES}
    directories: Dict[str, List[str]] = {j: [] for j in JUDGES}
    failure_records_by_judge: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        j: {} for j in JUDGES}
    if not judging.is_dir():
        raise TwoSiteJudgingError(f"{judging} is not a directory")
    for directory in sorted(p for p in judging.iterdir() if p.is_dir()):
        name = directory.name
        for judge in JUDGES:
            if not name.endswith(f"__{judge}"):
                continue
            directories[judge].append(name)
            for row_id, record in load_run(directory).items():
                if row_id in runs[judge]:
                    raise TwoSiteJudgingError(
                        f"{judge} has two records for {row_id}, in "
                        f"{runs[judge][row_id]['_dir']} and {name}")
                record["_dir"] = name
                runs[judge][row_id] = record
            for row_id, attempts in failed_rows(directory).items():
                failures[judge][row_id] = (failures[judge].get(row_id, 0)
                                           + attempts)
            for row_id, records in failure_records(directory).items():
                failure_records_by_judge[judge].setdefault(
                    row_id, []).extend(records)

    # The paired technical carry-forward, verified before anything is joined.
    # Nothing here reads a verdict; see routegate.twositeunresolved.
    carried = load_technical_unresolved(
        unresolved_manifest if unresolved_manifest is not None
        else TECHNICAL_UNRESOLVED_MANIFEST)
    carried_rows = set(unresolved_row_ids(carried))
    carried_index = unresolved_index(carried)
    run_dirs = {f"{j}:{r}": judging / f"packet__{r}__{j}"
                for j in JUDGES for r in RUBRICS_USED}
    carry_report = verify_technical_unresolved(
        carried, row_map, runs, failure_records_by_judge, run_dirs)
    checks.record(
        "technical_unresolved_verified", True,
        f"{carry_report['entries']} carried cell(s), all gates passed: "
        f"{sorted(carry_report['gates'])}" if carry_report["entries"]
        else "no cell is carried as technically unresolved")

    for judge in JUDGES:
        unresolved = [r for r in unresolved_failed_rows(failures[judge],
                                                        runs[judge])
                      if (r, judge) not in carried_index]
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
            f"{len(unresolved)} failed row(s) with no verdict and not "
            f"carried by the technical-unresolved manifest"
            + (f": {unresolved[:3]}. A failure is not a verdict; re-run the "
               f"judge to retry those rows, and note that a failed call is "
               f"never retried automatically." if unresolved else
               f"; every row that failed has since been answered or is "
               f"carried as technically unresolved"))

    cells: Dict[Any, Dict[str, Any]] = {}
    private: List[Dict[str, Any]] = []
    pairing_violations: List[str] = []
    for row_id, entry in sorted(row_map.items()):
        rubric = entry["rubric"]
        verdicts: Dict[str, Optional[str]] = {}
        consistent: Dict[str, Optional[bool]] = {}
        record_hashes: Dict[str, Optional[str]] = {}
        formats_used: Dict[str, str] = {}
        for judge in JUDGES:
            record = runs[judge].get(row_id)
            verdict, ok = cell_verdict(record, rubric, entry, row_id)
            verdicts[judge] = verdict
            consistent[judge] = ok
            record_hashes[judge] = (result_record_hash(record)
                                    if record is not None else None)
            if record is not None:
                formats_used[judge] = record["_result_format"]
            if ok is False:
                pairing_violations.append(f"{judge}:{row_id}")

        present = [j for j in JUDGES if verdicts[j] is not None]
        if row_id in carried_rows and len(present) < len(JUDGES):
            # Carried by the verified manifest: one judge never produced a
            # verdict. The cell is terminal for the join and decided by
            # nobody. adjudicate_cell will label it UNRESOLVED on its own
            # missing-evidence branch; nothing here decides anything.
            state, agree = STATE_UNRESOLVED_TECHNICAL, None
        elif not present:
            state, agree = "missing", None
        elif len(present) < len(JUDGES):
            state, agree = "incomplete", None
        else:
            agree = verdicts[JUDGES[0]] == verdicts[JUDGES[1]]
            state = "agree" if agree else "disagree"

        key = cell_key(entry, "the private row map")
        if key in cells:
            raise TwoSiteJudgingError(
                f"cell {key} is claimed by row {cells[key]['row_id']} and by "
                f"row {row_id}. This packet is not deduplicated, so one cell "
                f"is one row; two rows over one cell means the map is wrong, "
                f"and agreeing verdicts would only hide that.")
        cell = {
            "screen_row_id": entry["screen_row_id"],
            "arm_id": entry["arm_id"],
            "rubric": rubric,
            "row_class": entry["row_class"],
            "row_id": row_id,
            "state": state,
            "verdicts": verdicts,
            "record_sha256": record_hashes,
            "result_formats": dict(formats_used),
        }
        if state == STATE_UNRESOLVED_TECHNICAL:
            carried_entry = next(
                carried_index[(row_id, j)] for j in JUDGES
                if (row_id, j) in carried_index)
            cell["technically_unresolved"] = True
            cell["missing_judge"] = str(carried_entry["failed_judge"])
            cell["failure_type"] = str(carried_entry["failure_type"])
            cell["unresolved_id"] = str(carried_entry["unresolved_id"])
        for field_name in CELL_DIGEST_FIELDS[rubric]:
            cell[field_name] = entry[field_name]
        if rubric in PAIRWISE_RUBRICS:
            cell["pairing_consistent"] = dict(consistent)
        for field_name in CARRIED_CELL_FIELDS:
            cell[field_name] = entry.get(field_name)
        cells[key] = cell
        private.append({"row_id": row_id, "rubric": rubric, "state": state,
                        "verdicts": verdicts, "agree": agree,
                        "pairing_consistent": dict(consistent),
                        "result_formats": dict(formats_used),
                        "screen_row_id": entry["screen_row_id"],
                        "arm_id": entry["arm_id"]})

    by_state: Dict[str, int] = defaultdict(int)
    by_state_and_rubric: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    for cell in cells.values():
        by_state[cell["state"]] += 1
        by_state_and_rubric[cell["rubric"]][cell["state"]] += 1

    # The coverage a complete run must satisfy. Never widened to fit what the
    # run happened to produce.
    expect = expect if expect is not None else FROZEN_COVERAGE
    missing_by_judge: Dict[str, int] = {j: 0 for j in JUDGES}
    for (_row_id, judge) in carried_index:
        missing_by_judge[judge] = missing_by_judge.get(judge, 0) + 1
    coverage = check_coverage(cells, runs, expect, checks,
                              missing_by_judge, sorted(carried_rows))
    coverage["technical_unresolved"] = {
        "manifest": "two_site_actuator_judging_v1_technical_unresolved.json",
        "is_outcome_based_exclusion": False,
        "invents_no_verdict": True,
        "enters_human_review": False,
        "entries": carry_report["entries"],
        "row_ids": carry_report["row_ids"],
        "by_judge": dict(sorted(missing_by_judge.items())),
        "checked": carry_report["checked"],
        "gates": carry_report["gates"],
        "counts": derived_counts(carried),
    }

    summary = {
        "stage": STAGE_NAME,
        "protocol": PROTOCOL_NAME,
        "created_utc": utc_now(),
        "cells": len(cells),
        "by_state": dict(sorted(by_state.items())),
        "by_state_and_rubric": {r: dict(sorted(v.items()))
                                for r, v in sorted(by_state_and_rubric.items())},
        "rows": len(row_map),
        "records_by_judge": {j: len(runs[j]) for j in JUDGES},
        "failed_rows_by_judge": {j: sorted(failures[j]) for j in JUDGES},
        "failure_attempts_by_judge": {j: dict(sorted(failures[j].items()))
                                      for j in JUDGES},
        "unresolved_failed_rows_by_judge": {
            j: unresolved_failed_rows(failures[j], runs[j]) for j in JUDGES},
        "judges": list(JUDGES),
        "quality_decoding": {
            "where": "here, and nowhere else",
            "rule": "the private map says which presentation position held the "
                    "candidate; A_BETTER / B_BETTER become AFTER_BETTER / "
                    "BEFORE_BETTER accordingly, and EQUIVALENT is unchanged",
            "preserves_agreement": "a per-row bijection applied identically to "
                                   "both judges; two judges agree after "
                                   "decoding iff they agreed before it",
            "agreement_is_joint": "outcome AND strength. Naming the same "
                                  "winner at different strengths is not "
                                  "agreement.",
            "pairing_violations": sorted(pairing_violations),
            "pairing_violations_note": "recorded verbatim and flagged, never "
                                       "rewritten. A flagged decision is still "
                                       "that judge's decision.",
        },
        "coverage_expected": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in expect.items()},
        "coverage_observed": coverage,
        "disagreement_is_private": True,
        "disagreement_note": "per-cell judge verdicts and agreement live in "
                             f"private/{JOINED_CELLS_FILE} and are not read by "
                             "export_two_site_human_review.py except to select "
                             "rows. A reviewer who knew the judges had split "
                             "would not be blind.",
        "resolves_nothing": True,
        "resolves_nothing_note": "this stage produces no adjudicated verdict, "
                                 "no outcome count and no selected strength.",
        "adjudication_procedure_unchanged": DECLARED_PROCEDURE,
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
    parser.add_argument("--unresolved-manifest", type=Path,
                        default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = judging_root() / "judging"
        packet = args.packet or (root / "packet")
        judging = args.judging or (root / "runs")
        out_dir = args.output_dir or (root / "joined")
        join(packet, judging, out_dir, write=not args.dry_run,
             unresolved_manifest=args.unresolved_manifest)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
