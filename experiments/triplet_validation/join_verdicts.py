#!/usr/bin/env python3
"""Join two judges' verdicts back onto (case, arm, axis) cells. Deterministic.

Reads only run records and the private row maps. Calls nothing, generates
nothing, and decides nothing about inclusion -- that is
``build_inclusion_manifest.py``, and it runs after the human review.

What joining does
-----------------
Each judge produced one verdict per opaque row. A row may stand for more than
one occurrence, because identical requests were deduplicated at temperature 0.0
and the label is fanned back out. This script inverts that: it maps every row
back through the private map to the (case, arm, axis) cells it covers, and
records, per cell, the Sonnet verdict, the Opus verdict and whether they agree.

Two frozen layouts
------------------
The two packets store that map in two shapes, and both are read here.

* **supplemental** -- ``private/row_map.jsonl``, occurrences embedded in each
  entry, already carrying ``case_id``, ``arm`` and ``axis``.
* **historical** -- ``private/axis1_row_map.jsonl`` and
  ``private/axis2_row_map.jsonl``, whose entries list ``occurrence_source_keys``
  into a separate ``private/occurrences.jsonl``. An occurrence there records
  ``case_id``, ``role`` and ``condition``, and no arm at all.

The arm is resolved through the one exhaustive ``(role, condition)`` table in
``routegate.triplets``. Nothing is read out of a source key, a file name or a
row's position in a file: a source key reads like ``case::condition::role`` and
a file is named for its axis, but text that happens to be legible is not a
schema. An axis likewise comes from the entry's own ``axis`` field in the
inline layout, and in the separate layout from the frozen per-axis opaque
prefix each row id carries -- checked, not assumed from the file it was in.

The join fails closed on an occurrence key that resolves to nothing, a judged
row that fans out to no occurrence, one cell claimed by two rows, an occurrence
no row refers to, and a required cell that ends with no verdict.

Counting the result
-------------------
A layout the loader does not understand does not raise -- it yields no cells,
which reads exactly like a run nobody has done yet. That is the defect this
file was corrected for: 97 historical rows silently fanned out to nothing and
50 supplemental cells were reported as the whole experiment. So the run counts
itself against ``FROZEN_JOIN_COVERAGE``: 110 unique cells, 60 historical and 50
supplemental, over 10 cases, from 147 successful records per judge, with every
cell terminal in ``agree`` or ``disagree``. Any other shape is a failure.

Disagreement is **not** resolved here, and is **not** exposed to the human
reviewer. It is written to a private file that the review export never reads,
so that a reviewer cannot be anchored by knowing the judges split. The
adjudication importer reads it afterwards.

Incomplete runs
---------------
A cell with a verdict from only one judge is reported as ``incomplete`` and
carries no agreement value. A cell with no verdict is ``missing``. Neither is
silently treated as agreement, and neither becomes an inclusion input.

Usage
-----
    python experiments/triplet_validation/join_verdicts.py --dry-run
    python experiments/triplet_validation/join_verdicts.py
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

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ARMS,
    AXES,
    FROZEN_JOIN_COVERAGE,
    HISTORICAL_ROW_PREFIX,
    HISTORICAL_TEXT_FIELDS,
    INLINE_ROW_MAP,
    JUDGES,
    LAYOUT_INLINE,
    LAYOUT_SEPARATE,
    OCCURRENCE_KEY_FIELD,
    OCCURRENCE_KEYS_FIELD,
    RESULT_FILES,
    SEPARATE_OCCURRENCES,
    SEPARATE_ROW_MAPS,
    TripletError,
    assert_opaque,
    detect_result_format,
    occurrence_arm,
    read_jsonl,
    required_cells,
    result_is_success,
    result_join_key,
    result_record_hash,
    result_verdict,
    working_dir,
    write_jsonl,
)
from routegate.validation import Checklist                           # noqa: E402


def load_run(directory: Path) -> Dict[str, Dict[str, Any]]:
    """``row_id -> record`` for one judge/axis run. Successful rows only.

    Two runners can have produced this directory and their record schemas
    genuinely differ -- the historical retrospective runner keys its blinded
    id as ``item_id`` and carries no per-record hash; this repository's runner
    carries ``record_sha256`` and no ``item_id``. The format is **named** per
    record rather than inferred from a field that happens to be missing, so a
    schema drift is refused instead of read as an empty run.

    ``result_is_success`` matters as much: the historical runner stamps
    ``status = "OK"`` on a success, while older historical runs stamped no
    status at all. Both spellings must read as success; treating either one
    as a failure drops every historical row and refuses the join.
    """
    out: Dict[str, Dict[str, Any]] = {}
    formats: set = set()
    seen_any = False
    for name in sorted({spec["results"] for spec in RESULT_FILES.values()}):
        results = directory / name
        if not results.exists():
            continue
        for record in read_jsonl(results):
            seen_any = True
            fmt = detect_result_format(record)
            formats.add(fmt)
            if not result_is_success(record):
                continue
            row_id = result_join_key(record)
            if row_id in out:
                raise TripletError(
                    f"{results} holds two successful records for {row_id}")
            record["_result_format"] = fmt
            out[row_id] = record
    if len(formats) > 1:
        raise TripletError(
            f"{directory} mixes result formats {sorted(formats)}. One run "
            f"directory is one runner's output; joining two schemas from one "
            f"directory would make the provenance of a verdict unreadable.")
    if seen_any and not out:
        raise TripletError(
            f"{directory} holds records but none is a success. If these came "
            f"from the historical runner, its failures belong in "
            f"{RESULT_FILES['historical_rq1_retrospective_judge']['failures']}"
            f", not in results.jsonl; refusing to report an empty join.")
    return out


def _row_id_of(entry: Any, origin: str) -> str:
    """The opaque id of one row-map entry, or a refusal."""
    if not isinstance(entry, Mapping):
        raise TripletError(f"{origin}: a row map entry must be an object, got "
                           f"{type(entry).__name__}")
    row_id = entry.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise TripletError(f"{origin}: a row map entry carries no row_id")
    return row_id


def _cell_of(occurrence: Mapping[str, Any], axis: str,
             origin: str) -> Dict[str, str]:
    """One ``{case_id, arm, axis}`` cell, validated, and nothing else.

    Deliberately narrow. A separate-layout occurrence record carries the
    prompt and response text as well; projecting every occurrence down to
    three fields here is what keeps that text out of the joined output, which
    no later stage is allowed to see.
    """
    case_id = occurrence.get("case_id")
    arm = occurrence.get("arm")
    if not isinstance(case_id, str) or not case_id:
        raise TripletError(f"{origin}: an occurrence carries no case_id")
    if arm not in ARMS:
        raise TripletError(f"{origin}: occurrence arm {arm!r} is not one of "
                           f"{list(ARMS)}")
    return {"case_id": case_id, "arm": arm, "axis": axis}


def _entry(raw: Mapping[str, Any], row_id: str, axis: str, layout: str,
           packet: Path, cells: List[Dict[str, str]],
           covered: int) -> Dict[str, Any]:
    """One layout-independent row: what it is, and what it stands for.

    ``cells`` holds only the occurrences that are an inclusion arm. ``covered``
    counts every occurrence the row fans out to, arm or not, because those are
    different facts: 37 of the historical packet's 67 Axis-2 rows judge a
    *patched* response only, and a patched response is never an inclusion
    input. Such a row produces no cell and is not an error. A row that fans
    out to nothing at all is.
    """
    return {
        "row_id": row_id,
        "axis": axis,
        "layout": layout,
        "packet": packet.name,
        "occurrences": cells,
        "occurrences_covered": covered,
        "prompt_sha256": raw.get("prompt_sha256"),
        "response_sha256": raw.get("response_sha256"),
        # carried, never acted on; see the capped-response policy
        "reached_max_tokens": raw.get("reached_max_tokens"),
        "output_token_count": raw.get("output_token_count"),
    }


def _read_inline(packet: Path) -> List[Dict[str, Any]]:
    """The supplemental layout: one row map, occurrences embedded per entry.

    Every fact the join needs is already in the entry -- its axis, and for each
    occurrence the case and the arm. Nothing is resolved. An occurrence that
    also names its own axis, as every row of the frozen supplement does, is
    checked against the row's rather than assumed to agree with it.
    """
    path = packet / "private" / INLINE_ROW_MAP
    entries: List[Dict[str, Any]] = []
    for raw in read_jsonl(path):
        row_id = _row_id_of(raw, str(path))
        origin = f"{path} row {row_id}"
        axis = raw.get("axis")
        if axis not in AXES:
            raise TripletError(
                f"{origin}: declares axis {axis!r}. In this layout an entry "
                f"names its own axis, and it must be one of {list(AXES)}.")
        occurrences = raw.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise TripletError(
                f"{origin}: maps to no occurrence. A judged row that stands "
                f"for nothing cannot be joined; it is refused, not skipped.")
        cells = []
        for occurrence in occurrences:
            if not isinstance(occurrence, Mapping):
                raise TripletError(f"{origin}: an occurrence must be an object")
            declared = occurrence.get("axis")
            if declared is not None and declared != axis:
                raise TripletError(
                    f"{origin}: the row is on {axis} but one of its "
                    f"occurrences declares axis {declared!r}. One of the two "
                    f"is wrong and neither is preferred.")
            cells.append(_cell_of(occurrence, axis, origin))
        entries.append(_entry(raw, row_id, axis, LAYOUT_INLINE, packet,
                              cells, len(occurrences)))
    return entries


def _occurrence_index(packet: Path) -> Dict[str, Dict[str, Any]]:
    """``source_key -> occurrence`` for a separate-layout packet."""
    path = packet / "private" / SEPARATE_OCCURRENCES
    if not path.exists():
        raise TripletError(
            f"{packet} keeps its row maps per axis, so its occurrences live "
            f"in {SEPARATE_OCCURRENCES}, and that file is not there. Without "
            f"it no row resolves to any cell -- which reads exactly like a "
            f"packet nobody has judged yet, so it is refused instead.")
    index: Dict[str, Dict[str, Any]] = {}
    for occurrence in read_jsonl(path):
        if not isinstance(occurrence, Mapping):
            raise TripletError(f"{path}: an occurrence must be an object")
        key = occurrence.get(OCCURRENCE_KEY_FIELD)
        if not isinstance(key, str) or not key:
            raise TripletError(f"{path}: an occurrence carries no "
                               f"{OCCURRENCE_KEY_FIELD}")
        if key in index:
            raise TripletError(
                f"{path}: two occurrences share {OCCURRENCE_KEY_FIELD} "
                f"{key!r}. The key has to identify one occurrence; two would "
                f"merge two cells behind one judged row.")
        index[key] = occurrence
    if not index:
        raise TripletError(f"{path} holds no occurrence")
    return index


def _assert_digests_match(raw: Mapping[str, Any], occurrence: Mapping[str, Any],
                          origin: str) -> None:
    """The row and the occurrence it claims must be the same text.

    Both sides carry the digests, and the join key is a string that resolves
    without them. Checking them is what makes a resolved key evidence rather
    than an assumption: a row wired to the wrong occurrence would otherwise
    produce a confident verdict about a cell nobody judged.
    """
    for field_name in ("prompt_sha256", "response_sha256"):
        mine, theirs = raw.get(field_name), occurrence.get(field_name)
        if mine is None or theirs is None:
            continue
        if mine != theirs:
            raise TripletError(
                f"{origin}: the row's {field_name} does not match the "
                f"occurrence's. The row map and the occurrence file disagree "
                f"about what was judged.")


def _read_separate(packet: Path,
                   per_axis: Dict[str, Path]) -> List[Dict[str, Any]]:
    """The historical layout: a row map per axis, and one occurrences file.

    The axis of a map is not taken from its file name. Each map is read for
    the axis it is declared to hold, and every row id in it must carry that
    axis's frozen opaque prefix -- the packet's own record of which axis a row
    belongs to. A row filed under the wrong map fails the join rather than
    entering it under the wrong axis.

    The arm comes from the occurrence's recorded role and condition, through
    the one exhaustive table in ``routegate.triplets``. The source key is used
    to *find* the occurrence and is never parsed for what it appears to say.
    """
    index = _occurrence_index(packet)
    occurrences_path = packet / "private" / SEPARATE_OCCURRENCES
    referenced: set = set()
    entries: List[Dict[str, Any]] = []
    for axis, path in sorted(per_axis.items()):
        claimed: Dict[str, str] = {}
        for raw in read_jsonl(path):
            row_id = _row_id_of(raw, str(path))
            origin = f"{path} row {row_id}"
            assert_opaque(row_id, HISTORICAL_ROW_PREFIX[axis])
            keys = raw.get(OCCURRENCE_KEYS_FIELD)
            if not isinstance(keys, list) or not keys:
                raise TripletError(
                    f"{origin}: carries no {OCCURRENCE_KEYS_FIELD}, so it "
                    f"maps to no occurrence. A judged row that stands for "
                    f"nothing cannot be joined; it is refused, not skipped.")
            declared = raw.get("n_occurrences")
            if declared is not None and declared != len(keys):
                raise TripletError(
                    f"{origin}: says n_occurrences {declared!r} and lists "
                    f"{len(keys)} key(s). The packet contradicts itself.")
            cells = []
            for key in keys:
                if key not in index:
                    raise TripletError(
                        f"{origin}: refers to occurrence {key!r}, which "
                        f"{occurrences_path} does not hold.")
                if key in claimed:
                    raise TripletError(
                        f"{origin}: occurrence {key!r} is already claimed by "
                        f"row {claimed[key]} on {axis}. One occurrence is "
                        f"judged once per axis; two rows claiming it would "
                        f"give one cell two verdicts.")
                claimed[key] = row_id
                referenced.add(key)
                occurrence = index[key]
                _assert_digests_match(raw, occurrence, origin)
                arm = occurrence_arm(occurrence, f"{occurrences_path} {key}")
                if arm is None:
                    continue     # a steering outcome: covered, but not a cell
                cells.append(_cell_of(dict(occurrence, arm=arm), axis, origin))
            entries.append(_entry(raw, row_id, axis, LAYOUT_SEPARATE, packet,
                                  cells, len(keys)))
    orphans = sorted(set(index) - referenced)
    if orphans:
        raise TripletError(
            f"{occurrences_path}: {len(orphans)} occurrence(s) no row map "
            f"refers to, first {orphans[:3]}. An unreferenced occurrence is "
            f"an outcome no judge was ever asked about, and passing over it "
            f"is how a partial join comes to read as a complete one.")
    return entries


def load_row_maps(packets: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """``row_id -> {axis, occurrences, digests}`` across every packet.

    The two frozen packets store the same relation in two shapes, and both are
    read. The layout is chosen by which private files a packet actually holds,
    never by what the packet is called; a packet offering both is refused
    rather than half-read, and a packet offering neither is passed over.

    The defect this replaces: the loader understood the inline layout only and
    read the historical packet's entries through ``.get("occurrences", [])``.
    That key is absent there, so all 97 historical rows fanned out to nothing
    and the join reported 50 supplemental cells as though that were the whole
    experiment -- with every checklist entry passing.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for packet in packets:
        private = packet / "private"
        inline = private / INLINE_ROW_MAP
        per_axis = {axis: private / name
                    for axis, name in SEPARATE_ROW_MAPS.items()
                    if (private / name).exists()}
        if inline.exists() and per_axis:
            raise TripletError(
                f"{packet} holds both {INLINE_ROW_MAP} and "
                f"{sorted(path.name for path in per_axis.values())}. The two "
                f"frozen layouts are alternatives, not halves; reading one "
                f"packet as both would count its rows twice.")
        if inline.exists():
            entries = _read_inline(packet)
        elif per_axis:
            entries = _read_separate(packet, per_axis)
        else:
            continue
        for entry in entries:
            row_id = entry["row_id"]
            if row_id in rows:
                raise TripletError(
                    f"opaque id {row_id} appears in two row maps; ids must "
                    f"be unique across every packet")
            rows[row_id] = entry
    if not rows:
        raise TripletError(f"no private row map found under {list(packets)}")
    return rows


def check_coverage(cells: Mapping[Any, Mapping[str, Any]],
                   runs: Mapping[str, Mapping[str, Any]],
                   expect: Mapping[str, Any],
                   checks: Checklist) -> Dict[str, Any]:
    """The exact shape a complete join of this frozen experiment must have.

    Counting is the only check a joiner can make that a silent layout failure
    cannot pass. Each expected number is a property of the experiment and is
    derived in ``routegate.triplets.FROZEN_JOIN_COVERAGE``; nothing here is
    tuned to whatever the current inputs happen to produce.
    """
    def gate(name: str, passed: bool, detail: str) -> None:
        checks.record(name, passed, detail)
        if not passed:
            raise TripletError(f"{name}: {detail}")

    for judge in sorted(runs):
        gate(f"{judge}_record_count",
             len(runs[judge]) == expect["records_per_judge"],
             f"{judge} has {len(runs[judge])} successful record(s), and this "
             f"experiment has {expect['records_per_judge']}")

    # which cells, before how many: "60 required cells have no verdict, first
    # ..." names the gap, where a bare count leaves it to be looked for.
    wanted = {(case_id, arm, axis) for case_id in expect["case_ids"]
              for arm, axis in required_cells()}
    absent = sorted(wanted - set(cells))
    gate("required_cells_complete", not absent,
         f"{len(absent)} required cell(s) have no verdict, first "
         f"{absent[:3]}")
    extra = sorted(set(cells) - wanted)
    gate("no_cell_outside_the_frozen_grid", not extra,
         f"{len(extra)} cell(s) no inclusion rule asks for, first {extra[:3]}")

    gate("cells_total", len(cells) == expect["cells"],
         f"{len(cells)} unique cell(s), expected {expect['cells']}")

    by_layout: Dict[str, int] = defaultdict(int)
    for cell in cells.values():
        by_layout[cell["layout"]] += 1
    for layout, count in sorted(expect["cells_by_layout"].items()):
        gate(f"cells_from_{layout}", by_layout.get(layout, 0) == count,
             f"{by_layout.get(layout, 0)} cell(s) from the {layout} layout, "
             f"expected {count}")

    cases = sorted({case_id for case_id, _, _ in cells})
    gate("cases", len(cases) == expect["cases"],
         f"{len(cases)} case(s), expected {expect['cases']}")

    stray = sorted({cell["state"] for cell in cells.values()}
                   - set(expect["terminal_states"]))
    gate("every_cell_is_agree_or_disagree", not stray,
         f"{len(cells)} cell(s) hold the non-terminal state(s) {stray}; a "
         f"cell judged by one judge or by neither is not a result")

    return {"cells": len(cells), "cells_by_layout": dict(sorted(by_layout.items())),
            "cases": len(cases),
            "records_by_judge": {judge: len(runs[judge]) for judge in sorted(runs)}}


def join(root: Path, packets: Sequence[Path], judging: Path,
         out_dir: Path, write: bool = True,
         expect: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Join both frozen layouts onto cells, and count what came out.

    ``expect`` is the coverage this run must produce; ``main`` passes the
    frozen experiment's, so the real invocation always counts itself. Passing
    ``None`` joins whatever the inputs hold and checks only their internal
    consistency, which is what a fixture of a handful of rows needs.
    """
    checks = Checklist("triplet_join")
    row_map = load_row_maps(packets)

    runs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for directory in sorted(p for p in judging.iterdir() if p.is_dir()):
        name = directory.name
        for judge in JUDGES:
            if name.endswith(f"__{judge}"):
                for row_id, record in load_run(directory).items():
                    if row_id in runs[judge]:
                        raise TripletError(
                            f"{judge} has two records for {row_id}, in "
                            f"{runs[judge][row_id].get('_dir')} and {name}")
                    record["_dir"] = name
                    runs[judge][row_id] = record
    for judge in JUDGES:
        checks.record(f"{judge}_rows", True, f"{len(runs[judge])} successful row(s)")

    cells: Dict[tuple, Dict[str, Any]] = {}
    private: List[Dict[str, Any]] = []
    for row_id, entry in sorted(row_map.items()):
        axis = entry["axis"]
        verdicts: Dict[str, Optional[str]] = {}
        formats_used: Dict[str, str] = {}
        record_hashes: Dict[str, Optional[str]] = {}
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

        for occurrence in entry["occurrences"]:
            key = (occurrence["case_id"], occurrence["arm"], axis)
            if key in cells:
                raise TripletError(
                    f"cell {key} is claimed by row {cells[key]['row_id']} and "
                    f"by row {row_id}. Deduplication fans one row out to many "
                    f"cells, never one cell back to many rows; two rows over "
                    f"one cell means the deduplication key is not sound, and "
                    f"agreeing verdicts would only hide that.")
            cells[key] = {
                "case_id": occurrence["case_id"], "arm": occurrence["arm"],
                "axis": axis, "row_id": row_id, "state": state,
                "verdicts": verdicts, "record_sha256": record_hashes,
                "prompt_sha256": entry["prompt_sha256"],
                "response_sha256": entry["response_sha256"],
                "reached_max_tokens": entry.get("reached_max_tokens"),
                "output_token_count": entry.get("output_token_count"),
                "packet": entry["packet"], "layout": entry["layout"],
                "result_formats": dict(formats_used),
            }
        private.append({"row_id": row_id, "axis": axis, "state": state,
                        "layout": entry["layout"], "packet": entry["packet"],
                        "verdicts": verdicts, "agree": agree,
                        "result_formats": dict(formats_used),
                        "occurrences": entry["occurrences"],
                        "occurrences_covered": entry["occurrences_covered"]})

    by_state: Dict[str, int] = defaultdict(int)
    for cell in cells.values():
        by_state[cell["state"]] += 1
    by_layout: Dict[str, int] = defaultdict(int)
    for cell in cells.values():
        by_layout[cell["layout"]] += 1

    coverage: Optional[Dict[str, Any]] = None
    if expect is not None:
        coverage = check_coverage(cells, runs, expect, checks)

    summary = {
        "created_utc": utc_now(),
        "cells": len(cells),
        "by_state": dict(sorted(by_state.items())),
        "by_layout": dict(sorted(by_layout.items())),
        "rows": len(row_map),
        "records_by_judge": {judge: len(runs[judge]) for judge in JUDGES},
        "coverage_expected": dict(expect) if expect is not None else None,
        "coverage_observed": coverage,
        "judges": list(JUDGES),
        "disagreement_is_private": True,
        "disagreement_note": "per-cell judge verdicts and agreement live in "
                             "private/joined_cells.jsonl and are not read by "
                             "export_human_review.py. A reviewer who knew the "
                             "judges had split would not be blind.",
        "resolves_inclusion": False,
        "result_formats_seen": sorted({fmt for cell in cells.values()
                                       for fmt in cell["result_formats"].values()}),
        "text_never_copied": sorted(HISTORICAL_TEXT_FIELDS),
        "text_note": "the historical runner stores the rendered request body, "
                     "the raw API payload and the judge's raw text in its own "
                     "run directory. None of those fields is read or copied "
                     "here; only verdict labels, digests and opaque ids cross "
                     "into a joined cell.",
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        write_jsonl(out_dir / "private" / "joined_cells.jsonl",
                    [cells[k] for k in sorted(cells)])
        write_jsonl(out_dir / "private" / "joined_rows.jsonl", private)
        (out_dir / "join_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[join] wrote {out_dir}")
    for judge in JUDGES:
        print(f"[join] {judge + ' records':<22} {len(runs[judge])}")
    for layout, count in sorted(by_layout.items()):
        print(f"[join] {layout + ' cells':<22} {count}")
    print(f"[join] {'cells':<22} {len(cells)}")
    for state, count in sorted(by_state.items()):
        print(f"[join] {state:<22} {count}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, action="append", default=None)
    parser.add_argument("--judging", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = working_dir()
    packets = args.packet or [root / "historical_packet", root / "supplemental_packet"]
    judging = args.judging or (root / "judging")
    out_dir = args.output_dir or (root / "joined")
    try:
        join(root, packets, judging, out_dir, write=not args.dry_run,
             expect=FROZEN_JOIN_COVERAGE)
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
