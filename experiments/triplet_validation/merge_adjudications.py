#!/usr/bin/env python3
"""Merge the two completed adjudication records into one evidence record.

The experiment was reviewed in two passes. The supplement's 50 cells -- B_d2
and B_d3 on both axes, plus H on the safety axis -- were reviewed and
adjudicated first. The joiner's repair then produced the other 60, and those
were reviewed separately. This joins the two halves, checks them against the
authoritative judge record, and writes one 110-cell file.

It decides nothing. Every verdict here was decided upstream; this stage
concatenates, verifies and orders. It applies no inclusion rule and writes no
inclusion manifest -- that is ``build_inclusion_manifest.py``, and it runs
after this.

What is checked
---------------
Counting first, because a short merge is a well-formed file:

* exactly 50 supplemental cells, covering exactly the supplemental (arm, axis)
  cells; exactly 60 historical ones, covering exactly the historical ones;
* the two halves disjoint, and 110 unique cells between them;
* the merged key set **identical** to the authoritative joined record's, in
  both directions -- a missing cell and an unexpected one are different
  failures and are reported as such;
* and that key set identical to the frozen grid: the ten operative case ids
  crossed with the eleven required (arm, axis) cells.

Then every cell, against the joined record it came from: both judges'
verdicts, both judges' record digests, the prompt and response digests, and
the capped-response fields. A cell whose text or verdict has moved since the
join is refused, because a decision made about the old one is not a decision
about this one.

Then every cell on its own terms: a verdict present, a string, and in that
axis's frozen vocabulary; a label source that is a decision rather than a
recorded gap; and ``judges_agree`` consistent with both the two judge verdicts
and the joined cell's state.

Text
----
Nothing here reads a prompt or a response. The records at this stage carry
digests, labels and opaque ids, and a record that carries text is refused
rather than passed through -- this file has no code path that could copy one
on.

Usage
-----
    python experiments/triplet_validation/merge_adjudications.py --dry-run
    python experiments/triplet_validation/merge_adjudications.py \\
        --supplemental $TV/adjudicated \\
        --historical $TV/adjudicated_historical \\
        --joined $TV/joined \\
        --output-dir $TV/adjudicated_complete
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ADJUDICATED_CELLS_FILE,
    AXIS_VALUES,
    CARRIED_CELL_FIELDS,
    DECIDED_LABEL_SOURCES,
    EXPECTED_CASE_IDS,
    HISTORICAL_REQUIRED_CELLS,
    JOINED_CELLS_FILE,
    JUDGES,
    JUDGE_CELL_FIELDS,
    LABEL_SOURCES,
    SOURCE_UNRESOLVED,
    SUPPLEMENTAL_REQUIRED_CELLS,
    TripletError,
    assert_no_text_fields,
    assert_same_cell,
    index_cells,
    load_adjudicated_cells,
    read_jsonl,
    required_cells,
    working_dir,
)
from routegate.validation import Checklist, sha256_file, sha256_text  # noqa: E402

#: What this merge must produce, and what each half must bring to it. The
#: numbers are properties of the experiment: 5 supplemental (arm, axis) cells
#: and 6 historical ones, over the ten frozen case ids.
EXPECTED_HALVES: Dict[str, Dict[str, Any]] = {
    "supplemental": {"cells": SUPPLEMENTAL_REQUIRED_CELLS,
                     "count": len(SUPPLEMENTAL_REQUIRED_CELLS)
                     * len(EXPECTED_CASE_IDS)},
    "historical": {"cells": HISTORICAL_REQUIRED_CELLS,
                   "count": len(HISTORICAL_REQUIRED_CELLS)
                   * len(EXPECTED_CASE_IDS)},
}

#: Prompt and response text. Named only so it can be refused: a record at this
#: stage carries digests, and one that carries text is a record from some other
#: stage.
TEXT_FIELDS: Tuple[str, ...] = ("prompt_text", "response_text")

CELLS_OUT = ADJUDICATED_CELLS_FILE
SUMMARY_OUT = "adjudication_summary.json"
MANIFEST_OUT = "manifest.json"


def assert_carries_no_text(record: Mapping[str, Any], origin: str) -> None:
    """Refuse a cell record that carries prompt or response text.

    Skipping the field would be enough to keep it out of the output, but not
    enough to know it never arrived. The merge reads digests; a record holding
    text came from somewhere this stage does not read, and that is worth
    stopping for rather than tidying away.
    """
    present = sorted(f for f in TEXT_FIELDS if record.get(f) is not None)
    if present:
        raise TripletError(
            f"{origin}: a cell record carries {present}. This stage reads "
            f"digests, labels and opaque ids only; a record carrying text is "
            f"refused rather than read past.")


def cells_file(path: Path) -> Path:
    """The cell file a half names, whether it names the file or its directory.

    The same resolution ``load_adjudicated_cells`` performs, available on its
    own so the manifest can record which file was actually read rather than
    which path was typed.
    """
    return path if path.is_file() else path / ADJUDICATED_CELLS_FILE


def load_half(path: Path, name: str, flag: str
              ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """One completed adjudication record, keyed by stable identity.

    ``index_cells`` refuses a repeated ``(case_id, arm, axis)`` outright, so a
    duplicated cell fails here rather than being resolved by whichever line the
    reader happened to see last.
    """
    cells = load_adjudicated_cells(path, flag)
    expected = EXPECTED_HALVES[name]
    for key, record in sorted(cells.items()):
        assert_carries_no_text(record, f"{flag} {key}")

    if len(cells) != expected["count"]:
        raise TripletError(
            f"{flag} holds {len(cells)} cell(s); the {name} half of this "
            f"experiment is {expected['count']}. A half that is not the size "
            f"it should be is refused, not merged and counted afterwards.")

    covered = sorted({(arm, axis) for _, arm, axis in cells})
    if covered != sorted(expected["cells"]):
        raise TripletError(
            f"{flag} covers {covered}; the {name} half covers "
            f"{sorted(expected['cells'])}. The right number of the wrong "
            f"cells is not the right record.")
    return cells


def assert_matches_joined(record: Mapping[str, Any],
                          cell: Mapping[str, Any],
                          origin: str, joined_origin: str) -> Dict[str, Any]:
    """One adjudicated cell against the joined cell it was decided from.

    Identity and the prompt/response digests go through ``assert_same_cell``;
    the judge verdicts and record digests are shaped differently on the two
    sides -- flat fields here, maps there -- so the correspondence is applied
    from ``JUDGE_CELL_FIELDS`` rather than rebuilt in place.
    """
    verified: Dict[str, Any] = dict(
        assert_same_cell(record, cell, origin, joined_origin))

    verdicts = cell.get("verdicts")
    digests = cell.get("record_sha256")
    if not isinstance(verdicts, Mapping) or not isinstance(digests, Mapping):
        raise TripletError(
            f"{joined_origin}: the joined cell for {origin} carries no "
            f"verdicts or record_sha256 map, so there is nothing to check the "
            f"adjudicated cell against")

    for judge, fields in sorted(JUDGE_CELL_FIELDS.items()):
        for source, field in sorted(fields.items()):
            mine, theirs = record.get(field), (
                verdicts if source == "verdict" else digests).get(judge)
            if mine != theirs:
                raise TripletError(
                    f"{origin}: {field} is {mine!r} but {joined_origin} "
                    f"records {theirs!r} for {judge}. The adjudicated cell "
                    f"does not describe the judging this join holds.")
            verified[field] = mine

    for field in CARRIED_CELL_FIELDS:
        if field not in record and field not in cell:
            continue
        if record.get(field) != cell.get(field):
            raise TripletError(
                f"{origin}: {field} is {record.get(field)!r} but "
                f"{joined_origin} records {cell.get(field)!r}. It is carried "
                f"through every stage unchanged, so a difference is a stage "
                f"that changed it.")
        verified[field] = record.get(field)
    return verified


def assert_decided(record: Mapping[str, Any], cell: Mapping[str, Any],
                   origin: str) -> None:
    """The cell carries a real verdict, not a gap and not a malformed one.

    ``UNRESOLVED`` is a finding, and a true one; it is simply not a verdict.
    The merged record is the evidence the inclusion rules read, and a rule that
    met a blank there would have to decide what it meant.
    """
    axis = record["axis"]
    allowed = AXIS_VALUES[axis]

    source = record.get("label_source")
    if source not in LABEL_SOURCES:
        raise TripletError(
            f"{origin}: label_source {source!r} is not one of "
            f"{list(LABEL_SOURCES)}")
    if source not in DECIDED_LABEL_SOURCES:
        raise TripletError(
            f"{origin}: label_source is {SOURCE_UNRESOLVED}, so this cell was "
            f"never decided. That is a real finding and it is not a verdict; "
            f"the merged record carries decided cells only. Complete the "
            f"review that left it open rather than merging around it.")

    verdict = record.get("verdict")
    if not isinstance(verdict, str) or not verdict:
        raise TripletError(
            f"{origin}: verdict is {verdict!r}. A cell claiming {source} with "
            f"no verdict in it is malformed, and there is nothing here that "
            f"could supply one.")
    if verdict not in allowed:
        raise TripletError(
            f"{origin}: verdict {verdict!r} is not one of {list(allowed)} for "
            f"{axis}. An adjudication cannot introduce a label the frozen "
            f"schema does not define.")

    for judge in JUDGES:
        field = JUDGE_CELL_FIELDS[judge]["verdict"]
        value = record.get(field)
        if not isinstance(value, str) or value not in allowed:
            raise TripletError(
                f"{origin}: {field} is {value!r}, which is not one of "
                f"{list(allowed)} for {axis}")

    agree = record.get("judges_agree")
    both = tuple(record[JUDGE_CELL_FIELDS[j]["verdict"]] for j in JUDGES)
    if not isinstance(agree, bool) or agree != (both[0] == both[1]):
        raise TripletError(
            f"{origin}: judges_agree is {agree!r} but the two judge verdicts "
            f"are {list(both)}. The flag is a statement about them and cannot "
            f"disagree with them.")
    state = cell.get("state")
    if state is not None and state != ("agree" if agree else "disagree"):
        raise TripletError(
            f"{origin}: judges_agree is {agree!r} but the joined cell records "
            f"state {state!r}")


def assert_output_dir_is_new(out_dir: Path, inputs: Sequence[Path]) -> None:
    """Somewhere new, and never over an input.

    Both halves are completed human adjudications. Neither can be rebuilt from
    the judging runs, so the merge must not be able to land on one -- including
    by a path that only resolves to it after symlinks.
    """
    resolved = out_dir.resolve()
    for path in inputs:
        directory = (path if path.is_dir() else path.parent).resolve()
        if resolved == directory:
            raise TripletError(
                f"--output-dir resolves to {directory}, which is an input to "
                f"this merge. The merged record is written somewhere new; the "
                f"records it is built from are read and never written.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise TripletError(
            f"{out_dir} already exists and is not empty. An evidence record "
            f"is written once; name a new directory rather than one that "
            f"already holds something.")


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """The exact bytes ``write_jsonl`` produces, as text.

    Built here rather than written and then read back, so that ``--dry-run``
    can report the digests the real run would write without writing anything.
    A run that does write re-hashes the file afterwards and refuses a
    disagreement, so the two can never drift apart unnoticed.
    """
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                   for row in rows)


def _file_record(file_path: Path, text: str, **extra: Any) -> Dict[str, Any]:
    """One entry in the manifest: what a file is, and what it hashes to.

    The filesystem argument is ``file_path`` and not ``path`` because the
    manifest itself carries a ``path`` field -- the path the operator named, as
    against the file that was actually read. A caller supplying that field
    through ``extra`` would otherwise collide with this parameter.
    """
    payload = text.encode("utf-8")
    return {"name": file_path.name, "sha256": sha256_text(text),
            "bytes": len(payload), **extra}


def merge(supplemental: Path, historical: Path, joined: Path, out_dir: Path,
          write: bool = True) -> Dict[str, Any]:
    """The two halves, verified against the join, as one ordered record."""
    checks = Checklist("adjudication_merge")
    if write:
        assert_output_dir_is_new(out_dir, [supplemental, historical, joined])

    halves = {
        "supplemental": load_half(supplemental, "supplemental",
                                  "--supplemental"),
        "historical": load_half(historical, "historical", "--historical"),
    }
    for name, cells in sorted(halves.items()):
        checks.record(f"{name}_cells", True,
                      f"{len(cells)} cell(s) over "
                      f"{len({c for c, _, _ in cells})} case(s)")

    overlap = sorted(set(halves["supplemental"]) & set(halves["historical"]))
    if overlap:
        raise TripletError(
            f"{len(overlap)} cell(s) appear in both halves, first "
            f"{overlap[:3]}. Two decisions about one cell is not something to "
            f"resolve here: one of them is unaccounted for.")
    checks.record("halves_are_disjoint", True, "no cell in both records")

    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    source_of: Dict[Tuple[str, str, str], str] = {}
    for name, cells in sorted(halves.items()):
        for key, record in cells.items():
            merged[key] = record
            source_of[key] = name

    joined_path = joined / "private" / JOINED_CELLS_FILE
    if not joined_path.is_file():
        raise TripletError(
            f"{joined_path} is not a file. --joined names the join output "
            f"holding private/{JOINED_CELLS_FILE}; it is the authoritative "
            f"judge record every merged cell is checked against.")
    joined_cells = index_cells(read_jsonl(joined_path), str(joined_path))
    for key, cell in sorted(joined_cells.items()):
        assert_carries_no_text(cell, f"{joined_path} {key}")

    # Both directions, reported apart. A cell the join holds and the merge does
    # not is a judgement nobody made; a cell the merge holds and the join does
    # not is a judgement about nothing.
    absent = sorted(set(joined_cells) - set(merged))
    unexpected = sorted(set(merged) - set(joined_cells))
    if absent:
        raise TripletError(
            f"{len(absent)} cell(s) in {joined_path} have no adjudication, "
            f"first {absent[:3]}. The merged record must answer the whole "
            f"judge record; a short merge is not a smaller one.")
    if unexpected:
        raise TripletError(
            f"{len(unexpected)} adjudicated cell(s) are not in {joined_path}, "
            f"first {unexpected[:3]}. A decision about a cell the judges never "
            f"produced cannot enter the evidence record.")
    checks.record("keys_match_the_joined_record", True,
                  f"{len(merged)} cell(s), identical in both directions")

    grid = {(case_id, arm, axis) for case_id in EXPECTED_CASE_IDS
            for arm, axis in required_cells()}
    off_grid = sorted(set(merged) ^ grid)
    if off_grid:
        raise TripletError(
            f"{len(off_grid)} cell(s) differ from the frozen grid of "
            f"{len(grid)}, first {off_grid[:3]}. The ten operative cases and "
            f"the eleven required cells are fixed before any verdict exists.")
    checks.record("matches_the_frozen_grid", True,
                  f"{len(EXPECTED_CASE_IDS)} case(s) x "
                  f"{len(required_cells())} required cell(s) = {len(grid)}")

    verified_fields: Counter = Counter()
    for key, record in sorted(merged.items()):
        origin = f"{source_of[key]} {key}"
        for field in assert_matches_joined(record, joined_cells[key], origin,
                                           str(joined_path)):
            verified_fields[field] += 1
        assert_decided(record, joined_cells[key], origin)
    checks.record("verified_against_the_joined_record", True,
                  f"{len(merged)} cell(s); "
                  + ", ".join(f"{field} x{count}" for field, count
                              in sorted(verified_fields.items())))

    ordered = [merged[key] for key in sorted(merged)]
    by_source: Counter = Counter(record["label_source"]
                                 for record in ordered)
    summary = {
        "created_utc": utc_now(),
        "cells": len(ordered),
        "cells_by_input": {name: len(cells)
                           for name, cells in sorted(halves.items())},
        "cases": len({case_id for case_id, _, _ in merged}),
        "by_label_source": dict(sorted(by_source.items())),
        "by_arm_axis": {f"{arm}/{axis}": count for (arm, axis), count
                        in sorted(Counter((arm, axis) for _, arm, axis
                                          in merged).items())},
        "judges_agree": {"true": sum(1 for r in ordered if r["judges_agree"]),
                         "false": sum(1 for r in ordered
                                      if not r["judges_agree"])},
        "verified_against_joined": dict(sorted(verified_fields.items())),
        "ordering": "ascending by (case_id, arm, axis); keys sorted within "
                    "each record. The same inputs produce the same bytes.",
        "decides_nothing": "every verdict here was decided upstream. This "
                           "stage concatenates, verifies and orders.",
        "resolves_inclusion": False,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, SUMMARY_OUT)

    cells_text = _jsonl_text(ordered)
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    manifest = {
        "schema": "routegate/rq1_adjudication_merge/1",
        "description": "The two completed adjudication records, merged into "
                       "one evidence record and checked against the "
                       "authoritative joined judge record. Digests only: no "
                       "prompt text and no response text is read or written.",
        "created_utc": summary["created_utc"],
        # ``named`` is what the operator passed; ``resolved`` is the file that
        # was actually read. Both are recorded: a directory and the cell file
        # inside it are the same input named two ways, and the digest belongs
        # to the file.
        "inputs": {
            name: _file_record(
                resolved, resolved.read_text(encoding="utf-8"),
                path=str(named), cells=count)
            for name, named, resolved, count in (
                ("supplemental", supplemental, cells_file(supplemental),
                 len(halves["supplemental"])),
                ("historical", historical, cells_file(historical),
                 len(halves["historical"])),
                ("joined", joined, joined_path, len(joined_cells)))
        },
        "outputs": {
            "cells": _file_record(out_dir / CELLS_OUT, cells_text,
                                  rows=len(ordered)),
            "summary": _file_record(out_dir / SUMMARY_OUT, summary_text),
        },
        "inputs_note": "read, never written. --output-dir may not resolve to "
                       "any of them.",
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(manifest, MANIFEST_OUT)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / CELLS_OUT).write_text(cells_text, encoding="utf-8")
        (out_dir / SUMMARY_OUT).write_text(summary_text, encoding="utf-8")
        (out_dir / MANIFEST_OUT).write_text(manifest_text, encoding="utf-8")
        # what the manifest claims is on disk, re-read from disk
        for spec, name in ((manifest["outputs"]["cells"], CELLS_OUT),
                           (manifest["outputs"]["summary"], SUMMARY_OUT)):
            actual = sha256_file(out_dir / name)
            if actual != spec["sha256"]:
                raise TripletError(
                    f"{out_dir / name} hashes to {actual} but the manifest "
                    f"records {spec['sha256']}")
        print(f"[merge] wrote {out_dir}")

    for name, count in sorted(summary["cells_by_input"].items()):
        print(f"[merge] {name:<22} {count}")
    print(f"[merge] {'merged cells':<22} {len(ordered)}")
    for source, count in sorted(by_source.items()):
        print(f"[merge] {source:<22} {count}")
    print(f"[merge] {'cells sha256':<22} "
          f"{manifest['outputs']['cells']['sha256']}")
    return {"summary": summary, "manifest": manifest, "cells": ordered}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplemental", type=Path, default=None,
                        help="the first pass: the directory holding "
                             + ADJUDICATED_CELLS_FILE + ", or that file")
    parser.add_argument("--historical", type=Path, default=None,
                        help="the second pass, in the same shape")
    parser.add_argument("--joined", type=Path, default=None,
                        help="the join output holding private/"
                             + JOINED_CELLS_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = working_dir()
    supplemental = args.supplemental or (root / "adjudicated")
    historical = args.historical or (root / "adjudicated_historical")
    joined = args.joined or (root / "joined")
    out_dir = args.output_dir or (root / "adjudicated_complete")
    try:
        merge(supplemental, historical, joined, out_dir,
              write=not args.dry_run)
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
