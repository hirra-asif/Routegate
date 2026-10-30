#!/usr/bin/env python3
"""Export the blinded human-review packet. Calls nothing, decides nothing.

Covers every inclusion-critical baseline arm -- E, H, d1, d2, d3 -- for all ten
cases, on every axis those arms need. Patched arms are **not** included: a
patched response is a steering outcome, and this validation must not be able to
condition on one.

What the reviewer sees
----------------------
A prompt, or a prompt and one response, and an opaque id. That is all. The
export refuses to write a row carrying any of: case id, arm, E/B/H role, donor
number, case grouping, condition, layer, strength, expected verdict, previous
judge result, or repair outcome.

**Neither judge's verdict is in the packet, and neither is their agreement.**
``join_verdicts.py`` writes those to a private file; this script does not open
it. A reviewer told that Sonnet and Opus had split would be anchored, and
telling them only where the judges agreed would be worse -- it would make the
review a tie-break rather than an independent reading.

Rows are shuffled with an independent seed so that adjacent rows are not the
same case, and the ids are freshly minted so they cannot be joined to the
judging runs except through the private map.

Reviewing only what is left
---------------------------
``--exclude-adjudicated`` names a completed evidence record whose cells are
**not** to be reviewed again. Those cells are matched by their stable
scientific identity -- ``(case_id, arm, axis)`` -- because that is the only
thing that survives the blinding: the packets mint their own opaque ids, and
this export mints a fresh one per row precisely so an answer cannot be traced
back to a judging record.

Identity alone is not enough to skip a cell. Before a prior decision is allowed
to stand in for a review, every prompt and response digest either record
carries is checked against the joined cell. A cell whose text has moved is
refused, not skipped: a decision about the old text is not a decision about
this one. A prior cell the join does not contain is refused too, rather than
quietly removing that cell from the review as well.

For this frozen experiment the arithmetic is fixed and is checked: 110 joined
cells, 50 already adjudicated (B_d2 and B_d3 on both axes, plus H on the safety
axis), and exactly the 60 historical cells left -- E, H and B_d1 on Axis 1 and
Axis 2, across all ten cases.

Usage
-----
    python experiments/triplet_validation/export_human_review.py --dry-run
    python experiments/triplet_validation/export_human_review.py --seed 20260821
    python experiments/triplet_validation/export_human_review.py \\
        --joined-dir $TV/joined \\
        --exclude-adjudicated $TV/adjudicated \\
        --output-dir $TV/human_review_historical --seed 20260822
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ADJUDICATED_CELLS_FILE,
    ARMS,
    AXIS1,
    AXIS2,
    AXIS_VALUES,
    FROZEN_INCREMENTAL_REVIEW,
    HUMAN_ROW_MAP_FILE,
    HUMAN_ROW_PREFIX,
    JOINED_CELLS_FILE,
    JUDGE_ROW_FIELDS,
    SAFETY,
    TripletError,
    assert_ids_unique,
    assert_row_is_blind,
    assert_same_cell,
    assert_text_is_blind,
    index_cells,
    load_adjudicated_cells,
    load_protocol,
    new_opaque_ids,
    read_jsonl,
    required_cells,
    working_dir,
    write_jsonl,
)
from routegate.validation import Checklist, sha256_text              # noqa: E402

#: Fields that must never reach the reviewer's file. Superset of the judge list:
#: a human can read prose a judge would ignore. Every name a joined or
#: adjudicated cell can carry is here, so that a field added upstream is caught
#: by this gate rather than shipped.
FORBIDDEN = ("case_id", "arm", "role", "axis", "verdicts", "verdict", "agree",
             "judges_agree", "adjudication_rule", "state", "label_source",
             "sonnet", "opus", "sonnet_verdict", "opus_verdict",
             "record_sha256", "sonnet_record_sha256", "opus_record_sha256",
             "human_record_sha256", "prompt_sha256", "response_sha256",
             "packet", "layout", "row_id_source", "occurrences",
             "layer", "layers", "strength", "condition", "config_id",
             "steering", "expected", "expected_verdict")

#: Summary keys the operator's copy carries and the reviewer's does not. They
#: stratify the packet: a reviewer told the 60 rows cover six arm/axis cells
#: over ten cases knows they arrive in ten equal groups, and can start reading
#: the packet as a grouping exercise rather than one row at a time. Knowing
#: that 110 cells exist and 60 are here says the other 50 were decided
#: elsewhere, which is the same information again. The coverage record and the
#: checklist spell the arm and axis names out, so they go with them.
#:
#: The reviewer's manifest keeps the row count, the axes -- each row states its
#: own axis task anyway -- the seed, the instruction digests, and the list of
#: what the packet deliberately does not contain.
REVIEWER_WITHHELD: Tuple[str, ...] = (
    "arms_covered", "cases", "joined_cells", "cells_already_adjudicated",
    "coverage", "checklist")

#: The only fields ever carried out of a packet record. Everything else a
#: source record happens to hold -- digests, occurrence lists, token counts --
#: stays in the packet.
TEXT_FIELDS: Tuple[str, ...] = ("prompt_text", "response_text")

#: Where the two frozen packet layouts keep the text, relative to a packet root.
#:
#: The **historical** packet keeps it in the private per-axis row maps
#: (``private/axis1_row_map.jsonl``, ``private/axis2_row_map.jsonl``): each
#: entry carries ``prompt_text``, and on the response axes ``response_text``.
#:
#: The **supplemental** packet does not. Its ``private/row_map.jsonl`` is
#: digest-only by construction -- ``prompt_sha256``, ``response_sha256``, the
#: rendered-request digests, the occurrence list -- and its text lives in the
#: public per-axis judge-input files it actually sent, under the same
#: ``row_id``. Those files are already blind judge rows: ``row_id``,
#: ``prompt_text``, and on the response axes ``response_text``. Nothing else.
#:
#: Both layouts are read and neither is rewritten. The two packets mint their
#: ids from disjoint prefixes, so a ``row_id`` resolves in exactly one of them.
TEXT_SOURCE_GLOBS: Tuple[str, ...] = ("private/*row_map*.jsonl",
                                      "*_judge_input.jsonl")


def _text_of(entry: Mapping[str, Any],
             origin: str) -> Optional[Dict[str, str]]:
    """The text fields of one packet record, or ``None`` if it carries none.

    A record with no ``prompt_text`` is not a text source. That is the normal,
    intended state of the supplemental packet's digest-only private row map,
    and skipping it is what lets a single pass read both layouts.

    A record that *does* carry the field but whose value is not usable text is
    refused rather than skipped. Falling through to some other source would be
    choosing the reviewer's material by accident.
    """
    if not isinstance(entry, Mapping):
        raise TripletError(f"{origin}: a packet record must be an object, got "
                           f"{type(entry).__name__}")
    if "prompt_text" not in entry:
        return None
    out: Dict[str, str] = {}
    for field in TEXT_FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            raise TripletError(
                f"{origin}: {entry.get('row_id')!r} has an empty or non-text "
                f"{field}; the packet is malformed and no text is invented")
        out[field] = value
    return out


def index_text(packets: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    """``row_id`` -> its text, over every frozen layout, refusing ambiguity.

    A ``row_id`` that resolves to two different readings is refused: the
    exporter has no basis for choosing between them, and showing the reviewer
    the wrong one would silently break the comparison with the judges.
    """
    texts: Dict[str, Dict[str, str]] = {}
    origin_of: Dict[str, str] = {}
    for packet in packets:
        if not packet.is_dir():
            raise TripletError(
                f"{packet} is not a directory. A named packet that is not "
                f"there is refused, rather than silently contributing no text.")
        for pattern in TEXT_SOURCE_GLOBS:
            for path in sorted(packet.glob(pattern)):
                origin = f"{packet.name}/{path.relative_to(packet)}"
                for entry in read_jsonl(path):
                    payload = _text_of(entry, origin)
                    if payload is None:
                        continue
                    if "row_id" not in entry:
                        raise TripletError(
                            f"{origin}: a text-bearing record has no row_id, "
                            f"so it cannot be matched to a cell")
                    row_id = str(entry["row_id"])
                    previous = texts.get(row_id)
                    if previous is not None and previous != payload:
                        raise TripletError(
                            f"{row_id} reads differently in "
                            f"{origin_of[row_id]} and {origin}; the exporter "
                            f"will not choose between two readings of one row")
                    texts[row_id] = payload
                    origin_of[row_id] = origin
    return texts


def assert_output_dir_is_new(out_dir: Path) -> None:
    """Refuse to write into a directory that already holds something.

    A completed review is the only copy of a person's work, and the packet it
    was answered against is the only record of what they were asked. Writing a
    fresh export over either would leave answers keyed to rows that no longer
    exist, and there would be nothing left to notice it with.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise TripletError(
            f"{out_dir} already exists and is not empty. The export does not "
            f"write over a review packet -- a completed review cannot be "
            f"reconstructed from anything else. Name a new directory.")


def collect_rows(joined: Path, packets: Sequence[Path],
                 exclude: Optional[Mapping[Tuple[str, str, str],
                                           Mapping[str, Any]]] = None
                 ) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]],
                            List[Dict[str, Any]],
                            Dict[Tuple[str, str, str], Dict[str, str]]]:
    """``(every joined cell, the rows to review, the cells carried forward)``.

    Text is taken from the packets, in whichever of the two frozen layouts each
    packet uses -- see ``TEXT_SOURCE_GLOBS``. The joined cells supply the cell
    list and the digests to check the text against; never a verdict.

    Cells are walked in stable-identity order rather than file order, so the
    packet a given seed produces does not depend on how the joiner happened to
    write its output.
    """
    path = joined / "private" / JOINED_CELLS_FILE
    cells = index_cells(read_jsonl(path), str(path))
    wanted = set(required_cells())
    texts = index_text(packets)
    exclude = dict(exclude or {})

    stale = sorted(set(exclude) - set(cells))
    if stale:
        raise TripletError(
            f"{len(stale)} already-adjudicated cell(s) have no joined cell, "
            f"first {stale[:3]}. A prior decision about a cell this join does "
            f"not hold cannot be carried forward, and excluding it anyway "
            f"would drop that cell out of the review as well.")

    rows: List[Dict[str, Any]] = []
    carried: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for key, cell in sorted(cells.items()):
        if (cell["arm"], cell["axis"]) not in wanted:
            continue                       # patched arms and non-required axes
        prior = exclude.get(key)
        if prior is not None:
            # Identity says it is the same cell; the digests say it is the
            # same text. Only then does an earlier decision stand in for a
            # review that will not now happen.
            carried[key] = assert_same_cell(
                prior, cell, "the already-adjudicated record", str(path))
            continue
        source = texts.get(cell["row_id"])
        if source is None:
            raise TripletError(
                f"{cell['row_id']} has no text in any packet; it cannot be "
                f"shown to a reviewer without inventing it")
        payload: Dict[str, Any] = {}
        for field in JUDGE_ROW_FIELDS[cell["axis"]]:
            if field == "row_id":
                continue                   # a fresh opaque id is minted below
            if field not in source:
                raise TripletError(
                    f"{cell['row_id']}: the {cell['axis']} axis needs "
                    f"{field} and no packet holds one for that row")
            payload[field] = source[field]

        # What the human reads must be what the judges read. The joined cell
        # records the digest of the text that was actually judged, so a packet
        # whose text has drifted from it is refused here rather than quietly
        # turning the human control into a review of something else.
        for field, digest_field in (("prompt_text", "prompt_sha256"),
                                    ("response_text", "response_sha256")):
            if field not in payload:
                continue
            digest = cell.get(digest_field)
            if not isinstance(digest, str):
                raise TripletError(
                    f"{cell['row_id']}: the joined cell carries no "
                    f"{digest_field} to check {field} against; an "
                    f"unverifiable row is refused")
            if sha256_text(payload[field]) != digest:
                raise TripletError(
                    f"{cell['row_id']}: the {field} found in the packet does "
                    f"not match the {digest_field} the joined cell records; "
                    f"the reviewer is not shown text the judges did not see")
        rows.append({"_cell": cell, "_payload": payload})
    return cells, rows, carried


def reviewer_instructions(axes: Sequence[str],
                          protocol_dir: Optional[Path]) -> Dict[str, Any]:
    """The frozen judging instructions, for the human, per axis task.

    A reviewer who is shown only ``axis_task: "axis1"`` and a list of labels is
    not applying the same rubric the judges applied -- they are inventing one,
    and the comparison between them is then not a comparison. So the reviewer
    gets the **same frozen instruction text**, hash-verified from the same
    protocol files, plus the same allowed labels.

    The instruction text carries no E/B/H role, no arm, no donor number, no
    steering condition and no expected verdict -- ``preflight`` gate 5 and the
    blindness guards enforce that already. Passing it on therefore un-blinds
    nothing, and withholding it would only make the human a worse control.

    Written to the machine-local review directory, never to the repository:
    it is protocol text from the historical workspace.
    """
    out: Dict[str, Any] = {}
    for axis in axes:
        protocol = load_protocol(axis, directory=protocol_dir)
        out[axis] = {
            "instructions": protocol.instructions,
            "allowed_labels": list(protocol.allowed_values),
            "verdict_field": protocol.verdict_field,
            "instructions_sha256": protocol.instructions_sha256,
            "schema_sha256": protocol.schema_sha256,
            "note": "the frozen instruction text the judges received for "
                    "this task, verbatim and hash-verified.",
        }
    return out


def check_incremental(cells: Mapping[Tuple[str, str, str], Mapping[str, Any]],
                      carried: Mapping[Tuple[str, str, str], Any],
                      rows: Sequence[Mapping[str, Any]],
                      expect: Mapping[str, Any],
                      checks: Checklist) -> Dict[str, Any]:
    """The exact arithmetic of the incremental pass over this experiment.

    Counting is what a partial export cannot pass. An exporter that resolved
    fewer cells than it should still writes a well-formed packet; a reviewer
    would answer it, and nothing downstream would say the rest were never
    asked about.
    """
    def gate(name: str, passed: bool, detail: str) -> None:
        checks.record(name, passed, detail)
        if not passed:
            raise TripletError(f"{name}: {detail}")

    gate("joined_cells", len(cells) == expect["joined_cells"],
         f"{len(cells)} joined cell(s), expected {expect['joined_cells']}")
    gate("already_adjudicated",
         len(carried) == expect["already_adjudicated"],
         f"{len(carried)} already-adjudicated cell(s) carried forward, "
         f"expected {expect['already_adjudicated']}")
    gate("cells_to_review", len(rows) == expect["to_review"],
         f"{len(rows)} cell(s) to review, expected {expect['to_review']}")

    reviewed = {(r["_cell"]["case_id"], r["_cell"]["arm"], r["_cell"]["axis"])
                for r in rows}
    overlap = sorted(reviewed & set(carried))
    gate("no_cell_is_reviewed_twice", not overlap,
         f"{len(overlap)} cell(s) are both carried forward and re-exported, "
         f"first {overlap[:3]}")

    review_cells = sorted({(arm, axis) for _, arm, axis in reviewed})
    gate("review_cells", review_cells == sorted(expect["review_cells"]),
         f"the export covers {review_cells}, expected "
         f"{sorted(expect['review_cells'])}")
    prior_cells = sorted({(arm, axis) for _, arm, axis in carried})
    gate("prior_cells", prior_cells == sorted(expect["prior_cells"]),
         f"the prior record covers {prior_cells}, expected "
         f"{sorted(expect['prior_cells'])}")

    cases = sorted({case_id for case_id, _, _ in reviewed})
    gate("cases", (len(cases) == expect["cases"]
                   and cases == sorted(expect["case_ids"])),
         f"{len(cases)} case(s) in the export, expected {expect['cases']}: "
         f"{sorted(set(expect['case_ids']) - set(cases))} absent, "
         f"{sorted(set(cases) - set(expect['case_ids']))} unexpected")

    wanted = set(required_cells())
    required = {key for key, cell in cells.items()
                if (cell["arm"], cell["axis"]) in wanted}
    uncovered = sorted(required - reviewed - set(carried))
    gate("every_required_cell_is_covered", not uncovered,
         f"{len(uncovered)} required cell(s) are neither adjudicated nor "
         f"exported for review, first {uncovered[:3]}")

    return {"joined_cells": len(cells), "already_adjudicated": len(carried),
            "cells_to_review": len(rows), "cases": len(cases),
            "review_cells": review_cells, "prior_cells": prior_cells}


def export(joined: Path, packets: Sequence[Path], out_dir: Path,
           seed: int, write: bool = True,
           protocol_dir: Optional[Path] = None,
           exclude: Optional[Mapping[Tuple[str, str, str],
                                     Mapping[str, Any]]] = None,
           expect: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The blinded packet for whichever cells still need a human reading.

    ``exclude`` is a prior evidence record keyed by stable identity; its cells
    are carried forward instead of re-reviewed, and only after their digests
    have been checked against the joined cells. ``expect`` is the arithmetic
    the run must produce -- ``main`` passes the frozen experiment's whenever an
    exclusion is in play, so the real invocation always counts itself.
    """
    checks = Checklist("human_review_export")
    if write:
        assert_output_dir_is_new(out_dir)
    cells, rows, carried = collect_rows(joined, packets, exclude)
    if not rows:
        raise TripletError("no inclusion-critical cells were found to review")

    rng = random.Random(seed)
    rng.shuffle(rows)
    ids = new_opaque_ids(HUMAN_ROW_PREFIX, len(rows), rng=rng)
    assert_ids_unique(ids)

    case_ids = sorted({r["_cell"]["case_id"] for r in rows})
    public: List[Dict[str, Any]] = []
    private: List[Dict[str, Any]] = []
    for row_id, row in zip(ids, rows):
        cell = row["_cell"]
        axis = cell["axis"]
        record = {"row_id": row_id, "axis_task": axis, **row["_payload"]}

        leaked = sorted(set(record) & set(FORBIDDEN))
        if leaked:
            raise TripletError(f"human row {row_id} would expose {leaked}")
        for field in ("prompt_text", "response_text"):
            if field in record:
                assert_text_is_blind(record[field], case_ids,
                                     f"human row {row_id} {field}")
        public.append(record)
        private.append({"row_id": row_id, "case_id": cell["case_id"],
                        "arm": cell["arm"], "axis": axis,
                        "judge_row_id": cell["row_id"],
                        "prompt_sha256": cell["prompt_sha256"],
                        "response_sha256": cell["response_sha256"]})

    # the reviewer's answer sheet: allowed values only, no judge verdicts
    schema = {axis: list(AXIS_VALUES[axis]) for axis in (AXIS1, AXIS2, SAFETY)}
    axes_present = sorted({r["_cell"]["axis"] for r in rows})
    instructions = reviewer_instructions(axes_present, protocol_dir)
    checks.record("reviewer_gets_the_frozen_instructions", True,
                  f"{len(instructions)} axis task(s): "
                  + ", ".join(f"{a}={instructions[a]['instructions_sha256'][:12]}"
                              for a in axes_present))

    coverage = (check_incremental(cells, carried, rows, expect, checks)
                if expect is not None else None)

    summary = {
        "created_utc": utc_now(),
        "rows": len(public),
        "cases": len(case_ids),
        "arms_covered": sorted({r["_cell"]["arm"] for r in rows}),
        "axes_covered": sorted({r["_cell"]["axis"] for r in rows}),
        "joined_cells": len(cells),
        "cells_already_adjudicated": len(carried),
        "cells_already_adjudicated_note":
            "carried forward from a prior evidence record, matched on "
            "(case_id, arm, axis) and confirmed by every prompt and response "
            "digest either record carries. They are not re-reviewed and not "
            "re-decided.",
        "coverage": coverage,
        "patched_arms_included": False,
        "shuffle_seed": seed,
        "reviewer_sees": ["row_id", "axis_task", "prompt_text",
                          "response_text (response axes only)",
                          "the frozen instruction text for that axis task",
                          "the allowed labels for that axis task"],
        "instruction_hashes": {axis: spec["instructions_sha256"]
                               for axis, spec in sorted(instructions.items())},
        "instructions_note": "the human reviewer applies the same frozen "
                             "rubric the judges applied. A reviewer given only "
                             "a label list would be inventing a rubric, and "
                             "the human/judge comparison would not be one.",
        "reviewer_does_not_see": [
            "sonnet verdict", "opus verdict", "judge agreement or disagreement",
            "case id", "arm or E/B/H role", "donor number", "case grouping",
            "steering or patching condition", "layer", "strength",
            "expected verdict", "repair outcome"],
        "allowed_values": schema,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        write_jsonl(out_dir / "human_review_input.jsonl", public)
        write_jsonl(out_dir / "private" / HUMAN_ROW_MAP_FILE, private)
        # Two copies of one record. The operator's, beside the private row
        # map, is complete; the reviewer's omits the keys that would tell them
        # how the packet is grouped. Neither carries a judge verdict.
        (out_dir / "private" / "export_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        reviewer_copy = {k: v for k, v in summary.items()
                         if k not in REVIEWER_WITHHELD}
        (out_dir / "human_review_manifest.json").write_text(
            json.dumps(reviewer_copy, indent=2, sort_keys=True),
            encoding="utf-8")
        template = [{"row_id": r["row_id"], "verdict": "", "confidence": "",
                     "note": ""} for r in public]
        write_jsonl(out_dir / "human_review_TEMPLATE.jsonl", template)
        # Machine-local only: this file carries frozen protocol text.
        (out_dir / "REVIEWER_INSTRUCTIONS.json").write_text(
            json.dumps({"how_to_use": (
                "For each row in human_review_input.jsonl, read its "
                "axis_task, apply the instructions for that task below to the "
                "prompt_text (and response_text where present), and write one "
                "allowed label into the matching row of "
                "human_review_TEMPLATE.jsonl. Use only the labels listed for "
                "that task. If you cannot decide, leave the verdict empty and "
                "say why in note -- an empty verdict is recorded as "
                "unresolved, which is a real finding; a guessed label is not."),
                "tasks": instructions}, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"[human] wrote {out_dir}")
    print(f"[human] {len(cells)} joined cell(s); "
          f"{len(carried)} already adjudicated and carried forward")
    print(f"[human] {len(public)} row(s) over {len(case_ids)} case(s); "
          f"arms {summary['arms_covered']}")
    print("[human] no judge verdict and no agreement flag is in the packet")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-dir", "--joined", dest="joined_dir",
                        type=Path, default=None,
                        help="the join output holding private/"
                             + JOINED_CELLS_FILE)
    parser.add_argument("--exclude-adjudicated", type=Path, default=None,
                        help="a completed evidence record (directory or "
                             + ADJUDICATED_CELLS_FILE + ") whose cells are "
                             "not to be reviewed again")
    parser.add_argument("--packet", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None,
                        help="the hash-verified frozen protocol text, so the "
                             "human reviewer applies the judges' rubric")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = working_dir()
    joined = args.joined_dir or (root / "joined")
    packets = args.packet or [root / "historical_packet", root / "supplemental_packet"]
    out_dir = args.output_dir or (root / "human_review")
    try:
        # An exclusion means this is the incremental pass over the frozen
        # experiment, so the frozen arithmetic applies and is checked.
        exclude = (load_adjudicated_cells(args.exclude_adjudicated,
                                          "--exclude-adjudicated")
                   if args.exclude_adjudicated is not None else None)
        export(joined, packets, out_dir, args.seed, write=not args.dry_run,
               protocol_dir=args.protocol_dir, exclude=exclude,
               expect=FROZEN_INCREMENTAL_REVIEW if exclude else None)
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
