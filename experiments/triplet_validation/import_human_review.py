#!/usr/bin/env python3
"""Import the completed human review and produce the final adjudication.

Runs after the reviewer has finished. Only now are the judges' verdicts and
their disagreements read: until this point they were private, so the review was
an independent reading rather than a tie-break.

The adjudication rule, frozen before any verdict existed
--------------------------------------------------------
Per (case, arm, axis) cell:

``DUAL_LLM_AGREEMENT``
    Sonnet and Opus agree **and** the human agrees with them. The verdict is
    that value.

``HUMAN_ADJUDICATION``
    The judges disagree, or the human differs from an agreeing pair. **The
    human's verdict stands.** The disagreement is recorded, never averaged.

``UNRESOLVED``
    A judge verdict is missing, or the human left the cell blank. No verdict is
    produced. A gap is a gap; it is not a quiet pass and not a quiet fail.

The importer never invents a verdict, never breaks a tie by majority over two
judges, and never converts a recorded failure into a label.

Merging an earlier pass
-----------------------
``--base-adjudicated`` names an evidence record produced by an earlier run --
for this experiment, the 50 supplemental cells decided first. Those decisions
are **carried forward unchanged**: they are re-keyed by stable identity,
checked against the joined cells for identity and for every prompt and
response digest, and written out byte-for-byte as they were. Nothing re-decides
them, and a cell that appears in both the base record and this run's review is
refused rather than resolved -- two decisions about one cell means one of them
is unaccounted for.

The merge is written to a **new** directory. The base record is never opened
for writing and never named as the output; a human adjudication cannot be
reconstructed from anything else.

Usage
-----
    python experiments/triplet_validation/import_human_review.py \\
        --completed <file>.jsonl --dry-run

    python experiments/triplet_validation/import_human_review.py \\
        --review-dir $TV/human_review_historical \\
        --completed $TV/human_review_historical/human_review_COMPLETED.jsonl \\
        --base-adjudicated $TV/adjudicated \\
        --output-dir $TV/adjudicated_complete
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ADJUDICATED_CELLS_FILE,
    AXIS_VALUES,
    CONFIDENCE_VALUES,
    FROZEN_INCREMENTAL_REVIEW,
    HUMAN_ROW_MAP_FILE,
    JOINED_CELLS_FILE,
    JUDGES,
    SOURCE_DUAL,
    SOURCE_HUMAN,
    SOURCE_UNRESOLVED,
    TripletError,
    assert_same_cell,
    cell_key,
    index_cells,
    load_adjudicated_cells,
    read_jsonl,
    record_hash,
    required_cells,
    working_dir,
    write_jsonl,
)
from routegate.validation import Checklist, sha256_file              # noqa: E402


def adjudicate(cell: Dict[str, Any], human: Optional[Dict[str, Any]]
               ) -> Dict[str, Any]:
    axis = cell["axis"]
    allowed = AXIS_VALUES[axis]
    verdicts = cell["verdicts"]
    sonnet, opus = verdicts.get(JUDGES[0]), verdicts.get(JUDGES[1])
    human_verdict = (human or {}).get("verdict") or None

    if human_verdict is not None and human_verdict not in allowed:
        raise TripletError(
            f"{cell['case_id']}/{cell['arm']}/{axis}: the human verdict "
            f"{human_verdict!r} is not one of {list(allowed)}. A review cannot "
            f"introduce a label the frozen schema does not define.")

    if sonnet is None or opus is None or human_verdict is None:
        missing = [n for n, v in (("sonnet", sonnet), ("opus", opus),
                                  ("human", human_verdict)) if v is None]
        return {"verdict": None, "label_source": SOURCE_UNRESOLVED,
                "adjudication_rule": f"missing: {missing}",
                "judges_agree": None}

    if sonnet == opus == human_verdict:
        return {"verdict": human_verdict, "label_source": SOURCE_DUAL,
                "adjudication_rule": "sonnet == opus == human",
                "judges_agree": True}
    return {"verdict": human_verdict, "label_source": SOURCE_HUMAN,
            "adjudication_rule": (
                "human adjudication; judges "
                + ("agreed but the human differed" if sonnet == opus
                   else "disagreed")),
            "judges_agree": sonnet == opus}


def assert_output_dir_is_new(out_dir: Path, base: Optional[Path]) -> None:
    """The merge goes somewhere new, and never over the record it merges.

    The base record is a person's completed adjudication. There is no way to
    rebuild it from the judging runs, so the one thing this stage must not be
    able to do is write on top of it -- including by way of a path that only
    resolves to it after symlinks.
    """
    if base is not None:
        base_dir = (base if base.is_dir() else base.parent).resolve()
        if out_dir.resolve() == base_dir:
            raise TripletError(
                f"--output-dir resolves to {base_dir}, which is the record "
                f"being merged in. The merge is written to a new directory; "
                f"the earlier adjudication is never written over.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise TripletError(
            f"{out_dir} already exists and is not empty. An adjudication is "
            f"written once; name a new directory rather than overwriting one.")


def load_base(path: Path, cells: Mapping[Tuple[str, str, str], Mapping[str, Any]]
              ) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], str]:
    """A prior adjudication, verified against the joined cells, plus its digest.

    Every preserved decision has to be about a cell this join still holds, and
    about the same text: identity, then every digest either record carries. The
    file's own digest is returned so the summary can record which bytes were
    carried forward.
    """
    file = path / ADJUDICATED_CELLS_FILE if path.is_dir() else path
    base = load_adjudicated_cells(path, "--base-adjudicated")
    for key, record in sorted(base.items()):
        cell = cells.get(key)
        if cell is None:
            raise TripletError(
                f"{key} is adjudicated in {file} but this join holds no such "
                f"cell. A preserved decision about a cell that is not in the "
                f"evidence cannot be carried into the merged record.")
        assert_same_cell(record, cell, f"{file}", "the joined cell")
    return base, sha256_file(file)


def check_merge(merged: Mapping[Tuple[str, str, str], Mapping[str, Any]],
                base: Mapping[Tuple[str, str, str], Any],
                fresh: Mapping[Tuple[str, str, str], Any],
                expect: Mapping[str, Any], checks: Checklist) -> Dict[str, Any]:
    """The arithmetic the merged evidence record must satisfy."""
    def gate(name: str, passed: bool, detail: str) -> None:
        checks.record(name, passed, detail)
        if not passed:
            raise TripletError(f"{name}: {detail}")

    gate("preserved_decisions", len(base) == expect["already_adjudicated"],
         f"{len(base)} decision(s) carried forward, expected "
         f"{expect['already_adjudicated']}")
    gate("new_decisions", len(fresh) == expect["to_review"],
         f"{len(fresh)} new decision(s), expected {expect['to_review']}")
    gate("merged_cells", len(merged) == expect["merged_cells"],
         f"{len(merged)} merged cell(s), expected {expect['merged_cells']}")

    cases = sorted({case_id for case_id, _, _ in merged})
    gate("cases", (len(cases) == expect["cases"]
                   and cases == sorted(expect["case_ids"])),
         f"{len(cases)} case(s), expected {expect['cases']}: "
         f"{sorted(set(expect['case_ids']) - set(cases))} absent, "
         f"{sorted(set(cases) - set(expect['case_ids']))} unexpected")

    # Against the frozen grid, not against whatever arrived: a case that
    # produced nothing at all would satisfy a check derived from the record.
    wanted = {(case_id, arm, axis) for case_id in expect["case_ids"]
              for arm, axis in required_cells()}
    absent = sorted(wanted - set(merged))
    gate("every_required_cell_is_decided", not absent,
         f"{len(absent)} required cell(s) are absent from the merged record, "
         f"first {absent[:3]}")
    return {"preserved": len(base), "new": len(fresh), "merged": len(merged),
            "cases": len(cases)}


def run(joined: Path, review: Path, completed: Path, out_dir: Path,
        write: bool = True, base: Optional[Path] = None,
        expect: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    checks = Checklist("human_review_import")
    if write:
        assert_output_dir_is_new(out_dir, base)
    joined_path = joined / "private" / JOINED_CELLS_FILE
    cells = index_cells(read_jsonl(joined_path), str(joined_path))
    map_path = review / "private" / HUMAN_ROW_MAP_FILE
    row_map = {r["row_id"]: r for r in read_jsonl(map_path)}

    preserved: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    base_digest: Optional[str] = None
    if base is not None:
        preserved, base_digest = load_base(base, cells)
        checks.record("preserved_decisions_verified", True,
                      f"{len(preserved)} prior decision(s) matched to a joined "
                      f"cell by identity and digest")

    submitted: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(completed):
        row_id = row.get("row_id")
        if row_id not in row_map:
            raise TripletError(
                f"the completed review carries {row_id!r}, which is not a row "
                f"of the exported packet")
        if row_id in submitted:
            raise TripletError(f"{row_id} is answered twice")
        confidence = row.get("confidence") or ""
        if confidence and confidence not in CONFIDENCE_VALUES:
            raise TripletError(
                f"{row_id}: confidence {confidence!r} is not one of "
                f"{list(CONFIDENCE_VALUES)}")
        submitted[row_id] = row
    missing_rows = sorted(set(row_map) - set(submitted))
    checks.record("rows_answered", True,
                  f"{len(submitted)} of {len(row_map)}; "
                  f"{len(missing_rows)} unanswered")

    adjudicated: List[Dict[str, Any]] = []
    for row_id, mapping in sorted(row_map.items()):
        key = cell_key(mapping, str(map_path))
        cell = cells.get(key)
        if cell is None:
            raise TripletError(f"{key} has no joined cell")
        # The row map was written by the export from these same joined cells.
        # Checking it again costs nothing and catches the case that matters:
        # a review answered against a packet built from a different join.
        assert_same_cell(mapping, cell, str(map_path), str(joined_path))
        outcome = adjudicate(cell, submitted.get(row_id))
        record = {
            "case_id": mapping["case_id"], "arm": mapping["arm"],
            "axis": mapping["axis"],
            "prompt_sha256": mapping["prompt_sha256"],
            "response_sha256": mapping["response_sha256"],
            "reached_max_tokens": cell.get("reached_max_tokens"),
            "output_token_count": cell.get("output_token_count"),
            "sonnet_verdict": cell["verdicts"].get(JUDGES[0]),
            "opus_verdict": cell["verdicts"].get(JUDGES[1]),
            "sonnet_record_sha256": cell["record_sha256"].get(JUDGES[0]),
            "opus_record_sha256": cell["record_sha256"].get(JUDGES[1]),
            "human_row_id": row_id,
            **outcome,
        }
        record["human_record_sha256"] = record_hash(
            {"row_id": row_id, "verdict": outcome["verdict"],
             "confidence": submitted.get(row_id, {}).get("confidence", "")})
        adjudicated.append(record)

    fresh = index_cells(adjudicated, "this run's adjudicated cells")
    decided_twice = sorted(set(fresh) & set(preserved))
    if decided_twice:
        raise TripletError(
            f"{len(decided_twice)} cell(s) are decided both in the base "
            f"record and in this review, first {decided_twice[:3]}. Two "
            f"decisions about one cell is not something to resolve here: one "
            f"of them is unaccounted for, and the review that produced this "
            f"one should not have contained the cell at all.")
    merged = dict(preserved)
    merged.update(fresh)

    counts: Counter = Counter(r["label_source"] for r in merged.values())
    new_counts: Counter = Counter(r["label_source"] for r in fresh.values())
    kept_counts: Counter = Counter(r["label_source"]
                                   for r in preserved.values())
    coverage = (check_merge(merged, preserved, fresh, expect, checks)
                if expect is not None else None)

    summary = {
        "created_utc": utc_now(),
        "cells": len(merged),
        "by_label_source": dict(sorted(counts.items())),
        "decisions_from_this_review": dict(sorted(new_counts.items())),
        "decisions_carried_forward": dict(sorted(kept_counts.items())),
        "base_adjudicated_sha256": base_digest,
        "base_note": "the decisions carried forward are written out exactly as "
                     "they were read. They are matched to this join by "
                     "(case_id, arm, axis) and confirmed by every prompt and "
                     "response digest either record carries; nothing re-decides "
                     "them and no cell is decided twice.",
        "coverage": coverage,
        "unanswered_rows": missing_rows,
        "rule": {
            SOURCE_DUAL: "sonnet == opus == human",
            SOURCE_HUMAN: "the human verdict stands; judges disagreed, or the "
                          "human differed from an agreeing pair",
            SOURCE_UNRESOLVED: "a judge verdict or the human verdict is absent; "
                               "no verdict is produced",
        },
        "never_done": ["majority vote over two judges", "averaging",
                       "converting a recorded failure into a verdict",
                       "filling a blank cell with a default"],
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / ADJUDICATED_CELLS_FILE,
                    [merged[key] for key in sorted(merged)])
        (out_dir / "adjudication_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[adjudicate] wrote {out_dir}")
    if preserved:
        print(f"[adjudicate] {len(preserved)} decision(s) carried forward, "
              f"{len(fresh)} new, {len(merged)} in the merged record")
    for source, count in sorted(counts.items()):
        print(f"[adjudicate] {source:<20} {count}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--joined-dir", "--joined", dest="joined_dir",
                        type=Path, default=None)
    parser.add_argument("--review-dir", "--review", dest="review_dir",
                        type=Path, default=None,
                        help="the exported review packet holding private/"
                             + HUMAN_ROW_MAP_FILE)
    parser.add_argument("--base-adjudicated", type=Path, default=None,
                        help="an earlier evidence record whose decisions are "
                             "carried forward unchanged into the merge")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = working_dir()
    joined = args.joined_dir or (root / "joined")
    review = args.review_dir or (root / "human_review")
    out_dir = args.output_dir or (root / "adjudicated")
    try:
        # A base record means this is the merge of the incremental pass, so
        # the frozen arithmetic applies and is checked.
        run(joined, review, args.completed, out_dir, write=not args.dry_run,
            base=args.base_adjudicated,
            expect=(FROZEN_INCREMENTAL_REVIEW
                    if args.base_adjudicated is not None else None))
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
