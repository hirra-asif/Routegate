#!/usr/bin/env python3
"""Import the completed human review and produce the phase-1 adjudication.

Runs after the reviewer has finished. Only now are the judges' verdicts and
their disagreements read together with the human's: until this point the
disagreements were private.

The adjudication procedure is **declared, not defaulted**
---------------------------------------------------------
This project's frozen record holds two procedures. They share the label
vocabulary and the rule that a human verdict, where one exists, stands. They
differ in which cells a human reads, and that difference changes the result.
``--procedure`` is required, is written into every adjudicated record, and must
match the value declared in
``manifests/focused_direction_validation_v1_judging.json``.

``corpus_disagreement_only``
    ``DUAL_LLM_AGREEMENT`` when Sonnet and Opus agree. ``HUMAN_ADJUDICATION``
    when they disagree and a human answered, or when a human answered
    differently from an agreeing pair. ``UNRESOLVED`` when they disagree and no
    human answered, or when a judge verdict is missing.

``triplet_independent_review``
    ``DUAL_LLM_AGREEMENT`` only when Sonnet, Opus and the human all agree.
    ``HUMAN_ADJUDICATION`` for anything else a human answered.
    ``UNRESOLVED`` when a judge verdict or the human verdict is absent.

In both, the importer never invents a verdict, never breaks a tie by majority
over two judges, and never converts a recorded failure into a label. A gap is a
gap: it is not a quiet pass and not a quiet fail.

Every joined cell is adjudicated, not only the reviewed ones. A cell nobody
reviewed still gets a record -- with whatever label source it earns -- so the
evidence file is a complete statement about all 871 cells rather than a
silently shorter one.

Usage
-----
    python experiments/focused_direction_judging/import_human_review.py \\
        --procedure corpus_disagreement_only --completed <file>.jsonl --dry-run

    python experiments/focused_direction_judging/import_human_review.py \\
        --procedure corpus_disagreement_only \\
        --review-dir $FDV/judging/human_review \\
        --completed $FDV/judging/human_review/human_review_COMPLETED.jsonl \\
        --output-dir $FDV/judging/adjudicated
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

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.focusedjudging import (                              # noqa: E402
    ADJUDICATED_CELLS_FILE,
    ADJUDICATION_PROCEDURES,
    ADJUDICATION_PROCEDURE_EVIDENCE,
    CARRIED_CELL_FIELDS,
    FROZEN_COVERAGE,
    HUMAN_ROW_MAP_FILE,
    JOINED_CELLS_FILE,
    PROTOCOL_NAME,
    FocusedJudgingError,
    adjudicate_cell,
    analysable_cells,
    assert_declared_procedure,
    assert_same_cell,
    cell_key,
    cells_needing_human_review,
    index_cells,
    judging_root,
    load_technical_exclusions,
)
from routegate.triplets import (                                    # noqa: E402
    CONFIDENCE_VALUES,
    JUDGES,
    SOURCE_UNRESOLVED,
    assert_no_text_fields,
    read_jsonl,
    record_hash,
    write_jsonl,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
)


def assert_output_dir_is_new(out_dir: Path, review: Optional[Path]) -> None:
    """The adjudication goes somewhere new, and never over what it reads.

    A completed review is a person's work and cannot be rebuilt from the
    judging runs, so the one thing this stage must not be able to do is write
    on top of it -- including by way of a path that only resolves to it after
    symlinks.
    """
    if review is not None:
        review_dir = (review if review.is_dir() else review.parent).resolve()
        if out_dir.resolve() == review_dir:
            raise FocusedJudgingError(
                f"--output-dir resolves to the review directory {review_dir}. "
                f"A completed human review is not reconstructible; it is never "
                f"written over.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FocusedJudgingError(
            f"{out_dir} already holds files. An adjudication is written to a "
            f"new directory so an earlier one is never partially overwritten.")


def load_review(review: Optional[Path], completed: Optional[Path],
                cells: Mapping[Tuple[str, str, str], Mapping[str, Any]],
                checks: Checklist
                ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """``cell identity -> the human's answer``, checked against the join.

    The row map was written by the export from these same joined cells.
    Checking it again costs nothing and catches the case that matters: a review
    answered against a packet built from a different join.
    """
    if review is None and completed is None:
        checks.record("human_review", True,
                      "no human review was supplied; every cell that needs one "
                      "under this procedure will be UNRESOLVED")
        return {}
    if review is None or completed is None:
        raise FocusedJudgingError(
            "--review-dir and --completed are supplied together: the map says "
            "which cell a row is, the completed file says what the reviewer "
            "answered, and neither is usable alone.")

    map_path = review / "private" / HUMAN_ROW_MAP_FILE
    if not map_path.is_file():
        raise FocusedJudgingError(f"{map_path} is not a file")
    row_map = {r["row_id"]: r for r in read_jsonl(map_path)}
    if not row_map:
        raise FocusedJudgingError(f"{map_path} holds no row")

    submitted: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(completed):
        row_id = row.get("row_id")
        if row_id not in row_map:
            raise FocusedJudgingError(
                f"the completed review carries {row_id!r}, which is not a row "
                f"of the exported packet")
        if row_id in submitted:
            raise FocusedJudgingError(f"{row_id} is answered twice")
        confidence = row.get("confidence") or ""
        if confidence and confidence not in CONFIDENCE_VALUES:
            raise FocusedJudgingError(
                f"{row_id}: confidence {confidence!r} is not one of "
                f"{list(CONFIDENCE_VALUES)}")
        submitted[row_id] = row

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row_id, mapping in sorted(row_map.items()):
        key = cell_key(mapping, str(map_path))
        cell = cells.get(key)
        if cell is None:
            raise FocusedJudgingError(
                f"{key} was exported for review but this join holds no such "
                f"cell. A review answered against a different join is refused.")
        assert_same_cell(mapping, cell, str(map_path), "the joined cell")
        answer = submitted.get(row_id)
        if answer is None:
            continue
        verdict = answer.get("verdict") or None
        if verdict is None:
            continue
        if key in out:
            raise FocusedJudgingError(f"{key} is answered twice")
        out[key] = {"verdict": verdict,
                    "confidence": answer.get("confidence", ""),
                    "human_row_id": row_id}

    unanswered = len(row_map) - len(submitted)
    checks.record("rows_answered", True,
                  f"{len(submitted)} of {len(row_map)} exported row(s); "
                  f"{unanswered} unanswered")
    checks.record("verdicts_usable", True,
                  f"{len(out)} answered row(s) carried a non-blank verdict")
    return out


def run(joined: Path, review: Optional[Path], completed: Optional[Path],
        out_dir: Path, procedure: str, write: bool = True,
        expect: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    assert_declared_procedure(procedure)
    checks = Checklist("focused_phase1_adjudication")
    if write:
        assert_output_dir_is_new(out_dir, review)

    joined_path = joined / "private" / JOINED_CELLS_FILE
    cells = index_cells(read_jsonl(joined_path), str(joined_path))
    if expect is not None:
        checks.require("joined_cells", len(cells) == expect["cells"],
                       f"{len(cells)} cell(s), expected {expect['cells']}")

    human = load_review(review, completed, cells, checks)
    needed = set(cells_needing_human_review(cells.values(), procedure))
    stray = sorted(set(human) - needed)
    checks.require(
        "no_review_of_a_cell_this_procedure_does_not_send_to_a_human",
        not stray,
        f"{len(stray)} answered cell(s) the {procedure} procedure does not "
        f"send to a human, first {stray[:3]}")

    adjudicated: List[Dict[str, Any]] = []
    for key in sorted(cells):
        cell = cells[key]
        answer = human.get(key)
        outcome = adjudicate_cell(cell, answer, procedure)
        record = {
            "screen_row_id": cell["screen_row_id"],
            "arm_id": cell["arm_id"],
            "axis": cell["axis"],
            "row_class": cell["row_class"],
            "prompt_sha256": cell["prompt_sha256"],
            "response_sha256": cell["response_sha256"],
            "sonnet_verdict": cell["verdicts"].get(JUDGES[0]),
            "opus_verdict": cell["verdicts"].get(JUDGES[1]),
            "sonnet_record_sha256": cell["record_sha256"].get(JUDGES[0]),
            "opus_record_sha256": cell["record_sha256"].get(JUDGES[1]),
            "human_row_id": (answer or {}).get("human_row_id"),
            "judged_row_id": cell["row_id"],
            **outcome,
        }
        for field_name in CARRIED_CELL_FIELDS:
            record[field_name] = cell.get(field_name)
        record["human_record_sha256"] = record_hash(
            {"row_id": (answer or {}).get("human_row_id"),
             "verdict": outcome["verdict"],
             "confidence": (answer or {}).get("confidence", "")})
        adjudicated.append(record)

    merged = index_cells(adjudicated, "this run's adjudicated cells")
    by_source: Counter = Counter(r["label_source"] for r in merged.values())
    unresolved = sorted(k for k, r in merged.items()
                        if r["label_source"] == SOURCE_UNRESOLVED)
    agree = sum(1 for c in cells.values() if c["state"] == "agree")
    disagree = sum(1 for c in cells.values() if c["state"] == "disagree")

    summary: Dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "created_utc": utc_now(),
        "adjudication_procedure": procedure,
        "adjudication_procedure_note":
            ADJUDICATION_PROCEDURE_EVIDENCE[procedure]["what"],
        "cells": len(merged),
        "by_label_source": dict(sorted(by_source.items())),
        "by_class": dict(sorted(Counter(r["row_class"]
                                        for r in merged.values()).items())),
        "by_arm": dict(sorted(Counter(r["arm_id"]
                                      for r in merged.values()).items())),
        "judges_agree": {"true": agree, "false": disagree},
        "human_verdicts_used": len(human),
        "cells_this_procedure_sends_to_a_human": len(needed),
        "unresolved": len(unresolved),
        "unresolved_cells": [list(k) for k in unresolved[:50]],
        "unresolved_cells_truncated": max(0, len(unresolved) - 50),
        "complete_adjudication": not unresolved,
        "complete_adjudication_note":
            "every cell carries a decided verdict" if not unresolved else
            f"{len(unresolved)} cell(s) are UNRESOLVED. No reading is produced "
            f"from an incomplete adjudication; report_phase1.py refuses.",
        "rule": {
            "never_invents_a_verdict": True,
            "never_breaks_a_tie_by_majority_over_two_judges": True,
            "never_converts_a_failure_into_a_label": True,
            "human_verdict_stands_where_one_exists": True,
        },
        "inputs": {
            "joined_cells": str(joined_path),
            "joined_cells_sha256": sha256_file(joined_path),
            "completed_review": str(completed) if completed else None,
            "completed_review_sha256": (sha256_file(completed)
                                        if completed else None),
        },
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the adjudication summary")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / ADJUDICATED_CELLS_FILE,
                    [merged[k] for k in sorted(merged)])
        (out_dir / "adjudication_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[adjudicate] wrote {out_dir}")
    print(f"[adjudicate] procedure {procedure}")
    for source, count in sorted(by_source.items()):
        print(f"[adjudicate] {source:<22} {count}")
    print(f"[adjudicate] {'cells':<22} {len(merged)}")
    if unresolved:
        print(f"[adjudicate] {len(unresolved)} cell(s) UNRESOLVED; no reading "
              f"is produced from an incomplete adjudication", file=sys.stderr)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedure", required=True,
                        choices=list(ADJUDICATION_PROCEDURES),
                        help="the declared adjudication procedure; no default")
    parser.add_argument("--joined-dir", type=Path, default=None)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--completed", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = judging_root() / "judging"
        joined = args.joined_dir or (root / "joined")
        out_dir = args.output_dir or (root / "adjudicated")
        run(joined, args.review_dir, args.completed, out_dir, args.procedure,
            write=not args.dry_run,
            expect={"cells": analysable_cells(load_technical_exclusions())})
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
