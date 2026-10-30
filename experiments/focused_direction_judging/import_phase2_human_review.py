#!/usr/bin/env python3
"""Import the completed phase-2 human review and produce the adjudication.

Runs after the reviewer has finished. Only now are the judges' verdicts and
their disagreements read together with the human's: until this point the
disagreements were private.

Same rule, same vocabulary, same validation as phase 1. The adjudication rule
itself is ``routegate.focusedjudging.adjudicate_cell`` -- imported, not
reimplemented -- applied against the phase-2 arm table.

The adjudication procedure is **declared, not defaulted**
---------------------------------------------------------
``--procedure`` is required, is written into every adjudicated record, and must
match the value declared in
``manifests/focused_direction_validation_v1_judging.json``. That declaration
was made before the first billed call of phase 1 and is unchanged:
``corpus_disagreement_only``.

``corpus_disagreement_only``
    ``DUAL_LLM_AGREEMENT`` when Sonnet and Opus agree. ``HUMAN_ADJUDICATION``
    when they disagree and a human answered, or when a human answered
    differently from an agreeing pair. ``UNRESOLVED`` when they disagree and no
    human answered, or when a judge verdict is missing.

The importer never invents a verdict, never breaks a tie by majority over two
judges, and never converts a recorded failure into a label. A missing judge
verdict is an unresolved **technical** failure and stays one: it is never
replaced by a human's reading, because a human answer to a cell one judge never
answered is not the same evidence as two judges agreeing.

Every joined cell is adjudicated, not only the reviewed ones, so the evidence
file is a complete statement about all 144 phase-2 cells rather than a silently
shorter one. The 36 inherited phase-1 reference cells are **not** here: they
were adjudicated in phase 1 and are read, never re-adjudicated.

Usage
-----
    python experiments/focused_direction_judging/import_phase2_human_review.py \\
        --procedure corpus_disagreement_only --dry-run

    python experiments/focused_direction_judging/import_phase2_human_review.py \\
        --procedure corpus_disagreement_only \\
        --review-dir $FDV/phase2/judging/human_review \\
        --completed $FDV/phase2/judging/human_review/human_review_COMPLETED.jsonl \\
        --output-dir $FDV/phase2/judging/adjudicated
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
    HUMAN_ROW_MAP_FILE,
    JOINED_CELLS_FILE,
    adjudicate_cell,
    assert_declared_procedure,
    assert_same_cell,
    cells_needing_human_review,
)
from routegate.phase2judging import (                               # noqa: E402
    FROZEN_COVERAGE,
    JUDGING_SUBDIR,
    PHASE2_CLASS_ARMS,
    PHASE2_REFERENCE_ARM_IDS,
    PROTOCOL_NAME,
    Phase2JudgingError,
    index_phase2_cells,
    phase2_cell_key,
    phase2_root,
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
    """The adjudication goes somewhere new, and never over what it reads."""
    if review is not None:
        review_dir = (review if review.is_dir() else review.parent).resolve()
        if out_dir.resolve() == review_dir:
            raise Phase2JudgingError(
                f"--output-dir resolves to the review directory {review_dir}. "
                f"A completed human review is not reconstructible; it is never "
                f"written over.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise Phase2JudgingError(
            f"{out_dir} already holds files. An adjudication is written to a "
            f"new directory so an earlier one is never partially overwritten.")


def load_review(review: Optional[Path], completed: Optional[Path],
                cells: Mapping[Tuple[str, str, str], Mapping[str, Any]],
                checks: Checklist
                ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """``cell identity -> the human's answer``, checked against the join."""
    if review is None and completed is None:
        checks.record("human_review", True,
                      "no human review was supplied; every cell that needs one "
                      "under this procedure will be UNRESOLVED")
        return {}
    if review is None or completed is None:
        raise Phase2JudgingError(
            "--review-dir and --completed are supplied together: the map says "
            "which cell a row is, the completed file says what the reviewer "
            "answered, and neither is usable alone.")

    map_path = review / "private" / HUMAN_ROW_MAP_FILE
    if not map_path.is_file():
        raise Phase2JudgingError(f"{map_path} is not a file")
    row_map = {r["row_id"]: r for r in read_jsonl(map_path)}
    if not row_map:
        raise Phase2JudgingError(f"{map_path} holds no row")

    submitted: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(completed):
        row_id = row.get("row_id")
        if row_id not in row_map:
            raise Phase2JudgingError(
                f"the completed review carries {row_id!r}, which is not a row "
                f"of the exported packet")
        if row_id in submitted:
            raise Phase2JudgingError(f"{row_id} is answered twice")
        confidence = row.get("confidence") or ""
        if confidence and confidence not in CONFIDENCE_VALUES:
            raise Phase2JudgingError(
                f"{row_id}: confidence {confidence!r} is not one of "
                f"{list(CONFIDENCE_VALUES)}")
        submitted[row_id] = row

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row_id, mapping in sorted(row_map.items()):
        key = phase2_cell_key(mapping, str(map_path))
        cell = cells.get(key)
        if cell is None:
            raise Phase2JudgingError(
                f"{key} was exported for review but this join holds no such "
                f"cell. A review answered against a different join is refused.")
        assert_same_cell(mapping, cell, str(map_path), "the joined cell",
                         arms=PHASE2_CLASS_ARMS)
        answer = submitted.get(row_id)
        if answer is None:
            continue
        verdict = answer.get("verdict") or None
        if verdict is None:
            continue
        if key in out:
            raise Phase2JudgingError(f"{key} is answered twice")
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
    checks = Checklist("focused_phase2_adjudication")
    if write:
        assert_output_dir_is_new(out_dir, review)

    joined_path = joined / "private" / JOINED_CELLS_FILE
    cells = index_phase2_cells(read_jsonl(joined_path), str(joined_path))
    expect = expect if expect is not None else FROZEN_COVERAGE
    checks.require("joined_cells", len(cells) == expect["cells"],
                   f"{len(cells)} cell(s), expected {expect['cells']}")
    stray_arms = sorted({key[1] for key in cells}
                        & set(PHASE2_REFERENCE_ARM_IDS))
    checks.require(
        "no_inherited_reference_arm_is_adjudicated_here", not stray_arms,
        f"{stray_arms}; {list(PHASE2_REFERENCE_ARM_IDS)} were adjudicated in "
        f"phase 1 and are read, never re-adjudicated")

    human = load_review(review, completed, cells, checks)
    needed = set(cells_needing_human_review(cells.values(), procedure,
                                            arms=PHASE2_CLASS_ARMS))
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
        outcome = adjudicate_cell(cell, answer, procedure,
                                  arms=PHASE2_CLASS_ARMS)
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

    merged = index_phase2_cells(adjudicated, "this run's adjudicated cells")
    by_source: Counter = Counter(r["label_source"] for r in merged.values())
    unresolved = sorted(k for k, r in merged.items()
                        if r["label_source"] == SOURCE_UNRESOLVED)
    agree = sum(1 for c in cells.values() if c["state"] == "agree")
    disagree = sum(1 for c in cells.values() if c["state"] == "disagree")

    summary: Dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "phase": 2,
        "created_utc": utc_now(),
        "adjudication_procedure": procedure,
        "adjudication_procedure_note":
            ADJUDICATION_PROCEDURE_EVIDENCE[procedure]["what"],
        "cells": len(merged),
        "by_label_source": dict(sorted(by_source.items())),
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
            f"{len(unresolved)} cell(s) are UNRESOLVED. Under the frozen "
            f"phase-2 decision rule an undecided row makes its whole strength "
            f"undetermined, and report_phase2.py reports that rather than a "
            f"count taken over what happens to be decided.",
        "inherited_reference_cells_adjudicated_here": 0,
        "inherited_reference_cells_note":
            "baseline and repair_015 for these 18 rows were adjudicated in "
            "phase 1 under this same procedure. They are read by "
            "report_phase2.py and are never re-judged or re-adjudicated.",
        "rule": {
            "never_invents_a_verdict": True,
            "never_breaks_a_tie_by_majority_over_two_judges": True,
            "never_converts_a_failure_into_a_label": True,
            "human_verdict_stands_where_one_exists": True,
            "missing_judge_verdict_is_an_unresolved_technical_failure": True,
            "implemented_in": "routegate.focusedjudging.adjudicate_cell",
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
    assert_no_text_fields(summary, "the phase-2 adjudication summary")

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
        print(f"[adjudicate] {len(unresolved)} cell(s) UNRESOLVED; every "
              f"strength holding one is UNDETERMINED under the frozen rule",
              file=sys.stderr)
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
        root = phase2_root() / JUDGING_SUBDIR
        joined = args.joined_dir or (root / "joined")
        out_dir = args.output_dir or (root / "adjudicated")
        run(joined, args.review_dir, args.completed, out_dir, args.procedure,
            write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
