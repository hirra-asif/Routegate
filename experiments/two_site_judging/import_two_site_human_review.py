#!/usr/bin/env python3
"""Import the completed review and produce the two-site adjudication.

Runs after the reviewer has finished. Only now are the judges' verdicts and
their disagreements read together with the human's: until this point the
disagreements were private.

The adjudication procedure is **declared, not defaulted**
---------------------------------------------------------
This project's frozen record holds two procedures. They share the label
vocabulary and the rule that a human verdict, where one exists, stands. They
differ in which cells a human reads, and that difference changes the result.
``--procedure`` is required, is written into every adjudicated record, and must
match the value declared in ``manifests/two_site_actuator_judging_v1.json`` --
``corpus_disagreement_only``.

The rule itself is ``routegate.focusedjudging.adjudicate_cell``, reached
through ``routegate.twositejudging.adjudicate``. It is **imported and called,
never reimplemented and never altered**. It never invents a verdict, never
breaks a tie by majority over two judges, and never converts a recorded failure
into a label. A gap is a gap: it is not a quiet pass and not a quiet fail.

Decoding the reviewer's quality answers
---------------------------------------
The reviewer answered in the same A/B space the judges did. Their answer is
decoded here, through the **packet's** private orientation map, exactly as the
join decoded the judges'. That is why ``--packet`` is required: the review
packet deliberately carries no orientation of its own.

Every joined cell is adjudicated, not only the reviewed ones. A cell nobody
reviewed still gets a record -- with whatever label source it earns -- so the
evidence file is a complete statement about all 480 cells rather than a
silently shorter one.

Usage
-----
    python experiments/two_site_judging/import_two_site_human_review.py \\
        --procedure corpus_disagreement_only --dry-run

    python experiments/two_site_judging/import_two_site_human_review.py \\
        --procedure corpus_disagreement_only \\
        --review-dir $TSJ/judging/human_review \\
        --completed $TSJ/judging/human_review/human_review_COMPLETED.jsonl
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
from routegate.twositejudging import (                              # noqa: E402
    ADJUDICATED_CELLS_FILE,
    ADJUDICATION_PROCEDURES,
    ADJUDICATION_PROCEDURE_EVIDENCE,
    CARRIED_CELL_FIELDS,
    CELL_DIGEST_FIELDS,
    FROZEN_COVERAGE,
    HUMAN_ROW_MAP_FILE,
    JOINED_CELLS_FILE,
    PAIRWISE_RUBRICS,
    POSITION_A,
    POSITION_B,
    PROTOCOL_NAME,
    ROW_MAP_FILE,
    STAGE_NAME,
    TwoSiteJudgingError,
    adjudicate,
    assert_declared_procedure,
    assert_orientation,
    assert_output_dir_is_new,
    assert_same_cell,
    cell_key,
    cells_needing_human_review,
    compose_quality_verdict,
    decode_preference,
    index_cells,
    judging_root,
    quality_pairing_is_consistent,
    split_quality_verdict,
)
from routegate.triplets import (                                    # noqa: E402
    CONFIDENCE_VALUES,
    JUDGES,
    QUALITY,
    QUALITY_DIFFERENCE_VALUES,
    QUALITY_PREFERENCE_VALUES,
    SOURCE_UNRESOLVED,
    assert_no_text_fields,
    read_jsonl,
    record_hash,
    write_jsonl,
)
from routegate.twositeunresolved import (                           # noqa: E402
    STATE_UNRESOLVED_TECHNICAL,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
)


def load_packet_orientation(packet: Path) -> Dict[str, Dict[str, str]]:
    """``judged row_id -> orientation``, from the packet's private map.

    The one place the A/B assignment lives. Read here so a reviewer's A/B
    answer can be decoded the same way a judge's was.
    """
    path = packet / "private" / ROW_MAP_FILE
    if not path.is_file():
        raise TwoSiteJudgingError(
            f"{path} is not a file. --packet names the directory the two-site "
            f"packet was built into; the review packet deliberately carries no "
            f"orientation of its own.")
    out: Dict[str, Dict[str, str]] = {}
    for entry in read_jsonl(path):
        if entry.get("rubric") not in PAIRWISE_RUBRICS:
            continue
        out[str(entry["row_id"])] = assert_orientation(
            {POSITION_A: entry["response_a_origin"],
             POSITION_B: entry["response_b_origin"]},
            f"{path}: row {entry['row_id']!r}")
    return out


def read_answer(row: Mapping[str, Any], rubric: str, row_id: str,
                orientation: Optional[Mapping[str, str]]
                ) -> Optional[Dict[str, Any]]:
    """One reviewer answer, validated and -- for quality -- decoded.

    Returns ``None`` when the reviewer left the row blank. A blank row is
    recorded as UNRESOLVED, which is a finding; it is never guessed at.
    """
    if rubric not in PAIRWISE_RUBRICS:
        verdict = row.get("verdict") or None
        if verdict is None:
            return None
        confidence = row.get("confidence") or ""
        if confidence and confidence not in CONFIDENCE_VALUES:
            raise TwoSiteJudgingError(
                f"{row_id}: confidence {confidence!r} is not one of "
                f"{list(CONFIDENCE_VALUES)}")
        return {"verdict": verdict, "confidence": confidence,
                "human_row_id": row_id}

    preference = row.get("preference") or None
    difference = row.get("difference") or None
    if preference is None and difference is None:
        return None
    if preference is None or difference is None:
        raise TwoSiteJudgingError(
            f"{row_id}: a quality answer carries a preference and a "
            f"difference. This one carries {'a preference' if preference else 'a difference'} "
            f"only, and half a decision is not a decision.")
    if preference not in QUALITY_PREFERENCE_VALUES:
        raise TwoSiteJudgingError(
            f"{row_id}: preference {preference!r} is not one of "
            f"{list(QUALITY_PREFERENCE_VALUES)}")
    if difference not in QUALITY_DIFFERENCE_VALUES:
        raise TwoSiteJudgingError(
            f"{row_id}: difference {difference!r} is not one of "
            f"{list(QUALITY_DIFFERENCE_VALUES)}")
    if orientation is None:
        raise TwoSiteJudgingError(
            f"{row_id}: no orientation is known for the judged row this "
            f"answer belongs to, so an A/B answer cannot be decoded. Refused "
            f"rather than decoded by a guess.")
    outcome = decode_preference(preference, orientation, f"{row_id}")
    verdict = compose_quality_verdict(outcome, difference, f"{row_id}")
    return {"verdict": verdict, "confidence": row.get("confidence", "") or "",
            "human_row_id": row_id,
            "human_preference": preference,
            "human_difference": difference,
            "pairing_consistent": quality_pairing_is_consistent(verdict)}


def load_review(review: Optional[Path], completed: Optional[Path],
                cells: Mapping[Tuple[str, str, str], Mapping[str, Any]],
                orientation: Mapping[str, Mapping[str, str]],
                checks: Checklist
                ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """``cell identity -> the human's answer``, checked against the join."""
    if review is None and completed is None:
        checks.record("human_review", True,
                      "no human review was supplied; every cell that needs one "
                      "under this procedure will be UNRESOLVED")
        return {}
    if review is None or completed is None:
        raise TwoSiteJudgingError(
            "--review-dir and --completed are supplied together: the map says "
            "which cell a row is, the completed file says what the reviewer "
            "answered, and neither is usable alone.")

    map_path = review / "private" / HUMAN_ROW_MAP_FILE
    if not map_path.is_file():
        raise TwoSiteJudgingError(f"{map_path} is not a file")
    row_map = {r["row_id"]: r for r in read_jsonl(map_path)}
    if not row_map:
        raise TwoSiteJudgingError(f"{map_path} holds no row")

    submitted: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(completed):
        row_id = row.get("row_id")
        if row_id not in row_map:
            raise TwoSiteJudgingError(
                f"the completed review carries {row_id!r}, which is not a row "
                f"of the exported packet")
        if row_id in submitted:
            raise TwoSiteJudgingError(f"{row_id} is answered twice")
        submitted[row_id] = row

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row_id, mapping in sorted(row_map.items()):
        key = cell_key(mapping, str(map_path))
        cell = cells.get(key)
        if cell is None:
            raise TwoSiteJudgingError(
                f"{key} was exported for review but this join holds no such "
                f"cell. A review answered against a different join is refused.")
        assert_same_cell(mapping, cell, str(map_path), "the joined cell")
        row = submitted.get(row_id)
        if row is None:
            continue
        answer = read_answer(row, key[2], row_id,
                             orientation.get(str(mapping["judged_row_id"])))
        if answer is None:
            continue
        if key in out:
            raise TwoSiteJudgingError(f"{key} is answered twice")
        out[key] = answer

    unanswered = len(row_map) - len(submitted)
    checks.record("rows_answered", True,
                  f"{len(submitted)} of {len(row_map)} exported row(s); "
                  f"{unanswered} unanswered")
    checks.record("verdicts_usable", True,
                  f"{len(out)} answered row(s) carried a usable verdict")
    return out


def run(joined: Path, packet: Path, review: Optional[Path],
        completed: Optional[Path], out_dir: Path, procedure: str,
        write: bool = True, expect: Optional[Mapping[str, Any]] = None
        ) -> Dict[str, Any]:
    assert_declared_procedure(procedure)
    checks = Checklist(f"{STAGE_NAME}_adjudication")
    if write:
        assert_output_dir_is_new(out_dir, (review, completed, joined, packet),
                                 "the adjudication")

    joined_path = joined / "private" / JOINED_CELLS_FILE
    if not joined_path.is_file():
        raise TwoSiteJudgingError(
            f"{joined_path} is not a file. --joined-dir names the directory "
            f"join_two_site_verdicts.py wrote.")
    cells = index_cells(read_jsonl(joined_path), str(joined_path))
    expect = expect if expect is not None else FROZEN_COVERAGE
    checks.require("joined_cells", len(cells) == expect["cells"],
                   f"{len(cells)} cell(s), expected {expect['cells']}")

    orientation = load_packet_orientation(packet)
    checks.record("packet_orientation_loaded", True,
                  f"{len(orientation)} pairwise row(s); the review packet "
                  f"carries no orientation of its own")

    human = load_review(review, completed, cells, orientation, checks)
    needed = set(cells_needing_human_review(cells.values(), procedure))
    stray = sorted(set(human) - needed)
    checks.require(
        "no_review_of_a_cell_this_procedure_does_not_send_to_a_human",
        not stray,
        f"{len(stray)} answered cell(s) the {procedure} procedure does not "
        f"send to a human, first {stray[:3]}")

    # A technically unresolved cell is one a judge never answered. A human
    # cannot stand in for a missing judge verdict: that would decide a cell
    # under a rule invented after the failure was known, which
    # docs/decision_log.md 2026-08-25 rejected. Refused by name, not merely
    # by falling outside the disagreement selection above.
    technical = sorted(k for k, c in cells.items()
                       if c.get("state") == STATE_UNRESOLVED_TECHNICAL
                       or c.get("technically_unresolved") is True)
    answered_technical = sorted(set(human) & set(technical))
    checks.require(
        "no_human_verdict_replaces_a_missing_judge_verdict",
        not answered_technical,
        f"{len(answered_technical)} technically unresolved cell(s) carry a "
        f"human answer, first {answered_technical[:3]}. A human adjudicates a "
        f"disagreement between two judges; it does not supply a verdict a "
        f"judge never gave." if answered_technical else
        f"{len(technical)} technically unresolved cell(s), none answered")

    adjudicated: List[Dict[str, Any]] = []
    for key in sorted(cells):
        cell = cells[key]
        answer = human.get(key)
        outcome = adjudicate(cell, answer, procedure)
        record: Dict[str, Any] = {
            "screen_row_id": cell["screen_row_id"],
            "arm_id": cell["arm_id"],
            "rubric": cell["rubric"],
            "row_class": cell["row_class"],
            "sonnet_verdict": cell["verdicts"].get(JUDGES[0]),
            "opus_verdict": cell["verdicts"].get(JUDGES[1]),
            "sonnet_record_sha256": cell["record_sha256"].get(JUDGES[0]),
            "opus_record_sha256": cell["record_sha256"].get(JUDGES[1]),
            "human_row_id": (answer or {}).get("human_row_id"),
            "judged_row_id": cell["row_id"],
            "technically_unresolved": bool(
                cell.get("technically_unresolved")),
            "missing_judge": cell.get("missing_judge"),
            "failure_type": cell.get("failure_type"),
            **outcome,
        }
        for field_name in CELL_DIGEST_FIELDS[cell["rubric"]]:
            record[field_name] = cell[field_name]
        for field_name in CARRIED_CELL_FIELDS:
            record[field_name] = cell.get(field_name)
        if cell["rubric"] in PAIRWISE_RUBRICS:
            verdict = outcome["verdict"]
            if verdict is None:
                record["final_quality_outcome"] = None
                record["final_difference_strength"] = None
                record["pairing_consistent"] = None
            else:
                outcome_label, strength = split_quality_verdict(verdict)
                record["final_quality_outcome"] = outcome_label
                record["final_difference_strength"] = strength
                record["pairing_consistent"] = \
                    quality_pairing_is_consistent(verdict)
            record["judge_pairing_consistent"] = cell.get("pairing_consistent")
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
        "stage": STAGE_NAME,
        "protocol": PROTOCOL_NAME,
        "created_utc": utc_now(),
        "adjudication_procedure": procedure,
        "adjudication_procedure_note":
            ADJUDICATION_PROCEDURE_EVIDENCE[procedure]["what"],
        "cells": len(merged),
        "by_label_source": dict(sorted(by_source.items())),
        "by_class": dict(sorted(Counter(r["row_class"]
                                        for r in merged.values()).items())),
        "by_rubric": dict(sorted(Counter(r["rubric"]
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
            f"from an incomplete adjudication; report_two_site.py withholds "
            f"the affected class.",
        "rule": {
            "implemented_in": "routegate.focusedjudging.adjudicate_cell",
            "reused": "imported and called unchanged; not copied, not "
                      "reimplemented, not altered",
            "never_invents_a_verdict": True,
            "never_breaks_a_tie_by_majority_over_two_judges": True,
            "never_converts_a_failure_into_a_label": True,
            "human_verdict_stands_where_one_exists": True,
        },
        "quality_decoding": {
            "reviewer_answered_in": "A/B space, exactly as the judges did",
            "decoded_through": "the packet's private orientation map",
            "review_packet_carries_orientation": False,
        },
        "inputs": {
            "joined_cells": str(joined_path),
            "joined_cells_sha256": sha256_file(joined_path),
            "packet": str(packet),
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
        print(f"[adjudicate] {len(unresolved)} cell(s) UNRESOLVED; the "
              f"affected class's reading is withheld", file=sys.stderr)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedure", required=True,
                        choices=list(ADJUDICATION_PROCEDURES),
                        help="the declared adjudication procedure; no default")
    parser.add_argument("--joined-dir", type=Path, default=None)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--completed", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = judging_root() / "judging"
        joined = args.joined_dir or (root / "joined")
        packet = args.packet or (root / "packet")
        out_dir = args.output_dir or (root / "adjudicated")
        run(joined, packet, args.review_dir, args.completed, out_dir,
            args.procedure, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
