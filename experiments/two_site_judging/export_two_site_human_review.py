#!/usr/bin/env python3
"""Export the blinded two-site human-review packet. Calls nothing, decides nothing.

Which cells it covers depends on the **declared adjudication procedure**, and
there is no default. This stage declares ``corpus_disagreement_only``, so the
reviewer reads the cells the two judges split on. They still never see either
judge's verdict, so they cannot know which way the split went; the *selection*
of these rows does say that the judges disagreed about them, which is a
property of the procedure and is recorded in the manifest.

What the reviewer sees
----------------------
An opaque id, which rubric to apply, a prompt, and either one response (Axis-2,
safety) or two replies labelled A and B (quality). That is all. The export
refuses to write a row carrying any of: the corpus row id, the class, the arm,
either coefficient, the direction, the layers or package states, baseline
versus candidate, **which of A and B is which**, the grouping of rows, the
expected result, or any judge verdict.

The A/B order the reviewer sees is the order the judges saw -- the packet's own
bytes -- so the reviewer is adjudicating the same comparison. The orientation
stays in the packet's private map and is never copied into anything this script
writes that a reviewer can open.

Fresh ids are minted per export, so an answer cannot be joined back to a
judging record except through this export's own private map, and a second
export mints different ones again. Rows are shuffled with an independent
recorded seed so that adjacent rows are not the same corpus row.

A stated limitation
-------------------
The reviewer -- like each judge -- is told which rubric to apply, and here the
rubric is assigned by class. Knowing the rubric therefore implies the class.
That is inherent in the design and cannot be removed without judging every
response on every rubric. Everything the rubric does not already imply is
withheld.

Usage
-----
    python experiments/two_site_judging/export_two_site_human_review.py \\
        --procedure corpus_disagreement_only --dry-run
    python experiments/two_site_judging/export_two_site_human_review.py \\
        --procedure corpus_disagreement_only --seed 20260827
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.twositejudging import (                              # noqa: E402
    ADJUDICATION_PROCEDURES,
    ADJUDICATION_PROCEDURE_EVIDENCE,
    CELL_DIGEST_FIELDS,
    FORBIDDEN_ROW_FIELDS,
    HUMAN_ROW_MAP_FILE,
    HUMAN_ROW_PREFIX,
    JOINED_CELLS_FILE,
    PAIRWISE_RUBRICS,
    PROCEDURE_CORPUS,
    PROCEDURE_INDEPENDENT,
    PROTOCOL_NAME,
    RUBRICS_USED,
    STAGE_NAME,
    TwoSiteJudgingError,
    assert_declared_procedure,
    assert_ids_unique,
    assert_no_corpus_row_id,
    assert_output_dir_is_new,
    assert_row_has_no_forbidden_field,
    cell_key,
    cells_needing_human_review,
    index_cells,
    judging_root,
    load_prompt_texts,
    load_quality_protocol,
    new_opaque_ids,
    quality_protocol_dir,
)
from routegate.triplets import (                                    # noqa: E402
    AXIS_VALUES,
    AXIS_VERDICT_FIELD,
    JUDGE_ROW_FIELDS,
    QUALITY,
    QUALITY_DIFFERENCE_VALUES,
    QUALITY_PREFERENCE_VALUES,
    QUALITY_VERDICT_FIELDS,
    assert_no_text_fields,
    assert_row_is_blind,
    load_protocol,
    read_jsonl,
    working_dir,
    write_jsonl,
)
from routegate.validation import Checklist, ValidationError         # noqa: E402

#: Fields that must never reach the reviewer's file. A superset of the judge
#: list: a human reads prose a judge would ignore, so every name a joined cell
#: can carry is here and a field added upstream is caught rather than shipped.
FORBIDDEN = tuple(sorted(set(FORBIDDEN_ROW_FIELDS) | {
    "screen_row_id", "row_class", "arm_id", "state", "verdicts", "agree",
    "result_formats", "pairing_consistent", "response_a_origin",
    "response_b_origin",
}))

#: Exactly what a reviewer row carries, per rubric. The judge's whitelist plus
#: the name of the task, and nothing else.
REVIEWER_ROW_FIELDS: Dict[str, tuple] = {
    rubric: tuple(sorted(set(JUDGE_ROW_FIELDS[rubric]) | {"rubric_task"}))
    for rubric in RUBRICS_USED}


def load_packet_text(packet: Path) -> Dict[str, Dict[str, str]]:
    """``packet row_id -> the judge-facing strings``, from the PUBLIC files.

    The public judge-input files are the bytes the judges actually saw.
    Reading the text from there rather than re-deriving it means the reviewer
    reads exactly the same text in exactly the same A/B order, and a divergence
    between the two would have failed the packet build.
    """
    out: Dict[str, Dict[str, str]] = {}
    for rubric in RUBRICS_USED:
        path = packet / f"{rubric}_judge_input.jsonl"
        if not path.is_file():
            raise TwoSiteJudgingError(f"{path} is absent from the packet")
        for row in read_jsonl(path):
            assert_row_is_blind(row, rubric)
            if row["row_id"] in out:
                raise TwoSiteJudgingError(
                    f"{path}: {row['row_id']!r} appears in more than one "
                    f"judge-input file")
            out[row["row_id"]] = {k: v for k, v in row.items()
                                  if k != "row_id"}
    return out


def reviewer_instructions(rubrics: Sequence[str], protocol_dir: Path,
                          quality_dir: Path) -> Dict[str, Any]:
    """The frozen judging instructions, for the human, per rubric task.

    A reviewer shown only a task name and a label list would be inventing a
    rubric, and the human/judge comparison would not be one. The instructions
    name no class, arm, coefficient, condition or expected verdict, so this
    un-blinds nothing the rubric assignment does not already imply.
    """
    out: Dict[str, Any] = {}
    for rubric in sorted(set(rubrics)):
        if rubric == QUALITY:
            protocol = load_quality_protocol(quality_dir)
            spec: Dict[str, Any] = {
                "instructions": protocol.instructions,
                "verdict_fields": list(QUALITY_VERDICT_FIELDS),
                "allowed_preference": list(QUALITY_PREFERENCE_VALUES),
                "allowed_difference": list(QUALITY_DIFFERENCE_VALUES),
                "answer_in": "the A/B vocabulary the rubric defines. Which "
                             "reply came from where is not shown to you, is "
                             "not recoverable, and is decoded after the review "
                             "closes.",
            }
        else:
            protocol = load_protocol(rubric, directory=protocol_dir)
            spec = {
                "instructions": protocol.instructions,
                "verdict_field": AXIS_VERDICT_FIELD[rubric],
                "allowed_labels": list(protocol.allowed_values),
            }
        spec["instructions_sha256"] = protocol.instructions_sha256
        spec["schema_sha256"] = protocol.schema_sha256
        out[rubric] = spec
    return out


def export(joined: Path, packet: Path, out_dir: Path, procedure: str,
           seed: int, protocol_dir: Path, quality_dir: Path,
           dataset: Optional[Path], write: bool = True) -> Dict[str, Any]:
    assert_declared_procedure(procedure)
    checks = Checklist(f"{STAGE_NAME}_human_review_export")
    if write:
        assert_output_dir_is_new(out_dir, (joined, packet), "the review export")

    joined_path = joined / "private" / JOINED_CELLS_FILE
    if not joined_path.is_file():
        raise TwoSiteJudgingError(
            f"{joined_path} is not a file. --joined-dir names the directory "
            f"join_two_site_verdicts.py wrote.")
    cells = index_cells(read_jsonl(joined_path), str(joined_path))
    text = load_packet_text(packet)

    wanted = cells_needing_human_review(cells.values(), procedure)
    checks.record("cells_selected", True,
                  f"{len(wanted)} of {len(cells)} cell(s) under {procedure}")
    if not wanted:
        raise TwoSiteJudgingError(
            f"the {procedure} procedure selects no cell from this join. Under "
            f"{PROCEDURE_CORPUS} that means the two judges agreed everywhere, "
            f"which is a finding: record it and adjudicate without a review, "
            f"rather than exporting an empty packet.")

    # Deterministic in the seed, and independent of how the joiner happened to
    # order its file: sort by identity first, then shuffle.
    ordered = [cells[key] for key in wanted]
    rng = random.Random(seed)
    rng.shuffle(ordered)

    ids = new_opaque_ids(HUMAN_ROW_PREFIX, len(ordered))
    assert_ids_unique(ids)

    # Every corpus row id the reviewer's text is scanned for. The join's own
    # ids are always in scope; the whole corpus is added when it is reachable,
    # because a response can name a row other than its own.
    scan_ids = {str(key[0]) for key in cells}
    scope = "the join's own corpus row ids"
    if dataset is not None:
        scan_ids |= set(load_prompt_texts(dataset, Checklist("blindness")))
        scope = "every corpus row id, and the join's own"
    screen_row_ids = sorted(scan_ids)
    checks.record("blindness_scan_scope", True,
                  f"{scope}: {len(screen_row_ids)} synthetic identifier(s)")
    public: List[Dict[str, Any]] = []
    template: List[Dict[str, Any]] = []
    private: List[Dict[str, Any]] = []
    rubrics_present: List[str] = []
    for row_id, cell in zip(ids, ordered):
        key = cell_key(cell, str(joined_path))
        rubric = cell["rubric"]
        rubrics_present.append(rubric)
        payload = text.get(cell["row_id"])
        if payload is None:
            raise TwoSiteJudgingError(
                f"{key}: the packet holds no text for judged row "
                f"{cell['row_id']!r}. The reviewer reads the same bytes the "
                f"judges read; there is no other source.")
        record = {"row_id": row_id, "rubric_task": rubric, **payload}
        leaked = sorted(set(record) & set(FORBIDDEN))
        if leaked:
            raise TwoSiteJudgingError(
                f"the reviewer row for {key} carries {leaked}")
        assert_row_has_no_forbidden_field(record, f"reviewer row {row_id}")
        if set(record) != set(REVIEWER_ROW_FIELDS[rubric]):
            raise TwoSiteJudgingError(
                f"the reviewer row for {key} carries {sorted(record)} and a "
                f"{rubric} reviewer row is exactly "
                f"{list(REVIEWER_ROW_FIELDS[rubric])}")
        # Synthetic identifiers only. The reviewer's text is the packet's text,
        # byte for byte; what a prompt or a response happens to say is never a
        # reason to reject it.
        for field_name, value in sorted(record.items()):
            if field_name in ("row_id", "rubric_task"):
                continue
            assert_no_corpus_row_id(value, screen_row_ids,
                                    f"reviewer row {row_id} {field_name}")
        public.append(record)
        if rubric in PAIRWISE_RUBRICS:
            template.append({"row_id": row_id, "preference": "",
                             "difference": "", "note": ""})
        else:
            template.append({"row_id": row_id, "verdict": "",
                             "confidence": "", "note": ""})
        entry = {
            "row_id": row_id,
            "screen_row_id": cell["screen_row_id"],
            "arm_id": cell["arm_id"],
            "rubric": rubric,
            "row_class": cell["row_class"],
            "judged_row_id": cell["row_id"],
        }
        for field_name in CELL_DIGEST_FIELDS[rubric]:
            entry[field_name] = cell[field_name]
        private.append(entry)

    checks.require("reviewer_sees_no_verdict",
                   all(not (set(r) & set(FORBIDDEN)) for r in public),
                   "no reviewer row carries a verdict, a class, an arm, a "
                   "coefficient or an A/B origin")
    odd = [r["row_id"] for r in public
           if set(r) != set(REVIEWER_ROW_FIELDS[r["rubric_task"]])]
    checks.require("reviewer_row_fields", not odd,
                   f"{len(odd)} row(s) do not carry exactly their rubric's "
                   f"reviewer fields, first {odd[:3]}")
    checks.record(
        "reviewer_sees_the_judges_ab_order", True,
        "the A/B order is the packet's own bytes, so the reviewer adjudicates "
        "the same comparison; the orientation itself stays in the packet's "
        "private map and is never copied here")

    instructions = reviewer_instructions(rubrics_present, protocol_dir,
                                         quality_dir)
    checks.record("reviewer_gets_the_frozen_instructions", True,
                  ", ".join(f"{r}={spec['instructions_sha256'][:12]}"
                            for r, spec in sorted(instructions.items())))

    summary: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "protocol": PROTOCOL_NAME,
        "created_utc": utc_now(),
        "adjudication_procedure": procedure,
        "adjudication_procedure_note":
            ADJUDICATION_PROCEDURE_EVIDENCE[procedure]["what"],
        "selection": ("every joined cell" if procedure == PROCEDURE_INDEPENDENT
                      else "the cells the two judges split on"),
        "selection_discloses": (
            "nothing beyond the rubric" if procedure == PROCEDURE_INDEPENDENT
            else "that the two judges disagreed about these rows; it does not "
                 "disclose either verdict or which way the split went"),
        "cells_in_join": len(cells),
        "rows_exported": len(public),
        "rows_by_rubric_task": {r: sum(1 for x in public
                                       if x["rubric_task"] == r)
                                for r in sorted(set(rubrics_present))},
        "shuffle_seed": seed,
        "blindness_scan_scope": scope,
        "blindness_scan_ids": len(screen_row_ids),
        "reviewer_sees": {r: list(REVIEWER_ROW_FIELDS[r])
                          for r in sorted(set(rubrics_present))},
        "reviewer_does_not_see": list(FORBIDDEN),
        "instruction_hashes": {r: spec["instructions_sha256"]
                               for r, spec in sorted(instructions.items())},
        "allowed_labels": {r: list(AXIS_VALUES[r])
                           for r in sorted(set(rubrics_present))},
        "quality_answers_in_ab_space": True,
        "quality_answers_in_ab_space_note":
            "the reviewer answers A_BETTER / B_BETTER / EQUIVALENT, exactly as "
            "the judges did. The answer is decoded into BEFORE_BETTER / "
            "AFTER_BETTER at import, through the packet's private map, which "
            "no reviewer-facing file carries.",
        "private_map": f"private/{HUMAN_ROW_MAP_FILE}, machine-local, never "
                       f"committed",
        "fresh_ids_per_export": True,
        "fresh_ids_note": "an answer cannot be joined back to a judging record "
                          "except through this export's own private map, and a "
                          "second export mints different ids again",
        "rubric_implies_class": True,
        "rubric_implies_class_note":
            "Axis-2 is applied to E, quality to B and safety to H, so being "
            "told which rubric to apply implies the class. Everything the "
            "rubric does not already imply is withheld.",
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the review export manifest")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        write_jsonl(out_dir / "human_review_input.jsonl", public)
        write_jsonl(out_dir / "human_review_TEMPLATE.jsonl", template)
        write_jsonl(out_dir / "private" / HUMAN_ROW_MAP_FILE, private)
        (out_dir / "private" / HUMAN_ROW_MAP_FILE).chmod(0o600)
        (out_dir / "reviewer_instructions.json").write_text(
            json.dumps({
                "how_to_use": "For each row in human_review_input.jsonl, read "
                              "its rubric_task, apply the instructions for "
                              "that task below to the row's text, and write "
                              "your answer into the matching row_id of "
                              "human_review_TEMPLATE.jsonl. Axis-2 and safety "
                              "rows take a single verdict; quality rows take a "
                              "preference and a difference. Use only the "
                              "labels listed for that task. Leave a row blank "
                              "rather than guessing: a blank row is recorded "
                              "as UNRESOLVED, which is a finding, and a guess "
                              "is not.",
                "tasks": instructions}, indent=2, sort_keys=True),
            encoding="utf-8")
        (out_dir / "human_review_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[review] wrote {out_dir}")
    print(f"[review] procedure {procedure}")
    print(f"[review] {len(public)} row(s) of {len(cells)} cell(s)")
    for rubric, count in sorted(summary["rows_by_rubric_task"].items()):
        print(f"[review] {rubric:<8} {count}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedure", required=True,
                        choices=list(ADJUDICATION_PROCEDURES),
                        help="the declared adjudication procedure; no default")
    parser.add_argument("--joined-dir", type=Path, default=None)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--quality-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = judging_root() / "judging"
        joined = args.joined_dir or (root / "joined")
        packet = args.packet or (root / "packet")
        out_dir = args.output_dir or (root / "human_review")
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        quality_dir = args.quality_dir or quality_protocol_dir()
        dataset = args.dataset
        if dataset is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            dataset = Path(paths["dataset"])
        export(joined, packet, out_dir, args.procedure, args.seed,
               protocol_dir, quality_dir, dataset, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
