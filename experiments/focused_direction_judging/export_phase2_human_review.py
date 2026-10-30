#!/usr/bin/env python3
"""Export the blinded phase-2 human-review packet. Calls nothing, decides nothing.

Same schema and same validation as phase 1. Which cells it covers depends on
the **declared adjudication procedure**, and there is no default:

``corpus_disagreement_only``
    the cells the two judges split on. The reviewer is a tie-breaker by
    design; they still never see either judge's verdict, so they cannot know
    which way the split went, but the *selection* of these rows does say that
    the judges disagreed about them. That is a property of the procedure, not
    of this script, and it is recorded in the manifest.

``triplet_independent_review``
    every joined cell. It is not the procedure this experiment declared, and
    every stage refuses a ``--procedure`` that is not the declared one.

What the reviewer sees
----------------------
A prompt, one response, an opaque id, and which rubric to apply -- the frozen
actionable-harmful-help safety rubric, the only one phase 2 uses. That is all.
The export refuses to write a row carrying any of: the corpus row id, the arm,
either coefficient, the direction, the layers, the grouping of a row's arms,
the expected result, any judge verdict, or the fact that this row is one of the
18 that regressed under the repair direction.

**Neither judge's verdict is in the packet, and neither is their agreement.**
``join_phase2_verdicts.py`` writes those to a private file; this script does
not open it except to read the state that selects rows under the corpus
procedure, and that state never reaches the reviewer's file.

Rows are shuffled with an independent recorded seed, and the ids are freshly
minted so an answer cannot be joined back to a judging record except through
the private map.

A stated limitation
-------------------
The reviewer is told which rubric to apply. In phase 2 there is only one
rubric, so -- unlike phase 1, where the rubric implied the class -- being told
it discloses nothing: every phase-2 cell is an H response. Everything else --
arm, coefficient, safety-only versus combined, which rows belong to the same
corpus row, and the outcome -- is withheld.

Usage
-----
    python experiments/focused_direction_judging/export_phase2_human_review.py \\
        --procedure corpus_disagreement_only --dry-run
    python experiments/focused_direction_judging/export_phase2_human_review.py \\
        --procedure corpus_disagreement_only --seed 20260826
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
from routegate.focusedjudging import (                              # noqa: E402
    ADJUDICATION_PROCEDURES,
    ADJUDICATION_PROCEDURE_EVIDENCE,
    HUMAN_ROW_MAP_FILE,
    JOINED_CELLS_FILE,
    PROCEDURE_CORPUS,
    PROCEDURE_INDEPENDENT,
    assert_declared_procedure,
    assert_no_corpus_row_id,
    cells_needing_human_review,
    load_prompt_texts,
)
from routegate.phase2judging import (                               # noqa: E402
    AXES_USED,
    JUDGING_SUBDIR,
    PHASE2_ARM_IDS,
    PHASE2_AXIS,
    PHASE2_CLASS_ARMS,
    PHASE2_FORBIDDEN_ROW_FIELDS,
    PHASE2_HUMAN_ROW_PREFIX,
    PROTOCOL_NAME,
    Phase2JudgingError,
    assert_phase2_row_has_no_forbidden_field,
    index_phase2_cells,
    phase2_cell_key,
    phase2_root,
)
from routegate.triplets import (                                    # noqa: E402
    AXIS_VALUES,
    assert_ids_unique,
    assert_no_text_fields,
    assert_row_is_blind,
    load_protocol,
    new_opaque_ids,
    read_jsonl,
    working_dir,
    write_jsonl,
)
from routegate.validation import Checklist, ValidationError         # noqa: E402

#: Fields that must never reach the reviewer's file. The phase-2 forbidden set
#: -- which already covers the selection status -- plus the names a joined cell
#: can carry, so a field added upstream is caught rather than shipped.
FORBIDDEN = tuple(sorted(set(PHASE2_FORBIDDEN_ROW_FIELDS) | {
    "result_formats", "judged_row_id", "state", "verdicts", "agree",
    "sonnet_verdict", "opus_verdict", "record_sha256", "row_class",
    "screen_row_id", "arm_id", "axis", "prompt_sha256", "response_sha256",
}))

#: The reviewer file's exact shape. Identical to phase 1's, so one reviewing
#: workflow serves both.
REVIEWER_FIELDS = ("row_id", "axis_task", "prompt_text", "response_text")


def load_packet_text(packet: Path) -> Dict[str, Dict[str, str]]:
    """``packet row_id -> {prompt_text, response_text}`` from the PUBLIC file.

    The public judge-input file is the bytes the judges actually saw. Reading
    the text from there rather than re-deriving it from the corpus and the
    screen records means the reviewer reads exactly the same text, and a
    divergence between the two would have failed the packet build.
    """
    out: Dict[str, Dict[str, str]] = {}
    for axis in AXES_USED:
        path = packet / f"{axis}_judge_input.jsonl"
        if not path.is_file():
            raise Phase2JudgingError(f"{path} is absent from the packet")
        for row in read_jsonl(path):
            assert_row_is_blind(row, axis)
            if row["row_id"] in out:
                raise Phase2JudgingError(
                    f"{path}: {row['row_id']!r} appears more than once")
            out[row["row_id"]] = {"prompt_text": row["prompt_text"],
                                  "response_text": row["response_text"]}
    return out


def reviewer_instructions(protocol_dir: Path) -> Dict[str, Any]:
    """The frozen safety instructions, for the human, as one rubric task.

    A reviewer shown only a task name and a label list would be inventing a
    rubric, and the human/judge comparison would not be one. The instructions
    name no arm, coefficient, condition or expected verdict.
    """
    protocol = load_protocol(PHASE2_AXIS, directory=protocol_dir)
    return {PHASE2_AXIS: {
        "instructions": protocol.instructions,
        "allowed_labels": list(protocol.allowed_values),
        "verdict_field": protocol.verdict_field,
        "instructions_sha256": protocol.instructions_sha256,
        "schema_sha256": protocol.schema_sha256,
    }}


def export(joined: Path, packet: Path, out_dir: Path, procedure: str,
           seed: int, protocol_dir: Path, dataset: Path,
           write: bool = True) -> Dict[str, Any]:
    assert_declared_procedure(procedure)
    checks = Checklist("focused_phase2_human_review_export")
    joined_path = joined / "private" / JOINED_CELLS_FILE
    cells = index_phase2_cells(read_jsonl(joined_path), str(joined_path))
    text = load_packet_text(packet)

    wanted = cells_needing_human_review(cells.values(), procedure,
                                        arms=PHASE2_CLASS_ARMS)
    checks.record("cells_selected", True,
                  f"{len(wanted)} of {len(cells)} cell(s) under {procedure}")
    if not wanted:
        raise Phase2JudgingError(
            f"the {procedure} procedure selects no cell from this join. Under "
            f"{PROCEDURE_CORPUS} that means the two judges agreed everywhere, "
            f"which is a finding: record it and adjudicate without a review, "
            f"rather than exporting an empty packet.")

    # Deterministic in the seed, and independent of how the joiner happened to
    # order its file: sort by identity first, then shuffle.
    ordered = [cells[key] for key in wanted]
    rng = random.Random(seed)
    rng.shuffle(ordered)

    ids = new_opaque_ids(PHASE2_HUMAN_ROW_PREFIX, len(ordered))
    assert_ids_unique(ids)

    screen_row_ids = sorted(load_prompt_texts(dataset, Checklist("blindness")))
    public: List[Dict[str, Any]] = []
    template: List[Dict[str, Any]] = []
    private: List[Dict[str, Any]] = []
    for row_id, cell in zip(ids, ordered):
        key = phase2_cell_key(cell, str(joined_path))
        payload = text.get(cell["row_id"])
        if payload is None:
            raise Phase2JudgingError(
                f"{key}: the packet holds no text for judged row "
                f"{cell['row_id']!r}. The reviewer reads the same bytes the "
                f"judges read; there is no other source.")
        record = {"row_id": row_id, "axis_task": cell["axis"], **payload}
        leaked = sorted(set(record) & set(FORBIDDEN))
        if leaked:
            raise Phase2JudgingError(
                f"the reviewer row for {key} carries {leaked}")
        assert_phase2_row_has_no_forbidden_field(record,
                                                 f"reviewer row {row_id}")
        # Synthetic identifiers only. The reviewer's text is the packet's text,
        # byte for byte; what a prompt or a response happens to say is never a
        # reason to reject it.
        assert_no_corpus_row_id(record["prompt_text"], screen_row_ids,
                                f"reviewer row {row_id} prompt")
        assert_no_corpus_row_id(record["response_text"], screen_row_ids,
                                f"reviewer row {row_id} response")
        public.append(record)
        template.append({"row_id": row_id, "verdict": "", "confidence": "",
                         "note": ""})
        private.append({
            "row_id": row_id,
            "screen_row_id": cell["screen_row_id"],
            "arm_id": cell["arm_id"],
            "axis": cell["axis"],
            "row_class": cell["row_class"],
            "prompt_sha256": cell["prompt_sha256"],
            "response_sha256": cell["response_sha256"],
            "judged_row_id": cell["row_id"],
        })

    checks.require("reviewer_sees_no_verdict",
                   all(not (set(r) & set(FORBIDDEN)) for r in public),
                   "no reviewer row carries a verdict, an arm, a coefficient "
                   "or the fact that its row was selected")
    odd = [r["row_id"] for r in public if set(r) != set(REVIEWER_FIELDS)]
    checks.require("reviewer_row_fields", not odd,
                   f"{len(odd)} row(s) do not carry exactly "
                   f"{sorted(REVIEWER_FIELDS)}, first {odd[:3]}")
    exported_arms = sorted({e["arm_id"] for e in private})
    checks.require(
        "every_exported_cell_is_a_generated_arm",
        set(exported_arms) <= set(PHASE2_ARM_IDS),
        f"{exported_arms}; the inherited phase-1 reference arms are never "
        f"reviewed here")

    instructions = reviewer_instructions(protocol_dir)
    checks.record("reviewer_gets_the_frozen_instructions", True,
                  ", ".join(f"{a}={spec['instructions_sha256'][:12]}"
                            for a, spec in sorted(instructions.items())))

    summary: Dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "phase": 2,
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
        "rows_by_axis_task": {PHASE2_AXIS: len(public)},
        "shuffle_seed": seed,
        "reviewer_sees": list(REVIEWER_FIELDS),
        "reviewer_does_not_see": sorted(FORBIDDEN),
        "instruction_hashes": {axis: spec["instructions_sha256"]
                               for axis, spec in sorted(instructions.items())},
        "allowed_labels": {PHASE2_AXIS: list(AXIS_VALUES[PHASE2_AXIS])},
        "private_map": f"private/{HUMAN_ROW_MAP_FILE}, machine-local, never "
                       f"committed",
        "rubric_implies_class": False,
        "rubric_implies_class_note":
            "phase 2 has one rubric and one class, so being told which rubric "
            "to apply discloses nothing a reviewer could not infer from the "
            "packet existing. The arm, the coefficients, the safety-only "
            "versus combined condition, the row grouping and the outcome are "
            "withheld -- as is the fact that every row here regressed under "
            "the repair direction.",
        "same_schema_as_phase1": True,
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the phase-2 review export manifest")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        write_jsonl(out_dir / "human_review_input.jsonl", public)
        write_jsonl(out_dir / "human_review_TEMPLATE.jsonl", template)
        write_jsonl(out_dir / "private" / HUMAN_ROW_MAP_FILE, private)
        (out_dir / "reviewer_instructions.json").write_text(
            json.dumps({
                "how_to_use": "For each row in human_review_input.jsonl, read "
                              "its axis_task, apply the instructions for that "
                              "task below to the prompt and response in that "
                              "row, and write the label into the matching "
                              "row_id of human_review_TEMPLATE.jsonl. Use only "
                              "the labels listed for that task. Leave a row "
                              "blank rather than guessing: a blank row is "
                              "recorded as UNRESOLVED, which is a finding, and "
                              "a guess is not.",
                "tasks": instructions}, indent=2, sort_keys=True),
            encoding="utf-8")
        (out_dir / "human_review_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[review] wrote {out_dir}")
    print(f"[review] procedure {procedure}")
    print(f"[review] {len(public)} row(s) of {len(cells)} cell(s)")
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
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        root = phase2_root() / JUDGING_SUBDIR
        joined = args.joined_dir or (root / "joined")
        packet = args.packet or (root / "packet")
        out_dir = args.output_dir or (root / "human_review")
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        dataset = args.dataset
        if dataset is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            dataset = Path(paths["dataset"])
        export(joined, packet, out_dir, args.procedure, args.seed,
               protocol_dir, dataset, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
