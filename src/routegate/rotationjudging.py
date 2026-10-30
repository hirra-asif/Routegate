"""Registering the completed rotation screen with the common judging engine.

The screen is finished. 144 responses -- 18 targeted H rows x 8 predeclared
rotation arms -- were generated, consolidated and pinned, and its consolidation
record says in as many words that nothing in it judges anything. This module is
what lets the COMMON engine judge them, and it is deliberately small.

WHAT IS NEW HERE, AND WHY IT IS SO LITTLE
-----------------------------------------
``routegate.judgingprotocol`` and ``routegate.judgingstages`` are the nine
stages, and an experiment is supposed to supply a MANIFEST rather than code.
That is what ``manifests/rotation_safety_direction_v2_judging_v1.json`` is.
Two things a manifest cannot express, and only two, live here:

* **an adapter.** The engine reads source records carrying exactly
  ``judgingstages.SOURCE_FIELDS``. A rotation record carries an arm, a class,
  a response and the response's digest -- but its PROMPT only as a digest,
  because the screen never stored prompt text. :func:`adapt` resolves the text
  from the one hash-gated corpus the screen recorded it generated from, using
  ``focusedjudging.load_prompt_texts``, and checks every resolved prompt
  against the digest the record already carries. It converts; it decides
  nothing and it judges nothing.
* **this experiment's own reading.** The engine reports verdict tallies with
  explicit denominators and stops there, on purpose. The 180-cell analysis
  table, the four counts per beta and the decision belong to the experiment,
  and ``reporting.module`` in the manifest names this module for the record.

WHAT IS NOT NEW, AND IS NOT ALLOWED TO BE
-----------------------------------------
=========================== ===================================================
the rubric                  the frozen actionable-harmful-help safety rubric,
                            by digest, from the same frozen directory phase 1
                            and phase 2 read
the adjudication rule       ``focusedjudging.adjudicate_cell``, through the
                            engine, unchanged
the 36 reference cells      ``phase2judging.load_inherited_references``, called
                            unchanged. Phase 2 inherited the same two arms over
                            the same 18 rows under the same procedure, so the
                            reconciliation is that function and not a copy of
                            it
the four counts             ``phase2judging.counts_by_strength``, called with
                            this screen's arm ids. The arms are parameters; the
                            arithmetic, and the rule that UNCLEAR is not NO,
                            are not
the decision                ``rotationfreeze.evaluate_rotation_decision``,
                            which is phase 2's rule itself. Nothing here reads
                            a threshold, applies the veto, breaks a tie or
                            decides what UNDETERMINED means
=========================== ===================================================

THE REFERENCE ARMS ARE NEVER JUDGED AGAIN
-----------------------------------------
``baseline`` and ``repair_015`` for these 18 rows were generated in phase 1,
judged in phase 1 by the same two judges under the same rubric, and adjudicated
in phase 1 under the same procedure. They are READ. This module refuses a
candidate record on either arm, the manifest declares neither as a candidate
arm, and the manifest declares no ``reference_records`` input at all -- so
there is no path, not even a mistaken one, by which a reference response
reaches a judge. 144 judged + 36 inherited = 180, and 288 billed calls, not
360.

Nothing here calls a model, opens a socket or reads a credential.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple)

from .focusedjudging import (
    ADJUDICATED_CELLS_FILE,
    BASELINE_ARM,
    CARRIED_CELL_FIELDS,
    H_TEST_ARM,
    PROCEDURE_CORPUS,
    load_prompt_texts,
)
from .judgingprotocol import ROW_MAP_FILE, index_cells
from .judgingstages import SOURCE_FIELDS
from .phase2judging import (
    counts_by_strength,
    load_inherited_references,
    per_row_transitions,
    verdicts_by_arm,
)
from .rotationfreeze import (
    ADJUDICATION,
    NAME,
    PHASE1_ADJUDICATED_CELLS_SHA256,
    SAFETY_RUBRIC,
    WITHHELD,
    evaluate_rotation_decision,
    predeclaration_sha256,
)
from .rotationscreen import (
    ALPHA_REPAIR,
    ANALYSIS_TABLE_RECORDS,
    ARM_BY_ID_ROTATION,
    ARMS_BY_BETA,
    BETA_GRID,
    NEW_ARMS,
    NEW_GENERATIONS,
    N_ROWS,
    PROTOCOL,
    REFERENCE_ARM_IDS,
    REFERENCE_RECORDS,
    SELECTED_ROWS,
)
from .runrecord import environment_record, utc_now
from .triplets import (
    AXIS_VALUES,
    SAFETY,
    SOURCE_UNRESOLVED,
    read_jsonl,
    record_hash,
    sha256_text,
    write_jsonl,
)
from .validation import (
    Checklist,
    ValidationError,
    assert_not_sealed,
    sha256_file,
)

from . import __version__ as PACKAGE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]


class RotationJudgingError(ValidationError):
    """This stage was asked for something it must not do."""


# ============================================================== the identity
EXPERIMENT_ID = f"{NAME}_judging_v1"
PACKET_ID = f"{EXPERIMENT_ID}_packet"
JUDGING_MANIFEST = REPO_ROOT / "manifests" / f"{EXPERIMENT_ID}.json"

#: This packet's opaque-id prefixes. Independently randomised ids behind them,
#: and the prefixes themselves are new so a rotation row can never be joined
#: to a phase-1, phase-2, two-site or triplet row.
ROW_PREFIX = "RSD2SAFETY"
HUMAN_ROW_PREFIX = "RSD2HUMAN"

#: One class, one rubric. Every cell is an H response and receives the frozen
#: safety rubric; the rubric's NAME is the axis name the frozen text is filed
#: under, so the engine loads ``safety_instructions.txt`` and its two siblings
#: from the same directory phase 1 read them from.
RUBRIC = SAFETY
ROW_CLASS = "H"

#: The authoritative consolidated screen, by digest. This one line stands in
#: for the 6,247 checks the consolidation already ran: if the file still
#: hashes to this, every one of those checks was passed by this exact byte
#: sequence, and re-deriving them here would be a second opinion about a
#: settled artifact rather than a gate.
CONSOLIDATED_RECORDS_SHA256 = (
    "c6d72d7b09fa52143479f67c4a00552ec92d20db3d4d3d0c1a878c817c826922")
CONSOLIDATED_RECORDS = "rotation_records.jsonl"
CONSOLIDATION_MANIFEST = "run_manifest.json"
CONSOLIDATION_VALIDATION = "validation.json"

#: Where the adapted candidate records are written, under the judging root.
INPUTS_SUBDIR = "inputs"
CANDIDATE_RECORDS = "candidate_records.jsonl"

#: The ``configs/paths.local.json`` keys this stage and its manifest name.
PATHS_KEY_CONSOLIDATED = "rotation_safety_direction_v2_consolidated_dir"
PATHS_KEY_JUDGING = "rotation_judging_dir"

NEW_ARM_IDS: Tuple[str, ...] = tuple(arm.arm_id for arm in NEW_ARMS)

#: Both reference arms, in the two roles the counts read them in.
INHERITED_SOURCE = "phase1_adjudication"
ROTATION_SOURCE = "rotation_adjudication"


# ================================================================ the adapter
def adapt(records: Sequence[Mapping[str, Any]],
          prompts: Mapping[str, Mapping[str, str]],
          checks: Checklist) -> List[Dict[str, str]]:
    """Rotation screen records -> engine source records. A conversion only.

    The output carries exactly ``judgingstages.SOURCE_FIELDS`` and is built by
    whitelist, so an arm id, a beta, a state, a geometry block or an expected
    outcome cannot ride along into something a judge will see. What blinding
    the row further -- minting the opaque id, dropping the corpus row id --
    is the engine's ``build`` stage and is not reimplemented here.
    """
    checks.require("records", len(records) == NEW_GENERATIONS,
                   f"{len(records)} record(s), expected {NEW_GENERATIONS}")

    seen: Dict[Tuple[str, str], int] = {}
    out: List[Dict[str, str]] = []
    for number, record in enumerate(records, start=1):
        row_id = str(record.get("row_id", ""))
        arm_id = str(record.get("arm_id", ""))
        if arm_id in REFERENCE_ARM_IDS:
            raise RotationJudgingError(
                f"record {number} is on arm {arm_id!r}. baseline and "
                f"repair_015 for these {N_ROWS} rows were judged in phase 1 "
                f"and are inherited; submitting one to a judge again would "
                f"pay twice for a cell that already has a verdict, and would "
                f"produce a second verdict for it.")
        if arm_id not in NEW_ARM_IDS:
            raise RotationJudgingError(
                f"record {number} is on arm {arm_id!r}, which is not one of "
                f"the {len(NEW_ARM_IDS)} predeclared rotation arms")
        if row_id not in SELECTED_ROWS:
            raise RotationJudgingError(
                f"record {number} is row {row_id!r}, which is not one of the "
                f"{N_ROWS} frozen selected rows")
        if str(record.get("row_class")) != ROW_CLASS:
            raise RotationJudgingError(
                f"record {number} is class {record.get('row_class')!r}; every "
                f"cell in this screen is an {ROW_CLASS} response")
        if str(record.get("status")) != "ok":
            raise RotationJudgingError(
                f"record {number} has status {record.get('status')!r}. A "
                f"record that did not generate cleanly is not judged; the "
                f"consolidation would not have accepted one either.")
        key = (row_id, arm_id)
        if key in seen:
            raise RotationJudgingError(
                f"record {number}: {key} also appears at record {seen[key]}. "
                f"One cell is one record.")
        seen[key] = number

        response = str(record["response_text"])
        if sha256_text(response) != str(record.get("response_sha256")):
            raise RotationJudgingError(
                f"{key}: the response does not hash to the digest the record "
                f"carries. The stored response and its identity have parted "
                f"company; nothing is judged on it.")

        corpus = prompts.get(row_id)
        if corpus is None:
            raise RotationJudgingError(
                f"{key}: the accepted corpus holds no row {row_id!r}, so the "
                f"prompt this response answered cannot be resolved")
        if corpus["prompt_sha256"] != str(record.get("prompt_sha256")):
            raise RotationJudgingError(
                f"{key}: the record was generated over prompt "
                f"{str(record.get('prompt_sha256'))[:12]} and the corpus row "
                f"hashes {corpus['prompt_sha256'][:12]}. The prompt behind "
                f"the response has moved, so judging it would put a response "
                f"beside a question it did not answer.")
        if corpus["row_class"] != ROW_CLASS:
            raise RotationJudgingError(
                f"{key}: the corpus calls this row class "
                f"{corpus['row_class']!r}")

        out.append({"screen_row_id": row_id, "arm_id": arm_id,
                    "row_class": ROW_CLASS,
                    "prompt_text": corpus["prompt_text"],
                    "response_text": response})

    expected = {(row, arm) for row in SELECTED_ROWS for arm in NEW_ARM_IDS}
    checks.require(
        "every_row_at_every_arm", set(seen) == expected,
        f"{len(expected - set(seen))} absent, {len(set(seen) - expected)} "
        f"unexpected; {N_ROWS} row(s) x {len(NEW_ARM_IDS)} arm(s)")
    checks.require(
        "no_reference_arm_is_adapted",
        not (set(NEW_ARM_IDS) & set(REFERENCE_ARM_IDS)),
        f"{list(REFERENCE_ARM_IDS)} are inherited from phase 1 and are never "
        f"candidate records")
    stray = sorted({field for row in out for field in row} - set(SOURCE_FIELDS))
    checks.require(
        "source_fields_exactly", not stray,
        f"carries {stray}" if stray else
        f"every row carries exactly {list(SOURCE_FIELDS)}")

    # Deterministic on purpose: the manifest pins this file by digest, so the
    # same inputs must produce the same bytes on any machine.
    return sorted(out, key=lambda r: (r["arm_id"], r["screen_row_id"]))


def read_consolidation(consolidated: Path | str,
                       checks: Checklist) -> List[Dict[str, Any]]:
    """The 144 pinned records, refused unless the package still says so."""
    directory = Path(consolidated)
    assert_not_sealed(directory, "the consolidated rotation screen")
    records_path = directory / CONSOLIDATED_RECORDS
    for name in (CONSOLIDATED_RECORDS, CONSOLIDATION_MANIFEST,
                 CONSOLIDATION_VALIDATION):
        if not (directory / name).is_file():
            raise RotationJudgingError(
                f"{directory / name} is absent; --consolidated names the "
                f"completed consolidation package, which carries all three of "
                f"{CONSOLIDATED_RECORDS}, {CONSOLIDATION_MANIFEST} and "
                f"{CONSOLIDATION_VALIDATION}")

    digest = sha256_file(records_path)
    checks.require(
        "consolidated_records_sha256", digest == CONSOLIDATED_RECORDS_SHA256,
        f"{digest[:12]}, pinned {CONSOLIDATED_RECORDS_SHA256[:12]}")

    manifest = json.loads(
        (directory / CONSOLIDATION_MANIFEST).read_text(encoding="utf-8"))
    validation = json.loads(
        (directory / CONSOLIDATION_VALIDATION).read_text(encoding="utf-8"))
    checks.require("consolidation_protocol",
                   str(manifest.get("protocol")) == PROTOCOL,
                   f"{manifest.get('protocol')!r}, expected {PROTOCOL!r}")
    checks.require("consolidation_records_sha256",
                   str(manifest.get("records_sha256")) == digest,
                   "the package's own digest of its records agrees")
    checks.require("consolidation_complete_protocol",
                   validation.get("complete_protocol") is True,
                   "the consolidation reports the whole protocol present")
    checks.require(
        "consolidation_judged_nothing",
        validation.get("judges_responses") is False
        and manifest.get("judges_responses") is False,
        "the consolidation judged no response, which is why this stage exists")
    checks.require("consolidation_not_a_smoke_test",
                   manifest.get("smoke") is False,
                   "the package is a real run, not a smoke test")
    checks.require("consolidation_checks_all_passed",
                   validation.get("checklist", {}).get("all_passed") is True,
                   f"{validation.get('checklist', {}).get('checks_run')} "
                   f"check(s) passed at consolidation")
    checks.require(
        "predeclaration_sha256",
        str(manifest.get("predeclaration_sha256", "")) in
        ("", predeclaration_sha256()),
        "the package was produced under the committed predeclaration")
    return read_jsonl(records_path)


def build_judging_inputs(consolidated: Path | str, out_dir: Path | str,
                         dataset: Path | str, write: bool = True
                         ) -> Dict[str, Any]:
    """Write the engine's candidate records. Reads the screen; never writes it."""
    checks = Checklist(f"{EXPERIMENT_ID}_adapter")
    out_dir = Path(out_dir)
    if Path(consolidated).resolve() == out_dir.resolve():
        raise RotationJudgingError(
            f"--output-dir resolves to {consolidated}, which is the completed "
            f"consolidation this adapter reads. That package is read only: it "
            f"is never moved, renamed, copied into or written over.")

    records = read_consolidation(consolidated, checks)
    prompts = load_prompt_texts(dataset, checks)
    rows = adapt(records, prompts, checks)

    target = out_dir / CANDIDATE_RECORDS
    summary = {
        "stage": "adapt", "experiment_id": EXPERIMENT_ID,
        "protocol": PROTOCOL,
        "consolidated_records_sha256": CONSOLIDATED_RECORDS_SHA256,
        "records": len(rows),
        "rows": N_ROWS, "arms": len(NEW_ARM_IDS),
        "candidate_arms": list(NEW_ARM_IDS),
        "inherited_arms_not_adapted": list(REFERENCE_ARM_IDS),
        "source_fields": list(SOURCE_FIELDS),
        "output": str(target),
        "created_utc": utc_now(),
        "checks": checks.as_dict(),
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(target, rows)
        summary["candidate_records_sha256"] = sha256_file(target)
    return summary


# ================================================== the 36 inherited references
def inherited_references(phase1_adjudicated: Path | str,
                         prompt_sha256_by_row: Mapping[str, str],
                         checks: Checklist
                         ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Phase 1's ``baseline`` and ``repair_015`` cells for these 18 rows.

    ``phase2judging.load_inherited_references``, called unchanged. Phase 2
    inherited exactly this -- the same 18 rows, the same two arms, the same
    frozen safety rubric, the same declared procedure -- and every gate it
    applies is a gate this screen needs: 36 cells and no more, the bound
    selection and nothing else, one rubric, one procedure, every verdict
    decided, the verdicts the selection rule implies, and the prompt digest
    matching the record this screen generated for the same row.

    The last of those is the one that matters most here. An inherited verdict
    is only a reference for a rotation cell if both were about the same
    question, so the digests are compared rather than assumed.
    """
    path = Path(phase1_adjudicated)
    file = path / ADJUDICATED_CELLS_FILE if path.is_dir() else path
    digest = sha256_file(file)
    checks.require(
        "phase1_adjudication_sha256",
        digest == PHASE1_ADJUDICATED_CELLS_SHA256,
        f"{digest[:12]}, pinned {PHASE1_ADJUDICATED_CELLS_SHA256[:12]} by "
        f"routegate.rotationfreeze")
    inherited = load_inherited_references(
        file, prompt_sha256_by_row, PROCEDURE_CORPUS, checks)
    checks.require(
        "inherited_reference_cells",
        len(inherited) == REFERENCE_RECORDS,
        f"{len(inherited)} inherited cell(s), expected {REFERENCE_RECORDS} "
        f"({N_ROWS} row(s) x {list(REFERENCE_ARM_IDS)})")
    checks.record(
        "inherited_were_never_re_judged", True,
        f"{REFERENCE_RECORDS} cell(s) read from phase 1; no reference "
        f"response was sent to a judge by this stage, and the manifest "
        f"declares no reference_records input for one to be read from")
    return inherited


# ==================================================== the 180-record table
def _arm_metadata(arm_id: str) -> Dict[str, Any]:
    """Beta, alpha_R and kind for one arm, generated or inherited."""
    if arm_id in ARM_BY_ID_ROTATION:
        arm = ARM_BY_ID_ROTATION[arm_id]
        return {"beta": arm.beta, "alpha_R": arm.alpha_repair,
                "arm_kind": arm.kind}
    if arm_id == BASELINE_ARM:
        return {"beta": None, "alpha_R": 0.0, "arm_kind": "reference"}
    if arm_id == H_TEST_ARM:
        return {"beta": None, "alpha_R": ALPHA_REPAIR,
                "arm_kind": "reference"}
    raise RotationJudgingError(
        f"{arm_id!r} is neither a predeclared rotation arm nor one of the "
        f"inherited reference arms {list(REFERENCE_ARM_IDS)}")


def analysis_table(rotation_cells: Mapping[Tuple[str, str, str],
                                           Mapping[str, Any]],
                   inherited: Mapping[Tuple[str, str, str],
                                      Mapping[str, Any]],
                   digests: Mapping[Tuple[str, str], Mapping[str, str]],
                   checks: Checklist) -> List[Dict[str, Any]]:
    """180 records: 144 judged here, 36 inherited, none rewritten.

    Both sides are projected onto one field set so the table is one shape, and
    every row says which side it came from and carries the digest of the
    record it was projected from -- so "this was not modified" is checkable
    against phase 1's own file rather than asserted.

    A cell with no verdict keeps ``None``. That is the whole point of
    requirement 9: a technical failure is UNRESOLVED_TECHNICAL, it is never
    a NO, and the beta it belongs to goes UNDETERMINED because of it.
    """
    overlap = sorted(set(rotation_cells) & set(inherited))
    if overlap:
        raise RotationJudgingError(
            f"{len(overlap)} cell(s) are claimed by both this screen's "
            f"adjudication and the inherited references, first {overlap[:3]}. "
            f"An inherited arm is never judged here.")

    rows: List[Dict[str, Any]] = []
    for source, cells in ((ROTATION_SOURCE, rotation_cells),
                          (INHERITED_SOURCE, inherited)):
        for key in sorted(cells):
            record = cells[key]
            arm_id = str(record["arm_id"])
            row_id = str(record["screen_row_id"])
            unresolved = str(record.get("label_source")) == SOURCE_UNRESOLVED
            verdict = None if unresolved else record.get("verdict")
            if verdict is not None and verdict not in AXIS_VALUES[RUBRIC]:
                raise RotationJudgingError(
                    f"{key}: verdict {verdict!r} is not one of "
                    f"{list(AXIS_VALUES[RUBRIC])}")
            if record.get("technically_unresolved") and verdict is not None:
                raise RotationJudgingError(
                    f"{key}: the cell is technically unresolved and carries "
                    f"verdict {verdict!r}. A failure is never a verdict.")
            carried = digests.get((row_id, arm_id), {})
            row = {
                "screen_row_id": row_id,
                "arm_id": arm_id,
                "rubric": str(record.get("rubric", record.get("axis"))),
                "row_class": str(record["row_class"]),
                **_arm_metadata(arm_id),
                "verdict": verdict,
                "label_source": str(record["label_source"]),
                "adjudication_procedure": str(
                    record.get("adjudication_procedure", PROCEDURE_CORPUS)),
                "prompt_sha256": record.get(
                    "prompt_sha256", carried.get("prompt_sha256")),
                "response_sha256": record.get(
                    "response_sha256", carried.get("response_sha256")),
                "judged_row_id": record.get("judged_row_id"),
                "technically_unresolved": bool(
                    record.get("technically_unresolved")),
                "missing_judge": record.get("missing_judge"),
                "failure_type": record.get("failure_type"),
                "source": source,
                "source_record_sha256": record_hash(record),
                "is_inherited": source == INHERITED_SOURCE,
            }
            for field_name in CARRIED_CELL_FIELDS:
                row[field_name] = record.get(field_name)
            rows.append(row)

    checks.require(
        "analysis_table_records", len(rows) == ANALYSIS_TABLE_RECORDS,
        f"{len(rows)} record(s), expected {ANALYSIS_TABLE_RECORDS} "
        f"({NEW_GENERATIONS} judged here + {REFERENCE_RECORDS} inherited)")
    checks.require(
        "analysis_table_inherited_records",
        sum(1 for r in rows if r["is_inherited"]) == REFERENCE_RECORDS,
        f"{sum(1 for r in rows if r['is_inherited'])} inherited record(s), "
        f"expected {REFERENCE_RECORDS}")
    expected = {(row_id, arm_id, RUBRIC)
                for row_id in SELECTED_ROWS
                for arm_id in REFERENCE_ARM_IDS + NEW_ARM_IDS}
    present = {(r["screen_row_id"], r["arm_id"], r["rubric"]) for r in rows}
    checks.require(
        "analysis_table_covers_every_row_at_every_arm", present == expected,
        f"{len(expected - present)} absent, {len(present - expected)} "
        f"unexpected")
    return sorted(rows, key=lambda r: (r["screen_row_id"], r["arm_id"]))


# ==================================================== the counts and the rule
def counts_by_beta(verdicts: Mapping[Tuple[str, str], Optional[str]],
                   rows: Sequence[str] = SELECTED_ROWS
                   ) -> Dict[float, Dict[str, Any]]:
    """The four counts per beta, in this screen's vocabulary.

    ``phase2judging.counts_by_strength`` does the counting, called with this
    screen's arm ids. Phase 2's "safety-only" arm is this screen's
    "rotation-only" arm, so the shared function's output is relabelled on the
    way out -- exactly the way ``rotationfreeze.evaluate_rotation_decision``
    relabels phase 2's decision on the way back. The rename is the only
    rotation-specific step; the arithmetic, and the rule that an UNCLEAR or
    missing verdict is undetermined rather than favourable, are phase 2's.
    """
    shared = counts_by_strength(
        verdicts, rows,
        strengths=BETA_GRID,
        safety_arm_by_strength={beta: ARMS_BY_BETA[beta]["rotation_only"]
                                for beta in BETA_GRID},
        combined_arm_by_strength={beta: ARMS_BY_BETA[beta]["combined"]
                                  for beta in BETA_GRID},
        baseline_arm=BASELINE_ARM, treated_arm=H_TEST_ARM)
    out: Dict[float, Dict[str, Any]] = {}
    for beta, counts in shared.items():
        out[float(beta)] = {
            "combined_recovered": counts["combined_recovered"],
            "combined_undetermined": counts["combined_undetermined"],
            "rotation_only_harmful": counts["safety_only_harmful"],
            "rotation_only_undetermined": counts["safety_only_undetermined"],
            "combined_arm": counts["combined_arm"],
            "rotation_only_arm": counts["safety_only_arm"],
            "denominator": counts["denominator"],
            "counted_by": "routegate.phase2judging.counts_by_strength",
        }
    return out


# ==================================================== the final 180-cell report
FINAL_REPORT_FILE = "rotation_final_report.json"


def cell_digests(packet: Path | str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """``(row, arm) -> the prompt and response digests actually judged``.

    Read from the packet's private row map, which is the only record binding
    an opaque judged id back to a cell. Carrying the digests into the analysis
    table is what lets a reader check that the cell judged at
    ``repair015_rotation010`` held the response the screen recorded there,
    rather than taking the join's word for it.
    """
    path = Path(packet) / "private" / ROW_MAP_FILE
    if not path.is_file():
        raise RotationJudgingError(
            f"{path} is not a file. The private row map is the only record of "
            f"which cell each opaque judged id belongs to; the final report "
            f"is not assembled without it.")
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for entry in read_jsonl(path):
        key = (str(entry["screen_row_id"]), str(entry["arm_id"]))
        if key in out:
            raise RotationJudgingError(
                f"{key} appears twice in the private row map")
        out[key] = {"prompt_sha256": str(entry["prompt_sha256"]),
                    "response_sha256": str(entry["candidate_response_sha256"]),
                    "judged_row_id": str(entry["row_id"])}
    return out


def final_report(dirs: Mapping[str, Path], phase1_adjudicated: Path | str,
                 write: bool = True) -> Dict[str, Any]:
    """The 180-cell analysis table, the counts, and the frozen decision.

    The decision is not taken here. ``counts_by_beta`` produces the four
    counts per beta and ``rotationfreeze.evaluate_rotation_decision`` -- which
    is phase 2's rule itself, reached through
    ``routegate.phase2.evaluate_phase2_decision`` -- decides. No threshold, no
    veto, no tie-break and no UNDETERMINED branch is written, read or
    reinterpreted in this function.
    """
    checks = Checklist(f"{EXPERIMENT_ID}_final_report")
    adjudicated_dir = Path(dirs["adjudicated"])
    path = adjudicated_dir / ADJUDICATED_CELLS_FILE
    if not path.is_file():
        raise RotationJudgingError(
            f"{path} is not a file; run the engine's adjudicate stage first "
            f"(experiments/judging/run_protocol.py --stage adjudicate).")

    records = read_jsonl(path)
    checks.require("adjudicated_cells", len(records) == NEW_GENERATIONS,
                   f"{len(records)} cell(s), expected {NEW_GENERATIONS} "
                   f"judged by this stage")
    rotation_cells = index_cells(records, str(path))
    for key in rotation_cells:
        if key[1] in REFERENCE_ARM_IDS:
            raise RotationJudgingError(
                f"{key}: this stage adjudicated an inherited reference arm. "
                f"baseline and repair_015 are read from phase 1 and are never "
                f"judged again.")

    digests = cell_digests(Path(dirs["packet"]))
    checks.require(
        "row_map_covers_every_judged_cell",
        set(digests) == {(k[0], k[1]) for k in rotation_cells},
        f"{len(digests)} row-map entry/entries for {len(rotation_cells)} "
        f"adjudicated cell(s)")
    prompt_sha256_by_row = {row: entry["prompt_sha256"]
                            for (row, _arm), entry in digests.items()}

    inherited = inherited_references(
        phase1_adjudicated, prompt_sha256_by_row, checks)
    table = analysis_table(rotation_cells, inherited, digests, checks)

    verdicts = verdicts_by_arm(table)
    counts = counts_by_beta(verdicts)
    decision = evaluate_rotation_decision(counts)
    checks.record(
        "decision_rule_is_the_authoritative_one", True,
        f"{decision['evaluated_by']}, reached through "
        f"routegate.rotationfreeze.evaluate_rotation_decision")

    unresolved = [r for r in table if r["verdict"] is None]
    checks.record(
        "unresolved_cells", True,
        f"{len(unresolved)} cell(s) carry no verdict; each leaves its beta "
        f"undetermined rather than counting as a NO")
    decided = [r for r in table if r["verdict"] is not None]

    summary: Dict[str, Any] = {
        "stage": "final_report",
        "experiment_id": EXPERIMENT_ID,
        "protocol": PROTOCOL,
        "predeclaration_sha256": predeclaration_sha256(),
        "consolidated_records_sha256": CONSOLIDATED_RECORDS_SHA256,
        "phase1_adjudication_sha256": PHASE1_ADJUDICATED_CELLS_SHA256,
        "rubric": dict(SAFETY_RUBRIC),
        "adjudication": dict(ADJUDICATION),
        "counts": {
            "analysis_table_records": len(table),
            "judged_here": sum(1 for r in table if not r["is_inherited"]),
            "inherited": sum(1 for r in table if r["is_inherited"]),
            "arithmetic": f"{NEW_GENERATIONS} judged + {REFERENCE_RECORDS} "
                          f"inherited = {ANALYSIS_TABLE_RECORDS}",
            "billed_calls": NEW_GENERATIONS * 2,
            "billed_calls_note": "two judges over the 144 new cells only. The "
                                 "36 inherited cells were paid for in phase 1 "
                                 "and are not re-judged.",
            "decided_cells": len(decided),
            "unresolved_cells": len(unresolved),
        },
        "denominator": N_ROWS,
        "denominator_is": "the 18 selected rows, at every beta. Fixed by the "
                          "frozen selection and never rescaled.",
        "by_beta": {f"{beta:.2f}": counts[beta] for beta in sorted(counts)},
        "decision": decision,
        "per_row_transitions": per_row_transitions(
            verdicts, arms=REFERENCE_ARM_IDS + NEW_ARM_IDS),
        "unresolved_cells": sorted(
            ({"judged_row_id": r["judged_row_id"], "arm_id": r["arm_id"],
              "rubric": r["rubric"], "missing_judge": r["missing_judge"],
              "failure_type": r["failure_type"],
              "is_inherited": r["is_inherited"]}
             for r in unresolved),
            key=lambda d: (str(d["arm_id"]), str(d["judged_row_id"]))),
        "withheld": dict(WITHHELD),
        "checks": checks.as_dict(),
        "environment": environment_record(PACKAGE_VERSION),
        "created_utc": utc_now(),
    }
    if write:
        out_dir = Path(dirs["report"])
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "analysis_table.jsonl", table)
        (out_dir / FINAL_REPORT_FILE).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


__all__ = [
    "ANALYSIS_TABLE_RECORDS", "CANDIDATE_RECORDS",
    "CONSOLIDATED_RECORDS_SHA256", "EXPERIMENT_ID", "FINAL_REPORT_FILE",
    "HUMAN_ROW_PREFIX", "INPUTS_SUBDIR", "JUDGING_MANIFEST", "NEW_ARM_IDS",
    "PACKET_ID", "PATHS_KEY_CONSOLIDATED", "PATHS_KEY_JUDGING", "ROW_PREFIX",
    "RUBRIC", "RotationJudgingError", "adapt", "analysis_table",
    "build_judging_inputs", "cell_digests", "counts_by_beta", "final_report",
    "inherited_references", "read_consolidation",
]
