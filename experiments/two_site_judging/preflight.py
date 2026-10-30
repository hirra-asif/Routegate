#!/usr/bin/env python3
"""Everything that must be true before a single billed call. Offline, read-only.

Run this immediately before every judging step. It calls nothing, writes
nothing, loads no model and touches no GPU. It exits non-zero on the first
failure, and a failure means **stop** -- not "adapt the check".

The gates
---------
 1. HEAD is the expected commit and the working tree is clean.
 2. No git remote is configured, so no push or fetch can occur.
 3. The pinned consolidated input verifies: both SHA256s, 480 candidate
    records, 480 referenced baselines, 89 E / 178 B / 213 H on both sides, one
    arm, the candidate's coefficients, sites, historical blocks and hook count.
 4. The consolidation's own completeness claims are what this stage is pinned
    to, and its dataset, direction-bank and master digests match.
 5. The nine rubric digests -- Axis-2, quality and safety instructions,
    templates and schemas -- match the frozen record, and every instruction
    string is the recorded length. **The frozen B quality rubric is located by
    digest**; if it could not be, this gate is where the stage stops.
 6. Every packet request re-renders to the same instructions, template and
    schema hash.
 7. The packet holds exactly 89 Axis-2, 178 quality and 213 safety rows, 480 in
    all, one per ``(screen_row_id, arm_id, rubric)`` cell, and carries no
    Axis-1 file.
 8. Opaque ids are unique and collide with no other packet's prefix.
 9. No judge-facing row carries a class, an arm, a coefficient, a digest, an
    A/B origin or any other forbidden field, and no rendered request names a
    corpus row id.
10. Every B row's private entry names both origins exactly once, so no pair is
    one-sided and no orientation is missing.
11. The Sonnet and Opus identifiers, the temperature and the token cap match
    the frozen configuration, and the planned call count is 480 per judge.
12. No output directory already holds an incompatible run.
13. The predeclaration manifest exists, its rubric assignment and readings are
    byte-identical to the package's, and it declares the adjudication
    procedure ``corpus_disagreement_only``.
14. No configured path can resolve to sealed material, and nothing in this
    stage's code path reads the sealed-set commitment.
15. No file in this stage's code path imports or re-enters the Phase-1 packet
    builder.

Usage
-----
    python experiments/two_site_judging/preflight.py
    python experiments/two_site_judging/preflight.py --expect-head 2611335
    python experiments/two_site_judging/preflight.py --stage packet
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.twositejudging import (                              # noqa: E402
    ADJUDICATION_PROCEDURES,
    CANDIDATE_ARM_ID,
    CLASSES,
    CLASS_RUBRIC,
    DECLARED_PROCEDURE,
    FOREIGN_ROW_PREFIXES,
    FROZEN_INPUT,
    JUDGING_MANIFEST,
    ORIGIN_CANDIDATE,
    PAIRWISE_RUBRICS,
    PLANNED_CALLS_BOTH_JUDGES,
    PLANNED_CALLS_BY_RUBRIC,
    PLANNED_CALLS_PER_JUDGE,
    POSITION_A,
    POSITION_B,
    QUALITY_PROTOCOL,
    READING_RULES,
    ROW_MAP_FILE,
    ROW_PREFIX,
    RUBRICS_USED,
    assert_no_corpus_row_id,
    assert_orientation,
    assert_projection_is_exact,
    assert_rendered_request_is_projection,
    assert_row_has_no_forbidden_field,
    cell_key,
    consolidated_root,
    declared_procedure,
    judging_root,
    load_consolidation_claims,
    load_prompt_texts,
    load_quality_protocol,
    load_referenced_baselines,
    load_screen_records,
    quality_protocol_dir,
)
from routegate.triplets import (                                    # noqa: E402
    AXIS1,
    AXIS2,
    JUDGE_ROW_FIELDS,
    JUDGES,
    MAX_TOKENS,
    QUALITY,
    SAFETY,
    TEMPERATURE,
    assert_opaque,
    assert_row_is_blind,
    load_protocol,
    load_protocol_manifest,
    read_jsonl,
    working_dir,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    SEALED_PATH_TOKENS,
    looks_sealed,
)

STAGE_FILES = ("build_two_site_judge_packet.py", "run_two_site_judge.py",
               "join_two_site_verdicts.py",
               "export_two_site_human_review.py",
               "import_two_site_human_review.py", "report_two_site.py",
               "preflight.py")

#: The Phase-1 packet builder. This stage must not import it, re-enter it, or
#: reuse it with a looser argument: its checks belong to a different packet
#: over a different frozen input.
PHASE1_BUILDER_TOKENS = ("build_judge_packet", "focused_direction_judging")

#: The sealed-set commitment. Naming it in a declaration is the OPPOSITE of
#: reading it, so the check looks for the two ways a file could actually read
#: one, exactly as ``focusedjudging.reads_legacy_stress_manifest`` does.
SEALED_COMMITMENT_TOKEN = "sealed_set_commitment"

_FILE_OPENING_CALLS = (
    "open", "Path", "read_text", "read_bytes", "read_jsonl", "load", "loads",
    "sha256_file", "read_csv", "DictReader", "joinpath", "glob", "rglob",
)


def _reads_token(path: Path, token: str) -> bool:
    """Does this source file *read* a file whose path carries ``token``?"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(
            func, "id", "")
        if name not in _FILE_OPENING_CALLS:
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)
                    and token in inner.value):
                return True
    return False


def _imports_token(path: Path, tokens: Sequence[str]) -> bool:
    """Does this source file IMPORT a module whose name carries a token?"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(t in alias.name for t in tokens):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and any(t in node.module for t in tokens):
                return True
    return False


class Gate:
    """A pass/fail ledger that prints as it goes and never softens a failure."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" - {detail}" if detail else ""))
        return bool(ok)


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=REPO_ROOT, check=False,
                          capture_output=True, text=True).stdout.strip()


def gate_repository(gate: Gate, expect_head: Optional[str]) -> None:
    head = _git("rev-parse", "HEAD")
    if expect_head:
        gate.check("head_is_the_expected_commit", head.startswith(expect_head),
                   f"HEAD {head[:12]}, expected {expect_head}")
    else:
        gate.check("head_recorded", bool(head), f"HEAD {head[:12]}")
    status = _git("status", "--porcelain")
    gate.check("working_tree_is_clean", status == "",
               "clean" if status == "" else
               f"{len(status.splitlines())} modified path(s)")
    remotes = _git("remote")
    gate.check("no_git_remote_is_configured", remotes == "",
               "no remote" if remotes == "" else remotes.replace("\n", ", "))


def gate_input(gate: Gate, consolidated: Path, dataset: Path) -> None:
    checks = Checklist("preflight_input")
    try:
        records = load_screen_records(consolidated, checks)
        baselines = load_referenced_baselines(consolidated, checks)
        load_consolidation_claims(consolidated, checks)
        prompts = load_prompt_texts(dataset, checks)
    except Exception as exc:                        # noqa: BLE001 fail closed
        gate.check("pinned_two_site_input", False, f"{type(exc).__name__}: {exc}")
        return
    gate.check("pinned_two_site_input", checks.all_passed,
               f"{len(records)} candidate record(s), {len(baselines)} "
               f"referenced baseline(s), "
               f"{checks.as_dict()['checks_run']} gate(s)")
    missing = sorted({r["row_id"] for r in records} - set(prompts))
    gate.check("every_record_resolves_to_a_corpus_prompt", not missing,
               f"{len(missing)} unresolved, first {missing[:3]}")
    unpaired = sorted({r["row_id"] for r in records
                       if r["row_class"] == "B"} - set(baselines))
    gate.check("every_b_row_has_a_referenced_baseline", not unpaired,
               f"{len(unpaired)} B row(s) with no baseline, first "
               f"{unpaired[:3]}" if unpaired else
               "a pairwise comparison with one side missing is not a "
               "comparison")


def gate_protocol(gate: Gate, protocol_dir: Path,
                  quality_dir: Path) -> Dict[str, Any]:
    manifest = load_protocol_manifest()
    protocols: Dict[str, Any] = {}
    for rubric in RUBRICS_USED:
        try:
            if rubric == QUALITY:
                protocol = load_quality_protocol(quality_dir)
            else:
                protocol = load_protocol(rubric, directory=protocol_dir,
                                         manifest=manifest)
        except Exception as exc:                    # noqa: BLE001 fail closed
            gate.check(f"{rubric}_rubric_digests", False, f"{exc}")
            continue
        protocols[rubric] = protocol
        gate.check(f"{rubric}_rubric_digests", True,
                   f"instructions {protocol.instructions_sha256[:12]}, "
                   f"template {protocol.user_template_sha256[:12]}, "
                   f"schema {protocol.schema_sha256[:12]}")
    if QUALITY in protocols:
        gate.check(
            "frozen_b_quality_rubric_located_by_digest",
            protocols[QUALITY].instructions_sha256
            == QUALITY_PROTOCOL["instructions_sha256"],
            f"{QUALITY_PROTOCOL['name']}, the rubric that produced "
            f"final_quality_outcome and final_difference_strength in the "
            f"static master")
    return protocols


def gate_packet(gate: Gate, packet: Path, protocols: Dict[str, Any],
                dataset: Path) -> None:
    if not packet.is_dir():
        gate.check("packet_exists", False, f"{packet} is not a directory")
        return
    gate.check("packet_carries_no_axis1_file",
               not (packet / f"{AXIS1}_judge_input.jsonl").exists(),
               "Axis 1 is prompt-only and is not run in this stage")
    if sorted(protocols) != sorted(RUBRICS_USED):
        gate.check("packet_rows_are_blind", False,
                   "a rubric failed to load; the packet is not checked")
        return

    screen_row_ids = sorted(load_prompt_texts(dataset, Checklist("preflight")))
    all_ids: List[str] = []
    ok_render = True
    for rubric in RUBRICS_USED:
        path = packet / f"{rubric}_judge_input.jsonl"
        if not path.is_file():
            gate.check(f"{rubric}_rows", False, f"{path} is absent")
            ok_render = False
            continue
        rows = read_jsonl(path)
        gate.check(f"{rubric}_row_count",
                   len(rows) == PLANNED_CALLS_BY_RUBRIC[rubric],
                   f"{len(rows)} row(s), expected "
                   f"{PLANNED_CALLS_BY_RUBRIC[rubric]}")
        for row in rows:
            all_ids.append(row["row_id"])
            try:
                label = f"{rubric} row {row['row_id']}"
                assert_projection_is_exact(row, rubric, label=label,
                                           prefix=ROW_PREFIX[rubric])
                assert_row_is_blind(row, rubric)
                assert_row_has_no_forbidden_field(row, f"{rubric} row")
                assert_opaque(row["row_id"], ROW_PREFIX[rubric])
                system, user = protocols[rubric].render(row)
                assert_rendered_request_is_projection(system, user, row,
                                                      protocols[rubric], label)
                assert_no_corpus_row_id(user, screen_row_ids, label)
            except Exception as exc:                # noqa: BLE001 fail closed
                gate.check(f"{rubric}_rows_are_blind", False, f"{exc}")
                ok_render = False
                break
        else:
            gate.check(f"{rubric}_rows_are_blind", True,
                       f"{len(rows)} row(s) carry exactly "
                       f"{list(JUDGE_ROW_FIELDS[rubric])}, render to the "
                       f"frozen template filled with exactly those values, and "
                       f"name no corpus row")
    gate.check("every_request_uses_one_rubric_generation", ok_render,
               "every rendered request resolves to the frozen instructions, "
               "template and schema for its rubric")

    gate.check("rows_total", len(all_ids) == FROZEN_INPUT["rows"],
               f"{len(all_ids)} row(s), expected {FROZEN_INPUT['rows']}")
    gate.check("opaque_ids_unique", len(all_ids) == len(set(all_ids)),
               f"{len(set(all_ids))} unique of {len(all_ids)}")
    gate.check("ids_do_not_collide_with_another_packet",
               not any(i.startswith(FOREIGN_ROW_PREFIXES) for i in all_ids),
               f"prefixes {sorted(ROW_PREFIX.values())}")

    map_path = packet / "private" / ROW_MAP_FILE
    if not map_path.is_file():
        gate.check("private_row_map", False, f"{map_path} is absent")
        return
    entries = read_jsonl(map_path)
    cells = set()
    orientations = {POSITION_A: 0, POSITION_B: 0}
    ok_cells = ok_orientation = True
    for entry in entries:
        try:
            key = cell_key(entry, str(map_path))
        except Exception as exc:                    # noqa: BLE001 fail closed
            gate.check("private_row_map_cells", False, f"{exc}")
            ok_cells = False
            break
        if key in cells:
            gate.check("private_row_map_cells", False, f"{key} appears twice")
            ok_cells = False
            break
        cells.add(key)
        if entry["rubric"] in PAIRWISE_RUBRICS:
            try:
                orientation = assert_orientation(
                    {POSITION_A: entry["response_a_origin"],
                     POSITION_B: entry["response_b_origin"]},
                    f"row {entry['row_id']}")
            except Exception as exc:                # noqa: BLE001 fail closed
                gate.check("every_b_row_names_both_origins_once", False,
                           f"{exc}")
                ok_orientation = False
                break
            for position, origin in orientation.items():
                if origin == ORIGIN_CANDIDATE:
                    orientations[position] += 1
    if ok_cells:
        gate.check("private_row_map_cells",
                   len(cells) == FROZEN_INPUT["rows"],
                   f"{len(cells)} distinct (screen_row_id, arm_id, rubric) "
                   f"cell(s) over {len(entries)} row(s)")
    if ok_orientation:
        total = orientations[POSITION_A] + orientations[POSITION_B]
        gate.check("every_b_row_names_both_origins_once",
                   total == PLANNED_CALLS_BY_RUBRIC[QUALITY],
                   f"candidate as A {orientations[POSITION_A]}, as B "
                   f"{orientations[POSITION_B]}, total {total}, expected "
                   f"{PLANNED_CALLS_BY_RUBRIC[QUALITY]}")
    gate.check("orientation_is_only_in_the_private_map",
               all(not (set(row) & {"response_a_origin", "response_b_origin",
                                    "orientation"})
                   for rubric in RUBRICS_USED
                   for row in read_jsonl(packet / f"{rubric}_judge_input.jsonl")
                   if (packet / f"{rubric}_judge_input.jsonl").is_file()),
               "no judge-facing row carries an origin field")


def gate_configuration(gate: Gate) -> None:
    manifest = load_protocol_manifest()
    judges = manifest["judges"]
    gate.check("judges_are_the_frozen_pair",
               judges["primary"]["key"] == JUDGES[0]
               and judges["secondary"]["key"] == JUDGES[1],
               f"{judges['primary']['model']} and "
               f"{judges['secondary']['model']}")
    settings = manifest["generation_settings"]
    gate.check("temperature_and_token_cap",
               settings["temperature"] == TEMPERATURE
               and settings["max_tokens"] == MAX_TOKENS,
               f"T={TEMPERATURE}, max_tokens={MAX_TOKENS}")
    gate.check("planned_calls",
               PLANNED_CALLS_PER_JUDGE == FROZEN_INPUT["rows"]
               and PLANNED_CALLS_BOTH_JUDGES
               == FROZEN_INPUT["rows"] * len(JUDGES),
               f"{PLANNED_CALLS_PER_JUDGE} per judge, "
               f"{PLANNED_CALLS_BOTH_JUDGES} across both")


def gate_manifest(gate: Gate) -> None:
    if not JUDGING_MANIFEST.exists():
        gate.check("predeclaration_manifest_exists", False,
                   f"{JUDGING_MANIFEST} is absent")
        return
    manifest = json.loads(JUDGING_MANIFEST.read_text(encoding="utf-8"))
    gate.check("predeclaration_manifest_exists", True, JUDGING_MANIFEST.name)
    gate.check("manifest_holds_no_text",
               manifest.get("contains_prompt_or_response_text") is False,
               "metadata only")
    try:
        declared = declared_procedure()
    except Exception as exc:                        # noqa: BLE001 fail closed
        gate.check("manifest_declares_an_adjudication_procedure", False,
                   f"{exc}")
        return
    gate.check("manifest_declares_an_adjudication_procedure",
               declared == DECLARED_PROCEDURE,
               f"{declared!r} (of {list(ADJUDICATION_PROCEDURES)})")
    gate.check("manifest_rubric_assignment_matches_the_package",
               manifest["rubrics"]["by_class"]
               == {c: CLASS_RUBRIC[c] for c in CLASSES},
               f"{manifest['rubrics']['by_class']}")
    gate.check("manifest_readings_match_the_package",
               all(manifest["readings"][c] == READING_RULES[c]
                   for c in CLASSES),
               "the three readings are byte-identical to "
               "routegate.twositejudging")
    gate.check("manifest_arm_matches_the_package",
               manifest["arm"]["arm_id"] == CANDIDATE_ARM_ID
               and manifest["arm"]["package_states_edited"]
               == FROZEN_INPUT["states"],
               f"{manifest['arm']['arm_id']} at states "
               f"{manifest['arm']['package_states_edited']}")
    gate.check("manifest_quality_digests_match_the_package",
               all(manifest["rubrics"]["quality"][k] == QUALITY_PROTOCOL[k]
                   for k in ("instructions_sha256", "user_template_sha256",
                             "schema_sha256")),
               "the three B rubric digests")


def gate_code_path(gate: Gate) -> None:
    here = Path(__file__).resolve().parent
    reads_sealed = sorted(f for f in STAGE_FILES
                          if (here / f).is_file()
                          and _reads_token(here / f, SEALED_COMMITMENT_TOKEN))
    gate.check("no_stage_file_reads_the_sealed_set_commitment",
               not reads_sealed, f"{len(STAGE_FILES)} file(s) parsed"
               if not reads_sealed else f"{reads_sealed}")
    imports_phase1 = sorted(f for f in STAGE_FILES
                            if (here / f).is_file()
                            and _imports_token(here / f, PHASE1_BUILDER_TOKENS))
    gate.check("no_stage_file_imports_the_phase1_packet_builder",
               not imports_phase1,
               "the Phase-1 builder is a different packet over a different "
               "frozen input; none of its checks is reused with a looser "
               "argument" if not imports_phase1 else f"{imports_phase1}")


def gate_paths(gate: Gate) -> None:
    paths_file = REPO_ROOT / "configs" / "paths.local.json"
    if not paths_file.exists():
        gate.check("no_configured_path_is_sealed", False,
                   f"{paths_file} is absent")
        return
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    offending = sorted(k for k, v in paths.items()
                       if isinstance(v, str) and looks_sealed(v))
    gate.check("no_configured_path_is_sealed", not offending,
               f"{len(SEALED_PATH_TOKENS)} token(s) checked over "
               f"{len(paths)} entry(ies)" if not offending
               else f"{offending}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-head", default=None)
    parser.add_argument("--stage", default="all",
                        choices=("all", "input", "protocol", "packet",
                                 "manifest", "config"))
    parser.add_argument("--consolidated", type=Path, default=None)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--quality-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="accepted and ignored; this stage never writes")
    args = parser.parse_args(argv)

    gate = Gate()
    try:
        root = judging_root()
        consolidated = args.consolidated or consolidated_root()
        packet = args.packet or (root / "judging" / "packet")
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        quality_dir = args.quality_dir or quality_protocol_dir()
        dataset = args.dataset
        if dataset is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            dataset = Path(paths["dataset"])
    except Exception as exc:                        # noqa: BLE001 fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("== repository ==")
    gate_repository(gate, args.expect_head)
    print("== code path ==")
    gate_code_path(gate)
    print("== configured paths ==")
    gate_paths(gate)

    protocols: Dict[str, Any] = {}
    if args.stage in ("all", "input"):
        print("== pinned input ==")
        gate_input(gate, consolidated, dataset)
    if args.stage in ("all", "protocol", "packet"):
        print("== frozen rubrics ==")
        protocols = gate_protocol(gate, protocol_dir, quality_dir)
    if args.stage in ("all", "packet"):
        print("== packet ==")
        gate_packet(gate, packet, protocols, dataset)
    if args.stage in ("all", "config"):
        print("== judge configuration ==")
        gate_configuration(gate)
    if args.stage in ("all", "manifest"):
        print("== predeclaration ==")
        gate_manifest(gate)

    print(f"\n{len(gate.rows) - gate.failed}/{len(gate.rows)} gate(s) passed")
    if gate.failed:
        print(f"[FATAL] {gate.failed} gate(s) failed. Stop; do not adapt the "
              f"check.", file=sys.stderr)
        return 1
    print("preflight passed; nothing was called, written or billed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
