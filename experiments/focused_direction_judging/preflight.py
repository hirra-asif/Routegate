#!/usr/bin/env python3
"""Everything that must be true before a single billed call. Offline, read-only.

Run this immediately before every judging step. It calls nothing, writes
nothing, loads no model and touches no GPU. It exits non-zero on the first
failure, and a failure means **stop** -- not "adapt the check".

The gates
---------
 1. HEAD is the expected commit and the working tree is clean.
 2. No git remote is configured, so no push or fetch can occur.
 3. The pinned phase-1 records verify: SHA256, byte count, 871 records,
    445 E / 426 H, the per-arm split, and every row carrying every arm of its
    class.
 4. The consolidation's own completeness claims are what this stage is pinned
    to, and its dataset and direction-bank digests match.
 5. The six protocol digests -- Axis-2 and safety instructions, templates and
    schemas -- match the frozen record, and the two instruction strings are the
    recorded length.
 6. **Every packet request re-renders to the same instructions, template and
    schema hash.** This is the check that no second Axis-2 generation is in
    play anywhere.
 7. The packet holds exactly 445 Axis-2 rows and 426 safety rows, 871 in all,
    one per ``(screen_row_id, arm_id, axis)`` cell, and carries no Axis-1 file.
 8. Opaque ids are unique and collide with no other packet's prefix.
 9. No judge-facing row carries a class, an arm, a coefficient, a digest or any
    other forbidden field, and no rendered request names a corpus row id or a
    phase-1 metadata token.
10. The Sonnet and Opus identifiers, the temperature and the token cap match the
    frozen configuration.
11. No output directory already holds an incompatible run.
12. The predeclaration manifest exists, its two rules are byte-identical to the
    package's, and it **declares an adjudication procedure** -- 2026-08-24,
    `corpus_disagreement_only`. The gate fails if it is ever null again, and
    every adjudicating stage refuses a `--procedure` that is not the declared
    one.
13. No file in this stage's code path reads the legacy stress-set manifest.
14. No configured path can resolve to sealed material.

Usage
-----
    python experiments/focused_direction_judging/preflight.py
    python experiments/focused_direction_judging/preflight.py --expect-head 559706e
    python experiments/focused_direction_judging/preflight.py --stage packet
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.focusedjudging import (                              # noqa: E402
    ADJUDICATION_PROCEDURES,
    AXES_USED,
    CONSOLIDATED_SUBDIR,
    FOREIGN_ROW_PREFIXES,
    FROZEN_INPUT,
    H_REGRESSION_RULE,
    JUDGING_MANIFEST,
    PLANNED_CALLS_BOTH_JUDGES,
    PLANNED_CALLS_BY_AXIS,
    PLANNED_CALLS_PER_JUDGE,
    ROW_MAP_FILE,
    ROW_PREFIX,
    REPAIR_OPERATING_POINT_RULE,
    assert_no_corpus_row_id,
    assert_projection_is_exact,
    assert_rendered_request_is_projection,
    assert_row_has_no_forbidden_field,
    cell_key,
    judging_root,
    load_consolidation_claims,
    load_prompt_texts,
    load_screen_records,
    reads_legacy_stress_manifest,
)
from routegate.triplets import (                                    # noqa: E402
    AXIS1,
    JUDGE_ROW_FIELDS,
    JUDGES,
    MAX_TOKENS,
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

STAGE_FILES = ("build_judge_packet.py", "join_verdicts.py",
               "export_human_review.py", "import_human_review.py",
               "report_phase1.py", "preflight.py")


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
              + (f" — {detail}" if detail else ""))
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
        load_consolidation_claims(consolidated, checks)
        prompts = load_prompt_texts(dataset, checks)
    except Exception as exc:                        # noqa: BLE001 fail closed
        gate.check("pinned_phase1_input", False, f"{type(exc).__name__}: {exc}")
        return
    gate.check("pinned_phase1_input", checks.all_passed,
               f"{len(records)} record(s), {checks.as_dict()['checks_run']} "
               f"gate(s)")
    missing = sorted({r["row_id"] for r in records} - set(prompts))
    gate.check("every_record_resolves_to_a_corpus_prompt", not missing,
               f"{len(missing)} unresolved, first {missing[:3]}")


def gate_protocol(gate: Gate, protocol_dir: Path) -> Dict[str, str]:
    manifest = load_protocol_manifest()
    hashes: Dict[str, str] = {}
    for axis in AXES_USED:
        try:
            protocol = load_protocol(axis, directory=protocol_dir,
                                     manifest=manifest)
        except Exception as exc:                    # noqa: BLE001 fail closed
            gate.check(f"{axis}_protocol_digests", False, f"{exc}")
            continue
        hashes.update(protocol.hashes())
        gate.check(f"{axis}_protocol_digests", True,
                   f"instructions {protocol.instructions_sha256[:12]}, "
                   f"template {protocol.user_template_sha256[:12]}, "
                   f"schema {protocol.schema_sha256[:12]}")
    return hashes


def gate_packet(gate: Gate, packet: Path, protocol_dir: Path,
                dataset: Path) -> None:
    if not packet.is_dir():
        gate.check("packet_exists", False, f"{packet} is not a directory")
        return
    gate.check("packet_carries_no_axis1_file",
               not (packet / f"{AXIS1}_judge_input.jsonl").exists(),
               "Axis 1 is prompt-only and is not run in this phase")

    manifest = load_protocol_manifest()
    protocols = {axis: load_protocol(axis, directory=protocol_dir,
                                     manifest=manifest)
                 for axis in AXES_USED}
    screen_row_ids = sorted(load_prompt_texts(dataset, Checklist("preflight")))

    all_ids: List[str] = []
    ok_render = True
    for axis in AXES_USED:
        path = packet / f"{axis}_judge_input.jsonl"
        if not path.is_file():
            gate.check(f"{axis}_rows", False, f"{path} is absent")
            ok_render = False
            continue
        rows = read_jsonl(path)
        gate.check(f"{axis}_row_count",
                   len(rows) == PLANNED_CALLS_BY_AXIS[axis],
                   f"{len(rows)} row(s), expected "
                   f"{PLANNED_CALLS_BY_AXIS[axis]}")
        for row in rows:
            all_ids.append(row["row_id"])
            try:
                label = f"{axis} row {row['row_id']}"
                assert_projection_is_exact(row, axis, label=label,
                                           prefix=ROW_PREFIX[axis])
                assert_row_is_blind(row, axis)
                assert_row_has_no_forbidden_field(row, f"{axis} row")
                assert_opaque(row["row_id"], ROW_PREFIX[axis])
                system, user = protocols[axis].render(row)
                assert_rendered_request_is_projection(system, user, row,
                                                      protocols[axis], label)
                assert_no_corpus_row_id(user, screen_row_ids, label)
            except Exception as exc:                # noqa: BLE001 fail closed
                gate.check(f"{axis}_rows_are_blind", False, f"{exc}")
                ok_render = False
                break
        else:
            gate.check(f"{axis}_rows_are_blind", True,
                       f"{len(rows)} row(s) carry exactly "
                       f"{list(JUDGE_ROW_FIELDS[axis])}, render to the frozen "
                       f"template filled with exactly those values, and name "
                       f"no corpus row")
    gate.check("every_request_uses_one_protocol_generation", ok_render,
               "every rendered request resolves to the frozen instructions, "
               "template and schema for its axis")

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
    ok_cells = True
    for entry in entries:
        try:
            cells.add(cell_key(entry, str(map_path)))
        except Exception as exc:                    # noqa: BLE001 fail closed
            gate.check("private_row_map_cells", False, f"{exc}")
            ok_cells = False
            break
    if ok_cells:
        gate.check("private_row_map_cells", True,
                   f"{len(cells)} cell(s) over {len(entries)} row(s)")
    gate.check("one_row_per_cell", len(cells) == len(entries) == len(all_ids),
               f"{len(entries)} row(s), {len(cells)} cell(s), "
               f"{len(all_ids)} judged row(s)")
    gate.check("private_map_is_outside_git",
               REPO_ROOT not in map_path.resolve().parents,
               f"{map_path.resolve().parent}")


def gate_judges(gate: Gate) -> None:
    manifest = load_protocol_manifest()
    for role, key in (("primary", JUDGES[0]), ("secondary", JUDGES[1])):
        spec = manifest["judges"][role]
        gate.check(f"{key}_identifier", spec["key"] == key,
                   f"{spec['model']} (accepts {spec['expected_returned']})")
    gate.check("both_judges_run_on_every_row",
               manifest["judges"]["both_run_on_every_row"] is True, "yes")
    settings = manifest["generation_settings"]
    gate.check("temperature", float(settings["temperature"]) == TEMPERATURE,
               str(TEMPERATURE))
    gate.check("max_tokens", int(settings["max_tokens"]) == MAX_TOKENS,
               str(MAX_TOKENS))
    gate.check("auto_retry_disabled",
               "disabled" in settings["auto_retry"], settings["auto_retry"])
    gate.check("planned_calls", True,
               f"{PLANNED_CALLS_PER_JUDGE} per judge, "
               f"{PLANNED_CALLS_BOTH_JUDGES} across both")


def gate_output_dirs(gate: Gate, runs: Path, protocol_hashes: Dict[str, str]
                     ) -> None:
    if not runs.exists():
        gate.check("no_incompatible_run_directory", True,
                   f"{runs} does not exist yet")
        return
    bad: List[str] = []
    for directory in sorted(p for p in runs.iterdir() if p.is_dir()):
        path = directory / "run_manifest.json"
        if not path.is_file():
            continue
        recorded = json.loads(path.read_text(encoding="utf-8")).get(
            "protocol_hashes", {})
        for key, value in recorded.items():
            if key in protocol_hashes and protocol_hashes[key] != value:
                bad.append(f"{directory.name}:{key}")
    gate.check("no_incompatible_run_directory", not bad,
               "no existing run used a different protocol digest" if not bad
               else f"{bad[:3]}")


def gate_predeclaration(gate: Gate) -> None:
    if not JUDGING_MANIFEST.exists():
        gate.check("predeclaration_exists", False,
                   f"{JUDGING_MANIFEST} is absent")
        return
    manifest = json.loads(JUDGING_MANIFEST.read_text(encoding="utf-8"))
    gate.check("predeclaration_exists", True, JUDGING_MANIFEST.name)
    gate.check("repair_operating_point_rule_unchanged",
               manifest.get("repair_operating_point_rule")
               == REPAIR_OPERATING_POINT_RULE,
               "byte-identical to routegate.focusedjudging; alpha_R = 0.15 is "
               "fixed in advance and is not selected from the E outcomes")
    gate.check("h_regression_rule_unchanged",
               manifest.get("h_regression_rule") == H_REGRESSION_RULE,
               "byte-identical to routegate.focusedjudging")
    gate.check("input_records_sha256_pinned",
               manifest.get("input", {}).get("records_sha256")
               == FROZEN_INPUT["records_sha256"],
               FROZEN_INPUT["records_sha256"])
    procedure = manifest.get("adjudication_procedure")
    gate.check(
        "adjudication_procedure_declared",
        procedure in ADJUDICATION_PROCEDURES,
        f"{procedure!r}. Two frozen procedures exist, "
        f"{list(ADJUDICATION_PROCEDURES)}, they produce different results, and "
        f"neither is a default. Declare one in {JUDGING_MANIFEST.name} before "
        f"spending anything.")


def gate_code(gate: Gate) -> None:
    here = Path(__file__).resolve().parent
    files = [here / name for name in STAGE_FILES if (here / name).is_file()]
    files.append(REPO_ROOT / "src" / "routegate" / "focusedjudging.py")
    offenders = [f.name for f in files if reads_legacy_stress_manifest(f)]
    gate.check("legacy_stress_set_is_not_read", not offenders,
               f"{len(files)} file(s) checked; none opens it"
               if not offenders else f"{offenders}")


def gate_paths(gate: Gate) -> None:
    path = REPO_ROOT / "configs" / "paths.local.json"
    if not path.is_file():
        gate.check("no_configured_path_is_sealed", False, f"{path} is absent")
        return
    paths = json.loads(path.read_text(encoding="utf-8"))
    sealed = sorted(k for k, v in paths.items()
                    if isinstance(v, str) and looks_sealed(v))
    gate.check("no_configured_path_is_sealed", not sealed,
               f"checked against {len(SEALED_PATH_TOKENS)} token(s)"
               if not sealed else f"{sealed}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-head", default=None)
    parser.add_argument("--consolidated", type=Path, default=None)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--runs", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--stage", choices=("input", "packet", "judging"),
                        default="judging",
                        help="input: before the packet is built. packet: after "
                             "it is built. judging: everything, before a call.")
    args = parser.parse_args(argv)

    gate = Gate()
    try:
        root = judging_root()
        consolidated = args.consolidated or (root / CONSOLIDATED_SUBDIR)
        packet = args.packet or (root / "judging" / "packet")
        runs = args.runs or (root / "judging" / "runs")
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        dataset = args.dataset
        if dataset is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            dataset = Path(paths["dataset"])
    except Exception as exc:                        # noqa: BLE001 fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("repository")
    gate_repository(gate, args.expect_head)
    print("paths")
    gate_paths(gate)
    print("code")
    gate_code(gate)
    print("predeclaration")
    gate_predeclaration(gate)
    print("input")
    gate_input(gate, consolidated, dataset)
    print("protocol")
    hashes = gate_protocol(gate, protocol_dir)
    print("judges")
    gate_judges(gate)
    if args.stage in ("packet", "judging"):
        print("packet")
        gate_packet(gate, packet, protocol_dir, dataset)
    if args.stage == "judging":
        print("output")
        gate_output_dirs(gate, runs, hashes)

    print()
    passed = len(gate.rows) - gate.failed
    print(f"{passed}/{len(gate.rows)} gate(s) passed")
    if gate.failed:
        print(f"[STOP] {gate.failed} gate(s) failed. A failure means stop, not "
              f"adapt the check.", file=sys.stderr)
        return 1
    print("[OK] nothing was called, nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
