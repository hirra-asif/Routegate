#!/usr/bin/env python3
"""Write the phase-2 predeclaration, once, and prove it says what runs.

Offline and deterministic. Loads no model, generates nothing, judges nothing
and calls no service. It takes what ``routegate.phase2`` declares, checks it
against the committed selection manifest, and writes
``manifests/focused_direction_validation_v1_phase2.json``.

Why it refuses to overwrite
---------------------------
A predeclaration that can be rewritten is not one. If the file already exists
and differs from what the package now declares, this stops and says so rather
than replacing it: the difference is either a change somebody made on purpose
-- which is a decision to record, not a file to silently refresh -- or a drift
nobody meant, which is exactly what a freeze exists to catch. Re-freezing is
deliberate: remove the file by hand, with a reason, and run this again.

An identical file is not an overwrite. Re-running on an unchanged tree writes
nothing and re-verifies what is there, so this is safe in a runbook.

Usage
-----
    python experiments/dual_direction_controller/freeze_phase2.py --dry-run
    python experiments/dual_direction_controller/freeze_phase2.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.phase2 import (                                      # noqa: E402
    PHASE2_PREDECLARATION,
    PHASE2_SELECTION_MANIFEST,
    Phase2Error,
    build_phase2_predeclaration,
    verify_phase2_predeclaration,
)
from routegate.validation import Checklist, ValidationError         # noqa: E402


def serialise(payload: dict) -> str:
    """One canonical rendering, so "identical" is a byte comparison."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_atomically(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename.

    A half-written predeclaration is worse than none: it would be read as the
    frozen record by whatever ran next. ``os.replace`` is atomic within a
    filesystem, so a reader sees either the old file or the whole new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent),
        prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def print_arm_table(payload: dict) -> None:
    """The arm table, with the status each arm is actually in.

    "generated" would be a lie in the present tense: zero phase-2 records
    exist. The eight arms are TO GENERATE; the two reference arms are already
    recorded, in phase 1, and are joined afterwards.
    """
    arms = payload["arms"]
    rows = payload["counts"]["rows"]
    print(f"[freeze] arm table ({payload['name']})")
    print(f"[freeze]   {'arm_id':<22} {'alpha_R':>7} {'alpha_S':>7} "
          f"{'cells':>6}  status")
    for arm in arms["generated"]:
        kind = "safety-only" if arm["alpha_R"] == 0.0 else "combined"
        print(f"[freeze]   {arm['arm_id']:<22} {arm['alpha_R']:>7.2f} "
              f"{arm['alpha_S']:>7.2f} {rows:>6}  TO GENERATE ({kind})")
    for arm in arms["reference"]:
        print(f"[freeze]   {arm['arm_id']:<22} {arm['alpha_R']:>7.2f} "
              f"{arm['alpha_S']:>7.2f} {rows:>6}  already recorded in phase 1, "
              f"not regenerated")
    counts = payload["counts"]
    print(f"[freeze]   to generate {counts['new_generations']}  |  inherited "
          f"{counts['reference_records']}  |  analysis table "
          f"{counts['analysis_table_records']}")
    print(f"[freeze]   phase-2 generations that exist today: "
          f"{payload['phase2_generations_run']}")


def freeze(path: Path, write: bool) -> int:
    checks = Checklist("phase2_freeze")

    # 1. what the package declares, with the selection gates run inside it
    payload = build_phase2_predeclaration()
    text = serialise(payload)
    checks.record("selection_verified", True,
                  f"{payload['selection']['rows']} row(s), "
                  f"{payload['selection']['selected_rows_sha256'][:16]}")
    checks.record(
        "counts_derived", True,
        f"{payload['counts']['rows']} x {payload['counts']['new_arms']} = "
        f"{payload['counts']['new_generations']} generated, "
        f"{payload['counts']['reference_records']} inherited, "
        f"{payload['counts']['analysis_table_records']} in the table")

    # 2. an existing file is compared, never assumed replaceable
    existing = path.read_text(encoding="utf-8") if path.is_file() else None
    identical = existing == text
    if existing is not None and not identical:
        raise Phase2Error(
            f"{path.name} already exists and differs from what "
            f"routegate.phase2 now declares. A predeclaration is not "
            f"refreshed in place: either the package changed after the freeze "
            f"-- which is a decision to record -- or something drifted. "
            f"Re-freezing is deliberate: remove {path} by hand, with a "
            f"reason, and run this again.")
    checks.record("existing_manifest", True,
                  "absent" if existing is None else
                  "present and identical to what this run would write")

    print_arm_table(payload)

    if not write:
        print(f"[freeze] dry run: nothing written")
        print(f"[freeze] would write {path}"
              if existing is None else
              f"[freeze] {path.name} is already identical; a real run would "
              f"write nothing")
        print(json.dumps(payload["counts"], indent=2, sort_keys=True))
        return 0

    if identical:
        print(f"[freeze] {path.name} is already identical; nothing written")
    else:
        write_atomically(path, text)
        print(f"[freeze] wrote {path}")

    # 3. verify what is ON DISK, not what was in memory
    verify_phase2_predeclaration(path, checks=checks)
    reread = path.read_text(encoding="utf-8")
    checks.require("written_file_is_canonical", reread == text,
                   "the file on disk is not the canonical rendering")
    print(f"[freeze] verified {path.name}: "
          f"{checks.as_dict()['checks_run']} check(s), all passed")
    for entry in checks.entries:
        print(f"[freeze]   [{entry['status']}] {entry['check']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PHASE2_PREDECLARATION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(f"[freeze] selection {PHASE2_SELECTION_MANIFEST.name}")
    try:
        return freeze(args.output, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
