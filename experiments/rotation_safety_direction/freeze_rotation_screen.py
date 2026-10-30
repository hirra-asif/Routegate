#!/usr/bin/env python3
"""Write the rotation screen's predeclaration, once, and prove it says what runs.

Offline and deterministic. Loads no model, generates nothing, judges nothing
and calls no service. It takes what ``routegate.rotationfreeze`` declares,
checks it against the committed phase-2 selection and against the digests of
every completed artifact, and writes
``manifests/rotation_safety_direction_v2.json``.

v1 is superseded and is NOT edited: it stays on disk as written, is
pinned by digest among the preserved artifacts, and is what a reader
consults to see what the screen said before its decision rule was
corrected to phase 2's own. Nothing ever ran under it.

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
    python experiments/rotation_safety_direction/freeze_rotation_screen.py --dry-run
    python experiments/rotation_safety_direction/freeze_rotation_screen.py
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

from routegate.phase2 import verify_phase2_selection                # noqa: E402
from routegate.rotationfreeze import (                              # noqa: E402
    DIGEST_FIELD,
    NAME,
    PREDECLARATION,
    RotationFreezeError,
    build_predeclaration,
    verify_predeclaration,
    verify_preserved_artifacts,
)
from routegate.validation import Checklist, ValidationError          # noqa: E402


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

    "generated" would be a lie in the present tense: zero rotation records
    exist. The eight arms are TO GENERATE; the two reference arms are already
    recorded, in phase 1, and are joined afterwards.
    """
    arms = payload["arms"]
    rows = payload["counts"]["rows"]
    print(f"[freeze] arm table ({payload['name']})")
    print(f"[freeze]   {'arm_id':<24} {'beta':>6} {'alpha_R':>8} "
          f"{'cells':>6}  status")
    for arm in arms["generated"]:
        print(f"[freeze]   {arm['arm_id']:<24} {arm['beta']:>6.2f} "
              f"{arm['alpha_R']:>8.2f} {rows:>6}  TO GENERATE ({arm['kind']}, "
              f"{arm['intervention_order']})")
    for arm_id in arms["reference"]:
        print(f"[freeze]   {arm_id:<24} {'-':>6} {'-':>8} {rows:>6}  already "
              f"recorded in phase 1, not regenerated")
    counts = payload["counts"]
    print(f"[freeze]   to generate {counts['new_generations']}  |  inherited "
          f"{counts['reference_records']}  |  analysis table "
          f"{counts['analysis_table_records']}")
    print(f"[freeze]   rotation generations that exist today: "
          f"{payload['generations_run']}")
    print(f"[freeze]   beta grid {payload['arms']['beta_grid']} -- a FRACTION "
          f"of the per-row, per-layer rotation angle. Not an additive alpha, "
          f"and not a matched perturbation dose.")
    print(f"[freeze]   order {payload['intervention']['intervention_order']} "
          f"at states {payload['intervention']['states']} "
          f"(historical {payload['intervention']['historical_layers']}), "
          f"final prompt token, prefill only")
    rule = payload["decision_rule"]
    print(f"[freeze]   rule {rule['name']} -- phase 2's own, via "
          f"{rule['implemented_by']}")
    print(f"[freeze]     qualify: recover >= {rule['recovery_threshold']} of "
          f"{rule['denominator']} combined AND "
          f"{rule['rotation_only_harm_limit']} harmful alone")
    print(f"[freeze]     select : {rule['selection_among_qualifiers']} "
          f"Tie: {rule['tie_break']}")
    print(f"[freeze]     else   : {' / '.join(rule['outcomes'])}")
    superseded = payload["supersedes"]
    print(f"[freeze]   supersedes {superseded['predeclaration']} "
          f"({superseded['generations_under_it']} generation(s) under it); "
          f"changed: {superseded['what_changed']['field']}")


def freeze(path: Path, write: bool) -> int:
    checks = Checklist("rotation_screen_freeze")

    # 1. the completed record is untouched, before anything else is derived
    preserved = verify_preserved_artifacts(checks=checks)
    print(f"[freeze] {preserved['verified']} completed artifact(s) verified "
          f"byte-for-byte")

    # 2. the selection this screen is bound to, gated by the phase-2 verifier
    verify_phase2_selection(checks=checks)

    # 3. what the package declares
    payload = build_predeclaration()
    text = serialise(payload)
    checks.record("selection_verified", True,
                  f"{payload['frozen_inputs']['selection']['rows']} row(s), "
                  f"{payload['frozen_inputs']['selection']['selected_rows_sha256'][:16]}")
    checks.record(
        "counts_derived", True,
        f"{payload['counts']['rows']} x {payload['counts']['new_arms']} = "
        f"{payload['counts']['new_generations']} generated, "
        f"{payload['counts']['reference_records']} inherited, "
        f"{payload['counts']['analysis_table_records']} in the table")
    checks.record("tolerances_predeclared", True,
                  f"version {payload['intervention']['tolerances_version']}: "
                  f"{json.dumps(payload['intervention']['tolerances'], sort_keys=True)}")

    # 4. an existing file is compared, never assumed replaceable
    existing = path.read_text(encoding="utf-8") if path.is_file() else None
    identical = existing == text
    if existing is not None and not identical:
        raise RotationFreezeError(
            f"{path.name} already exists and differs from what "
            f"routegate.rotationfreeze now declares. A predeclaration is not "
            f"refreshed in place: either the package changed after the freeze "
            f"-- which is a decision to record -- or something drifted. "
            f"Re-freezing is deliberate: remove {path} by hand, with a "
            f"reason, and run this again.")
    checks.record("existing_manifest", True,
                  "absent" if existing is None else
                  "present and identical to what this run would write")

    print_arm_table(payload)
    print(f"[freeze] self digest {payload[DIGEST_FIELD]}")

    if not write:
        print("[freeze] dry run: nothing written")
        print(f"[freeze] would write {path}" if existing is None else
              f"[freeze] {path.name} is already identical; a real run would "
              f"write nothing")
        print(json.dumps(payload["counts"], indent=2, sort_keys=True))
        return 0

    if identical:
        print(f"[freeze] {path.name} is already identical; nothing written")
    else:
        write_atomically(path, text)
        print(f"[freeze] wrote {path}")

    # 5. verify what is ON DISK, not what was in memory
    verify_predeclaration(path, checks=checks)
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
    parser.add_argument("--output", type=Path, default=PREDECLARATION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(f"[freeze] {NAME} -- a CANDIDATE screen; nothing here has "
          f"been run, judged or validated")
    try:
        return freeze(args.output, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
