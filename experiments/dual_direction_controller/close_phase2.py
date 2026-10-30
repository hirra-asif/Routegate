#!/usr/bin/env python3
"""Close phase 2: record what it decided, and declare what continues.

Offline and deterministic. Loads no model, generates nothing, judges nothing,
trains nothing and calls no service. It reads one completed result file --
``phase2_readings.json`` -- checks that it is the result this closure claims,
and writes two committed artifacts:

    manifests/focused_direction_validation_v1_phase2_closure.json
        what phase 2 decided: NO_STRENGTH_QUALIFIES, the readings digest,
        d_S = H - E retired as an actuator, d_R = B_d1 - E retained, the
        H-versus-(E+B) probe retained as a sensor, the protocol closed, and
        the identity of the retained 254/254 probe-head run.

    configs/single_direction_controller.json
        the one-direction controller that continues: one sigmoid gate g_R,
        alpha_R = 0.15 * g_R, read at state 32 on the final prompt token, with
        the out-of-fold and deployment protocols fixed.

What it does not touch
----------------------
``manifests/focused_direction_validation_v1_phase2.json`` (the predeclaration),
``manifests/focused_direction_validation_v1_phase2_selection.json`` and
``configs/dual_direction_controller.json`` are historical artifacts. They are
read, checked for presence, and never written. A predeclaration edited to carry
its own outcome is not a predeclaration.

Why it refuses to overwrite
---------------------------
Both files are immutable once written. If one exists and differs from what the
package now declares, this stops and says so rather than replacing it: the
difference is either a change somebody made on purpose -- which is a decision
to record, as a new version -- or a drift nobody meant. An identical file is
not an overwrite; re-running on an unchanged tree writes nothing and re-verifies
what is there, so this is safe in a runbook.

Usage
-----
    python experiments/dual_direction_controller/close_phase2.py --dry-run
    python experiments/dual_direction_controller/close_phase2.py \\
        --readings /path/to/phase2_readings.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.phase2closure import (                              # noqa: E402
    PHASE2_CLOSURE_MANIFEST,
    PHASE2_READINGS_FILENAME,
    PHASE2_READINGS_PATH,
    Phase2ClosureError,
    assert_historical_artifacts_are_preserved,
    build_phase2_closure,
    closure_summary_lines,
    verify_phase2_closure,
    verify_phase2_readings,
    verify_probe_head_run,
)
from routegate.singledirection import (                            # noqa: E402
    SINGLE_DIRECTION_CONFIG,
    build_single_direction_specification,
    verify_single_direction_specification,
)
from routegate.validation import Checklist, ValidationError        # noqa: E402


def serialise(payload: Dict[str, Any]) -> str:
    """One canonical rendering, so "identical" is a byte comparison."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_atomically(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename.

    A half-written closure is worse than none: it would be read as the record
    of what phase 2 decided by whatever ran next. ``os.replace`` is atomic
    within a filesystem, so a reader sees either the old file or the whole new
    one.
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


def default_readings_path() -> Optional[Path]:
    """Where the completed readings sit on this machine, if it says.

    ``configs/paths.local.json`` is the only place an absolute path appears.
    Its absence is not an error here: ``--readings`` names the file directly.
    """
    try:
        from routegate.data import load_local_paths
        paths = load_local_paths()
    except Exception:
        return None
    root = paths.get("focused_direction_validation_dir")
    if not root:
        return None
    return Path(root) / "phase2" / "judging" / "readings" / PHASE2_READINGS_FILENAME


def write_one(path: Path, payload: Dict[str, Any], write: bool,
              checks: Checklist, label: str) -> None:
    """Write one immutable artifact, or prove the one on disk is identical."""
    text = serialise(payload)
    existing = path.read_text(encoding="utf-8") if path.is_file() else None
    identical = existing == text
    if existing is not None and not identical:
        raise Phase2ClosureError(
            f"{path.name} already exists and differs from what the package now "
            f"declares. It is immutable: it is not refreshed in place. Either "
            f"the package changed after the closure -- which is a decision to "
            f"record, as a new version beside this one -- or something "
            f"drifted. Remove {path} by hand, with a reason, and run this "
            f"again.")
    checks.record(f"existing_{label}", True,
                  "absent" if existing is None else
                  "present and identical to what this run would write")
    if not write:
        print(f"[close] dry run: would "
              f"{'write' if existing is None else 'leave'} {path}")
        return
    if identical:
        print(f"[close] {path.name} is already identical; nothing written")
    else:
        write_atomically(path, text)
        print(f"[close] wrote {path}")


def close(readings: Optional[Path], write: bool) -> int:
    checks = Checklist("phase2_closure")

    # 1. the historical artifacts are still there. Closing is not erasing.
    preserved = assert_historical_artifacts_are_preserved(checks)
    for name, entry in sorted(preserved.items()):
        print(f"[close] preserved {entry['path']}  {entry['sha256'][:16]}")

    # 2. the readings say what this closure says they say
    if readings is None:
        raise Phase2ClosureError(
            f"no readings file given and none derivable from "
            f"configs/paths.local.json. Pass --readings; the closure is taken "
            f"from the completed readings ({PHASE2_READINGS_PATH}) and from "
            f"nothing else.")
    summary = verify_phase2_readings(readings, checks)
    print(f"[close] readings {summary['path']}  {summary['sha256']}")
    print(f"[close] outcome  {summary['outcome']}  "
          f"(qualifying strengths {summary['qualifying_strengths']}, "
          f"selected {summary['selected']!r}, "
          f"{summary['unresolved_cells']} unresolved cell(s))")

    # 3. the probe-head run this closure names, when it is reachable
    probe = verify_probe_head_run(checks=checks)
    standing = ("verified on disk" if probe["reachable"]
                else "not present on this machine; the digests are pinned")
    print(f"[close] probe heads {standing}")

    # 4. the two artifacts
    closure = build_phase2_closure()
    specification = build_single_direction_specification()
    for line in closure_summary_lines(closure):
        print(f"[close]   {line}")

    write_one(PHASE2_CLOSURE_MANIFEST, closure, write, checks, "closure")
    write_one(SINGLE_DIRECTION_CONFIG, specification, write, checks,
              "specification")

    if not write:
        print("[close] dry run: nothing written")
        return 0

    # 5. verify what is ON DISK, not what was in memory
    verify_phase2_closure(PHASE2_CLOSURE_MANIFEST, checks=checks)
    verify_single_direction_specification(SINGLE_DIRECTION_CONFIG, checks=checks)
    print(f"[close] verified: {checks.as_dict()['checks_run']} check(s), all "
          f"passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, default=None,
                        help="the completed phase2_readings.json; defaults to "
                             "the one named by configs/paths.local.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    readings = args.readings or default_readings_path()
    try:
        return close(readings, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
