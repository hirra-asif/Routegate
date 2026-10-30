#!/usr/bin/env python3
"""The final 180-cell reading of the rotation screen, and the frozen decision.

    python experiments/rotation_safety_direction/report_rotation_judging.py \
        --manifest manifests/rotation_safety_direction_v2_judging_v1.json \
        --dry-run

Run after the common engine's ``adjudicate`` stage. It assembles the analysis
table -- 144 cells adjudicated here plus the 36 ``baseline`` and
``repair_015`` cells inherited from the completed phase-1 adjudication --
derives the four counts per beta, and hands them to the authoritative decision
function.

**The decision is not taken here.**
``routegate.rotationfreeze.evaluate_rotation_decision`` is phase 2's own rule,
reached through ``routegate.phase2.evaluate_phase2_decision``. Its recovery
threshold, its rotation-only veto, its tie-break and its UNDETERMINED branch
are not copied, restated, reinterpreted or weakened by anything in this path.

Nothing here re-judges a cell, writes to the phase-1 adjudication or touches
the completed screen.
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.judgingstages import resolved_dirs                  # noqa: E402
from routegate.judgingprotocol import (                            # noqa: E402
    load_manifest,
    stage_dir,
)
from routegate.rotationjudging import (                            # noqa: E402
    FINAL_REPORT_FILE,
    JUDGING_MANIFEST,
    final_report,
)
from routegate.validation import ValidationError                   # noqa: E402

#: Where phase 1's adjudicated cells live, under the phase-1 data root.
PHASE1_ADJUDICATED = Path("judging") / "adjudicated"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=JUDGING_MANIFEST)
    parser.add_argument("--phase1-adjudicated", type=Path, default=None,
                        help="phase 1's adjudicated_cells.jsonl or its "
                             "directory; defaults to paths.local.json")
    parser.add_argument("--root", type=Path, default=None,
                        help="override every root; for fixtures and tests")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every gate and write nothing")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        if args.root is not None:
            dirs = resolved_dirs(manifest, args.root)
        else:
            dirs = {name: stage_dir(manifest, name)
                    for name in manifest.outputs if name != "root_key"}
        phase1 = args.phase1_adjudicated
        if phase1 is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            phase1 = (Path(paths["focused_direction_validation_dir"])
                      / PHASE1_ADJUDICATED)
        summary = final_report(dirs, phase1, write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                       # noqa: BLE001 - fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    counts = summary["counts"]
    decision = summary["decision"]
    print(f"[report] {counts['analysis_table_records']} cell(s): "
          f"{counts['judged_here']} judged here + {counts['inherited']} "
          f"inherited")
    print(f"[report] {counts['decided_cells']} decided, "
          f"{counts['unresolved_cells']} unresolved")
    for beta, entry in sorted(summary["by_beta"].items()):
        print(f"[report] beta {beta}: "
              f"recovered {entry['combined_recovered']}/{entry['denominator']}"
              f" at {entry['combined_arm']}, "
              f"harmful alone {entry['rotation_only_harmful']} at "
              f"{entry['rotation_only_arm']}")
    print(f"[report] rule {decision['rule']} via {decision['evaluated_by']}")
    print(f"[report] OUTCOME {decision['outcome']}")
    if decision.get("selected_beta") is not None:
        print(f"[report] selected beta {decision['selected_beta']} "
              f"({decision['selected_arm']})")
    print(f"[report] {decision['statement']}")
    if args.dry_run:
        print("[report] dry run: every gate passed; nothing was written")
        return 0
    print(f"[report] wrote {Path(dirs['report']) / FINAL_REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
