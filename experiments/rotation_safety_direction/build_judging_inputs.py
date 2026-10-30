#!/usr/bin/env python3
"""Adapt the completed rotation screen into the common engine's input shape.

    python experiments/rotation_safety_direction/build_judging_inputs.py \
        --consolidated <the completed consolidation package> \
        --output-dir <judging root>/inputs --dry-run

The screen is finished and its consolidation is pinned by digest. This script
READS it and writes one file: ``candidate_records.jsonl``, the 144 rows the
engine's ``build`` stage projects into a blind packet, carrying exactly
``screen_row_id``, ``arm_id``, ``row_class``, ``prompt_text`` and
``response_text``.

The prompt text is not in a rotation record -- the screen stored the prompt's
digest, not the prompt -- so it is resolved from the one hash-gated corpus the
screen recorded it was generated from, and every resolved prompt is checked
against the digest the record already carries.

It judges nothing, decides nothing, calls nothing and writes nowhere near the
completed package. ``--dry-run`` runs every gate and writes no file.

The output is deterministic: the manifest pins it by SHA256, so print the
digest after writing it and put that value in the manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.judgingprotocol import resolve_paths                # noqa: E402
from routegate.rotationjudging import (                            # noqa: E402
    CANDIDATE_RECORDS,
    INPUTS_SUBDIR,
    PATHS_KEY_CONSOLIDATED,
    PATHS_KEY_JUDGING,
    build_judging_inputs,
)
from routegate.validation import ValidationError                   # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated", type=Path, default=None,
                        help="the completed consolidation package; defaults "
                             f"to paths.local.json[{PATHS_KEY_CONSOLIDATED}]")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"defaults to paths.local.json"
                             f"[{PATHS_KEY_JUDGING}]/{INPUTS_SUBDIR}")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="the accepted corpus; defaults to paths.local.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every gate and write nothing")
    args = parser.parse_args(argv)

    try:
        paths = resolve_paths()
        consolidated = args.consolidated or Path(
            paths[PATHS_KEY_CONSOLIDATED])
        out_dir = args.output_dir or (
            Path(paths[PATHS_KEY_JUDGING]) / INPUTS_SUBDIR)
        dataset = args.dataset or Path(paths["dataset"])
        summary = build_judging_inputs(consolidated, out_dir, dataset,
                                       write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                       # noqa: BLE001 - fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    checks = summary["checks"]
    print(f"[adapt] {summary['records']} record(s) from "
          f"{summary['rows']} row(s) x {summary['arms']} arm(s)")
    print(f"[adapt] inherited arms not adapted: "
          f"{json.dumps(summary['inherited_arms_not_adapted'])}")
    print(f"[adapt] {checks['checks_run']} check(s), all passed: "
          f"{checks['all_passed']}")
    if args.dry_run:
        print("[adapt] dry run: every gate passed; nothing was written")
        return 0
    print(f"[adapt] wrote {summary['output']}")
    print(f"[adapt] {CANDIDATE_RECORDS} sha256 "
          f"{summary['candidate_records_sha256']}")
    print("[adapt] put that digest in the manifest's "
          "inputs.candidate_records.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
