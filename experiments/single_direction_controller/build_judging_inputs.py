#!/usr/bin/env python3
"""Adapt the completed dynamic controller arm into the engine's input shape.

    python experiments/single_direction_controller/build_judging_inputs.py \
        --run-dir <the completed dynamic_raw output> \
        --output-dir <judging root>/inputs --dry-run

The arm is finished and every one of its four files is pinned by digest. This
script READS them and writes three things:

  inputs/candidate_records.jsonl   480 rows the engine's ``build`` stage
                                   projects into a blind packet, carrying
                                   exactly ``screen_row_id``, ``arm_id``,
                                   ``row_class``, ``prompt_text`` and
                                   ``response_text``
  inputs/reference_records.jsonl   the 178 B-class unsteered responses, the
                                   other side of the pairwise comparison. Only
                                   B: a reference is read for a pairwise rubric
                                   and for nothing else
  frozen/judge_rubrics/            the nine frozen rubric files, copied byte
                                   for byte and verified on copy

The prompt text is not in a dynamic record -- the arm stored the prompt's
digest, not the prompt -- so it is resolved from the same hash-gated
development corpus the arm itself read, and every resolved prompt is checked
against the digest the record already carries.

Before it converts anything it re-derives the audit: that the 213 "already
present" rows are a RESUME prefix in sorted row-id order rather than a class,
that every record is on this arm with both hooks fired, that no record applied
an exactly zero edit, that one coefficient was reused at both sites at
``alpha_R * g_R``, and that every coefficient came from the fold controller
that never trained on its row. Any of those failing stops the stage.

It judges nothing, decides nothing, calls nothing and writes nowhere near the
completed run. ``--dry-run`` runs every gate and writes no file.

The output is deterministic: the manifest pins both files by SHA256, so print
the digests after writing them and put those values in the manifest.
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

from routegate.dynamicjudging import (                            # noqa: E402
    CANDIDATE_RECORDS,
    INPUTS_SUBDIR,
    PATHS_KEY_JUDGING,
    PATHS_KEY_RUN,
    PATHS_KEY_PHASE1,
    PATHS_KEY_STATIC,
    PATHS_KEY_STATIC_INPUT,
    PATHS_KEY_TRIPLETS,
    REFERENCE_RECORDS,
    build_judging_inputs,
)
from routegate.judgingprotocol import (                            # noqa: E402
    ADJUDICATED_CELLS_FILE,
    resolve_paths,
)
from routegate.validation import ValidationError                   # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="the completed dynamic_raw output directory; "
                             f"defaults to paths.local.json[{PATHS_KEY_RUN}]")
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="the consolidated two-site package holding the "
                             "referenced baselines; defaults to "
                             f"paths.local.json[{PATHS_KEY_STATIC_INPUT}]")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"defaults to paths.local.json"
                             f"[{PATHS_KEY_JUDGING}]/{INPUTS_SUBDIR}")
    parser.add_argument("--paths", type=Path, default=None,
                        help="the development input path file, for the corpus")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every gate and write nothing")
    args = parser.parse_args(argv)

    try:
        from routegate.data import load_development_inputs

        paths = resolve_paths()
        run_dir = args.run_dir or Path(paths[PATHS_KEY_RUN])
        baseline_dir = args.baseline_dir or Path(paths[PATHS_KEY_STATIC_INPUT])
        out_dir = args.output_dir or (
            Path(paths[PATHS_KEY_JUDGING]) / INPUTS_SUBDIR)
        inputs = load_development_inputs(
            **({"paths_path": args.paths} if args.paths else {}))
        adjudicated = "judging/adjudicated"
        summary = build_judging_inputs(
            run_dir, baseline_dir, out_dir,
            {PATHS_KEY_TRIPLETS: Path(paths[PATHS_KEY_TRIPLETS]),
             PATHS_KEY_STATIC: Path(paths[PATHS_KEY_STATIC])},
            inputs, write=not args.dry_run,
            prior_adjudications={
                key: Path(paths[key]) / adjudicated / ADJUDICATED_CELLS_FILE
                for key in (PATHS_KEY_STATIC, PATHS_KEY_PHASE1)
                if paths.get(key)})
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                       # noqa: BLE001 - fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    checks = summary["checks"]
    audit = summary["audit"]
    split = summary["resume_split"]
    print(f"[adapt] arm {summary['arm']}, {summary['records']} candidate "
          f"record(s) + {summary['references']} B reference(s)")
    print(f"[adapt] audit {audit['verdict']}: "
          f"{audit['generated_by_this_arm']} generated by this arm, "
          f"{audit['inherited_or_reused']} inherited or reused")
    print(f"[adapt] already present on resume "
          f"{json.dumps(split['already_present_by_class'])}; generated last "
          f"{json.dumps(split['generated_here_by_class'])}")
    reuse = summary["reuse"]
    print(f"[adapt] reuse rule evaluated: {reuse['evaluated']}; "
          f"{reuse['reusable']} cell(s) reusable, {reuse['scheduled_new']} "
          f"scheduled new"
          + (f"; {reuse['material_matches']} match on judge-facing material "
             f"alone, on another arm" if reuse.get("material_matches")
             else ""))
    print(f"[adapt] {summary['rubrics']['files']} frozen rubric file(s) "
          f"staged under {summary['rubrics']['directory']}")
    print(f"[adapt] {checks['checks_run']} check(s), all passed: "
          f"{checks['all_passed']}")
    if args.dry_run:
        print("[adapt] dry run: every gate passed; nothing was written")
        return 0
    print(f"[adapt] wrote {summary['output']}")
    print(f"[adapt] {CANDIDATE_RECORDS} sha256 "
          f"{summary['candidate_records_sha256']}")
    print(f"[adapt] {REFERENCE_RECORDS} sha256 "
          f"{summary['reference_records_sha256']}")
    print("[adapt] put those digests in the manifest's inputs.*.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
