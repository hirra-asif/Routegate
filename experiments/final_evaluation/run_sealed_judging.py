#!/usr/bin/env python3
"""Use the common judging pipeline on the released, completed v5 evaluation."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from routegate import finaljudging as F
from routegate.validation import ValidationError


@contextlib.contextmanager
def exclusive_run(root, write):
    """Prevent simultaneous mutating stages from duplicating paid calls."""
    if not write:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".judging.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise F.FinalJudgingError("Another judging stage is running in this output directory.")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=F.PHASES, default="baseline")
    parser.add_argument("--stage", required=True,
                        choices=["prepare", "preflight", "build", "dryrun", "judge", "join", "export", "import", "report", "progress"])
    parser.add_argument("--judge", choices=["sonnet", "opus"])
    parser.add_argument("--data-root", type=Path, default=Path.home() / "routegate-data")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--rubrics-root", type=Path, help="optional exported rubric folder for an offline audit")
    parser.add_argument("--completed", type=Path, help="blinded human-review COMPLETED file")
    parser.add_argument("--carried", type=Path, help="explicit technical-unresolved manifest, if separately reviewed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failures", action="store_true", help="deliberately retry recorded failed requests")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    root = F.phase_root(data_root, args.phase)
    write = not args.dry_run and args.stage not in ("preflight", "dryrun", "progress")
    try:
        with exclusive_run(data_root / F.OUTPUT_NAME, write):
            result = F.run_stage(data_root, args.phase, args.stage, judge=args.judge,
                                 completed=args.completed, carried=args.carried,
                                 dry_run=args.dry_run, retry_failures=args.retry_failures,
                                 adapter=args.adapter, rubrics_root=args.rubrics_root)
        shown = {k: v for k, v in result.items() if k not in ("checks", "environment")}
        print(json.dumps(shown, indent=2, sort_keys=True))
        if args.dry_run:
            print("Dry run passed: no files written and no API calls made.")
        if args.stage == "judge" and result.get("failed", 0):
            print("Some calls failed. Their attempts were preserved. Inspect failures before a deliberate retry.")
            return 2
        if args.stage == "export":
            print("Review files: " + str(root / "judging/review"))
        return 0
    except (ValidationError, OSError, ValueError, KeyError) as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
