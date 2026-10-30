#!/usr/bin/env python3
"""Judge only the complete new cancellation study through the common engine."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from routegate import cancellationjudging as C
from routegate.validation import ValidationError


@contextlib.contextmanager
def exclusive_stage(root, enabled):
    if not enabled:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".judging.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise C.CancellationJudgingError("Another judging stage is using this study.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=C.STAGES)
    parser.add_argument("--judge", choices=("sonnet", "opus"))
    parser.add_argument("--completed", type=Path, help="Completed exported blinded disagreement review.")
    parser.add_argument("--reviewer-provenance", type=Path, help="Truthful reviewer/AI-assistance metadata bound to completed file SHA-256.")
    parser.add_argument("--carried", type=Path, help="Explicit technical-unresolved JSON with an unresolved list.")
    parser.add_argument("--rubrics-root", type=Path, help="Optional directory containing the three byte-identical frozen safety files.")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--retry-failures", action="store_true", help="Deliberately retry previously recorded failed calls.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the selected stage without writes or API calls.")
    args = parser.parse_args(argv)
    study_root = args.study_root.expanduser().resolve()
    mutate = not args.dry_run and args.stage not in ("dryrun", "preflight", "progress")
    try:
        with exclusive_stage(study_root / C.OUTPUT_NAME, mutate):
            result = C.run_stage(study_root, args.stage, judge=args.judge, completed=args.completed,
                                 reviewer_provenance=args.reviewer_provenance, carried=args.carried,
                                 dry_run=args.dry_run, retry_failures=args.retry_failures,
                                 repo_root=args.repo_root, rubrics_root=args.rubrics_root)
        print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "environment")}, indent=2, sort_keys=True))
        return 2 if args.stage == "judge" and result.get("failed", 0) else 0
    except (ValidationError, OSError, ValueError, KeyError) as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
