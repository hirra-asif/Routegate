#!/usr/bin/env python3
"""Prepare and assess the next E/B/H comparison; generation is a later step."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def status(root, diagnostic_root):
    from routegate import followupinputs as inputs
    result = {"stage": "status", "steps": []}
    diagnostic = {"step": 1, "name": "Cancellation diagnostic", "state": "Awaiting initialisation"}
    if (diagnostic_root / "study_manifest.json").is_file():
        from routegate.cancellationexperiment import progress
        recorded = progress(diagnostic_root, REPO)
        diagnostic.update(state="Generation complete; assessment pending" if recorded["complete"] else "Generation incomplete",
                          completed=recorded["successful"], expected=162)
        report = diagnostic_root / "assessment_v1/scientific_report.json"
        if report.is_file():
            diagnostic["state"] = "Assessment report present; interpret before choosing the next edit"
    result["steps"].append(diagnostic)
    second = {"step": 2, "name": "E/B/H development comparison", "state": "Inputs not yet prepared"}
    if (root / "input_manifest.json").is_file():
        manifest, rows = inputs.verify(root)
        second.update(state="Inputs ready; comparison and fresh generations pending",
                      prompts=len(rows), class_counts=manifest["class_counts"], folds=5)
        from routegate.followupjudging import PLAN_FILE, verify_plan, load_generation_records
        if (root / PLAN_FILE).is_file():
            plan, selected = verify_plan(root, REPO)
            second.update(state="Comparison fixed; generation pending", expected_generations=plan["expected_generations"],
                          arms=plan["arm_ids"])
            record_path = root / "generations/records.jsonl"
            if record_path.is_file():
                lines = [line for line in record_path.read_text().splitlines() if line.strip()]
                second.update(state="Generation records present; full validation follows at assessment preparation",
                              recorded_lines=len(lines))
                if len(lines) == plan["expected_generations"]:
                    load_generation_records(root, plan, selected)
                    second["state"] = "Generation complete and verified; assessment/reporting next"
            if (root / "results/report.json").is_file():
                second["state"] = "Paired results report present; interpret before selector training"
    result["steps"].extend([
        second,
        {"step": 3, "name": "Learn which prompts benefit from the same edit", "state": "Not started; needs useful paired response outcomes"},
        {"step": 4, "name": "Compare selectors using the same edit", "state": "Not started"},
        {"step": 5, "name": "External 380-prompt evaluation", "state": "Later, after development choices are fixed; disclose earlier benchmark use"},
    ])
    result["next_action"] = "Finish the 162-generation diagnostic and its assessment; prepare the 480-row input inventory in parallel."
    return result


@contextlib.contextmanager
def writer_lock(root, enabled):
    if not enabled:
        yield
        return
    import fcntl
    if not (root / "input_manifest.json").is_file():
        raise ValueError("Run prepare-inputs first")
    with (root / "followup.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another follow-up stage is using this output folder") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-inputs", "verify-inputs", "status", "selftest", "bind-plan",
                                          "prepare", "build", "preflight", "dryrun", "judge", "progress",
                                          "join", "export", "adjudicate", "report", "score"))
    parser.add_argument("--study-root", type=Path, default=Path.home() / "routegate-data/repair_preservation_v1")
    parser.add_argument("--diagnostic-root", type=Path, default=Path.home() / "routegate-data/cancellation_controls_v1")
    parser.add_argument("--data-root", type=Path, default=Path.home() / "routegate-data")
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--judge", choices=("sonnet", "opus"))
    parser.add_argument("--completed", type=Path)
    parser.add_argument("--reviewer-provenance", type=Path)
    parser.add_argument("--carried", type=Path)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from routegate import followupinputs as inputs
    from routegate.validation import ValidationError
    try:
        if args.dry_run and args.stage in ("prepare-inputs", "verify-inputs", "status", "selftest", "bind-plan", "score"):
            raise ValueError("--dry-run applies to the judging stages; use verify-inputs or status for a read-only inventory check")
        if args.stage == "prepare-inputs":
            manifest = inputs.prepare(REPO, args.data_root, args.study_root)
            result = {"stage": args.stage, "rows": manifest["rows"], "class_counts": manifest["class_counts"],
                      "fold_counts": manifest["fold_counts"], "inventory_sha256": manifest["inventory_sha256"],
                      "candidate_arms": [], "new_generations": 0, "api_calls": 0}
        elif args.stage == "verify-inputs":
            manifest, rows = inputs.verify(args.study_root)
            result = {"stage": args.stage, "rows": len(rows), "verified": True,
                      "inventory_sha256": manifest["inventory_sha256"]}
        elif args.stage == "status":
            result = status(args.study_root, args.diagnostic_root)
        elif args.stage == "selftest":
            os.environ["CANCELLATION_TEST_DATA_ROOT"] = str(args.data_root.resolve())
            os.environ["FOLLOWUP_TEST_DATA_ROOT"] = str(args.data_root.resolve())
            os.environ["ROUTEGATE_FOLLOWUP_TEST_REPO_ROOT"] = str(REPO)
            os.environ["ROUTEGATE_FOLLOWUP_TEST_DATA_ROOT"] = str(args.data_root.resolve())
            inputs.verify(args.study_root)
            suite = unittest.defaultTestLoader.discover(str(REPO / "tests"), pattern="test_followup*.py")
            checked = unittest.TextTestRunner(verbosity=2).run(suite)
            if not checked.wasSuccessful() or checked.skipped:
                raise ValueError("Follow-up tests failed or skipped a required check")
            result = {"stage": "selftest", "passed": checked.testsRun, "new_generations": 0, "api_calls": 0}
        elif args.stage == "bind-plan":
            if args.plan_file is None:
                raise ValueError("bind-plan requires --plan-file after the diagnostic is reviewed")
            from routegate.followupjudging import bind_plan
            with writer_lock(args.study_root, True):
                result = bind_plan(args.study_root, args.plan_file, repo_root=REPO, data_root=args.data_root)
        elif args.stage == "score":
            from routegate.followupscoring import build_report
            with writer_lock(args.study_root, not args.dry_run):
                result = build_report(args.study_root, args.study_root / "assessment_v1", args.study_root / "results")
        else:
            from routegate.followupjudging import run_stage
            mutate = not args.dry_run and args.stage not in ("preflight", "dryrun", "progress")
            with writer_lock(args.study_root, mutate):
                result = run_stage(args.study_root, args.stage, judge=args.judge, completed=args.completed,
                                   reviewer_provenance=args.reviewer_provenance, carried=args.carried,
                                   dry_run=args.dry_run, retry_failures=args.retry_failures, repo_root=REPO)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2 if args.stage == "judge" and result.get("failed", 0) else 0
    except (ValueError, OSError, KeyError, ImportError, ValidationError) as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
