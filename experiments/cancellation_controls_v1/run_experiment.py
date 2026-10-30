#!/usr/bin/env python3
"""Run the fixed development cancellation diagnostic; never train a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "preflight", "selftest", "generate", "progress"))
    parser.add_argument("--study-root", type=Path, default=Path("/home/ucabh21/routegate-data/cancellation_controls_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/ucabh21/routegate-data"))
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--check-model", action="store_true", help="Verify the local checkpoint revision and tokeniser hashes without loading the model")
    parser.add_argument("--limit", type=int, help="Stop after this many successful new generations; resume uses the same study")
    parser.add_argument("--retry-failures", action="store_true", help="Deliberate technical retry after inspecting preserved errors")
    args = parser.parse_args()
    from routegate import cancellationexperiment as experiment
    try:
        if args.stage == "selftest":
            import torch  # Mandatory here: skipping the toy hook tests is not a passed gate.
            os.environ["ROUTEGATE_CANCELLATION_DATA_ROOT"] = str(args.data_root.resolve())
            os.environ["CANCELLATION_TEST_DATA_ROOT"] = str(args.data_root.resolve())
            from routegate.cancellationcontrols import load_selected_rows, build_control_bank, BANK_RELATIVE_PATH
            load_selected_rows(args.repo_root, args.data_root)
            build_control_bank(args.data_root / BANK_RELATIVE_PATH)
            suite = unittest.defaultTestLoader.discover(str(args.repo_root / "tests"), pattern="test_cancellation*.py")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful() or result.skipped:
                raise experiment.StudyError("Selftest failed or skipped a required check; do not start generation")
            print(json.dumps({"stage": "selftest", "passed": result.testsRun,
                              "torch": torch.__version__, "model_loaded": False,
                              "new_generations": 0, "api_calls": 0}, indent=2))
            return 0
        if args.stage == "init":
            manifest = experiment.init_study(args.repo_root, args.data_root, args.study_root)
            result = {"stage": "init", "study_root": str(args.study_root),
                      "study_sha256": manifest["study_sha256"], "rows": 18, "arms": 9,
                      "planned_fresh_generations": 162, "api_calls": 0}
        elif args.stage == "preflight":
            result = experiment.preflight(args.study_root, args.repo_root, args.check_model)
        elif args.stage == "progress":
            result = experiment.progress(args.study_root, args.repo_root)
        else:
            result = experiment.generate(args.study_root, args.repo_root, args.limit, args.retry_failures)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
