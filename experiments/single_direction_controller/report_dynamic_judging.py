#!/usr/bin/env python3
"""The three-arm reading of the dynamic controller arm. Offline; billed never.

    python experiments/single_direction_controller/report_dynamic_judging.py \
        --dry-run

The common engine emits verdict tallies with explicit denominators and stops
there, on purpose. This is the experiment's own reading, and it compares three
arms over the same 480 development rows:

    baseline                 unsteered
    repair_015_blocks31_32   static B_d1 - E repair at alpha_R = 0.15 on
                             historical blocks 31 and 32
    dynamic_raw              the audited dynamic coefficient controller

It reports, per arm: E substantive and full repair, B material and
non-material degradation, and the H actionable-harmful-help split; then the
baseline-relative H regressions (NO at baseline, YES at the arm), the per-row
H transition table, the intervention rate and the g_R distribution.

Nothing here restates a rule. The E, B and H readings are
``routegate.twositejudging.e_reading``, ``b_reading`` and ``h_reading``, which
were frozen before any two-site verdict existed and are called unchanged for
all three arms. The per-row table is ``phase2judging.per_row_transitions``.

It fails closed. A missing adjudication is a diagnosis, not an empty column: a
comparison that quietly lost an arm reads as that arm having no verdicts, and
that is the one wrong answer this report must never give. ``--dry-run``
validates every input it can reach and writes nothing.

No judge is called, no response is generated, no completed artifact is
modified and the held-out split is not reachable from here.
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
    FINAL_REPORT_FILE,
    PATHS_KEY_JUDGING,
    PATHS_KEY_PHASE1,
    PATHS_KEY_RUN,
    PATHS_KEY_STATIC,
    PHASE1_ADJUDICATED_SHA256,
    STATIC_ADJUDICATED_SHA256,
    comparison_report,
    load_adjudicated,
    load_coefficients,
    normalise_prior_cells,
)
from routegate.judgingprotocol import (                           # noqa: E402
    ADJUDICATED_CELLS_FILE,
    resolve_paths,
)
from routegate.validation import Checklist, ValidationError       # noqa: E402

ADJUDICATED_SUBDIR = "judging/adjudicated"
REPORT_SUBDIR = "judging/report"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudicated", type=Path, default=None,
                        help="this run's adjudicated cells; defaults to "
                             f"paths.local.json[{PATHS_KEY_JUDGING}]/"
                             f"{ADJUDICATED_SUBDIR}/{ADJUDICATED_CELLS_FILE}")
    parser.add_argument("--static", type=Path, default=None,
                        help="the completed two-site adjudication")
    parser.add_argument("--phase1", type=Path, default=None,
                        help="the completed phase-1 adjudication, which holds "
                             "the unsteered baseline verdicts")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="the completed dynamic_raw output, for the "
                             "applied-coefficient table")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"defaults to paths.local.json"
                             f"[{PATHS_KEY_JUDGING}]/{REPORT_SUBDIR}")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every gate and write nothing")
    args = parser.parse_args(argv)

    checks = Checklist("dynamic_arms_v1_judging_v1_final_report")
    try:
        paths = resolve_paths()
        judging = Path(paths[PATHS_KEY_JUDGING])
        adjudicated = args.adjudicated or (
            judging / ADJUDICATED_SUBDIR / ADJUDICATED_CELLS_FILE)
        static = args.static or (
            Path(paths[PATHS_KEY_STATIC]) / ADJUDICATED_SUBDIR
            / ADJUDICATED_CELLS_FILE)
        phase1 = args.phase1 or (
            Path(paths[PATHS_KEY_PHASE1]) / ADJUDICATED_SUBDIR
            / ADJUDICATED_CELLS_FILE)
        run_dir = args.run_dir or Path(paths[PATHS_KEY_RUN])
        out_dir = args.output_dir or (judging / REPORT_SUBDIR)

        coefficients = load_coefficients(run_dir, checks)
        static_cells = normalise_prior_cells(load_adjudicated(
            static, STATIC_ADJUDICATED_SHA256, "static_adjudication", checks))
        phase1_cells = normalise_prior_cells(load_adjudicated(
            phase1, PHASE1_ADJUDICATED_SHA256, "phase1_adjudication", checks))
        dynamic_cells = load_adjudicated(
            adjudicated, None, "dynamic_adjudication", checks)
        report = comparison_report(dynamic_cells, static_cells, phase1_cells,
                                   coefficients, checks)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        _explain(checks)
        return 1
    except Exception as exc:                       # noqa: BLE001 - fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        _explain(checks)
        return 1

    for arm in report["arms"]:
        reading = report["arm_readings"][arm]
        line = [f"[report] {arm}"]
        for name in ("E", "B", "H"):
            block = reading.get(name)
            line.append(f"{name}=" + ("-" if block is None else
                                      f"{block['decided_denominator']} decided"))
        print(" ".join(line))
    for arm, block in sorted(report["h_regressions"].items()):
        print(f"[report] {arm}: {block['regressions']} H regression(s) over "
              f"{block['decided_denominator']} paired decided row(s)")
    every = report["coefficients"]["all"]
    print(f"[report] g_R median {every['median']:.6f}, mean {every['mean']:.6f}"
          f", max {every['max']:.6f}; applied mean "
          f"{every['applied_mean']:.6f}")
    checklist = report["checks"]
    print(f"[report] {checklist['checks_run']} check(s), all passed: "
          f"{checklist['all_passed']}")
    if args.dry_run:
        print("[report] dry run: every gate passed; nothing was written")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FINAL_REPORT_FILE
    target.write_text(json.dumps(report, indent=2, sort_keys=True),
                      encoding="utf-8")
    print(f"[report] wrote {target}")
    return 0


def _explain(checks: Checklist) -> None:
    """What had already passed when it stopped, so the failure is locatable."""
    print(f"[report] {len(checks.entries)} check(s) ran before the stop; "
          f"{len(checks.failures)} failed", file=sys.stderr)
    for entry in checks.entries[-3:]:
        print(f"[report]   {entry['status']} {entry['check']}: "
              f"{entry['detail']}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
