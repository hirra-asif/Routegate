#!/usr/bin/env python3
"""Close the one-pass budget result and freeze the two-site candidate.

Offline and deterministic. Loads no model, generates nothing, judges nothing,
trains nothing and calls no service. It verifies the completed v2 run against
its committed digests, derives the candidate's sites from that run's own
per-block verdicts, and writes two immutable records:

    manifests/one_pass_budget_equivalence_closure_v1.json
        what the question was, what the answer is, block 33's infeasibility,
        the rejected runtime-input alternative, that no tolerance moved, and
        the candidate consequence.

    manifests/two_site_actuator_candidate_v1.json
        historical blocks 31 and 32 -- exactly the sites v2 cleared -- bound to
        the v2 summary digest, at the unchanged validated direction and
        strength, with no controller and no safety coefficient.

Why it refuses to overwrite
---------------------------
Both files are immutable once written. If one exists and differs from what the
package now declares, this stops and says so rather than replacing it. An
identical file is not an overwrite: re-running on an unchanged tree writes
nothing and re-verifies what is there.

Usage
-----
    python experiments/single_direction_controller/freeze_two_site_candidate.py --dry-run
    python experiments/single_direction_controller/freeze_two_site_candidate.py
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

from routegate.twosite import (                                     # noqa: E402
    CANDIDATE_BLOCKS,
    CANDIDATE_BUDGET_STATES,
    CANDIDATE_MANIFEST,
    CANDIDATE_STATES,
    CLOSURE_MANIFEST,
    DROPPED_BLOCKS,
    TwoSiteError,
    assert_sites_are_the_passing_sites,
    build_candidate_record,
    build_closure_record,
    passing_blocks_from_summary,
    verify_records,
    verify_v2_result,
)
from routegate.validation import Checklist, ValidationError        # noqa: E402


def serialise(payload: Dict[str, Any]) -> str:
    """One canonical rendering, so "identical" is a byte comparison."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_atomically(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename."""
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


def write_one(path: Path, payload: Dict[str, Any], write: bool,
              checks: Checklist, label: str) -> None:
    text = serialise(payload)
    existing = path.read_text(encoding="utf-8") if path.is_file() else None
    identical = existing == text
    if existing is not None and not identical:
        raise TwoSiteError(
            f"{path.name} already exists and differs from what the package now "
            f"declares. It is immutable and is not refreshed in place: either "
            f"the package changed after it was written -- which is a decision "
            f"to record, as a new version beside it -- or something drifted. "
            f"Remove {path} by hand, with a reason, and run this again.")
    checks.record(f"existing_{label}", True,
                  "absent" if existing is None else
                  "present and identical to what this run would write")
    if not write:
        print(f"[freeze] dry run: would "
              f"{'write' if existing is None else 'leave'} {path}")
        return
    if identical:
        print(f"[freeze] {path.name} is already identical; nothing written")
    else:
        write_atomically(path, text)
        print(f"[freeze] wrote {path}")


def freeze(write: bool) -> int:
    checks = Checklist("two_site_candidate_freeze")

    # 1. the completed run, against its committed digests and its own numbers
    outcome = verify_v2_result(checks=checks)
    if not outcome["reachable"]:
        raise TwoSiteError(
            "the v2 run is not present on this machine. The candidate's sites "
            "are derived from that run's per-block verdicts; they are not "
            "typed in, and this freeze does not proceed without it.")
    summary = outcome["summary"]
    print(f"[freeze] v2 verified: decision "
          f"{summary['decision']['decision']}, {summary['rows']} row(s), "
          f"tolerance {summary['decision']['tolerance']}")
    for block in sorted(summary["per_block"], key=int):
        entry = summary["per_block"][block]
        print(f"[freeze]   block {block}: max {entry['max']:.6e}  "
              f"{entry['n_exceeding_tolerance']:>3} over tolerance  "
              f"{'PASS' if entry['within_tolerance'] else 'FAIL'}")
    variant = summary["block_33_runtime_input_variant"]
    print(f"[freeze]   block 33 runtime-input variant: max "
          f"{variant['max']:.6e}  {variant['n_exceeding_tolerance']} over "
          f"tolerance  REJECTED (worse)")

    # 2. the sites, derived from that run and not asserted beside it
    derived = passing_blocks_from_summary(summary)
    assert_sites_are_the_passing_sites(CANDIDATE_BLOCKS, summary, checks)
    print(f"[freeze] sites derived from the v2 summary: {list(derived)}")
    print(f"[freeze] candidate historical blocks {list(CANDIDATE_BLOCKS)} -> "
          f"package states {list(CANDIDATE_STATES)}, budgeted against states "
          f"{list(CANDIDATE_BUDGET_STATES)}")
    print(f"[freeze] dropped historical block(s) {list(DROPPED_BLOCKS)}")

    # 3. the two records
    closure = build_closure_record()
    candidate = build_candidate_record()
    write_one(CLOSURE_MANIFEST, closure, write, checks, "closure")
    write_one(CANDIDATE_MANIFEST, candidate, write, checks, "candidate")

    if not write:
        print("[freeze] dry run: nothing written")
        return 0

    verify_records(checks=checks)
    print(f"[freeze] verified: {checks.as_dict()['checks_run']} check(s), all "
          f"passed: {checks.all_passed}")
    return 0 if checks.all_passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        return freeze(write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
