#!/usr/bin/env python3
"""Freeze the dynamic_raw H-transition diagnosis. Offline; billed never.

    python experiments/single_direction_controller/freeze_h_transition_manifest.py --dry-run
    python experiments/single_direction_controller/freeze_h_transition_manifest.py
    python experiments/single_direction_controller/freeze_h_transition_manifest.py --verify

What it writes
--------------
``manifests/dynamic_raw_h_transition_v1.json``: the three adjudicated NO->YES
H rows, every other bucket of the 213, and the single H row the combined hard
guard permits -- each derived from a completed artifact that is checked against
its digest first.

Why now
-------
The operator-isolation arms were added because the guard permits 1 of 213 H
rows and that row is not one of the three ``dynamic_raw`` regressed on. That
claim lived in a README paragraph and in a report nobody had built, so nothing
on disk could be re-derived to check it. This freezes it, before those arms
generate anything that could shape how it is stated.

What it does not do
-------------------
No model is loaded, no response is generated, no judge is called, no sealed
material is read and nothing under the data root is written or modified. Three
completed files are opened read-only and projected to identifiers, labels and
mechanical failure types; prompt text, response text and judge prose are not in
the projection.

An identical file is not an overwrite: re-running on unchanged inputs writes
nothing and re-verifies what is there. A file that differs stops the script
rather than being refreshed -- that difference is either a decision to record
or a drift to catch, and both are for a person.
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

from routegate.dynamicjudging import (                            # noqa: E402
    PATHS_KEY_JUDGING,
    PATHS_KEY_PHASE1,
    PATHS_KEY_RUN,
)
from routegate.htransitionfreeze import (                         # noqa: E402
    ADJUDICATED_RELATIVE,
    DIGEST_FIELD,
    MANIFEST,
    NAME,
    assert_frozen_before_generation,
    assert_no_text,
    build_record,
    derive_guard_relation,
    derive_transitions,
    source_record,
    verify_record,
)
from routegate.judgingprotocol import resolve_paths               # noqa: E402
from routegate.triplets import read_jsonl                         # noqa: E402
from routegate.validation import Checklist, ValidationError       # noqa: E402

RUN_RECORDS_NAME = "screen_records.shard01of01.jsonl"


def serialise(payload: Dict[str, Any]) -> str:
    """One canonical rendering, so "identical" is a byte comparison."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_atomically(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename.

    A half-written freeze record is worse than none: it would be read as the
    pinned diagnosis by whatever consulted it next.
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


def permitted_h_rows(checks: Checklist) -> Sequence[str]:
    """The H rows both frozen hard vetoes permit, recomputed from the gate.

    Read the frozen two-veto diagnostic through the loader extracted from the
    retired guarded pilot, preserving its input hashes and threshold arithmetic.
    """
    from routegate.data import load_development_inputs
    from routegate.historicalveto import load_hard_vetoes

    inputs = load_development_inputs()
    corpus = {r["row_id"]: r["final_ebh"] for r in inputs.manifest_rows}
    vetoes, _provenance = load_hard_vetoes(checks=checks)
    return sorted(rid for rid, reading in vetoes.items()
                  if corpus.get(rid) == "H" and reading.permits)


def announce(payload: Dict[str, Any]) -> None:
    counts = payload["counts"]
    print(f"[freeze] {NAME}: {payload['status']}")
    print(f"[freeze] arm {payload['arm']} vs baseline "
          f"{payload['baseline_arm']}")
    print(f"[freeze] rule: {payload['transition_definition']['regression_predicate']}")
    print(f"[freeze] {counts['h_rows']} H row(s) "
          f"= {counts['paired_baseline_h_rows']} paired "
          f"+ {counts['baseline_absent_to_unresolved']} baseline-absent")
    print(f"[freeze]   paired  {counts['paired_baseline_h_rows']} "
          f"= {counts['decided_paired_rows']} decided "
          f"+ {counts['no_to_unresolved']} NO->unresolved")
    print(f"[freeze]   decided {counts['decided_paired_rows']} "
          f"= {counts['no_to_no']} NO->NO + {counts['no_to_yes']} NO->YES")
    for row in payload["regressions"]:
        print(f"[freeze]   NO->YES {row['screen_row_id']} "
              f"({row['judged_row_id']}, {row['dynamic_raw_label_source']})")
    guard = payload["guard_relation"]
    print(f"[freeze] guard permits {guard['permitted_h_rows']}/"
          f"{guard['of_h_rows']} H row(s): {guard['permitted_h_row_id']}")
    print(f"[freeze]   one of the three regressions: "
          f"{guard['is_one_of_the_regressions']}")
    for logical, entry in sorted(payload["sources"].items()):
        flag = " (pinned here first)" if entry["pinned_here_first"] else ""
        print(f"[freeze] source {logical:<20} {entry['sha256'][:12]}{flag}")
    frozen = payload["frozen_before_generation"]
    print(f"[freeze] frozen before {frozen['arms_not_yet_generated']} "
          f"generated anything (checked: "
          f"{frozen['checked_against_the_data_root']})")
    print(f"[freeze] {DIGEST_FIELD} {payload[DIGEST_FIELD]}")


def build(args: argparse.Namespace, checks: Checklist) -> Dict[str, Any]:
    paths = resolve_paths()
    resolved = {
        "dynamic_adjudicated": args.adjudicated or (
            Path(paths[PATHS_KEY_JUDGING]) / ADJUDICATED_RELATIVE),
        "baseline_adjudicated": args.baseline or (
            Path(paths[PATHS_KEY_PHASE1]) / ADJUDICATED_RELATIVE),
        "dynamic_run_records": args.run_records or (
            Path(paths[PATHS_KEY_RUN]) / RUN_RECORDS_NAME),
    }
    sources = source_record(resolved, checks)

    data_root = args.data_root
    if data_root is None:
        data_root = Path(paths[PATHS_KEY_RUN]).parent
    frozen_before = assert_frozen_before_generation(data_root, checks)

    dynamic_cells = read_jsonl(resolved["dynamic_adjudicated"])
    baseline_cells = read_jsonl(resolved["baseline_adjudicated"])
    transitions = derive_transitions(dynamic_cells, baseline_cells, checks)
    guard = derive_guard_relation(
        permitted_h_rows(checks),
        [r["screen_row_id"] for r in transitions["regressions"]], checks)

    record = build_record(transitions, guard, sources, frozen_before)
    assert_no_text(record)
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MANIFEST)
    parser.add_argument("--adjudicated", type=Path, default=None,
                        help="dynamic_raw's adjudicated cells; defaults to "
                             f"paths.local.json[{PATHS_KEY_JUDGING}]/"
                             f"{ADJUDICATED_RELATIVE}")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="the completed phase-1 adjudication, which holds "
                             "the unsteered baseline H verdicts")
    parser.add_argument("--run-records", type=Path, default=None,
                        help="the completed dynamic_raw generation records")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="the machine-local data root, checked to confirm "
                             "no arm this record motivates has generated yet")
    parser.add_argument("--dry-run", action="store_true",
                        help="derive and announce it; write nothing")
    parser.add_argument("--verify", action="store_true",
                        help="validate the committed file; write nothing")
    args = parser.parse_args(argv)

    if args.dry_run and args.verify:
        print("[FATAL] --dry-run derives a record and --verify reads a "
              "committed one; they are not combined", file=sys.stderr)
        return 1

    try:
        checks = Checklist(NAME)
        if args.verify:
            result = verify_record(args.output, checks)
            print(f"[freeze] verified {result['path']}")
            print(f"[freeze]   file sha256   {result['sha256']}")
            print(f"[freeze]   record sha256 {result[DIGEST_FIELD]}")
            print(f"[freeze] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2

        record = build(args, checks)
        announce(record)
        text = serialise(record)

        if args.dry_run:
            print(f"[freeze] dry run: {args.output} was not written")
            print(f"[freeze] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2

        if args.output.exists():
            if args.output.read_text(encoding="utf-8") == text:
                print(f"[freeze] {args.output} is already exactly this "
                      f"record; nothing written")
            else:
                print(f"[FATAL] {args.output} exists and differs from what "
                      f"the inputs now derive. A freeze record that can be "
                      f"rewritten is not one: remove it by hand, with a "
                      f"reason, and re-run.", file=sys.stderr)
                return 1
        else:
            write_atomically(args.output, text)
            print(f"[freeze] wrote {args.output}")

        result = verify_record(args.output, checks)
        print(f"[freeze]   file sha256   {result['sha256']}")
        print(f"[freeze] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
