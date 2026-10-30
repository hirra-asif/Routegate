#!/usr/bin/env python3
"""Offline, outcome-blind recovery of judge replies whose ``item_id`` echo slipped.

**This script makes no network call, loads no model, allocates no GPU and reads
no credential.** It reads what the synchronous call already wrote -- the request
digest, the response digest and the preserved body -- and asks one question:

    can this response be bound to exactly one frozen packet row by something
    other than the identifier the judge echoed back?

Why this exists
---------------
``run_judge.py`` refuses a reply whose ``item_id`` does not match the row it
asked about, and that refusal stays: a mismatched echo means the reply *may*
belong to another row, and string similarity is not evidence about which. But
the runner recorded, at the moment of the call, three things the echo cannot
contradict -- the digest of the request it sent, the digest of the body that
came back, and the body itself, preserved byte for byte. Re-render the frozen
packet, re-hash it, and the request digest either pins the response to one row
or it does not.

When it does, the echo is a transcription error in a copied field, not a claim
about identity, and the row is recovered with the correction recorded in the
open. When it does not -- a digest that will not recompute, a request that binds
two rows, an echo that is itself a real packet id, a reply the frozen schema
rejects -- the row stays failed. There is no partial credit.

The ten conditions
------------------
Every one must hold; each is named in the record and in the report, so a
refusal says which gate stopped it. See ``routegate.triplets.RECOVERY_GATES``.

What it will not do
-------------------
*It never looks at a verdict.* The decision is a function of two opaque id
strings and a chain of digests. ``correction_is_verdict_blind`` proves that at
run time by re-deciding under every other value the frozen vocabulary allows
and requiring the same answer. The report prints ids, digests, gate names and
malformation kinds, and never a verdict.

*It never rewrites history.* ``failures.jsonl`` and everything under ``raw/``
are opened read-only. A recovered row is **appended** to ``results.jsonl``,
carrying its original failure history inside it.

*It never runs twice into two records.* A row that already has a successful
record is reported as already recovered and nothing is appended, so a second
run is a no-op and ``run_judge.py``'s duplicate-resume guard stays satisfied.

Usage
-----
    python experiments/triplet_validation/recover_failed_rows.py \\
        --packet <packet dir> --protocol-dir <frozen protocol> \\
        --runs <runs root> --dry-run
    python experiments/triplet_validation/recover_failed_rows.py \\
        --packet <packet dir> --protocol-dir <frozen protocol> \\
        --run-dir <one run dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    AXES,
    JUDGES,
    RECOVERY_GATES,
    RECOVERY_MECHANISM,
    TripletError,
    assess_failed_row,
    build_request_index,
    load_protocol,
    load_protocol_manifest,
    read_jsonl,
    working_dir,
)

# The resume machinery is the runner's, not a second copy of it: `already_done`
# is the same function that decides what run_judge.py must not call again, and
# it is what makes this script idempotent. Importing run_judge does not call
# anything -- its module body only imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_judge import already_done, judge_model                      # noqa: E402


def load_run_context(run_dir: Path) -> Dict[str, Any]:
    """Axis, judge and model for one run directory, cross-checked, not guessed.

    The run manifest and the failure records each say which axis and judge the
    directory holds. They must agree: a directory whose manifest and records
    disagree is a mixed or moved directory, and recovering from it would bind a
    response to a row rendered under a different protocol.
    """
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise TripletError(f"{manifest_path} is absent; this is not a run "
                           f"directory written by run_judge.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    axis, judge = manifest.get("axis"), manifest.get("judge")
    if axis not in AXES or judge not in JUDGES:
        raise TripletError(f"{manifest_path} names axis={axis!r} judge="
                           f"{judge!r}, which is not a valid pair")
    failures_path = run_dir / "failures.jsonl"
    failures = read_jsonl(failures_path) if failures_path.is_file() else []
    stated = {(r.get("axis"), r.get("judge")) for r in failures}
    if stated - {(axis, judge)}:
        raise TripletError(
            f"{failures_path} holds records for {sorted(stated)}, but the run "
            f"manifest says ({axis!r}, {judge!r}). Refusing to recover from a "
            f"directory whose records and manifest disagree.")
    return {"axis": axis, "judge": judge, "manifest": manifest,
            "failures": failures}


def claim_maps(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """Which rows in this run claim which raw-response path and request digest.

    Condition 8 is not "the digest looks unique in the packet"; it is "no
    second row in this run could have received this response". That is a fact
    about the run's own records, so it is read from them.
    """
    by_path: Dict[str, List[str]] = {}
    by_request: Dict[str, List[str]] = {}
    for record in records:
        row_id = str(record.get("row_id", ""))
        path = record.get("raw_response_path")
        request = record.get("request_sha256")
        if path:
            by_path.setdefault(str(path), []).append(row_id)
        if request:
            by_request.setdefault(str(request), []).append(row_id)
    return {"paths": by_path, "requests": by_request}


def assess_run(run_dir: Path, packet: Path, protocol_dir: Path,
               protocol_manifest: Dict[str, Any],
               recovered_utc: str) -> Dict[str, Any]:
    """Every failed row in one run directory, decided. Writes nothing."""
    context = load_run_context(run_dir)
    axis, judge = context["axis"], context["judge"]
    spec = judge_model(protocol_manifest, judge)
    protocol = load_protocol(axis, directory=protocol_dir,
                             manifest=protocol_manifest)

    rows = read_jsonl(packet / f"{axis}_judge_input.jsonl")
    if not rows:
        raise TripletError(f"{packet} holds no {axis} rows")
    packet_ids = [str(r["row_id"]) for r in rows]
    request_index = build_request_index(rows, protocol, spec["model"])

    results_path = run_dir / "results.jsonl"
    recorded_ok = already_done(results_path)
    existing = read_jsonl(results_path) if results_path.is_file() else []
    claims = claim_maps(list(existing) + list(context["failures"]))

    # Attempt history, grouped in the order the runner wrote it. Repeated
    # entries for one row are retries of that row, not extra missing rows.
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in context["failures"]:
        grouped.setdefault(str(record.get("row_id", "")), []).append(record)

    decisions = [
        assess_failed_row(
            row_id, attempts, run_dir, protocol, spec["model"], request_index,
            packet_ids, recovered_utc, recorded_ok=recorded_ok.keys(),
            claimed_paths=claims["paths"], claimed_requests=claims["requests"])
        for row_id, attempts in grouped.items()
    ]
    return {"run_dir": run_dir, "axis": axis, "judge": judge,
            "packet_rows": len(rows), "recorded_ok": len(recorded_ok),
            "failure_entries": len(context["failures"]),
            "unresolved_rows": len(grouped), "decisions": decisions,
            "protocol_hashes": protocol.hashes(),
            "model_requested": spec["model"]}


def apply_run(report: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """Append the recoverable records. ``failures.jsonl`` and ``raw/`` are not opened.

    Appending, never rewriting: the recovered record carries its own failure
    history, so the row's past is inside the result rather than edited out of
    the failure log.
    """
    run_dir: Path = report["run_dir"]
    recoverable = [d for d in report["decisions"] if d.recoverable]
    if not dry_run and recoverable:
        with open(run_dir / "results.jsonl", "a", encoding="utf-8") as handle:
            for decision in recoverable:
                handle.write(json.dumps(decision.record, sort_keys=True) + "\n")
                handle.flush()
    manifest = {
        "mechanism": RECOVERY_MECHANISM,
        "axis": report["axis"], "judge": report["judge"],
        "model_requested": report["model_requested"],
        "protocol_hashes": report["protocol_hashes"],
        "packet_rows": report["packet_rows"],
        "successful_before": report["recorded_ok"],
        "failure_entries": report["failure_entries"],
        "unresolved_rows_before": report["unresolved_rows"],
        "recovered": [d.row_id for d in recoverable],
        "still_unresolved": [d.row_id for d in report["decisions"]
                             if not d.recoverable and not d.already_recovered],
        "already_recovered": [d.row_id for d in report["decisions"]
                              if d.already_recovered],
        "gates": list(RECOVERY_GATES),
        # Verdict-free by construction: summary() is the printable projection.
        "decisions": [d.summary() for d in report["decisions"]],
        "dry_run": dry_run,
        "network_calls": 0,
        "files_never_modified": ["failures.jsonl", "raw/"],
        "created_utc": utc_now(),
        "environment": environment_record(PACKAGE_VERSION),
    }
    if not dry_run:
        (run_dir / "recovery_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def print_report(report: Dict[str, Any], dry_run: bool) -> None:
    """The operator's view. Reads only ``summary()``, so it cannot print a verdict."""
    tag = f"[{report['judge']}/{report['axis']}]"
    successful = report["recorded_ok"]
    print(f"{tag} {report['packet_rows']} packet row(s); {successful} already "
          f"successful; {report['failure_entries']} failure entry/entries over "
          f"{report['unresolved_rows']} unresolved row(s)")
    for decision in report["decisions"]:
        s = decision.summary()
        if s["already_recovered"]:
            state = "ALREADY-RECOVERED"
        elif s["recoverable"]:
            state = "RECOVERABLE" if dry_run else "RECOVERED"
        else:
            state = "STILL-UNRESOLVED"
        detail = (f"correction={s['correction_kind']} echoed={s['echoed_item_id']}"
                  if s["correction_kind"] else f"blocked_by={s['blocked_by'][:2]}")
        print(f"  {state:<18} {s['row_id']}  attempts={s['attempts']}  {detail}")
        if not s["recoverable"] and not s["already_recovered"]:
            print(f"    {s['reason']}")
    recovered = sum(1 for d in report["decisions"] if d.recoverable)
    unresolved = sum(1 for d in report["decisions"]
                     if not d.recoverable and not d.already_recovered)
    verb = "recoverable" if dry_run else "recovered"
    print(f"{tag} {verb} {recovered}; still unresolved {unresolved}; "
          f"would total {successful + recovered}/{report['packet_rows']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, action="append", default=None,
                        help="one run directory; repeatable")
    parser.add_argument("--runs", type=Path, default=None,
                        help="a runs root; every run directory under it")
    parser.add_argument("--dry-run", action="store_true",
                        help="decide and report, write nothing")
    args = parser.parse_args(argv)

    try:
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        run_dirs: List[Path] = list(args.run_dir or [])
        if args.runs:
            run_dirs += sorted(p for p in args.runs.iterdir()
                               if p.is_dir() and (p / "run_manifest.json").is_file())
        if not run_dirs:
            raise TripletError("pass --run-dir (repeatable) or --runs")
        protocol_manifest = load_protocol_manifest()
        stamp = utc_now()

        # Decide everything before writing anything: a run that would refuse
        # halfway leaves no half-applied recovery behind it.
        reports = [assess_run(d, args.packet, protocol_dir, protocol_manifest,
                              stamp) for d in run_dirs]
        total_recovered = total_unresolved = 0
        for report in reports:
            apply_run(report, args.dry_run)
            print_report(report, args.dry_run)
            total_recovered += sum(1 for d in report["decisions"] if d.recoverable)
            total_unresolved += sum(
                1 for d in report["decisions"]
                if not d.recoverable and not d.already_recovered)
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    mode = "dry run: nothing written" if args.dry_run else "applied"
    print(f"[recovery] {mode}; {total_recovered} row(s) "
          f"{'recoverable' if args.dry_run else 'recovered'}, "
          f"{total_unresolved} still unresolved; 0 network call(s)")
    if total_unresolved:
        print("[recovery] a still-unresolved row is not a verdict and is not "
              "made one. It stays in failures.jsonl.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
