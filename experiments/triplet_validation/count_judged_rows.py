#!/usr/bin/env python3
"""Count the successful rows in each judging run directory. Offline, verdict-blind.

Independent of the recovery tool: it re-reads ``results.jsonl`` through the
shared format dispatcher and counts *distinct* successful ``row_id`` values,
so a duplicated resume key is an error rather than a larger number. It reads no
verdict and prints none -- only counts, and how many of them were recovered
rather than answered on the first call.

    python experiments/triplet_validation/count_judged_rows.py \\
        --runs <runs root> --expect packet__axis2__sonnet=445 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.triplets import (                                     # noqa: E402
    TripletError,
    read_jsonl,
    result_is_success,
    result_join_key,
)


def count_run(directory: Path) -> Dict[str, int]:
    """Distinct successful rows, and how many carry a recorded id correction."""
    path = directory / "results.jsonl"
    if not path.is_file():
        raise TripletError(f"{path} is absent; refusing to read a missing run "
                           f"as an empty one")
    records = read_jsonl(path)
    successes = [r for r in records if result_is_success(r)]
    rows = {result_join_key(r) for r in successes}
    if len(rows) != len(successes):
        raise TripletError(
            f"{path} holds {len(successes)} successful records for "
            f"{len(rows)} distinct row(s). A duplicate successful record means "
            f"the resume key is not unique.")
    return {"successful": len(rows),
            "recovered": sum(1 for r in successes if "item_id_correction" in r),
            "failed_records": sum(1 for r in records if not result_is_success(r))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--expect", action="append", default=[],
                        metavar="DIR=N",
                        help="a run directory name and the count it must have; "
                             "repeatable. Without any, counts are reported and "
                             "nothing is asserted.")
    args = parser.parse_args(argv)

    expected: Dict[str, int] = {}
    for item in args.expect:
        name, _, value = item.partition("=")
        if not value.isdigit():
            print(f"[FATAL] --expect wants DIR=N, got {item!r}", file=sys.stderr)
            return 1
        expected[name] = int(value)

    names = sorted(expected) or sorted(
        p.name for p in args.runs.iterdir()
        if p.is_dir() and (p / "results.jsonl").is_file())
    short = 0
    for name in names:
        try:
            counts = count_run(args.runs / name)
        except TripletError as exc:
            print(f"[FATAL] {exc}", file=sys.stderr)
            return 1
        want = expected.get(name)
        mark = "  " if want is None else ("OK" if counts["successful"] == want
                                          else "!!")
        short += want is not None and counts["successful"] != want
        target = "" if want is None else f"/{want}"
        print(f"[{mark}] {name:<26} {counts['successful']:>4}{target}"
              f"   recovered={counts['recovered']}")
    if expected:
        print("[count] every directory matches its expected total" if not short
              else f"[count] {short} directory/ies short of the expected total")
    return 1 if short else 0


if __name__ == "__main__":
    raise SystemExit(main())
