#!/usr/bin/env python3
"""Read-only: does the same prompt text appear in more than one fold?

Two rows carrying identical prompt text sit in the corpus under two row ids. If
the frozen five-fold division puts them in different folds, then every
cross-fitted quantity built on that division -- the direction bank, both probe
heads, and any controller trained under the out-of-fold protocol -- is fitted on
one copy and scored on the other. The held-out score is then not held out, and
nothing downstream can see it: each row is individually well formed, each fold
is individually the right size, and the number that comes out is simply too
good.

This tool answers the question and does nothing about it. It **never** edits a
fold assignment, never writes into the direction bank, and never proposes a
re-split. A duplicate that crosses folds is reported as a **BLOCKER**, and what
to do about it is a decision with consequences for every artifact already built
on that division -- which is not a decision a diagnostic gets to take.

Normalisation
-------------
Two prompts are "identical" here when their normalised text is equal:

    Unicode NFKC, leading and trailing whitespace stripped, every run of
    whitespace collapsed to a single space, then case-folded.

Exact byte-identical duplicates are counted separately, so a normalised-only
collision -- same words, different spacing or case -- is visible as such rather
than being reported as though the two rows were the same string.

Usage
-----
    python tools/diagnostics/duplicate_prompt_folds.py
    python tools/diagnostics/duplicate_prompt_folds.py --json report.json

Exit status is 1 when a normalised duplicate crosses a fold boundary, so this
can gate a runbook step. It is 0 when duplicates exist inside a single fold --
that is a corpus fact, not a leak -- and the report says so either way.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import load_development_inputs                 # noqa: E402
from routegate.probeheads import load_bank_fold_assignment          # noqa: E402
from routegate.validation import ValidationError                    # noqa: E402

SCHEMA = "routegate/duplicate_prompt_fold_diagnostic/1"

DEFAULT_BANK = (REPO_ROOT / "outputs" / "causal_repair"
                / "orthogonalized_population_scan" / "direction_bank")

NORMALISATION = ("NFKC; strip; collapse internal whitespace runs to one "
                 "space; casefold")


def normalise(text: str) -> str:
    """The one declared normalisation. Written down, not applied from memory."""
    folded = unicodedata.normalize("NFKC", text)
    return " ".join(folded.split()).casefold()


def group_duplicates(rows: Sequence[Dict[str, str]],
                     fold_of_row: Dict[str, int],
                     where: str = "the fold assignment"
                     ) -> List[Dict[str, Any]]:
    """Rows sharing normalised prompt text, with the folds they land in.

    Pure: it takes rows and a fold mapping and returns groups. Kept separate
    from ``diagnose`` so the grouping -- including the crossing case, which the
    corpus does not currently exhibit -- can be exercised without the corpus.
    """
    missing = [row["row_id"] for row in rows
               if row["row_id"] not in fold_of_row]
    if missing:
        raise ValidationError(
            f"{len(missing)} corpus row(s) have no fold in {where}, first "
            f"{missing[:3]}. The diagnostic compares the corpus against the "
            f"division the directions were built on; a row absent from it "
            f"cannot be checked, and is not assumed safe.")

    by_normalised: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_normalised[normalise(row["prompt_text"])].append({
            "row_id": row["row_id"],
            "final_ebh": row["final_ebh"],
            "fold": fold_of_row[row["row_id"]],
            "prompt_sha256": row["prompt_sha256"],
        })

    groups: List[Dict[str, Any]] = []
    for _, members in sorted(by_normalised.items()):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda entry: entry["row_id"])
        folds = sorted({entry["fold"] for entry in members})
        digests = sorted({entry["prompt_sha256"] for entry in members})
        groups.append({
            "rows": len(members),
            "row_ids": [entry["row_id"] for entry in members],
            "labels": sorted({entry["final_ebh"] for entry in members}),
            "folds": folds,
            "crosses_folds": len(folds) > 1,
            "byte_identical": len(digests) == 1,
            "distinct_prompt_digests": len(digests),
            "members": members,
        })
    return groups


def diagnose(bank_directory: Path | str = DEFAULT_BANK) -> Dict[str, Any]:
    """Group the corpus by normalised prompt text and report fold spread.

    Reads the hash-verified development inputs and the direction bank's own
    frozen fold assignment, both through the package's existing gates. Nothing
    here writes.
    """
    inputs = load_development_inputs()
    assignment, source = load_bank_fold_assignment(bank_directory)

    rows = inputs.manifest_rows
    groups = group_duplicates(rows, assignment.fold_of_row,
                              str(source.fold_assignment_path))
    crossing = [group for group in groups if group["crosses_folds"]]
    return {
        "schema": SCHEMA,
        "read_only": True,
        "folds_were_not_altered": True,
        "normalisation": NORMALISATION,
        "inputs": {
            "dataset_sha256": inputs.dataset_sha256,
            "extraction_manifest_sha256": inputs.extraction_manifest_sha256,
            "rows": len(rows),
        },
        "fold_source": {
            "directory": str(source.directory),
            "fold_assignment_sha256": source.fold_assignment_sha256,
            "manifest_sha256": source.manifest_sha256,
            "n_splits": source.n_splits,
        },
        "duplicate_groups": len(groups),
        "duplicate_rows": sum(group["rows"] for group in groups),
        "groups_crossing_folds": len(crossing),
        "rows_in_groups_crossing_folds": sum(group["rows"]
                                             for group in crossing),
        "blocker": bool(crossing),
        "verdict": (
            "BLOCKER: identical normalised prompt text occurs across folds. "
            "Every cross-fitted artifact built on this division -- the "
            "direction bank, both probe heads, and any controller trained "
            "under the out-of-fold protocol -- is fitted on one copy and "
            "scored on the other. Nothing is altered here."
            if crossing else
            "CLEAR: no normalised prompt text occurs in more than one fold."),
        "groups": groups,
    }


def report(payload: Dict[str, Any]) -> None:
    print(f"[dup] corpus rows          {payload['inputs']['rows']}")
    print(f"[dup] dataset sha256       "
          f"{payload['inputs']['dataset_sha256'][:16]}")
    print(f"[dup] fold assignment      "
          f"{payload['fold_source']['fold_assignment_sha256'][:16]} "
          f"({payload['fold_source']['n_splits']} folds)")
    print(f"[dup] normalisation        {payload['normalisation']}")
    print(f"[dup] duplicate groups     {payload['duplicate_groups']} "
          f"({payload['duplicate_rows']} row(s))")
    print(f"[dup] crossing folds       {payload['groups_crossing_folds']} "
          f"({payload['rows_in_groups_crossing_folds']} row(s))")
    for group in payload["groups"]:
        mark = "CROSSES FOLDS" if group["crosses_folds"] else "one fold"
        print(f"[dup]   {mark}: folds {group['folds']}, labels "
              f"{group['labels']}, "
              f"{'byte-identical' if group['byte_identical'] else 'normalised only'}")
        for member in group["members"]:
            print(f"[dup]     fold {member['fold']}  {member['final_ebh']}  "
                  f"{member['row_id']}")
    print(f"[dup] {payload['verdict']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-bank", type=Path, default=DEFAULT_BANK,
                        help="the frozen division to check the corpus against")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the report here")
    args = parser.parse_args(argv)
    try:
        payload = diagnose(args.direction_bank)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    report(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True)
                             + "\n", encoding="utf-8")
        print(f"[dup] wrote {args.json}")
    return 1 if payload["blocker"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
