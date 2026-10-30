#!/usr/bin/env python3
"""Freeze the metadata-only compatibility release record. Offline.

    python experiments/final_evaluation/freeze_sealed_input_release.py \
        --input /path/to/the/manifest.csv --dry-run
    python experiments/final_evaluation/freeze_sealed_input_release.py \
        --input /path/to/the/manifest.csv
    python experiments/final_evaluation/freeze_sealed_input_release.py --verify

Pins the v5 manifest's file and record digests, the input's filename and
content digest, the superseded predecessor's digest and the reason it was
refused, the adapter's data-root-relative path and digest, both header
schemas, the transformation rule, the counts and the preparation script's
digest. It carries no prompt, no response, no gold and no score.

``--verify`` needs no input file: the committed record's own consistency, the
v5 manifest and the preparation script are all it checks.
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

from routegate.sealedinputrelease import (                        # noqa: E402
    CHANGES,
    DIGEST_FIELD,
    NAME,
    RELEASE,
    build_release,
    default_adapter_path,
    verify_release,
    write_atomically,
)
from routegate.validation import Checklist, ValidationError        # noqa: E402


def announce(payload: dict) -> None:
    print(f"[release] {NAME}: applies to {payload['applies_to']}")
    print(f"[release]   v5 manifest   "
          f"{payload['applies_to_manifest_sha256'][:16]} "
          f"(record {payload['applies_to_manifest_record_sha256'][:16]})")
    print(f"[release]   input         {payload['source']['sha256'][:16]}, "
          f"{payload['source']['bytes']} byte(s)")
    print(f"[release]   rejected      "
          f"{payload['rejected_predecessor']['sha256'][:16]}, "
          f"{payload['rejected_predecessor']['rows']} row(s)")
    print(f"[release]   adapter       {payload['adapter']['sha256'][:16]}, "
          f"{payload['adapter']['bytes']} byte(s)")
    print(f"[release]   path          {payload['adapter']['relative_path']}")
    print(f"[release]   rename        "
          f"{payload['header_schema']['renamed_from']} -> "
          f"{payload['header_schema']['renamed_to']} "
          f"({payload['header_schema']['columns']} columns)")
    counts = payload["counts"]
    print(f"[release]   counts        {counts['rows']} rows, "
          f"{counts['safe_source']} safe_source, "
          f"{counts['harmful_source']} harmful_source, "
          f"{counts['unique_row_ids']} unique ids, "
          f"{counts['unique_prompts']} unique prompts")
    print(f"[release]   prep script   "
          f"{payload['preparation_script_sha256'][:16]}")
    print(f"[release]   scope         "
          f"{', '.join(f'{k}={payload[k]}' for k in sorted(CHANGES))}")
    print(f"[release] {DIGEST_FIELD} {payload[DIGEST_FIELD]}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="the archival input, supplied at launch; "
                             "required unless --verify is given")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="the derived adapter; defaults to the release "
                             "path for this input")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=RELEASE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    try:
        checks = Checklist(NAME)
        if args.verify:
            result = verify_release(args.output, checks)
            print(f"[release] verified {result['path']}")
            print(f"[release]   file sha256   {result['sha256']}")
            print(f"[release]   record sha256 {result[DIGEST_FIELD]}")
            print(f"[release] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2

        if args.input is None:
            parser.error("--input is required unless --verify is given")
        adapter = (args.adapter if args.adapter is not None
                   else default_adapter_path(args.input, args.data_root))
        payload = build_release(args.input, adapter, checks)
        announce(payload)
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

        if args.dry_run:
            print(f"[release] dry run: {args.output} was not written")
            return 0
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") == text:
                print(f"[release] {args.output} is already exactly this "
                      f"record; nothing written")
            else:
                print(f"[FATAL] {args.output} exists and differs from what "
                      f"the tree now describes. A release record that can be "
                      f"rewritten is not one: remove it by hand, with a "
                      f"reason, and re-run.", file=sys.stderr)
                return 1
        else:
            write_atomically(args.output, text.encode("utf-8"))
            print(f"[release] wrote {args.output}")

        result = verify_release(args.output, checks)
        print(f"[release]   file sha256   {result['sha256']}")
        print(f"[release] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
