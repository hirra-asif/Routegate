#!/usr/bin/env python3
"""Derive and verify the header-only compatibility adapter for the v5 runner.

Offline. Loads no model, generates nothing, judges nothing, calls no service
and prints no prompt. It reads one archival input, gates it on a content
digest supplied at launch, and writes a copy whose only difference is that the
prompt column's header token reads ``prompt_text`` instead of
``canonical_prompt``.

Usage
-----
    python experiments/final_evaluation/prepare_sealed_input_release.py \
        --input /path/to/the/manifest.csv --dry-run
    python experiments/final_evaluation/prepare_sealed_input_release.py \
        --input /path/to/the/manifest.csv
    python experiments/final_evaluation/prepare_sealed_input_release.py \
        --input /path/to/the/manifest.csv --verify

The input path is supplied by the operator and is never written into this
repository, into the adapter, or into the release record. The output filename
is derived from the input's own stem, so nothing here names the file either.

Re-running on an unchanged input writes nothing and re-verifies what is there.
An adapter that exists and differs stops the script rather than being
refreshed: that difference is either a decision to record or a drift to catch,
and both are for a person.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.sealedinputrelease import (                        # noqa: E402
    ADAPTER_COLUMN,
    BYTE_DELTA,
    NAME,
    RELEASE_RELATIVE,
    REQUIRED_SOURCE_SHA256,
    SOURCE_COLUMN,
    default_adapter_path,
    prepare_adapter,
    verify_adapter,
)
from routegate.validation import Checklist, ValidationError        # noqa: E402


def announce(result: dict) -> None:
    """Digests, counts and column names. Never a value from a row."""
    print(f"[release]   rows            {result['rows']}")
    print(f"[release]   composition     {result['safe_source']} safe_source + "
          f"{result['harmful_source']} harmful_source")
    print(f"[release]   unique row ids  {result['unique_row_ids']}")
    print(f"[release]   unique prompts  {result['unique_prompts']} "
          f"(counted by digest, never displayed)")
    print(f"[release]   columns         {len(result['adapter_fields'])}, "
          f"{SOURCE_COLUMN} -> {ADAPTER_COLUMN}")
    print(f"[release]   bytes           {result['source_bytes']} -> "
          f"{result['adapter_bytes']} ({BYTE_DELTA:+d})")
    print(f"[release]   after header    {result['bytes_after_header']} "
          f"byte(s), identical")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="the archival input, supplied at launch; gated "
                             "on its content digest and never stored here")
    parser.add_argument("--output", type=Path, default=None,
                        help="the derived adapter; defaults to the release "
                             "directory under the data root, named from the "
                             "input's own stem")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="the machine-local data root the adapter is "
                             "written beneath")
    parser.add_argument("--dry-run", action="store_true",
                        help="derive and check it; write nothing")
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing adapter; write nothing")
    args = parser.parse_args(argv)

    target = (args.output if args.output is not None
              else default_adapter_path(args.input, args.data_root))
    try:
        checks = Checklist(NAME)
        print(f"[release] {NAME}")
        print(f"[release] required input sha256 {REQUIRED_SOURCE_SHA256}")
        print(f"[release] adapter {RELEASE_RELATIVE}/{target.name}")

        if args.verify:
            result = verify_adapter(args.input, target, checks)
            print(f"[release] verified {target}")
            print(f"[release]   input   sha256 {result['source_sha256']}")
            print(f"[release]   adapter sha256 {result['adapter_sha256']}")
            announce(result)
            print(f"[release] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2

        prepared = prepare_adapter(args.input, target, checks,
                                   data_root=args.data_root,
                                   dry_run=args.dry_run)
        print(f"[release]   input   sha256 {prepared['source_sha256']}")
        print(f"[release]   adapter sha256 {prepared['adapter_sha256']}")
        if args.dry_run:
            print(f"[release] dry run: {target} was not written")
            print(f"[release] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2
        print(f"[release] {'wrote' if prepared['written'] else 'already'} "
              f"{target}")

        result = verify_adapter(args.input, target, checks)
        announce(result)
        print(f"[release] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
