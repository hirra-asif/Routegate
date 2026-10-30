#!/usr/bin/env python3
"""Freeze the v5.1 verifier erratum. Offline; billed never.

    python experiments/final_evaluation/freeze_verifier_erratum.py --dry-run
    python experiments/final_evaluation/freeze_verifier_erratum.py
    python experiments/final_evaluation/freeze_verifier_erratum.py --verify

Pins the v5 manifest digest, both verifier digests and the six existing smoke
records, and states that this changes technical validation only. It reads no
model, no gold, no score and no sealed material, and modifies no record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.verifiererratum import (                          # noqa: E402
    DIGEST_FIELD, ERRATUM, NAME, build_erratum, verify_erratum)
from routegate.validation import Checklist, ValidationError       # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ERRATUM)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        checks = Checklist(NAME)
        if args.verify:
            r = verify_erratum(args.output, checks)
            print(f"[erratum] verified {r['path']}")
            print(f"[erratum]   file sha256   {r['sha256']}")
            print(f"[erratum]   record sha256 {r[DIGEST_FIELD]}")
            print(f"[erratum] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2
        payload = build_erratum()
        print(f"[erratum] {NAME}: applies to {payload['applies_to']}")
        print(f"[erratum]   v5 manifest   {payload['applies_to_manifest_sha256'][:16]}")
        print(f"[erratum]   old verifier  {payload['original_verifier_sha256'][:16]}")
        print(f"[erratum]   new verifier  {payload['corrected_verifier_sha256'][:16]}")
        for arm, digest in sorted(payload["smoke_record_sha256"].items()):
            print(f"[erratum]   record {arm:<38} {digest[:16]}")
        print(f"[erratum]   scope: technical validation only; generation "
              f"{payload['scope']['changes_generation']}, scoring "
              f"{payload['scope']['changes_scoring']}")
        print(f"[erratum] {DIGEST_FIELD} {payload[DIGEST_FIELD]}")
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.dry_run:
            print(f"[erratum] dry run: {args.output} was not written")
            return 0
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") == text:
                print(f"[erratum] {args.output} is already exactly this "
                      f"record; nothing written")
            else:
                print(f"[FATAL] {args.output} exists and differs.",
                      file=sys.stderr)
                return 1
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(args.output.parent),
                prefix=f".{args.output.name}.", suffix=".tmp", delete=False)
            with handle:
                handle.write(text); handle.flush(); os.fsync(handle.fileno())
            os.replace(handle.name, args.output)
            print(f"[erratum] wrote {args.output}")
        r = verify_erratum(args.output, checks)
        print(f"[erratum]   file sha256   {r['sha256']}")
        print(f"[erratum] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
