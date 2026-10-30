#!/usr/bin/env python3
"""Freeze the final-evaluation protocol. Offline; billed never.

    python experiments/final_evaluation/freeze_final_evaluation_manifest.py --dry-run
    python experiments/final_evaluation/freeze_final_evaluation_manifest.py
    python experiments/final_evaluation/freeze_final_evaluation_manifest.py --verify

Writes ``manifests/routegate_final_evaluation_v1.json``: three conditions,
three datasets, nine arms, 1,740 generations, the dataset pins, the sensor
digests and the two rejected hybrids recorded as rejected.

Frozen BEFORE any sealed prompt is read and before any token is generated. It
loads no model, generates nothing, judges nothing, downloads nothing and holds
no sealed path.
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

from routegate.finaleval import (                                # noqa: E402
    DIGEST_FIELD, NAME, PREDECLARATION, build_protocol, verify_protocol)
from routegate.validation import Checklist, ValidationError       # noqa: E402


def write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                    prefix=f".{path.name}.", suffix=".tmp",
                                    delete=False)
    try:
        with h:
            h.write(text); h.flush(); os.fsync(h.fileno())
        os.replace(h.name, path)
    except BaseException:
        Path(h.name).unlink(missing_ok=True); raise


def announce(p: Dict[str, Any]) -> None:
    print(f"[freeze] {NAME}: {p['status']}")
    print(f"[freeze] supersedes {p['supersedes']}; v1 generated "
          f"{p['v1_generated_rows']} row(s), v2 generated "
          f"{p['v2_generated_rows']} row(s)")
    print(f"[freeze] scientific protocol unchanged from v2: "
          f"{p['scientific_protocol_unchanged_from_v2']}")
    for change in p["v3_changes"]:
        print(f"[freeze]   v3 adds: {change[:100]}")
    print(f"[freeze] conditions {p['conditions']} (complete)")
    for name, why in sorted(p["rejected_candidates"].items()):
        print(f"[freeze] REJECTED {name}: {why}")
    for a in p["arms"]:
        print(f"[freeze]   {a['arm_id']:<38} {a['rows']:>4} rows -> "
              f"{a['output_dir']}")
    g = p["generations"]
    print(f"[freeze] generations sealed {g['sealed']} + tinymmlu "
          f"{g['tinymmlu']} + tinygsm8k {g['tinygsm8k']} = {g['total']} "
          f"({g['per_condition']} per condition), {g['inherited_or_reused']} "
          f"inherited")
    for d in ("tinymmlu", "tinygsm8k"):
        s = p["dataset_spec"][d]
        print(f"[freeze] {d:<10} {s['repo']} @ {s['revision'][:12]} "
              f"{s['rows']} items sha {s['sha256'][:12]} "
              f"prompt {s['prompt_mode']}")
    print(f"[freeze] sealed     prompt "
          f"{p['dataset_spec']['sealed']['prompt_mode']}")
    for relative, digest in sorted(p["source_sha256"].items()):
        print(f"[freeze] source {relative:<52} {digest[:12]}")
    s = p["sealed_set"]
    print(f"[freeze] sealed     {s['expected_row_count']} rows; path stored "
          f"here {s['path_stored_in_this_repository']}; prompt read "
          f"{s['prompt_content_read_by_this_module']}")
    print(f"[freeze] excluded   {p['excluded']['benchmarks']}")
    print(f"[freeze]            {p['excluded']['operators']}")
    print(f"[freeze] {DIGEST_FIELD} {p[DIGEST_FIELD]}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PREDECLARATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        checks = Checklist(NAME)
        if args.verify:
            r = verify_protocol(args.output, checks)
            print(f"[freeze] verified {r['path']}")
            print(f"[freeze]   file sha256   {r['sha256']}")
            print(f"[freeze]   record sha256 {r[DIGEST_FIELD]}")
            print(f"[freeze] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2
        payload = build_protocol()
        announce(payload)
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.dry_run:
            print(f"[freeze] dry run: {args.output} was not written")
            return 0
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") == text:
                print(f"[freeze] {args.output} is already exactly this "
                      f"protocol; nothing written")
            else:
                print(f"[FATAL] {args.output} exists and differs. A frozen "
                      f"protocol is not rewritten: remove it by hand, with a "
                      f"reason, and re-run.", file=sys.stderr)
                return 1
        else:
            write_atomically(args.output, text)
            print(f"[freeze] wrote {args.output}")
        r = verify_protocol(args.output, checks)
        print(f"[freeze]   file sha256   {r['sha256']}")
        print(f"[freeze] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
