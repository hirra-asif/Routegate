#!/usr/bin/env python3
"""Freeze the gate-deployment predeclaration. Offline; billed never.

    python experiments/routegate_deployment/freeze_gate_deployment_manifest.py --dry-run
    python experiments/routegate_deployment/freeze_gate_deployment_manifest.py
    python experiments/routegate_deployment/freeze_gate_deployment_manifest.py --verify

Writes ``manifests/routegate_gate_deployment_v1.json``: what will be refitted,
from which digests, under which unchanged rule, to which files, and what the
result may never be used to claim -- plus the capability suite, the two
capability conditions and the sealed set that follow it.

Loads no model, fits nothing, generates nothing, judges nothing, downloads
nothing and reads no sealed material. An identical file is not an overwrite; a
file that differs stops the script.
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

from routegate.gatefreeze import (                               # noqa: E402
    DIGEST_FIELD,
    NAME,
    PREDECLARATION,
    build_predeclaration,
    verify_predeclaration,
)
from routegate.validation import Checklist, ValidationError       # noqa: E402


def serialise(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_atomically(path: Path, text: str) -> None:
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


def announce(p: Dict[str, Any]) -> None:
    print(f"[freeze] {NAME}: {p['status']}")
    print(f"[freeze] closes: {p['closes_blocker']}")
    print(f"[freeze] unchanged: {', '.join(p['changes_nothing_else'])}")
    g = p["gate"]
    print(f"[freeze] gate   {g['rule']}")
    print(f"[freeze]        threshold_E {g['threshold_E']!r} "
          f"threshold_H {g['threshold_H']!r}  (sha {g['thresholds_sha256'][:12]})")
    a = p["actuator"]
    print(f"[freeze] actuator {a['direction_variant']} donors "
          f"{'/'.join(a['donors'])} alpha_R {a['alpha_R']} layers "
          f"{a['historical_layers']}")
    t = p["training"]
    print(f"[freeze] refit on all {p['representation']['rows']} rows; C grid "
          f"{t['C_grid']}; seed {t['seed']}")
    print(f"[freeze] reproduction gate atol "
          f"{t['reproduction_gate']['atol']:g} against the frozen OOF table")
    d = p["development_estimates"]
    print(f"[freeze] development estimates UNCHANGED (out-of-fold): "
          f"E {d['E_substantive_repair'][0]}/{d['E_substantive_repair'][1]}, "
          f"B {d['B_material_degradation'][0]}/{d['B_material_degradation'][1]}, "
          f"H {d['H_actionable_regression'][0]}/{d['H_actionable_regression'][1]}")
    c = p["capability_evaluation"]
    print(f"[freeze] capability + conditions declared in {c['protocol']} "
          f"({c['protocol_manifest']})")
    s = p["sealed_evaluation"]
    print(f"[freeze] sealed: {s['expected_row_count']} rows, opened here "
          f"{s['opened_by_this_stage']}, prompt content read "
          f"{s['prompt_content_read']}")
    for logical, entry in sorted(p["sources"].items()):
        print(f"[freeze] source {logical:<30} {entry['sha256'][:12]}")
    print(f"[freeze] output {p['output']['directory']}")
    print(f"[freeze] {DIGEST_FIELD} {p[DIGEST_FIELD]}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PREDECLARATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.verify:
        print("[FATAL] --dry-run builds and --verify reads; not combined",
              file=sys.stderr)
        return 1
    try:
        checks = Checklist(NAME)
        if args.verify:
            result = verify_predeclaration(args.output, checks)
            print(f"[freeze] verified {result['path']}")
            print(f"[freeze]   file sha256   {result['sha256']}")
            print(f"[freeze]   record sha256 {result[DIGEST_FIELD]}")
            print(f"[freeze] {checks.as_dict()['checks_run']} check(s), all "
                  f"passed: {checks.all_passed}")
            return 0 if checks.all_passed else 2

        payload = build_predeclaration()
        announce(payload)
        text = serialise(payload)
        if args.dry_run:
            print(f"[freeze] dry run: {args.output} was not written")
            return 0
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") == text:
                print(f"[freeze] {args.output} is already exactly this "
                      f"declaration; nothing written")
            else:
                print(f"[FATAL] {args.output} exists and differs from what "
                      f"the package now declares. A predeclaration that can "
                      f"be rewritten is not one: remove it by hand, with a "
                      f"reason, and re-run.", file=sys.stderr)
                return 1
        else:
            write_atomically(args.output, text)
            print(f"[freeze] wrote {args.output}")
        result = verify_predeclaration(args.output, checks)
        print(f"[freeze]   file sha256   {result['sha256']}")
        print(f"[freeze] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
