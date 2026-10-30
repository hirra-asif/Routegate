#!/usr/bin/env python3
"""One command for every judging stage. The operator selects only a manifest.

    python experiments/judging/run_protocol.py --manifest <path> --stage <name>

Stages, in order:

    preflight    every gate that can be checked before anything is built
    build        mint opaque ids and write the blind packet
    dryrun       render and hash every request; calls nothing
    judge        the billed stage; --judge sonnet, then --judge opus
    join         bind both judges' verdicts onto cells
    export       the blinded human-review packet: disagreements only
    import       read the reviewer's answers and adjudicate
    adjudicate   alias for import, for readability in a runbook
    report       verdict tallies with explicit denominators

Every stage accepts ``--dry-run``, which runs the gates and writes nothing.
``judge`` is the only stage that costs money, and it refuses to run without a
credential in the environment; the credential is never read from a file, never
placed on a command line, and never written into a record.

Nothing here is experiment-specific. A new steering mechanism -- a controller,
a CAST-style gate, a layer ablation -- is a new manifest, not new code.
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

from routegate import judgingstages as S                          # noqa: E402
from routegate.judgingprotocol import (                           # noqa: E402
    STAGES,
    JudgingProtocolError,
    load_manifest,
    resolve_root,
    stage_dir,
)
from routegate.validation import ValidationError                  # noqa: E402


def _dirs(manifest, root: Optional[Path]):
    if root is not None:
        return S.resolved_dirs(manifest, root)
    return {name: stage_dir(manifest, name)
            for name in manifest.outputs if name != "root_key"}


def _roots(manifest, root: Optional[Path]):
    """The input root and the frozen-rubric root, or a single test root."""
    if root is not None:
        return Path(root), Path(root)
    input_root = resolve_root(manifest, str(manifest.inputs["paths_key"]))
    first = next(iter(sorted(manifest.rubrics)))
    protocol_root = resolve_root(
        manifest, manifest.rubrics[first].protocol_dir_key)
    return input_root, protocol_root


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="the frozen protocol manifest for one experiment")
    parser.add_argument("--stage", required=True,
                        choices=list(STAGES) + ["adjudicate"])
    parser.add_argument("--judge", default=None,
                        help="for --stage judge: which judge to run")
    parser.add_argument("--completed", type=Path, default=None,
                        help="for --stage import: the reviewer's file")
    parser.add_argument("--root", type=Path, default=None,
                        help="override every root; for fixtures and tests")
    parser.add_argument("--carried", type=Path, default=None,
                        help="a technical-unresolved manifest, if one exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every gate and write nothing")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        dirs = _dirs(manifest, args.root)
        input_root, protocol_root = _roots(manifest, args.root)
        write = not args.dry_run
        stage = args.stage

        carried = ()
        source = args.carried or (
            REPO_ROOT / manifest.technical_manifest
            if manifest.technical_manifest else None)
        if source is not None and Path(source).is_file():
            carried = tuple(json.loads(Path(source).read_text(
                encoding="utf-8")).get("unresolved", ()))

        if stage == "preflight":
            out = S.preflight(manifest, dirs, input_root, protocol_root)
        elif stage == "build":
            out = S.build_packet(manifest, dirs, input_root, write=write)
        elif stage == "dryrun":
            out = S.dry_run(manifest, dirs, protocol_root, args.judge)
        elif stage == "judge":
            if not args.judge:
                print("[FATAL] --stage judge needs --judge sonnet|opus",
                      file=sys.stderr)
                return 2
            out = S.run_judge(manifest, dirs, protocol_root, args.judge,
                              write=write)
        elif stage == "join":
            out = S.join_evidence(manifest, dirs, write=write,
                                  carried=carried)
        elif stage == "export":
            out = S.export_review(manifest, dirs, write=write)
        elif stage in ("import", "adjudicate"):
            out = S.import_review(manifest, dirs, args.completed, write=write)
        elif stage == "report":
            out = S.report(manifest, dirs, write=write)
        else:                                            # pragma: no cover
            raise JudgingProtocolError(f"unknown stage {stage!r}")
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    label = f"{manifest.experiment_id}/{stage}"
    for key, value in sorted(out.items()):
        if key in ("checks", "readings", "environment"):
            continue
        shown = (json.dumps(value) if isinstance(value, (dict, list))
                 else value)
        print(f"[{label}] {key:<28} {shown}")
    if args.dry_run:
        print(f"[{label}] dry run: every gate passed; "
              f"nothing was written or called")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
