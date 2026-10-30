#!/usr/bin/env python3
"""Freeze the donor pairing. No model is loaded, no GPU is touched.

This runs first, on its own, and it is the only place a target is paired with a
donor. The pairing is a pure function of the accepted dataset's metadata --
source_family, topic_standardized, task_frame and prompt length in characters --
so it can be recomputed, diffed and argued with before a single token is
generated.

What it deliberately cannot see: activations, generated responses, patching
results, and anything belonging to the held-out benchmark split. The withheld
class H is refused rather than filtered.

The manifest it writes carries its own SHA256 over a canonical serialisation.
The run script takes that hash on the command line and refuses to execute
against anything else, so the pairing cannot drift between a run and its
re-run.

Usage:
    python experiments/causal_repair/build_donor_manifest.py
    python experiments/causal_repair/build_donor_manifest.py --output-dir DIR
    python experiments/causal_repair/build_donor_manifest.py --print-summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.data import (                                        # noqa: E402
    COVARIATE_COLUMNS,
    load_input_manifest,
    load_local_paths,
)
from routegate.donors import (                                      # noqa: E402
    MATCH_TIERS,
    build_manifest,
    build_pairs,
    load_pairing_rows,
    split_targets_and_donors,
    write_manifest,
)
from routegate.runrecord import utc_now, write_csv                  # noqa: E402
from routegate.validation import (                                  # noqa: E402
    ValidationError,
    assert_config_is_sealed_free,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "causal_repair.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "causal_repair"


def build(config_path: Path = DEFAULT_CONFIG,
          output_dir: Optional[Path] = None,
          paths_path: Optional[Path] = None,
          write: bool = True) -> Dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(config, "causal repair config")

    input_manifest = load_input_manifest()
    local_paths = load_local_paths(paths_path) if paths_path else load_local_paths()
    dataset_path = Path(local_paths["dataset"])
    expected_sha = input_manifest["dataset"]["sha256"]

    print(f"[pairing] reading {dataset_path.name} (no model, no activations)")
    rows, dataset_sha = load_pairing_rows(dataset_path, expected_sha)
    targets, donors = split_targets_and_donors(rows)
    print(f"[pairing] {len(targets)} E target(s), {len(donors)} B donor(s)")

    pairs = build_pairs(targets, donors)
    manifest = build_manifest(pairs, dataset_path, dataset_sha, utc_now(),
                              n_targets=len(targets), n_donors=len(donors))

    counts = manifest["counts"]
    print(f"[pairing] {counts['pairs']} pair(s): "
          f"{counts['exact_matches']} exact on "
          f"{list(COVARIATE_COLUMNS.values())}, "
          f"{counts['fallback_matches']} by fallback")
    for tier, n in counts["by_tier"].items():
        if n:
            keys = dict(MATCH_TIERS)[tier]
            print(f"           {tier:<18s} {n:4d}  keys={list(keys) or 'none'}")
    print(f"[pairing] {counts['distinct_donors_used']} distinct donor(s); the "
          f"most-used donor appears {counts['max_uses_of_one_donor']} time(s)")

    if not write:
        return {"manifest": manifest, "output_dir": None}

    output_dir = Path(output_dir or DEFAULT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "donor_manifest.json"
    digest = write_manifest(manifest_path, manifest)

    write_csv(output_dir / "donor_pairs.csv",
              ["target_row_id", "target_label", "donor_row_id", "donor_label",
               "tier", "tier_index", "exact_match", "relaxed_keys",
               "target_prompt_chars", "donor_prompt_chars",
               "length_delta_chars", "candidates_in_tier"],
              [{**{k: v for k, v in pair.as_dict().items()
                   if k not in ("matched_values", "target_values",
                                "donor_values")},
                "relaxed_keys": "|".join(pair.relaxed_keys)}
               for pair in pairs])

    print(f"[pairing] frozen: {manifest_path}")
    print(f"[pairing] manifest_sha256 = {digest}")
    print()
    print("Pass that hash to the run script, which refuses any other manifest:")
    print(f"  --donor-manifest {manifest_path.relative_to(REPO_ROOT)} \\")
    print(f"  --expect-manifest-sha256 {digest}")
    return {"manifest": manifest, "output_dir": output_dir,
            "manifest_sha256": digest}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="compute the pairing and print the summary without "
                             "writing the manifest")
    parser.add_argument("--print-summary", action="store_true",
                        help="also print the per-tier pair counts as JSON")
    args = parser.parse_args(argv)

    outcome = build(config_path=args.config, output_dir=args.output_dir,
                    paths_path=args.paths, write=not args.dry_run)
    if args.print_summary:
        print(json.dumps(outcome["manifest"]["counts"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
