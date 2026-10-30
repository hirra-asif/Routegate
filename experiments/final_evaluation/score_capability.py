#!/usr/bin/env python3
"""Score a completed capability arm offline. Deterministic; billed never.

    python experiments/final_evaluation/score_capability.py --dataset tinymmlu
    python experiments/final_evaluation/score_capability.py --dataset tinygsm8k

Reads each condition's ``records.jsonl`` and the pinned gold column, parses
every completion by the rule frozen in the protocol, and writes the ordered
100-element correctness vector, the raw accuracy and the paired item
transitions against baseline.

The gold column is read HERE and nowhere else. The generating runner never
imports this module and never touches a gold field; ``--check-hygiene``
asserts that from its source.

Loads no model, generates nothing, calls no judge and reads no sealed material.
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

from routegate.capabilityscoring import (                        # noqa: E402
    assert_generation_cannot_read_gold, paired_transitions, score_arm,
    tinybenchmarks_estimate)
from routegate.finaleval import (                                # noqa: E402
    CONDITIONS, COND_BASELINE, DATASET_SPEC, DS_GSM8K, DS_MMLU, RECORDS_FILE,
    arm_id, arm_output_dir)
from routegate.validation import (                               # noqa: E402
    Checklist, ValidationError, sha256_file)

RUNNER = REPO_ROOT / "experiments/final_evaluation/run_final_evaluation.py"
TINYBENCHMARKS_NAME = {DS_MMLU: "mmlu", DS_GSM8K: "gsm8k"}


def read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"{path} is absent; this arm has not run")
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[DS_MMLU, DS_GSM8K], required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "routegate-data")
    parser.add_argument("--check-hygiene", action="store_true",
                        help="assert the generating runner reads no gold field")
    args = parser.parse_args(argv)

    checks = Checklist(f"score_{args.dataset}")
    try:
        assert_generation_cannot_read_gold(
            RUNNER.read_text(encoding="utf-8"), RUNNER.name)
        checks.record("generation_cannot_read_gold", True,
                      f"{RUNNER.name} references no gold column")
        if args.check_hygiene:
            print(f"[score] {RUNNER.name}: reads no gold column")
            return 0

        import pandas as pd
        spec = DATASET_SPEC[args.dataset]
        local = Path.home() / spec["local"]
        digest = sha256_file(local)
        checks.require(f"{args.dataset}_sha256", digest == spec["sha256"],
                       f"{digest}, pinned {spec['sha256']}")
        golds = pd.read_parquet(local)[spec["answer_column"]].tolist()
        checks.require("gold_rows", len(golds) == spec["rows"],
                       f"{len(golds)}")

        readings: Dict[str, Any] = {}
        for condition in CONDITIONS:
            path = (args.data_root / arm_output_dir(args.dataset, condition)
                    / RECORDS_FILE)
            reading = score_arm(args.dataset, read_records(path), golds)
            readings[condition] = reading
            print(f"[score] {arm_id(args.dataset, condition):<40} "
                  f"{reading['correct']}/{reading['denominator']} "
                  f"= {reading['accuracy']:.3f}  parse "
                  f"{reading['parse_status_counts']}")

        base = readings[COND_BASELINE]["correctness_vector"]
        print()
        for condition in CONDITIONS:
            if condition == COND_BASELINE:
                continue
            paired = paired_transitions(
                base, readings[condition]["correctness_vector"])
            readings[condition]["paired_vs_baseline"] = paired
            interval = paired["delta_confidence_interval"]
            print(f"[score] {condition} vs baseline: "
                  f"+{paired['incorrect_to_correct']} "
                  f"-{paired['correct_to_incorrect']} "
                  f"delta {paired['accuracy_delta']:+.3f}"
                  + (f"  discordant {interval['discordant']} "
                     f"[{interval['low']:.3f}, {interval['high']:.3f}]"
                     if interval else "  (no discordant pairs)"))
        for condition in CONDITIONS:
            readings[condition]["tinybenchmarks"] = tinybenchmarks_estimate(
                TINYBENCHMARKS_NAME[args.dataset],
                readings[condition]["correctness_vector"])
        print()
        print(f"[score] primary: raw accuracy and paired transitions. "
              f"tinyBenchmarks IRT is secondary.")
        print(f"[score] {checks.as_dict()['checks_run']} check(s), all "
              f"passed: {checks.all_passed}")
        return 0 if checks.all_passed else 2
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
