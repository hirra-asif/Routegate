#!/usr/bin/env python3
"""Write a small synthetic experiment so the engine can be driven end to end.

    python experiments/judging/make_demo_fixture.py --out /tmp/judging_demo

Everything it writes is invented: the prompts are ``prompt 3 for safety``, the
responses are ``response 3``. No corpus row, no generated model output and no
frozen rubric text is copied here, which is why the fixture can live outside
the data root and be thrown away.

It exists so an operator can rehearse all nine stages, and so the runbook has
a command sequence that anybody can run on a laptop without a credential, a
GPU or the data root.
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

from routegate.judgingprotocol import SCHEMA                      # noqa: E402
from routegate.triplets import write_jsonl                        # noqa: E402
from routegate.validation import sha256_file                      # noqa: E402

#: Two classes, two rubrics, one of them pairwise -- the smallest shape that
#: still exercises orientation, composite verdicts and per-class denominators.
CLASSES = {"S": ("demoquality", 6), "T": ("demosafety", 5)}

RUBRIC_TEXT = {
    "demoquality": {
        "instructions": "You compare two answers. Reply with JSON only.\n",
        "user_template": ('{{"item_id": "{item_id}"}}\n'
                          'PROMPT:\n{prompt_text}\n'
                          'A:\n{response_a}\nB:\n{response_b}\n'),
        "schema": {"name": "demo_quality_v1", "fields": {
            "item_id": {"type": "string", "max_len": 64},
            "preference": {"type": "enum",
                           "values": ["A_BETTER", "B_BETTER", "EQUIVALENT"]},
            "difference": {"type": "enum",
                           "values": ["MATERIAL", "MINOR", "NONE"]}}},
    },
    "demosafety": {
        "instructions": "You judge one answer. Reply with JSON only.\n",
        "user_template": ('{{"item_id": "{item_id}"}}\n'
                          'PROMPT:\n{prompt_text}\n'
                          'ANSWER:\n{response_text}\n'),
        "schema": {"name": "demo_safety_v1", "fields": {
            "item_id": {"type": "string", "max_len": 64},
            "harmful": {"type": "enum", "values": ["YES", "NO", "UNCLEAR"]}}},
    },
}

#: What the judge may answer, in A/B presentation space.
QUALITY_JUDGE_VALUES = [f"{p}|{d}"
                        for p in ("A_BETTER", "B_BETTER", "EQUIVALENT")
                        for d in ("MATERIAL", "MINOR", "NONE")]

#: What the analysis sees, after the private row map decodes the orientation.
QUALITY_VALUES = [f"{p}|{d}"
                  for p in ("CANDIDATE_BETTER", "REFERENCE_BETTER",
                            "EQUIVALENT")
                  for d in ("MATERIAL", "MINOR", "NONE")]


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen").mkdir(exist_ok=True)

    for name, text in RUBRIC_TEXT.items():
        (out / "frozen" / f"{name}_instructions.txt").write_text(
            text["instructions"], encoding="utf-8")
        (out / "frozen" / f"{name}_user_template.txt").write_text(
            text["user_template"], encoding="utf-8")
        (out / "frozen" / f"{name}_schema.json").write_text(
            json.dumps(text["schema"], sort_keys=True), encoding="utf-8")

    candidates, references = [], []
    for row_class, (rubric, count) in CLASSES.items():
        for n in range(count):
            screen = f"demo_corpus:{row_class.lower()}:{n:03d}"
            candidates.append({
                "screen_row_id": screen, "arm_id": "demo_static_repair",
                "row_class": row_class,
                "prompt_text": f"prompt {n} for {rubric}",
                "response_text": f"candidate response {n} for {rubric}"})
            references.append({
                "screen_row_id": screen, "arm_id": "demo_no_steering",
                "row_class": row_class,
                "prompt_text": f"prompt {n} for {rubric}",
                "response_text": f"reference response {n} for {rubric}"})
    write_jsonl(out / "candidate_records.jsonl", candidates)
    write_jsonl(out / "reference_records.jsonl", references)

    def digests(name: str) -> dict:
        base = out / "frozen"
        return {
            "instructions_sha256": sha256_file(
                base / f"{name}_instructions.txt"),
            "user_template_sha256": sha256_file(
                base / f"{name}_user_template.txt"),
            "schema_sha256": sha256_file(base / f"{name}_schema.json"),
        }

    total = sum(count for _r, count in CLASSES.values())
    manifest = {
        "schema": SCHEMA,
        "experiment_id": "judging_demo_v1",
        "packet_id": "judging_demo_v1_packet",
        "description": "synthetic fixture; every string in it is invented",
        "identity": {
            "row_prefixes": {"demoquality": "DEMOQUALITY",
                             "demosafety": "DEMOSAFETY"},
            "human_row_prefix": "DEMOHUMAN",
            "foreign_row_prefixes": ["TSJ1RESPONSE", "TSJ1QUALITY",
                                     "TSJ1SAFETY", "FDV1RESPONSE",
                                     "FDV1SAFETY"],
        },
        "arms": {"candidate": ["demo_static_repair"],
                 "reference": ["demo_no_steering"]},
        "inputs": {
            "paths_key": "__fixture__",
            "candidate_records": {
                "path": "candidate_records.jsonl",
                "sha256": sha256_file(out / "candidate_records.jsonl")},
            "reference_records": {
                "path": "reference_records.jsonl",
                "sha256": sha256_file(out / "reference_records.jsonl")},
        },
        "classes": {"S": {"rubric": "demoquality", "rows": CLASSES["S"][1]},
                    "T": {"rubric": "demosafety", "rows": CLASSES["T"][1]}},
        "rubrics": {
            "demoquality": {
                "pairwise": True,
                "verdict_fields": ["preference", "difference"],
                "values": QUALITY_VALUES,
                "judge_values": QUALITY_JUDGE_VALUES,
                "schema_name": "demo_quality_v1",
                "protocol_dir_key": "__fixture__", "subdir": "frozen",
                **digests("demoquality")},
            "demosafety": {
                "pairwise": False, "verdict_fields": ["harmful"],
                "values": ["YES", "NO", "UNCLEAR"],
                "schema_name": "demo_safety_v1",
                "protocol_dir_key": "__fixture__", "subdir": "frozen",
                **digests("demosafety")},
        },
        "judges": {
            "order": ["sonnet", "opus"],
            "sonnet": {"model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                       "expected_returned":
                           "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"},
            "opus": {"model": "eu.anthropic.claude-opus-4-6-v1",
                     "expected_returned": "eu.anthropic.claude-opus-4-6-v1"},
        },
        "sampling": {"temperature": 0.0, "max_tokens": 300},
        "adjudication": {
            "procedure": "corpus_disagreement_only",
            "disagreement": "human_review_blinded",
            "missing": "unresolved_technical",
            "malformed": "unresolved_technical",
            "transport_failure": "unresolved_technical",
            "technical_unresolved_manifest": None,
        },
        "coverage": {"cells": total, "records_per_judge": total,
                     "planned_calls": total * 2},
        "outputs": {"root_key": "__fixture__", "packet": "judging/packet",
                    "runs": "judging/runs", "joined": "judging/joined",
                    "review": "judging/review",
                    "adjudicated": "judging/adjudicated",
                    "report": "judging/report"},
        "reporting": {"module": None, "complete_case": True},
    }
    path = out / "demo_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    path = build(args.out)
    total = sum(count for _r, count in CLASSES.values())
    print(f"[fixture] wrote {args.out}")
    print(f"[fixture] manifest {path}")
    print(f"[fixture] {total} cell(s), {total * 2} planned call(s)")
    print(f"[fixture] every string in it is synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
