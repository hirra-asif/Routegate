#!/usr/bin/env python3
"""Build the blind phase-1 judging packet. Calls nothing, generates nothing.

No model is loaded, no GPU is touched, no API is called and **no response is
generated**. Every prompt and every response already exists, frozen, and is
reconciled by SHA256 before it is written into a row.

What it builds
--------------
One judged row per phase-1 screen record, 871 of them, split by the rubric its
class receives::

    E records (445)  ->  axis2_judge_input.jsonl    the frozen Axis-2
                                                    response-behaviour rubric
    H records (426)  ->  safety_judge_input.jsonl   the frozen actionable-
                                                    harmful-help safety rubric

Axis 1 is **not** run. It is prompt-only, every phase-1 prompt is a row of the
accepted corpus whose prompt status was settled when that corpus was built, and
re-asking it here would answer a question this phase is not about.

Blindness
---------
A judge row carries exactly ``row_id``, ``prompt_text`` and ``response_text``.
The class, the arm, the coefficients, the direction, the layers, the
baseline/steered status, the grouping of a row's arms, the expected result and
the repair outcome are all absent, and the build refuses a row that carries any
of them. The opaque ids are freshly minted, independently randomised, and
prefixed so they cannot collide with either retrospective packet's ids.

Rows are written in ascending opaque-id order. Because the ids are
cryptographically random and are minted before the sort, that order is
independent of class, of arm and of corpus position, so a judge reading the
file in order sees no grouping. The scientific identity survives only in
``private/row_map.jsonl``, which is written under the machine-local data root
and is never committed.

Fail-closed
-----------
The build stops, rather than adapting, if any of these differs from its frozen
value: the records' SHA256 or byte count; the record count, per-class count,
per-arm count or per-row arm coverage; the corpus digest; a prompt digest; a
response's own digest; an instruction, template or schema digest; the
consolidation's own completeness claims. It also stops if a row would carry a
field a judge must not see, or if a rendered request names a corpus row id or a
phase-1 metadata token.

Usage
-----
    python experiments/focused_direction_judging/build_judge_packet.py --dry-run
    python experiments/focused_direction_judging/build_judge_packet.py
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

from routegate import __version__ as PACKAGE_VERSION                # noqa: E402
from routegate.runrecord import environment_record, utc_now         # noqa: E402
from routegate.focusedjudging import (                              # noqa: E402
    AXES_USED,
    CAPPED_RESPONSE_POLICY,
    CLASS_ARMS,
    CLASS_AXIS,
    CONSOLIDATED_SUBDIR,
    DEDUPLICATION,
    FOREIGN_ROW_PREFIXES,
    FROZEN_COVERAGE,
    FROZEN_INPUT,
    PLANNED_CALLS_BOTH_JUDGES,
    PLANNED_CALLS_BY_AXIS,
    PLANNED_CALLS_PER_JUDGE,
    PROTOCOL_NAME,
    ROW_MAP_FILE,
    ROW_PREFIX,
    FocusedJudgingError,
    assert_no_corpus_row_id,
    assert_projection_is_exact,
    assert_rendered_request_is_projection,
    assert_row_has_no_forbidden_field,
    project_judge_row,
    judging_root,
    load_consolidation_claims,
    load_prompt_texts,
    load_screen_records,
)
from routegate.triplets import (                                    # noqa: E402
    JUDGE_ROW_FIELDS,
    JUDGES,
    MAX_TOKENS,
    TEMPERATURE,
    assert_ids_unique,
    assert_no_text_fields,
    assert_row_is_blind,
    load_protocol,
    load_protocol_manifest,
    new_opaque_ids,
    working_dir,
    write_jsonl,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
    sha256_text,
)

PACKET_NAME = "focused_direction_validation_v1_packet"


def collect_items(records: Sequence[Dict[str, Any]],
                  prompts: Dict[str, Dict[str, str]],
                  checks: Checklist) -> List[Dict[str, Any]]:
    """One judged item per screen record, with its text reconciled by digest."""
    items: List[Dict[str, Any]] = []
    for number, record in enumerate(records, start=1):
        origin = f"screen record {number}"
        screen_row_id = record["row_id"]
        row_class = record["row_class"]
        corpus = prompts.get(screen_row_id)
        if corpus is None:
            raise FocusedJudgingError(
                f"{origin}: {screen_row_id!r} is not a row of the pinned "
                f"corpus. The prompt text is resolved from the corpus and is "
                f"never reconstructed from anywhere else.")
        if corpus["row_class"] != row_class:
            raise FocusedJudgingError(
                f"{origin}: {screen_row_id!r} is class {row_class!r} in the "
                f"screen and {corpus['row_class']!r} in the corpus")
        if corpus["prompt_sha256"] != record["prompt_sha256"]:
            raise FocusedJudgingError(
                f"{origin}: the corpus prompt for {screen_row_id!r} hashes to "
                f"{corpus['prompt_sha256']} and the record recorded "
                f"{record['prompt_sha256']}. The prompt behind this response "
                f"has moved; refused rather than judged against the wrong one.")
        items.append({
            "screen_row_id": screen_row_id,
            "row_class": row_class,
            "arm_id": record["arm_id"],
            "axis": CLASS_AXIS[row_class],
            "prompt_text": corpus["prompt_text"],
            "prompt_sha256": record["prompt_sha256"],
            "response_text": record["response_text"],
            "response_sha256": record["response_sha256"],
            "hit_max_new_tokens": bool(record["hit_max_new_tokens"]),
            "completion_token_count": int(record["completion_token_count"]),
        })
    checks.record("items_collected", True,
                  f"{len(items)} item(s) over {len(prompts)} corpus row(s)")
    return items


def build(consolidated: Path, out_dir: Path, protocol_dir: Path,
          dataset: Path, write: bool = True) -> Dict[str, Any]:
    checks = Checklist(PACKET_NAME)
    manifest = load_protocol_manifest()

    protocols = {axis: load_protocol(axis, directory=protocol_dir,
                                     manifest=manifest)
                 for axis in AXES_USED}
    for protocol in protocols.values():
        for key, value in protocol.hashes().items():
            checks.record(key, True, value)

    records = load_screen_records(consolidated, checks)
    claims = load_consolidation_claims(consolidated, checks)
    prompts = load_prompt_texts(dataset, checks)
    items = collect_items(records, prompts, checks)

    screen_row_ids = sorted(prompts)

    by_axis: Dict[str, List[Dict[str, Any]]] = {axis: [] for axis in AXES_USED}
    for item in items:
        by_axis[item["axis"]].append(item)

    judge_rows: Dict[str, List[Dict[str, Any]]] = {}
    private_rows: List[Dict[str, Any]] = []
    for axis in AXES_USED:
        group = by_axis[axis]
        ids = new_opaque_ids(ROW_PREFIX[axis], len(group))
        assert_ids_unique(ids)
        minted: List[Dict[str, Any]] = []
        for row_id, item in zip(ids, group):
            label = f"{axis} row {row_id}"
            # Whitelist projection. The row is CONSTRUCTED from three named
            # values; the screen record is never copied, filtered or stripped,
            # so no treatment field can reach the judge by omission.
            row = project_judge_row(item, axis, row_id)
            assert_projection_is_exact(row, axis, item, label,
                                       prefix=ROW_PREFIX[axis])
            assert_row_is_blind(row, axis)
            assert_row_has_no_forbidden_field(row, label)
            system, user = protocols[axis].render(row)
            # The request is the frozen instructions plus the frozen template
            # filled with exactly those three values, byte for byte. Authentic
            # prompt and response text is never inspected for what it says.
            assert_rendered_request_is_projection(system, user, row,
                                                  protocols[axis], label)
            assert_no_corpus_row_id(user, screen_row_ids, label)
            minted.append(row)
            private_rows.append({
                "row_id": row_id,
                "axis": axis,
                "screen_row_id": item["screen_row_id"],
                "row_class": item["row_class"],
                "arm_id": item["arm_id"],
                "prompt_sha256": item["prompt_sha256"],
                "response_sha256": item["response_sha256"],
                "hit_max_new_tokens": item["hit_max_new_tokens"],
                "completion_token_count": item["completion_token_count"],
                "rendered_user_sha256": sha256_text(user),
                "rendered_system_sha256": sha256_text(system),
            })
        # ascending opaque-id order: random with respect to class, arm and
        # corpus position, so the file itself carries no grouping
        judge_rows[axis] = sorted(minted, key=lambda r: r["row_id"])

    all_ids = [row["row_id"] for rows in judge_rows.values() for row in rows]
    assert_ids_unique(all_ids)
    checks.require("opaque_ids_unique", len(all_ids) == len(set(all_ids)),
                   f"{len(all_ids)} id(s), {len(set(all_ids))} unique")
    checks.require(
        "ids_do_not_collide_with_another_packet",
        not any(i.startswith(FOREIGN_ROW_PREFIXES) for i in all_ids),
        f"phase-1 ids use {sorted(ROW_PREFIX.values())} and are independently "
        f"randomised")
    for axis, want in PLANNED_CALLS_BY_AXIS.items():
        got = len(judge_rows[axis])
        checks.require(f"{axis}_row_count", got == want,
                       f"{got} row(s), expected {want}")
    checks.require("rows_total", len(all_ids) == FROZEN_INPUT["rows"],
                   f"{len(all_ids)} row(s), expected {FROZEN_INPUT['rows']}")

    private_by_cell = {(r["screen_row_id"], r["arm_id"], r["axis"])
                       for r in private_rows}
    checks.require("one_row_per_cell",
                   len(private_by_cell) == len(private_rows),
                   f"{len(private_rows)} row(s) over {len(private_by_cell)} "
                   f"distinct (screen_row_id, arm_id, axis) cell(s)")

    summary: Dict[str, Any] = {
        "packet": PACKET_NAME,
        "protocol": PROTOCOL_NAME,
        "created_utc": utc_now(),
        "purpose": "blind phase-1 judging: the frozen Axis-2 response-behaviour "
                   "rubric on every E response, and the frozen actionable-"
                   "harmful-help safety rubric on every H response",
        "generates_responses": False,
        "runs_axis1": False,
        "runs_axis1_note": "Axis 1 is prompt-only and was settled when the "
                           "corpus was built; re-asking it here would answer a "
                           "question this phase is not about",
        "rubric_by_class": {name: CLASS_AXIS[name] for name in CLASS_AXIS},
        "arms_by_class": {name: list(arms) for name, arms in CLASS_ARMS.items()},
        "rows_by_axis": {axis: len(judge_rows[axis]) for axis in AXES_USED},
        "rows_total": len(all_ids),
        "calls_per_judge": PLANNED_CALLS_PER_JUDGE,
        "calls_both_judges": PLANNED_CALLS_BOTH_JUDGES,
        "judges": list(JUDGES),
        "judge_settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        "protocol_hashes": {k: v for p in protocols.values()
                            for k, v in p.hashes().items()},
        "axis2_version": manifest["axis2_version"]["name"],
        "safety_version": manifest["safety_version"]["name"],
        "deduplication": dict(DEDUPLICATION),
        "capped_responses": dict(CAPPED_RESPONSE_POLICY),
        "input": {
            "consolidated_dir": str(consolidated),
            "records_sha256": FROZEN_INPUT["records_sha256"],
            "records_rows": FROZEN_INPUT["rows"],
            "run_manifest_sha256": claims["run_manifest_sha256"],
            "validation_sha256": claims["validation_sha256"],
            "dataset_sha256": FROZEN_INPUT["dataset_sha256"],
        },
        "coverage_expected": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in FROZEN_COVERAGE.items()},
        "blindness": {
            "judge_row_fields": {axis: list(JUDGE_ROW_FIELDS[axis])
                                 for axis in AXES_USED},
            "row_order": "ascending opaque id; the ids are cryptographically "
                         "random and are minted before the sort, so the order "
                         "is independent of class, arm and corpus position",
            "private_map": f"private/{ROW_MAP_FILE}, machine-local, never "
                           f"committed",
            "prompt_or_response_text_in_git": False,
        },
        "does_not_read": "manifests/targeted_h_legacy_regressions_v1.json",
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the packet manifest")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        for axis in AXES_USED:
            write_jsonl(out_dir / f"{axis}_judge_input.jsonl", judge_rows[axis])
        write_jsonl(out_dir / "private" / ROW_MAP_FILE,
                    sorted(private_rows, key=lambda r: r["row_id"]))
        (out_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        sums = []
        for path in sorted(p for p in out_dir.rglob("*")
                           if p.is_file() and p.name != "SHA256SUMS.txt"):
            sums.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
        (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n",
                                                encoding="utf-8")
        print(f"[packet] wrote {out_dir}")

    for axis in AXES_USED:
        print(f"[packet] {axis:<8} {len(judge_rows[axis])} row(s)")
    print(f"[packet] {PLANNED_CALLS_PER_JUDGE} call(s) per judge, "
          f"{PLANNED_CALLS_BOTH_JUDGES} across both")
    print("[packet] no model was loaded, no response was generated, "
          "no API was called")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated", type=Path, default=None,
                        help="the phase-1 consolidated package directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None,
                        help="the frozen judge-protocol text directory")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="the accepted corpus; defaults to paths.local.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every gate and write nothing")
    args = parser.parse_args(argv)

    try:
        root = judging_root()
        consolidated = args.consolidated or (root / CONSOLIDATED_SUBDIR)
        out_dir = args.output_dir or (root / "judging" / "packet")
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        dataset = args.dataset
        if dataset is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            dataset = Path(paths["dataset"])
        build(consolidated, out_dir, protocol_dir, dataset,
              write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                       # noqa: BLE001 - fail closed
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
