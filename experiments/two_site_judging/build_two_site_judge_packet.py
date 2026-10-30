#!/usr/bin/env python3
"""Build the blind two-site judging packet. Calls nothing, generates nothing.

No model is loaded, no GPU is touched, no API is called and **no response is
generated**. Every prompt and every response already exists, frozen, and is
reconciled by SHA256 before it is written into a row.

This is a NEW builder on a versioned path. It does not import, subclass, wrap
or re-enter ``experiments/focused_direction_judging/build_judge_packet.py``,
and none of that builder's Phase-1 checks is relaxed, widened or reused with a
looser argument: phase 1's packet is a different packet with a different frozen
input, and the two never share a code path.

What it builds
--------------
One judged row per candidate screen record, 480 of them, split by the rubric
its class receives::

    E records (89)   ->  axis2_judge_input.jsonl     the frozen Axis-2
                                                     response-behaviour rubric
    B records (178)  ->  quality_judge_input.jsonl   the frozen blinded
                                                     pairwise answer-quality
                                                     rubric
    H records (213)  ->  safety_judge_input.jsonl    the frozen actionable-
                                                     harmful-help safety rubric

Axis 1 is **not** run. It is prompt-only, every prompt here is a row of the
accepted corpus whose prompt status was settled when that corpus was built, and
re-asking it would answer a question this stage is not about.

The B rows are the only pairwise ones. Each carries the referenced baseline and
the candidate as ``response_a`` and ``response_b``, in an order drawn per row
from ``secrets``. Which position holds which is recorded **only** in
``private/row_map.jsonl`` under the machine-local data root.

Blindness
---------
An E or H judge row carries exactly ``row_id``, ``prompt_text`` and
``response_text``; a B judge row carries exactly ``row_id``, ``prompt_text``,
``response_a`` and ``response_b``. The class, the system, the arm, the
coefficients, the direction, the layers and package states, baseline versus
candidate, which side of a B pair is which, the grouping of rows, the expected
outcome and the corpus row id are all absent, and the build refuses a row that
carries any of them. The opaque ids are cryptographically random and prefixed
so they cannot collide with any other packet's ids.

Rows are written in ascending opaque-id order. Because the ids are random and
are minted before the sort, that order is independent of class, of corpus
position and of anything else.

Fail-closed
-----------
The build stops, rather than adapting, if any of these differs from its frozen
value: either input file's SHA256; the record count, per-class count, arm,
coefficients, sites, historical blocks or hook count; the corpus digest; a
prompt digest; a response's own digest; a baseline's source digest; an
instruction, template or schema digest; the consolidation's own completeness
claims. It also stops if a row would carry a field a judge must not see, if a
rendered request names a corpus row id, or if the output directory already
holds files.

Usage
-----
    python experiments/two_site_judging/build_two_site_judge_packet.py --dry-run
    python experiments/two_site_judging/build_two_site_judge_packet.py
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
from routegate.twositejudging import (                              # noqa: E402
    BASELINE_ARM,
    CANDIDATE_ARM_ID,
    CAPPED_RESPONSE_POLICY,
    CLASSES,
    CLASS_ARMS,
    CLASS_RUBRIC,
    DEDUPLICATION,
    FOREIGN_ROW_PREFIXES,
    FROZEN_COVERAGE,
    FROZEN_INPUT,
    ORIGIN_BASELINE,
    ORIGIN_CANDIDATE,
    ORIGIN_OUTCOME,
    PAIRWISE_RUBRICS,
    PLANNED_CALLS_BOTH_JUDGES,
    PLANNED_CALLS_BY_RUBRIC,
    PLANNED_CALLS_PER_JUDGE,
    POSITION_A,
    POSITION_B,
    PROTOCOL_NAME,
    QUALITY_PROTOCOL,
    ROW_MAP_FILE,
    ROW_PREFIX,
    RUBRICS_USED,
    STAGE_NAME,
    TwoSiteJudgingError,
    assert_ids_unique,
    assert_no_corpus_row_id,
    assert_orientation,
    assert_output_dir_is_new,
    assert_projection_is_exact,
    assert_rendered_request_is_projection,
    assert_row_has_no_forbidden_field,
    consolidated_root,
    draw_orientation,
    judging_root,
    load_consolidation_claims,
    load_prompt_texts,
    load_quality_protocol,
    load_referenced_baselines,
    load_screen_records,
    new_opaque_ids,
    project_judge_row,
    quality_protocol_dir,
)
from routegate.triplets import (                                    # noqa: E402
    AXIS2,
    JUDGE_ROW_FIELDS,
    JUDGES,
    MAX_TOKENS,
    QUALITY,
    SAFETY,
    TEMPERATURE,
    assert_no_text_fields,
    assert_row_is_blind,
    load_protocol,
    load_protocol_manifest,
    working_dir,
    write_jsonl,
)
from routegate.validation import (                                  # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
    sha256_text,
)

PACKET_NAME = f"{STAGE_NAME}_packet"


def collect_items(records: Sequence[Dict[str, Any]],
                  baselines: Dict[str, Dict[str, Any]],
                  prompts: Dict[str, Dict[str, str]],
                  checks: Checklist) -> List[Dict[str, Any]]:
    """One judged item per candidate record, with every text reconciled by digest.

    The prompt comes from the hash-gated corpus, the candidate response from
    the hash-gated screen record, and -- for B only -- the baseline response
    from the hash-gated referenced-baseline file. All three are checked against
    the digests their own source recorded before any of them reaches a row.
    """
    items: List[Dict[str, Any]] = []
    oriented = {POSITION_A: 0, POSITION_B: 0}
    for number, record in enumerate(records, start=1):
        origin = f"screen record {number}"
        screen_row_id = record["row_id"]
        row_class = record["row_class"]
        rubric = CLASS_RUBRIC[row_class]

        corpus = prompts.get(screen_row_id)
        if corpus is None:
            raise TwoSiteJudgingError(
                f"{origin}: {screen_row_id!r} is not a row of the pinned "
                f"corpus. The prompt text is resolved from the corpus and is "
                f"never reconstructed from anywhere else.")
        if corpus["row_class"] != row_class:
            raise TwoSiteJudgingError(
                f"{origin}: {screen_row_id!r} is class {row_class!r} in the "
                f"screen and {corpus['row_class']!r} in the corpus")
        if corpus["prompt_sha256"] != record["prompt_sha256"]:
            raise TwoSiteJudgingError(
                f"{origin}: the corpus prompt for {screen_row_id!r} hashes to "
                f"{corpus['prompt_sha256']} and the record recorded "
                f"{record['prompt_sha256']}. The prompt behind this response "
                f"has moved; refused rather than judged against the wrong one.")

        candidate_text = record["response_text"]
        item: Dict[str, Any] = {
            "screen_row_id": screen_row_id,
            "row_class": row_class,
            "arm_id": record["arm_id"],
            "rubric": rubric,
            "prompt_text": corpus["prompt_text"],
            "prompt_sha256": record["prompt_sha256"],
            "candidate_response_sha256": sha256_text(candidate_text),
            "hit_max_new_tokens": bool(record["hit_max_new_tokens"]),
            "completion_token_count": int(record["completion_token_count"]),
        }

        if rubric in PAIRWISE_RUBRICS:
            baseline = baselines.get(screen_row_id)
            if baseline is None:
                raise TwoSiteJudgingError(
                    f"{origin}: {screen_row_id!r} has no referenced baseline. "
                    f"A pairwise comparison with one side missing is not a "
                    f"comparison; refused rather than judged one-sided.")
            if baseline["row_class"] != row_class:
                raise TwoSiteJudgingError(
                    f"{origin}: the baseline for {screen_row_id!r} is class "
                    f"{baseline['row_class']!r} and the candidate is "
                    f"{row_class!r}")
            if baseline["prompt_sha256"] != record["prompt_sha256"]:
                raise TwoSiteJudgingError(
                    f"{origin}: the baseline for {screen_row_id!r} answers a "
                    f"prompt that hashes to {baseline['prompt_sha256']} and "
                    f"the candidate answers {record['prompt_sha256']}. Two "
                    f"replies to different prompts are not a comparison.")
            baseline_text = baseline["response_text"]
            if baseline_text == candidate_text:
                # Not an error. Recorded so the manifest can say how many, and
                # so a reader is never surprised by an EQUIVALENT that was
                # arithmetically certain.
                item["responses_are_byte_identical"] = True
            orientation = assert_orientation(
                draw_orientation(), f"{origin} orientation")
            texts = {ORIGIN_BASELINE: baseline_text,
                     ORIGIN_CANDIDATE: candidate_text}
            digests = {ORIGIN_BASELINE: sha256_text(baseline_text),
                       ORIGIN_CANDIDATE: sha256_text(candidate_text)}
            item.update({
                "baseline_response_sha256": digests[ORIGIN_BASELINE],
                "response_a": texts[orientation[POSITION_A]],
                "response_b": texts[orientation[POSITION_B]],
                "response_a_sha256": digests[orientation[POSITION_A]],
                "response_b_sha256": digests[orientation[POSITION_B]],
                "orientation": dict(orientation),
                "baseline_source": baseline["source"],
                "baseline_source_column": baseline["source_column"],
                "baseline_source_sha256": baseline["source_sha256"],
            })
            for position, origin_name in orientation.items():
                if origin_name == ORIGIN_CANDIDATE:
                    oriented[position] += 1
        else:
            item["response_text"] = candidate_text
            item["response_sha256"] = item["candidate_response_sha256"]
        items.append(item)

    checks.record("items_collected", True,
                  f"{len(items)} item(s) over {len(prompts)} corpus row(s)")
    checks.record(
        "b_orientation_drawn", True,
        f"candidate in position A for {oriented[POSITION_A]} row(s) and in "
        f"position B for {oriented[POSITION_B]} row(s); drawn per row from "
        f"secrets and recorded only in the private map")
    return items


def build(consolidated: Path, out_dir: Path, protocol_dir: Path,
          quality_dir: Path, dataset: Path, write: bool = True
          ) -> Dict[str, Any]:
    checks = Checklist(PACKET_NAME)
    if write:
        assert_output_dir_is_new(out_dir, (consolidated, quality_dir,
                                           protocol_dir),
                                 "the packet builder")

    manifest = load_protocol_manifest()
    protocols: Dict[str, Any] = {
        axis: load_protocol(axis, directory=protocol_dir, manifest=manifest)
        for axis in (AXIS2, SAFETY)}
    protocols[QUALITY] = load_quality_protocol(quality_dir)
    for protocol in protocols.values():
        for key, value in protocol.hashes().items():
            checks.record(key, True, value)

    records = load_screen_records(consolidated, checks)
    baselines = load_referenced_baselines(consolidated, checks)
    claims = load_consolidation_claims(consolidated, checks)
    prompts = load_prompt_texts(dataset, checks)
    items = collect_items(records, baselines, prompts, checks)

    screen_row_ids = sorted(prompts)

    by_rubric: Dict[str, List[Dict[str, Any]]] = {r: [] for r in RUBRICS_USED}
    for item in items:
        by_rubric[item["rubric"]].append(item)

    judge_rows: Dict[str, List[Dict[str, Any]]] = {}
    private_rows: List[Dict[str, Any]] = []
    for rubric in RUBRICS_USED:
        group = by_rubric[rubric]
        ids = new_opaque_ids(ROW_PREFIX[rubric], len(group))
        assert_ids_unique(ids)
        minted: List[Dict[str, Any]] = []
        for row_id, item in zip(ids, group):
            label = f"{rubric} row {row_id}"
            # Whitelist projection. The row is CONSTRUCTED from named values;
            # the screen record is never copied, filtered or stripped, so no
            # treatment field can reach the judge by omission.
            row = project_judge_row(item, rubric, row_id)
            assert_projection_is_exact(row, rubric, item, label,
                                       prefix=ROW_PREFIX[rubric])
            assert_row_is_blind(row, rubric)
            assert_row_has_no_forbidden_field(row, label)
            system, user = protocols[rubric].render(row)
            # The request is the frozen instructions plus the frozen template
            # filled with exactly the whitelisted values, byte for byte.
            # Authentic prompt and response text is never inspected for what
            # it happens to say.
            assert_rendered_request_is_projection(system, user, row,
                                                  protocols[rubric], label)
            assert_no_corpus_row_id(user, screen_row_ids, label)
            minted.append(row)

            entry: Dict[str, Any] = {
                "row_id": row_id,
                "rubric": rubric,
                "screen_row_id": item["screen_row_id"],
                "row_class": item["row_class"],
                "arm_id": item["arm_id"],
                "prompt_sha256": item["prompt_sha256"],
                "hit_max_new_tokens": item["hit_max_new_tokens"],
                "completion_token_count": item["completion_token_count"],
                "rendered_user_sha256": sha256_text(user),
                "rendered_system_sha256": sha256_text(system),
            }
            if rubric in PAIRWISE_RUBRICS:
                entry.update({
                    "baseline_response_sha256":
                        item["baseline_response_sha256"],
                    "candidate_response_sha256":
                        item["candidate_response_sha256"],
                    "response_a_origin": item["orientation"][POSITION_A],
                    "response_b_origin": item["orientation"][POSITION_B],
                    "responses_are_byte_identical":
                        bool(item.get("responses_are_byte_identical", False)),
                })
            else:
                entry["response_sha256"] = item["response_sha256"]
                entry["candidate_response_sha256"] = \
                    item["candidate_response_sha256"]
            private_rows.append(entry)
        # ascending opaque-id order: random with respect to class and corpus
        # position, so the file itself carries no grouping
        judge_rows[rubric] = sorted(minted, key=lambda r: r["row_id"])

    all_ids = [row["row_id"] for rows in judge_rows.values() for row in rows]
    assert_ids_unique(all_ids)
    checks.require("opaque_ids_unique", len(all_ids) == len(set(all_ids)),
                   f"{len(all_ids)} id(s), {len(set(all_ids))} unique")
    checks.require(
        "ids_do_not_collide_with_another_packet",
        not any(i.startswith(FOREIGN_ROW_PREFIXES) for i in all_ids),
        f"two-site ids use {sorted(ROW_PREFIX.values())} and are "
        f"cryptographically random")
    for rubric, want in sorted(PLANNED_CALLS_BY_RUBRIC.items()):
        got = len(judge_rows[rubric])
        checks.require(f"{rubric}_row_count", got == want,
                       f"{got} row(s), expected {want}")
    checks.require("rows_total", len(all_ids) == FROZEN_INPUT["rows"],
                   f"{len(all_ids)} row(s), expected {FROZEN_INPUT['rows']}")

    private_by_cell = {(r["screen_row_id"], r["arm_id"], r["rubric"])
                       for r in private_rows}
    checks.require("one_row_per_cell",
                   len(private_by_cell) == len(private_rows),
                   f"{len(private_rows)} row(s) over {len(private_by_cell)} "
                   f"distinct (screen_row_id, arm_id, rubric) cell(s)")

    candidate_as_a = sum(1 for r in private_rows
                         if r.get("response_a_origin") == ORIGIN_CANDIDATE)
    candidate_as_b = sum(1 for r in private_rows
                         if r.get("response_b_origin") == ORIGIN_CANDIDATE)
    checks.require(
        "every_b_row_has_both_origins_exactly_once",
        candidate_as_a + candidate_as_b == PLANNED_CALLS_BY_RUBRIC[QUALITY],
        f"{candidate_as_a} candidate-as-A + {candidate_as_b} candidate-as-B "
        f"= {candidate_as_a + candidate_as_b}, expected "
        f"{PLANNED_CALLS_BY_RUBRIC[QUALITY]}")
    identical = sum(1 for r in private_rows
                    if r.get("responses_are_byte_identical"))
    checks.record(
        "b_rows_whose_two_replies_are_byte_identical", True,
        f"{identical} row(s); judged as produced, never merged or skipped")

    summary: Dict[str, Any] = {
        "packet": PACKET_NAME,
        "stage": STAGE_NAME,
        "protocol": PROTOCOL_NAME,
        "created_utc": utc_now(),
        "purpose": "blind judging of the two-site static actuator candidate: "
                   "the frozen Axis-2 response-behaviour rubric on every E "
                   "candidate response, the frozen blinded pairwise "
                   "answer-quality rubric on every B baseline/candidate pair, "
                   "and the frozen actionable-harmful-help safety rubric on "
                   "every H candidate response",
        "generates_responses": False,
        "runs_axis1": False,
        "runs_axis1_note": "Axis 1 is prompt-only and was settled when the "
                           "corpus was built; re-asking it here would answer a "
                           "question this stage is not about",
        "reuses_the_phase1_packet_builder": False,
        "reuses_the_phase1_packet_builder_note":
            "experiments/focused_direction_judging/build_judge_packet.py is a "
            "different packet over a different frozen input. It is not "
            "imported, wrapped or re-entered here, and none of its checks is "
            "relaxed or reused with a looser argument.",
        "arm_id": CANDIDATE_ARM_ID,
        "rubric_by_class": {name: CLASS_RUBRIC[name] for name in CLASSES},
        "arms_by_class": {name: list(arms)
                          for name, arms in sorted(CLASS_ARMS.items())},
        "rows_by_rubric": {r: len(judge_rows[r]) for r in RUBRICS_USED},
        "rows_total": len(all_ids),
        "calls_per_judge": PLANNED_CALLS_PER_JUDGE,
        "calls_both_judges": PLANNED_CALLS_BOTH_JUDGES,
        "judges": list(JUDGES),
        "judge_settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        "protocol_hashes": {k: v for p in protocols.values()
                            for k, v in p.hashes().items()},
        "axis2_version": manifest["axis2_version"]["name"],
        "safety_version": manifest["safety_version"]["name"],
        "quality_rubric": {k: v for k, v in QUALITY_PROTOCOL.items()
                           if k != "pairing_rule"},
        "deduplication": dict(DEDUPLICATION),
        "capped_responses": dict(CAPPED_RESPONSE_POLICY),
        "input": {
            "consolidated_dir": str(consolidated),
            "records_sha256": FROZEN_INPUT["records_sha256"],
            "baseline_sha256": FROZEN_INPUT["baseline_sha256"],
            "records_rows": FROZEN_INPUT["rows"],
            "run_manifest_sha256": claims["run_manifest_sha256"],
            "validation_sha256": claims["validation_sha256"],
            "dataset_sha256": FROZEN_INPUT["dataset_sha256"],
            "master_sha256": FROZEN_INPUT["master_sha256"],
            "baseline_arm": BASELINE_ARM,
        },
        "coverage_expected": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in FROZEN_COVERAGE.items()},
        "b_orientation": {
            "randomised": True,
            "source": "secrets.randbelow, per row",
            "candidate_in_position_a": candidate_as_a,
            "candidate_in_position_b": candidate_as_b,
            "byte_identical_pairs": identical,
            "decoded_outcome_by_origin": dict(ORIGIN_OUTCOME),
            "recorded_only_in": f"private/{ROW_MAP_FILE}, machine-local, never "
                                f"committed",
        },
        "blindness": {
            "judge_row_fields": {r: list(JUDGE_ROW_FIELDS[r])
                                 for r in RUBRICS_USED},
            "row_order": "ascending opaque id; the ids are cryptographically "
                         "random and are minted before the sort, so the order "
                         "is independent of class and corpus position",
            "private_map": f"private/{ROW_MAP_FILE}, machine-local, never "
                           f"committed",
            "prompt_or_response_text_in_git": False,
        },
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    assert_no_text_fields(summary, "the packet manifest")

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        for rubric in RUBRICS_USED:
            write_jsonl(out_dir / f"{rubric}_judge_input.jsonl",
                        judge_rows[rubric])
        write_jsonl(out_dir / "private" / ROW_MAP_FILE,
                    sorted(private_rows, key=lambda r: r["row_id"]))
        (out_dir / "private" / ROW_MAP_FILE).chmod(0o600)
        (out_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        sums = []
        for path in sorted(p for p in out_dir.rglob("*")
                           if p.is_file() and p.name != "SHA256SUMS.txt"):
            sums.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
        (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n",
                                                encoding="utf-8")
        print(f"[packet] wrote {out_dir}")

    for rubric in RUBRICS_USED:
        print(f"[packet] {rubric:<8} {len(judge_rows[rubric]):>4} row(s)")
    print(f"[packet] {'B orientation':<8} candidate as A {candidate_as_a}, "
          f"as B {candidate_as_b}")
    print(f"[packet] {PLANNED_CALLS_PER_JUDGE} call(s) per judge, "
          f"{PLANNED_CALLS_BOTH_JUDGES} across both")
    print("[packet] no model was loaded, no response was generated, "
          "no API was called")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated", type=Path, default=None,
                        help="the consolidated two-site package directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None,
                        help="the frozen rq1 judge-protocol text directory")
    parser.add_argument("--quality-dir", type=Path, default=None,
                        help="the frozen B quality-rubric text directory")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="the accepted corpus; defaults to paths.local.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every gate and write nothing")
    args = parser.parse_args(argv)

    try:
        root = judging_root()
        consolidated = args.consolidated or consolidated_root()
        out_dir = args.output_dir or (root / "judging" / "packet")
        protocol_dir = args.protocol_dir or (
            working_dir() / "frozen" / "rq1_judge_protocol")
        quality_dir = args.quality_dir or quality_protocol_dir()
        dataset = args.dataset
        if dataset is None:
            paths = json.loads(
                (REPO_ROOT / "configs" / "paths.local.json").read_text(
                    encoding="utf-8"))
            dataset = Path(paths["dataset"])
        build(consolidated, out_dir, protocol_dir, quality_dir, dataset,
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
