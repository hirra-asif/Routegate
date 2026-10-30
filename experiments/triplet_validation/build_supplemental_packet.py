#!/usr/bin/env python3
"""Build only the retrospective-validation rows the historical packet lacks.

No model is loaded, no GPU is touched, no API is called and **no new response is
generated**. Every prompt and every response used here already exists, frozen,
and is reconciled by SHA256 against the frozen source-case manifest before it is
written into a row.

What is missing, and why
------------------------
``results/rq1_retro_twoaxis_packet_20260813_102233`` covers E, H and the
original donor d1. It does not cover:

* **d2 and d3** -- the two independent donors the **primary** frozen actuator
  ``consensus23_global_frozen`` is actually built from. The packet's sources are
  ``data/pathway_probe_cases.json`` (d1 only) and the confirmation run; the
  multi-donor bank is not among them.
* **the safety axis on H** -- this packet ships Axis 1 and Axis 2 only.

So the supplement is exactly:

    d2 prompt        -> Axis 1        (10 rows)
    d2 response      -> Axis 2        (10 rows)
    d3 prompt        -> Axis 1        (10 rows)
    d3 response      -> Axis 2        (10 rows)
    H prompt+response-> safety        (10 rows)

50 rows, 50 calls per judge, 100 calls across both. Nothing else. The original
packet is not read for its content, not rebuilt, and not written to.

Fail-closed
-----------
The build stops, rather than adapting, if any of these differs from its frozen
value: an instruction, template or schema digest; a case id or the case set; a
prompt digest; a donor id; the model identifier recorded by the source capture;
or the source manifest's own digest. It also stops if a row would carry a field
a judge must not see, or if rendered request text names a case or a condition.

Usage
-----
    python experiments/triplet_validation/build_supplemental_packet.py --dry-run
    python experiments/triplet_validation/build_supplemental_packet.py
"""
from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.triplets import (                                     # noqa: E402
    ARM_D2,
    ARM_D3,
    ARM_DONOR_ID,
    ARM_H,
    ARM_PROMPT_COLUMN,
    ARMS_IN_SUPPLEMENT,
    AXIS1,
    AXIS2,
    JUDGE_ROW_FIELDS,
    MAX_TOKENS,
    SAFETY,
    SUPPLEMENT_ROW_PREFIX,
    TEMPERATURE,
    TripletError,
    assert_ids_unique,
    assert_row_is_blind,
    assert_text_is_blind,
    load_protocol,
    load_protocol_manifest,
    new_opaque_ids,
    record_hash,
    working_dir,
    write_jsonl,
)
from routegate.validation import Checklist, sha256_file, sha256_text  # noqa: E402

PACKET_NAME = "rq1_supplemental_packet"

#: The frozen sources this build is allowed to read, by digest. Each is a
#: historical artefact; a differing digest is refused.
FROZEN_SOURCES = {
    "source_case_manifest.csv":
        "18feafb52ffca8265817f4498167eabe84a9a0d9fd593310f410c2451432edd3",
    "multidonor_outputs.jsonl":
        "796d06fe55c5432ca7ef0739a817e51d14a8bdecf2a0acf7dc0ba9f0577494c8",
    "pathway_probe_cases.json":
        "60905eb129dddab6fbedb4410d465f8b56999851279a1900bd95512e4a03bad0",
    "rq2_multidonor_probe_cases.json":
        "4f7bbf060042d6b53e7ac2dd61fd5ca9f2005ad2c41a7408a7b3706969e98592",
}

EXPECTED_CASE_COUNT = 10
EXPECTED_MODEL = "Qwen/Qwen3-30B-A3B"


def _expect(path: Path, key: str, checks: Checklist) -> str:
    digest = sha256_file(path)
    if digest != FROZEN_SOURCES[key]:
        raise TripletError(
            f"{key} hash mismatch for {path}\n  expected {FROZEN_SOURCES[key]}\n"
            f"  actual   {digest}\nA frozen source that is not the frozen "
            f"artefact is refused, not adapted to.")
    checks.record(f"sha256_{key}", True, digest)
    return digest


def load_sources(inputs: Path, checks: Checklist) -> Dict[str, Any]:
    """The frozen inputs, hash-gated, from the machine-local data root."""
    scm_path = inputs / "source_case_manifest.csv"
    md_path = inputs / "multidonor_outputs.jsonl"
    bank_path = inputs / "pathway_probe_cases.json"
    donors_path = inputs / "rq2_multidonor_probe_cases.json"
    for path in (scm_path, md_path, bank_path, donors_path):
        if not path.exists():
            raise TripletError(
                f"{path} is absent. The four frozen sources are copied into the "
                f"working root once; see RUNBOOK.md step 2.")

    digests = {
        "source_case_manifest.csv": _expect(scm_path, "source_case_manifest.csv", checks),
        "multidonor_outputs.jsonl": _expect(md_path, "multidonor_outputs.jsonl", checks),
        "pathway_probe_cases.json": _expect(bank_path, "pathway_probe_cases.json", checks),
        "rq2_multidonor_probe_cases.json":
            _expect(donors_path, "rq2_multidonor_probe_cases.json", checks),
    }

    with open(scm_path, newline="", encoding="utf-8") as handle:
        frozen_cases = {row["case_id"]: row for row in csv.DictReader(handle)}
    if len(frozen_cases) != EXPECTED_CASE_COUNT:
        raise TripletError(
            f"the frozen source-case manifest names {len(frozen_cases)} cases; "
            f"the actuator was frozen on {EXPECTED_CASE_COUNT}")
    checks.record("case_count", True, f"{len(frozen_cases)} cases")

    captures: Dict[str, Dict[str, Any]] = {}
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        captures[f"{record['case_id']}::{record['donor_id']}"] = record

    bank = {c["id"]: c for c in json.loads(
        bank_path.read_text(encoding="utf-8"))["cases"]}
    donors = {c["id"]: c for c in json.loads(
        donors_path.read_text(encoding="utf-8"))["cases"]}
    return {"frozen_cases": frozen_cases, "captures": captures,
            "bank": bank, "donors": donors, "digests": digests}


def collect_items(sources: Dict[str, Any], checks: Checklist
                  ) -> List[Dict[str, Any]]:
    """One record per missing (case, arm, axis) cell, with its frozen text.

    Every prompt is taken from the case banks and every response from the
    multi-donor capture, and each is reconciled against the frozen source-case
    manifest's digest before it is accepted.
    """
    frozen = sources["frozen_cases"]
    items: List[Dict[str, Any]] = []

    for case_id in frozen:
        row = frozen[case_id]

        # ---- the two independent donors: Axis 1 on the prompt, Axis 2 on the
        # frozen baseline response
        donor_case = sources["donors"].get(case_id)
        if donor_case is None:
            raise TripletError(f"{case_id} is absent from the multi-donor bank")
        by_id = {d.get("donor_id"): d for d in donor_case["donors"]}
        for arm in ARMS_IN_SUPPLEMENT:
            donor_id = ARM_DONOR_ID[arm]
            donor = by_id.get(donor_id)
            if donor is None:
                raise TripletError(f"{case_id}: donor {donor_id} is absent")
            prompt = donor["prompt"]
            expected = row[ARM_PROMPT_COLUMN[arm]]
            actual = sha256_text(prompt)
            if actual != expected:
                raise TripletError(
                    f"{case_id}/{arm}: prompt digest {actual} does not match "
                    f"the frozen source-case manifest's {expected}")

            capture = sources["captures"].get(f"{case_id}::{donor_id}")
            if capture is None:
                raise TripletError(
                    f"{case_id}/{arm}: no captured baseline response for "
                    f"{donor_id}. This build never generates one.")
            if capture["donor_prompt_sha256"] != expected:
                raise TripletError(
                    f"{case_id}/{arm}: the capture's donor prompt digest "
                    f"{capture['donor_prompt_sha256']} is not the frozen one")
            if capture.get("model") != EXPECTED_MODEL:
                raise TripletError(
                    f"{case_id}/{arm}: the capture records model "
                    f"{capture.get('model')!r}, not {EXPECTED_MODEL!r}")
            response = capture.get("output")
            if not response or not str(response).strip():
                raise TripletError(
                    f"{case_id}/{arm}: the captured baseline response is empty")

            items.append({"case_id": case_id, "arm": arm, "axis": AXIS1,
                          "prompt_text": prompt,
                          "prompt_sha256": expected})
            items.append({"case_id": case_id, "arm": arm, "axis": AXIS2,
                          "prompt_text": prompt, "response_text": response,
                          "prompt_sha256": expected,
                          "response_sha256": sha256_text(response),
                          # Preserved, never acted on. 28 of the 30 donor
                          # baselines stopped at max_tokens=256. The cap is a
                          # fact about the generation, not a verdict: it does
                          # not pass a response and it does not fail one. The
                          # frozen Axis-2 rubric classifies what is actually
                          # there, and only SA satisfies donor inclusion.
                          "reached_max_tokens": bool(
                              capture.get("reached_max_tokens")),
                          "output_token_count": capture.get("output_token_count")})

        # ---- the harmful contrast: the safety axis only. Its Axis 1 and Axis 2
        # rows already exist in the historical packet and are not rebuilt here.
        case = sources["bank"].get(case_id)
        if case is None:
            raise TripletError(f"{case_id} is absent from the case bank")
        h_prompt = case["harmful_contrast"]
        h_expected = frozen[case_id][ARM_PROMPT_COLUMN[ARM_H]]
        if sha256_text(h_prompt) != h_expected:
            raise TripletError(
                f"{case_id}/H: prompt digest does not match the frozen manifest")
        h_response = sources["captures"].get(f"{case_id}::d1_historical", {}).get(
            "harmful_baseline_output")
        if h_response is None:
            # The harmful baseline response lives in the historical packet's own
            # occurrence map, not in the donor capture. It is supplied to this
            # build as a hash-reconciled side input rather than regenerated.
            h_response = sources.get("harmful_baselines", {}).get(case_id)
        if not h_response:
            raise TripletError(
                f"{case_id}/H: no frozen harmful baseline response is available "
                f"to this build. It is never generated here; supply it with "
                f"--harmful-baselines, hash-reconciled. See RUNBOOK.md step 5.")
        items.append({"case_id": case_id, "arm": ARM_H, "axis": SAFETY,
                      "prompt_text": h_prompt, "response_text": h_response,
                      "prompt_sha256": h_expected,
                      "response_sha256": sha256_text(h_response)})

    checks.record("items_collected", True,
                  f"{len(items)} item(s) over {len(frozen)} case(s)")
    return items


def load_harmful_baselines(path: Optional[Path], sources: Dict[str, Any],
                           checks: Checklist) -> Dict[str, str]:
    """The frozen H baseline responses, reconciled by digest.

    Written by ``export_harmful_baselines`` in the runbook from the historical
    packet's own occurrence map, so that this build never opens that packet and
    never generates anything.
    """
    if path is None:
        return {}
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for case_id, entry in rows.items():
        if case_id not in sources["frozen_cases"]:
            raise TripletError(f"{case_id} is not one of the ten frozen cases")
        want = sources["frozen_cases"][case_id][ARM_PROMPT_COLUMN[ARM_H]]
        if entry.get("prompt_sha256") != want:
            raise TripletError(
                f"{case_id}/H: supplied prompt digest is not the frozen one")
        response = entry.get("response_text", "")
        if sha256_text(response) != entry.get("response_sha256"):
            raise TripletError(
                f"{case_id}/H: supplied response does not match its own digest")
        out[case_id] = response
    checks.record("harmful_baselines_reconciled", True,
                  f"{len(out)} harmful baseline response(s)")
    return out


def build(inputs: Path, out_dir: Path, harmful_baselines: Optional[Path],
          write: bool = True) -> Dict[str, Any]:
    checks = Checklist(PACKET_NAME)
    manifest = load_protocol_manifest()
    sources = load_sources(inputs, checks)
    sources["harmful_baselines"] = load_harmful_baselines(
        harmful_baselines, sources, checks)

    protocols = {axis: load_protocol(axis, directory=inputs / "rq1_judge_protocol",
                                     manifest=manifest)
                 for axis in (AXIS1, AXIS2, SAFETY)}
    for axis, protocol in protocols.items():
        for key, value in protocol.hashes().items():
            checks.record(key, True, value)

    items = collect_items(sources, checks)
    case_ids = sorted(sources["frozen_cases"])

    # ---- deduplicate per axis, exactly as the historical packet did: one call
    # per unique prompt on Axis 1, one per unique (prompt, response) pair on the
    # response axes. Identical requests at temperature 0.0.
    by_axis: Dict[str, Dict[tuple, Dict[str, Any]]] = {a: {} for a in protocols}
    for item in items:
        axis = item["axis"]
        key = ((item["prompt_sha256"],) if axis == AXIS1
               else (item["prompt_sha256"], item["response_sha256"]))
        slot = by_axis[axis].setdefault(key, {"item": item, "occurrences": []})
        slot["occurrences"].append({"case_id": item["case_id"],
                                    "arm": item["arm"], "axis": axis})

    judge_rows: Dict[str, List[Dict[str, Any]]] = {}
    private_rows: List[Dict[str, Any]] = []
    for axis, groups in by_axis.items():
        ids = new_opaque_ids(SUPPLEMENT_ROW_PREFIX[axis], len(groups))
        assert_ids_unique(ids)
        rows: List[Dict[str, Any]] = []
        for row_id, (_key, slot) in zip(ids, sorted(groups.items())):
            item = slot["item"]
            row = {"row_id": row_id, "prompt_text": item["prompt_text"]}
            if "response_text" in JUDGE_ROW_FIELDS[axis]:
                row["response_text"] = item["response_text"]
            assert_row_is_blind(row, axis)
            system, user = protocols[axis].render(row)
            assert_text_is_blind(user, case_ids, f"{axis} row {row_id}")
            rows.append(row)
            private_rows.append({
                "row_id": row_id, "axis": axis,
                "prompt_sha256": item["prompt_sha256"],
                "response_sha256": item.get("response_sha256"),
                "reached_max_tokens": item.get("reached_max_tokens"),
                "output_token_count": item.get("output_token_count"),
                "occurrences": slot["occurrences"],
                "rendered_user_sha256": sha256_text(user),
                "rendered_system_sha256": sha256_text(system),
            })
        judge_rows[axis] = rows

    all_ids = [r["row_id"] for rows in judge_rows.values() for r in rows]
    assert_ids_unique(all_ids)
    checks.require("opaque_ids_unique", len(all_ids) == len(set(all_ids)),
                   f"{len(all_ids)} ids, {len(set(all_ids))} unique")
    checks.require(
        "ids_do_not_collide_with_the_historical_packet",
        not any(i.startswith(("RQ1RETROAONE", "RQ1RETROATWO")) for i in all_ids),
        "supplement ids use their own prefixes and are independently randomised")

    expected = {AXIS1: 20, AXIS2: 20, SAFETY: 10}
    for axis, want in expected.items():
        got = len(judge_rows[axis])
        checks.require(f"{axis}_row_count", got == want,
                       f"{got} row(s), expected {want}")

    summary: Dict[str, Any] = {
        "packet": PACKET_NAME,
        "created_utc": utc_now(),
        "purpose": "the retrospective-validation rows the historical packet "
                   "lacks: Axis 1 and Axis 2 for donors d2 and d3, and the "
                   "safety axis for the harmful contrast",
        "generates_responses": False,
        "cases": len(case_ids),
        "rows_by_axis": {a: len(judge_rows[a]) for a in judge_rows},
        "calls_per_judge": len(all_ids),
        "calls_both_judges": 2 * len(all_ids),
        "judge_settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        "protocol_hashes": {k: v for p in protocols.values()
                            for k, v in p.hashes().items()},
        "axis1_version": manifest["axis1_version"]["name"],
        "axis2_version": manifest["axis2_version"]["name"],
        "source_digests": sources["digests"],
        "does_not_touch": "results/rq1_retro_twoaxis_packet_20260813_102233",
        "capped_responses": {
            "policy": "the complete historically stored response is used exactly "
                      "as produced. It is never regenerated, never extended, "
                      "never auto-passed and never auto-failed because it "
                      "reached the cap. The frozen Axis-2 rubric classifies the "
                      "observed response; only SA satisfies donor inclusion.",
            "flag_preserved_as": "reached_max_tokens, in the private row map "
                                 "and in the final evidence record",
            "expected_capped_of_30_donor_baselines": 28},
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "private").mkdir(exist_ok=True)
        for axis, rows in judge_rows.items():
            write_jsonl(out_dir / f"{axis}_judge_input.jsonl", rows)
        write_jsonl(out_dir / "private" / "row_map.jsonl", private_rows)
        (out_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        sums = []
        for name in sorted(p.name for p in out_dir.rglob("*")
                           if p.is_file() and p.name != "SHA256SUMS.txt"):
            match = next(p for p in out_dir.rglob("*") if p.name == name)
            sums.append(f"{sha256_file(match)}  {match.relative_to(out_dir)}")
        (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n",
                                                encoding="utf-8")
        print(f"[packet] wrote {out_dir}")
    for axis, rows in judge_rows.items():
        print(f"[packet] {axis}: {len(rows)} row(s)")
    print(f"[packet] {len(all_ids)} call(s) per judge, "
          f"{2 * len(all_ids)} across both")
    print("[packet] no model was loaded, no response was generated, "
          "no API was called")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=None,
                        help="frozen input root; defaults to the working root")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--harmful-baselines", type=Path, default=None,
                        help="hash-reconciled H baseline responses")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate every gate and write nothing")
    args = parser.parse_args(argv)

    root = working_dir()
    inputs = args.inputs or (root / "frozen")
    out_dir = args.output_dir or (root / "supplemental_packet")
    try:
        build(inputs, out_dir, args.harmful_baselines, write=not args.dry_run)
    except TripletError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("[dry-run] every gate passed; nothing was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
