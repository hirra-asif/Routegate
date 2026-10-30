#!/usr/bin/env python3
"""Everything that must be true before a single billed call. Offline, read-only.

Run this immediately before every judging step. It calls nothing, writes
nothing, loads no model and touches no GPU. It exits non-zero on the first
failure, and a failure means **stop** -- not "adapt the check".

The twelve gates
----------------
 1. HEAD is the expected commit, and the working tree is clean.
 2. No git remote is configured, so no push or fetch can occur.
 3. The original packet's manifest and every file in its SHA256SUMS verify.
 4. All nine protocol digests match the frozen record.
 5. **Every original and supplemental Axis-2 request resolves to the same
    frozen instructions, template and schema hash.** This is the check that
    the 2026-08-04 undocumented Axis-2 variant is not in play anywhere.
 6. The supplemental packet holds exactly 50 rows.
 7. The original packet holds exactly 97 requests per judge.
 8. Combined, exactly 147 requests per judge.
 9. The Sonnet and Opus identifiers match the frozen configuration.
10. No output directory already holds an incompatible run.
11. Opaque ids are unique across every packet.
12. No judge-facing row exposes an E/B/H role, donor, layer, strength or
    condition field; and no configured path can resolve to sealed material.

Usage
-----
    python experiments/triplet_validation/preflight.py
    python experiments/triplet_validation/preflight.py --expect-head df93dd7
    python experiments/triplet_validation/preflight.py --stage supplement
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate.triplets import (                                     # noqa: E402
    AXIS1,
    AXIS2,
    JUDGE_ROW_FIELDS,
    JUDGES,
    PACKET_MANIFEST,
    SAFETY,
    TripletError,
    assert_ids_unique,
    assert_opaque,
    assert_row_is_blind,
    load_protocol,
    load_protocol_manifest,
    read_jsonl,
    working_dir,
)
from routegate.validation import (                                   # noqa: E402
    SEALED_PATH_TOKENS,
    looks_sealed,
    sha256_file,
)

#: The counts this validation is preregistered on. A different count means the
#: packet is not the packet, and the run stops.
EXPECTED_ORIGINAL_PER_JUDGE = 97          # 30 Axis-1 + 67 Axis-2
EXPECTED_SUPPLEMENT_PER_JUDGE = 50        # 20 Axis-1 + 20 Axis-2 + 10 safety
EXPECTED_COMBINED_PER_JUDGE = 147
EXPECTED_ORIGINAL_BY_AXIS = {AXIS1: 30, AXIS2: 67}
EXPECTED_SUPPLEMENT_BY_AXIS = {AXIS1: 20, AXIS2: 20, SAFETY: 10}


class Gate:
    """A pass/fail ledger that prints as it goes and never softens a failure."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          capture_output=True, text=True, check=False).stdout.strip()


# ------------------------------------------------------------- 1, 2. the repo
def gate_repository(gate: Gate, expect_head: Optional[str]) -> None:
    print("\n[1-2] repository")
    head = _git("rev-parse", "HEAD")
    short = _git("rev-parse", "--short", "HEAD")
    if expect_head:
        gate.check("head_is_the_reviewed_commit",
                   head.startswith(expect_head) or short == expect_head,
                   f"HEAD {short}, expected {expect_head}")
    else:
        gate.check("head_recorded", bool(head), f"HEAD {short}")
    dirty = _git("status", "--porcelain")
    gate.check("working_tree_clean", not dirty,
               "clean" if not dirty else f"{len(dirty.splitlines())} modified path(s)")
    gate.check("no_remote_configured", not _git("remote"),
               "no remote, so no push or fetch is possible")


# ----------------------------------------------------------- 3, 4, 5. packets
def gate_original_packet(gate: Gate, packet: Path) -> Dict[str, Any]:
    print("\n[3] the original packet, unchanged")
    manifest = json.loads(PACKET_MANIFEST.read_text(encoding="utf-8"))
    # The frozen manifest's own key. The historical inventory is named
    # ``covered_by_packet_checksum`` and is NOT renamed here to suit the code:
    # a checked-in provenance record is evidence, and evidence is read as
    # written. A KeyError here means the manifest changed, which is a stop.
    covered = dict(manifest["inventory"]["covered_by_packet_checksum"])
    covered.update(manifest["inventory"]["the_checksum_file_itself"])
    gate.check("manifest_inventory_readable",
               bool(covered) and all("sha256" in spec for spec in covered.values()),
               f"{len(covered)} file(s) covered by the frozen manifest")
    if not packet.is_dir():
        gate.check("original_packet_present", False, f"{packet} is absent")
        return manifest
    gate.check("original_packet_present", True, str(packet.name))
    for name, spec in sorted(covered.items()):
        path = packet / name
        if not path.exists():
            gate.check(f"packet_{name}", False, "absent")
            continue
        digest = sha256_file(path)
        gate.check(f"packet_{name}", digest == spec["sha256"],
                   digest if digest == spec["sha256"]
                   else f"{digest} != {spec['sha256']}")
    return manifest


def gate_protocol(gate: Gate, protocol_dir: Path) -> Dict[str, Any]:
    print("\n[4] the frozen protocol")
    manifest = load_protocol_manifest()
    want = manifest["protocol_hashes"]
    loaded: Dict[str, Any] = {}
    for axis in (AXIS1, AXIS2, SAFETY):
        try:
            protocol = load_protocol(axis, directory=protocol_dir,
                                     manifest=manifest)
        except TripletError as exc:
            gate.check(f"protocol_{axis}", False, str(exc)[:110])
            continue
        loaded[axis] = protocol
        ok = all(protocol.hashes()[k] == want[k] for k in protocol.hashes())
        gate.check(f"protocol_{axis}", ok, protocol.instructions_sha256[:16] + "…")
    gate.check("axis1_version_is_chg004",
               manifest["axis1_version"]["name"] == "CHG-004",
               manifest["axis1_version"]["name"])
    gate.check("axis1_not_claimed_as_the_corpus_version",
               manifest["axis1_version"]["relationship_to_the_corpus_version"][
                   "identical"] is False,
               "CHG-004 is not the CHG-001 string the 501-row corpus run used")
    return loaded


def gate_axis2_single_protocol(gate: Gate, packets: Sequence[Path],
                               protocol: Dict[str, Any]) -> None:
    """Requirement 2: one Axis-2 protocol, everywhere, or stop.

    Every Axis-2 request in every packet -- original and supplemental -- is
    re-rendered from the frozen instructions and template, and the three digests
    are compared. The 2026-08-04 re-audit used an undocumented Axis-2 variant
    (``ed57133b…``) which is not available, is not reconstructed, and must not
    appear anywhere. This gate is what makes that checkable rather than assumed.
    """
    print("\n[5] one Axis-2 protocol across every packet")
    if AXIS2 not in protocol:
        gate.check("axis2_protocol_loaded", False, "the Axis-2 protocol did not load")
        return
    axis2 = protocol[AXIS2]
    frozen = load_protocol_manifest()["protocol_hashes"]
    forbidden = load_protocol_manifest()["known_divergence_not_resolved_here"]
    gate.check("undocumented_axis2_variant_is_named_and_excluded",
               "ed57133b" in json.dumps(forbidden),
               "the 2026-08-04 variant is recorded as a provenance limitation "
               "and is not used")

    rendered = 0
    consistent = True
    for packet in packets:
        path = packet / f"{AXIS2}_judge_input.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            system, _user = axis2.render(row)
            if system != axis2.instructions:
                consistent = False
            rendered += 1
    gate.check("every_axis2_request_uses_the_frozen_instructions",
               consistent and rendered > 0,
               f"{rendered} Axis-2 request(s) rendered from "
               f"{frozen['axis2_instructions_sha256'][:16]}…")
    for key in ("axis2_instructions_sha256", "axis2_user_template_sha256",
                "axis2_schema_sha256"):
        gate.check(f"single_{key}", axis2.hashes()[key] == frozen[key],
                   frozen[key][:16] + "…")


# ------------------------------------------------------- 6, 7, 8. the counts
def gate_counts(gate: Gate, original: Path, supplement: Path,
                stage: str = "all") -> None:
    """Row counts, per stage.

    ``--stage original`` is the state the operator is actually in before the
    supplement has been built, so it must not fail on a supplement that does
    not exist yet. It checks the original counts and stops there. The combined
    total is only meaningful once both packets are real, so it is a gate of
    ``all`` alone.
    """
    print("\n[6-8] request counts")
    original_total = 0
    for axis, want in EXPECTED_ORIGINAL_BY_AXIS.items():
        path = original / f"{axis}_judge_input.jsonl"
        got = len(read_jsonl(path)) if path.exists() else 0
        original_total += got
        gate.check(f"original_{axis}_rows", got == want, f"{got}, expected {want}")
    gate.check("original_requests_per_judge",
               original_total == EXPECTED_ORIGINAL_PER_JUDGE,
               f"{original_total}, expected {EXPECTED_ORIGINAL_PER_JUDGE}")

    if stage == "original":
        gate.check("supplement_not_required_at_this_stage", True,
                   "stage=original: the supplemental packet is not read, not "
                   "counted and not required to exist")
        return

    supplement_total = 0
    for axis, want in EXPECTED_SUPPLEMENT_BY_AXIS.items():
        path = supplement / f"{axis}_judge_input.jsonl"
        got = len(read_jsonl(path)) if path.exists() else 0
        supplement_total += got
        gate.check(f"supplement_{axis}_rows", got == want, f"{got}, expected {want}")
    gate.check("supplement_requests_per_judge",
               supplement_total == EXPECTED_SUPPLEMENT_PER_JUDGE,
               f"{supplement_total}, expected {EXPECTED_SUPPLEMENT_PER_JUDGE}")

    if stage != "all":
        return
    combined = original_total + supplement_total
    gate.check("combined_requests_per_judge",
               combined == EXPECTED_COMBINED_PER_JUDGE,
               f"{combined}, expected {EXPECTED_COMBINED_PER_JUDGE} "
               f"({2 * combined} across both judges)")


# ------------------------------------------------------------ 9. the models
def gate_models(gate: Gate) -> None:
    print("\n[9] judge identifiers")
    judges = load_protocol_manifest()["judges"]
    expected = {
        "sonnet": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "opus": "eu.anthropic.claude-opus-4-6-v1",
    }
    for role in ("primary", "secondary"):
        spec = judges[role]
        gate.check(f"{spec['key']}_model_identifier",
                   spec["model"] == expected[spec["key"]], spec["model"])
    gate.check("both_judges_declared", {j["key"] for j in judges.values()
                                        if isinstance(j, dict)} == set(JUDGES),
               ", ".join(JUDGES))


# ------------------------------------------------ 10. the output directories
def gate_output_dirs(gate: Gate, judging: Path, protocol: Dict[str, Any]) -> None:
    print("\n[10] output directories")
    if not judging.exists():
        gate.check("no_incompatible_existing_run", True, f"{judging} is new")
        return
    frozen = load_protocol_manifest()["protocol_hashes"]
    incompatible: List[str] = []
    inprogress: List[str] = []
    for directory in sorted(p for p in judging.iterdir() if p.is_dir()):
        if directory.name.endswith(".inprogress"):
            inprogress.append(directory.name)
        for record_name in ("run_manifest.json", "config.json"):
            record = directory / record_name
            if not record.exists():
                continue
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
            except Exception:
                incompatible.append(f"{directory.name}: unreadable {record_name}")
                continue
            hashes = payload.get("protocol_hashes") or {}
            for key, value in hashes.items():
                if key in frozen and value != frozen[key]:
                    incompatible.append(f"{directory.name}: {key} differs")
    gate.check("no_incompatible_existing_run", not incompatible,
               "; ".join(incompatible) if incompatible
               else f"{len(list(judging.iterdir()))} directory/ies, all compatible")
    if inprogress:
        print(f"       note: {len(inprogress)} resumable .inprogress run(s): "
              f"{inprogress}. Resume them; do not start a fresh name.")


# ------------------------------------------------------ 11, 12. rows and ids
def gate_rows(gate: Gate, packets: Sequence[Path]) -> None:
    print("\n[11-12] opaque ids, blindness, sealed paths")
    all_ids: List[str] = []
    leaks: List[str] = []
    for packet in packets:
        for axis in (AXIS1, AXIS2, SAFETY):
            path = packet / f"{axis}_judge_input.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                all_ids.append(row.get("row_id", ""))
                try:
                    assert_row_is_blind(row, axis)
                    assert_opaque(row["row_id"], row["row_id"][:11])
                except TripletError as exc:
                    leaks.append(str(exc)[:100])
    try:
        assert_ids_unique(all_ids)
        unique = True
    except TripletError:
        unique = False
    gate.check("opaque_ids_unique", unique and bool(all_ids),
               f"{len(all_ids)} id(s), {len(set(all_ids))} unique")
    gate.check("no_role_donor_layer_strength_or_condition_reaches_a_judge",
               not leaks, "; ".join(leaks[:3]) if leaks
               else f"{len(all_ids)} row(s) carry only their axis's fields")

    paths_file = REPO_ROOT / "configs" / "paths.local.json"
    sealed_hits: List[str] = []
    if paths_file.exists():
        for key, value in json.loads(paths_file.read_text(encoding="utf-8")).items():
            if key.startswith("_") or not isinstance(value, str):
                continue
            token = looks_sealed(value)
            if token:
                sealed_hits.append(f"{key} -> {token}")
            if Path(value).exists() and any(
                    t in str(p).lower() for p in Path(value).rglob("*")
                    for t in SEALED_PATH_TOKENS):
                sealed_hits.append(f"{key} contains sealed-looking material")
                break
    gate.check("no_configured_path_resolves_to_sealed_material", not sealed_hits,
               "; ".join(sealed_hits) if sealed_hits
               else "no configured path names or contains held-out material")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-head", default=None,
                        help="the reviewed commit this run is pinned to")
    parser.add_argument("--original", type=Path, default=None)
    parser.add_argument("--supplement", type=Path, default=None)
    parser.add_argument("--judging", type=Path, default=None)
    parser.add_argument("--protocol-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=["original", "supplement", "all"],
                        default="all",
                        help="which packets this run is gating. 'original' "
                             "checks the historical packet alone and does not "
                             "require the supplement to exist yet; "
                             "'supplement' and 'all' require both packets.")
    args = parser.parse_args(argv)

    # Only consult the machine-local path file for the paths not given. A run
    # that names all four should not fail because a config it never reads is
    # missing -- that failure would look like a gate failure and is not one.
    given = (args.original, args.supplement, args.judging, args.protocol_dir)
    root = working_dir() if not all(given) else None
    original = args.original or (root / "historical_packet")
    supplement = args.supplement or (root / "supplemental_packet")
    judging = args.judging or (root / "judging")
    protocol_dir = args.protocol_dir or (root / "frozen" / "rq1_judge_protocol")

    stage = args.stage
    print(f"preflight — offline, read-only, no API call, no model, no GPU "
          f"(stage={stage})")
    gate = Gate()
    gate_repository(gate, args.expect_head)
    gate_original_packet(gate, original)
    protocol = gate_protocol(gate, protocol_dir)

    # Which packets this stage is responsible for. `original` is the state
    # before the supplement exists, so it must neither read nor require it.
    # `supplement` and `all` require both, because a supplement is only
    # meaningful against the original it completes -- its opaque ids must not
    # collide with the original's, and that cannot be checked from one packet.
    required = [original] if stage == "original" else [original, supplement]
    for path in required:
        gate.check(f"packet_present:{path.name}", path.is_dir(),
                   str(path) if path.is_dir() else f"{path} is absent")
    if stage == "original" and supplement.is_dir():
        gate.check("supplement_ignored_at_this_stage", True,
                   f"{supplement.name} exists but stage=original does not "
                   f"read it")
    packets = [p for p in required if p.is_dir()]

    gate_axis2_single_protocol(gate, packets, protocol)
    gate_counts(gate, original, supplement, stage)
    gate_models(gate)
    gate_output_dirs(gate, judging, protocol)
    gate_rows(gate, packets)

    passed = sum(1 for r in gate.rows if r["ok"])
    print(f"\npreflight: {passed}/{len(gate.rows)} passed, {gate.failed} failed")
    if gate.failed:
        print("STOP. A preflight failure is not something to adapt around.",
              file=sys.stderr)
        return 1
    print("all gates passed — the judging steps may proceed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
