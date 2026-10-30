#!/usr/bin/env python3
"""Build the validated dual-direction bank. Loads no model; calls nothing.

Two directions over historical decoder blocks 31/32/33, at the final prompt
token, from the ten retrospectively validated E/B/H triplets::

    repair = unit( mean over 10 cases of unit( B_d1 - E ) )
    safety = unit( mean over 10 cases of unit( H    - E ) )

built independently at each layer. The arithmetic and the gates live in
``routegate.directionbank``; this file supplies the paths, checks the inputs
against their recorded digests, and writes the artefact.

Everything is read
------------------
No model is loaded, no GPU is touched, nothing is generated and nothing is
judged. The activations were captured on 2026-07-09 and are read as stored:
``torch.load(..., weights_only=True)``, so no code inside the capture runs.

What is checked before a direction is built
-------------------------------------------
1. every input file against the SHA256 in ``manifests/rq2_geometry_capture.json``;
2. the eligibility record's own digest -- it is the authoritative gate and is
   not read until it has been shown to be the file this build was pinned to;
3. exactly ten cases marked ``validated`` for each direction's own rule, in the
   manifest's order, and the same ten in the capture, in the same order;
4. the thirty prompt digests, three ways: eligibility record, capture manifest,
   RQ2 source-case manifest;
5. the model and revision, the layer band, the state numbering, the token
   position, the tensor shapes and the dtype;
6. that the rebuilt repair direction reproduces the historical frozen d1
   comparator to float32 rounding.

Any failure raises and nothing is written.

What this does not do
---------------------
It does not orthogonalise the two directions -- their per-layer cosine is
measured and recorded, and that is all. It does not choose, embed or default an
application strength: this phase constructs directions, and scaling one is a
separate decision taken elsewhere. It does not fit or train the coefficient
controller.

Usage
-----
    python experiments/dual_direction_controller/build_validated_direction_bank.py \\
        --capture-dir <the migrated capture directory> \\
        --inclusion-manifest manifests/rq1_triplet_inclusion.json \\
        --source-case-manifest <rq2_actuator>/source_case_manifest.csv \\
        --actuator-dir <rq2_actuator> \\
        --output-dir <a new directory> [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from routegate import __version__ as PACKAGE_VERSION      # noqa: E402
from routegate.actuator import (                                     # noqa: E402
    COMPARATOR_VARIANT,
    load_frozen_actuator,
)
from routegate.directionbank import (                                # noqa: E402
    ARM_TENSOR,
    ARTIFACT_ARRAYS,
    BANK_STATES,
    BASELINE_ARM,
    BASELINE_ONLY_ARMS,
    BASELINE_ONLY_RULE,
    CAPTURE_DTYPE,
    CAPTURE_SHAPE,
    DIRECTIONS,
    HISTORICAL_LAYERS,
    MODEL_NAME,
    MODEL_REVISION,
    SITE,
    SITE_KIND,
    TOKEN_POSITION,
    TOKEN_POSITION_NOTE,
    DirectionBankError,
    array_sha256,
    assert_layers_and_states,
    build_bank,
    capture_arrays,
    eligibility_record_sha256,
    eligible_case_ids,
    npz_bytes,
    parity_against_frozen,
    sha256_bytes,
)
from routegate.runrecord import environment_record, utc_now          # noqa: E402
from routegate.validation import (                                   # noqa: E402
    Checklist,
    ValidationError,
    sha256_file,
)

#: The committed record naming every input and its digest.
CAPTURE_MANIFEST = REPO_ROOT / "manifests" / "rq2_geometry_capture.json"

CAPTURE_FILE = "capture.pt"
CAPTURE_META_FILE = "manifest.json"

ARTIFACT_OUT = "validated_direction_bank.npz"
MANIFEST_OUT = "direction_bank_manifest.json"

#: How the capture's own manifest names each arm's prompt digest, and how the
#: RQ2 source-case manifest does. Two records, two vocabularies, one mapping.
CAPTURE_PROMPT_KEY: Dict[str, str] = {
    "E": "safe_target",
    "B_d1": "donor_prompt",
    "H": "harmful_contrast",
}
SOURCE_PROMPT_COLUMN: Dict[str, str] = {
    "E": "safe_target_sha256",
    "B_d1": "donor_d1_historical_sha256",
    "H": "harmful_contrast_sha256",
}


def gate(checks: Checklist, name: str, passed: bool, detail: str) -> None:
    """Record the check, and stop the build if it did not pass."""
    checks.record(name, passed, detail)
    if not passed:
        raise DirectionBankError(f"{name}: {detail}")


def assert_output_dir_is_new(out_dir: Path) -> None:
    """A bank is written once, into somewhere new.

    Overwriting one in place would leave its manifest describing bytes that are
    no longer there, and nothing downstream would be able to tell.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DirectionBankError(
            f"{out_dir} already exists and is not empty. The direction bank is "
            f"written once; name a new directory.")


def verify_file(path: Path, expected: Mapping[str, Any], label: str,
                checks: Checklist) -> str:
    """Re-hash one input and refuse it if the digest is not the recorded one."""
    if not path.is_file():
        raise DirectionBankError(
            f"{label}: {path} is not a file. Every input is named explicitly "
            f"and an absent one is refused, never skipped.")
    digest = sha256_file(path)
    gate(checks, f"{label}_sha256", digest == expected["sha256"],
         f"{path.name} is {digest}, recorded {expected['sha256']}")
    size = path.stat().st_size
    if "bytes" in expected:
        gate(checks, f"{label}_bytes", size == expected["bytes"],
             f"{path.name} is {size} byte(s), recorded {expected['bytes']}")
    return digest


def verify_capture_meta(meta: Mapping[str, Any], capture_meta: Mapping[str, Any],
                        pinned: Mapping[str, Any], checks: Checklist) -> None:
    """The capture describes the band, the position and the model it claims.

    ``meta`` is the record inside ``capture.pt``; ``capture_meta`` is the
    sidecar ``manifest.json``. They are checked separately and against each
    other: agreeing with itself is not the same as being the right capture.
    """
    assert_layers_and_states(meta.get("layers", ()), BANK_STATES,
                             "the capture's meta record")
    assert_layers_and_states(capture_meta.get("layers", ()), BANK_STATES,
                             "the capture's manifest")
    gate(checks, "historical_layers", True,
         f"{list(HISTORICAL_LAYERS)} -> states {list(BANK_STATES)}")

    slot = str(meta.get("slot_convention", ""))
    gate(checks, "token_position",
         "slot 0" in slot and "final prefill row" in slot,
         f"{TOKEN_POSITION}: {slot[:90]}")
    gate(checks, "window_is_one", int(meta.get("window", -1)) == 1,
         f"window={meta.get('window')}")

    gate(checks, "model_name", capture_meta.get("model") == MODEL_NAME,
         f"{capture_meta.get('model')!r}, expected {MODEL_NAME!r}")
    gate(checks, "model_revision_pinned",
         pinned["model"]["revision"] == MODEL_REVISION,
         f"{MODEL_REVISION} "
         f"({pinned['model']['revision_provenance'][:60]}...)")
    gate(checks, "capture_dtype",
         capture_meta.get("dtype") == pinned["model"]["dtype"],
         f"forward pass {capture_meta.get('dtype')}, stored as "
         f"{CAPTURE_DTYPE}")


def verify_case_agreement(inclusion: Mapping[str, Any],
                          capture_meta: Mapping[str, Any],
                          meta: Mapping[str, Any],
                          source_rows: Mapping[str, Mapping[str, str]],
                          checks: Checklist) -> Sequence[str]:
    """The same ten cases, in the same order, with the same thirty digests."""
    expected = list(inclusion["expected_case_ids"])
    gate(checks, "capture_case_order",
         list(meta.get("case_ids", [])) == expected,
         f"{len(expected)} case(s), identical order in the capture and the "
         f"eligibility record")
    gate(checks, "capture_manifest_case_order",
         list(capture_meta.get("case_ids", [])) == expected,
         "the capture's manifest lists the same ten, in the same order")
    gate(checks, "source_manifest_cases", set(source_rows) == set(expected),
         f"{len(source_rows)} case(s) in the RQ2 source-case manifest")

    mismatched = []
    for case_id in expected:
        for arm in sorted(ARM_TENSOR):
            a = inclusion["cases"][case_id]["arms"][arm]["prompt_sha256"]
            b = capture_meta["prompt_hashes"][case_id][CAPTURE_PROMPT_KEY[arm]]
            c = source_rows[case_id][SOURCE_PROMPT_COLUMN[arm]]
            if not a == b == c:
                mismatched.append((case_id, arm))
    gate(checks, "prompt_digests_agree_three_ways", not mismatched,
         f"{len(expected) * len(ARM_TENSOR)} digest(s) checked across the "
         f"eligibility record, the capture manifest and the RQ2 source-case "
         f"manifest"
         + (f"; mismatched {mismatched[:3]}" if mismatched else ""))
    return expected


def build(capture_dir: Path, inclusion_path: Path, source_case_path: Path,
          actuator_dir: Path, out_dir: Path, write: bool = True,
          capture_manifest_path: Path = CAPTURE_MANIFEST) -> Dict[str, Any]:
    checks = Checklist("validated_direction_bank")
    if write:
        assert_output_dir_is_new(out_dir)

    pinned = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    capture_path = capture_dir / CAPTURE_FILE
    capture_meta_path = capture_dir / CAPTURE_META_FILE

    # --- every input, against the digest this build was pinned to ---------
    input_digests: Dict[str, Dict[str, Any]] = {}
    for label, path, spec in (
            ("capture", capture_path, pinned["files"][CAPTURE_FILE]),
            ("capture_manifest", capture_meta_path,
             pinned["files"][CAPTURE_META_FILE]),
            ("inclusion_manifest", inclusion_path, pinned["eligibility_gate"]),
            ("source_case_manifest", source_case_path,
             pinned["case_digest_cross_check"])):
        digest = verify_file(path, spec, label, checks)
        input_digests[label] = {"path": str(path), "name": path.name,
                                "sha256": digest,
                                "bytes": path.stat().st_size}

    inclusion = json.loads(inclusion_path.read_text(encoding="utf-8"))
    capture_meta = json.loads(capture_meta_path.read_text(encoding="utf-8"))
    source_rows = {row["case_id"]: row for row in
                   csv.DictReader(source_case_path.open(encoding="utf-8"))}

    # --- eligibility, from the authoritative record and nowhere else -------
    case_ids: Dict[str, Sequence[str]] = {}
    eligibility: Dict[str, Any] = {}
    for direction in sorted(DIRECTIONS):
        rule = DIRECTIONS[direction]["rule"]
        ids = eligible_case_ids(inclusion, rule)
        case_ids[direction] = ids
        eligibility[direction] = {
            "rule": rule,
            "arm": DIRECTIONS[direction]["arm"],
            "n_cases": len(ids),
            "case_ids": list(ids),
            "eligibility_record_sha256": {
                case_id: eligibility_record_sha256(inclusion, case_id, rule)
                for case_id in ids},
        }
        gate(checks, f"{direction}_eligible_cases", len(ids) == CAPTURE_SHAPE[0],
             f"{len(ids)} case(s) validated for {rule}")

    # --- the capture itself ------------------------------------------------
    activations, meta = capture_arrays(capture_path)
    verify_capture_meta(meta, capture_meta, pinned, checks)
    expected = verify_case_agreement(inclusion, capture_meta, meta,
                                     source_rows, checks)

    for direction, ids in sorted(case_ids.items()):
        gate(checks, f"{direction}_case_order", list(ids) == list(expected),
             "the eligible cases are the capture's ten, in the capture's order")
    gate(checks, "no_baseline_only_arm_is_read",
         not (set(BASELINE_ONLY_ARMS) & set(activations)),
         f"{list(BASELINE_ONLY_ARMS)} ({BASELINE_ONLY_RULE}) are not read; "
         f"this bank reads {sorted(activations)}")
    for arm in sorted(ARM_TENSOR):
        gate(checks, f"{arm}_activations", arm in activations,
             f"{ARM_TENSOR[arm]} {CAPTURE_SHAPE} {CAPTURE_DTYPE}")

    # --- the two directions ------------------------------------------------
    bank = build_bank(activations, case_ids, checks)
    norms = bank.norms()
    cosine = bank.cosine()

    # --- parity with the historical frozen comparator ----------------------
    frozen = load_frozen_actuator(variant=COMPARATOR_VARIANT,
                                  directory=actuator_dir)
    parity = parity_against_frozen(bank.repair, frozen.directions)
    gate(checks, "historical_d1_parity", parity["agrees"],
         f"cosine {[round(v, 12) for v in parity['per_layer_cosine'].values()]}")

    arrays = bank.arrays()
    payload = npz_bytes(arrays)

    manifest = {
        "schema": "routegate/validated_direction_bank/1",
        "description": "Two unit directions over historical decoder blocks "
                       "31/32/33 at the final prompt token, built from the ten "
                       "retrospectively validated E/B/H triplets. Directions "
                       "only: no orthogonalisation and no application "
                       "strength.",
        "created_utc": utc_now(),
        "loads_a_model": False,
        "inputs": input_digests,
        # named and hashed, not path-arithmetic'd: the pin may sit outside
        # the repository, and its digest is what identifies it either way
        "pinned_by": {"record": capture_manifest_path.name,
                      "sha256": sha256_file(capture_manifest_path)},
        "eligibility": {
            "authoritative_record": str(
                inclusion_path.name),
            "authoritative_record_sha256": input_digests[
                "inclusion_manifest"]["sha256"],
            "rule": "a case enters a direction if and only if the eligibility "
                    "record marks that direction's own inclusion rule "
                    "'validated' for it",
            "directions": eligibility,
            "excluded": {
                "rule": BASELINE_ONLY_RULE,
                "arms": list(BASELINE_ONLY_ARMS),
                "reason": "the historical B_d2/B_d3 primary actuator is "
                          "validated for 3 of 10 cases; the other seven were "
                          "judged and failed. It remains a comparison "
                          "baseline and does not enter this bank.",
            },
        },
        "construction": {
            "subtraction_order": {
                direction: f"{DIRECTIONS[direction]['arm']} - {BASELINE_ARM}"
                for direction in sorted(DIRECTIONS)},
            "subtraction_order_note": "arm minus baseline, never baseline "
                                      "minus arm. A reversed subtraction "
                                      "yields a well-formed unit vector "
                                      "pointing the wrong way.",
            "recipe": [
                "per case: delta = arm_activation - E_activation",
                "per case: case_direction = delta / ||delta||_2",
                "average the ten case directions",
                "renormalise the average to unit L2 norm",
            ],
            "per_layer": "each layer is built independently; no information "
                         "crosses between 31, 32 and 33",
            "per_case_normalisation": True,
            "per_case_normalisation_note": "each case is normalised before "
                                           "averaging, so every case carries "
                                           "one vote and a large delta cannot "
                                           "dominate the bank",
            "identical_recipe_for_both_directions": True,
            "refuses": ["zero or near-zero deltas", "non-finite activations",
                        "malformed shapes or dtypes"],
        },
        "geometry": {
            "site": SITE,
            "site_kind": SITE_KIND,
            "historical_layers": list(HISTORICAL_LAYERS),
            "routegate_states": list(BANK_STATES),
            "layer_to_state": {str(layer): state for layer, state
                               in zip(HISTORICAL_LAYERS, BANK_STATES)},
            "token_position": TOKEN_POSITION,
            "token_position_note": TOKEN_POSITION_NOTE,
        },
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION,
                  "revision_provenance": pinned["model"]["revision_provenance"]},
        "tensors": {
            "read_from_capture": {arm: ARM_TENSOR[arm]
                                  for arm in sorted(ARM_TENSOR)},
            "activation_shape": list(CAPTURE_SHAPE),
            "activation_dtype": str(CAPTURE_DTYPE),
            "direction_shape": [len(HISTORICAL_LAYERS), CAPTURE_SHAPE[2]],
            "direction_dtype": str(CAPTURE_DTYPE),
        },
        "norms": {name: {str(layer): value for layer, value
                         in zip(HISTORICAL_LAYERS, values)}
                  for name, values in sorted(norms.items())},
        "diagnostics": {
            "repair_safety_cosine_per_layer": {
                str(layer): value for layer, value
                in zip(HISTORICAL_LAYERS, cosine)},
            "note": "measured and recorded only. Nothing reads this to alter "
                    "either direction; it is on the record so the relation "
                    "between them is known before anyone decides whether an "
                    "orthogonalised ablation is worth running.",
        },
        "historical_d1_parity": parity,
        "orthogonalisation": {
            "applied": False,
            "statement": "no orthogonalisation of any kind was applied. The "
                         "artefact holds the direct normalised B_d1 - E and "
                         "H - E directions. An orthogonalised variant is a "
                         "later declared ablation and is not this artefact.",
        },
        "application_strength": {
            "applied": False,
            "value": None,
            "statement": "no application strength was selected, embedded or "
                         "defaulted. This phase constructs directions; "
                         "scaling one is a separate decision taken elsewhere "
                         "on its own terms. No coefficient controller is "
                         "implemented or trained here.",
        },
        "outputs": {
            "artifact": {"name": ARTIFACT_OUT, "sha256": sha256_bytes(payload),
                         "bytes": len(payload), "format": "npz",
                         "members": list(ARTIFACT_ARRAYS),
                         "pickle": False,
                         "read_with": "numpy.load(allow_pickle=False)"},
            "arrays": {name: {"sha256": array_sha256(arrays[name]),
                              "shape": list(arrays[name].shape),
                              "dtype": str(arrays[name].dtype)}
                       for name in ARTIFACT_ARRAYS},
        },
        "environment": environment_record(PACKAGE_VERSION),
        "checklist": checks.as_dict(),
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ARTIFACT_OUT).write_bytes(payload)
        (out_dir / MANIFEST_OUT).write_text(manifest_text, encoding="utf-8")
        written = sha256_file(out_dir / ARTIFACT_OUT)
        if written != manifest["outputs"]["artifact"]["sha256"]:
            raise DirectionBankError(
                f"{out_dir / ARTIFACT_OUT} hashes to {written} but the "
                f"manifest records "
                f"{manifest['outputs']['artifact']['sha256']}")
        print(f"[bank] wrote {out_dir}")

    for direction in sorted(DIRECTIONS):
        print(f"[bank] {direction:<10} {DIRECTIONS[direction]['arm']} - "
              f"{BASELINE_ARM} over {len(case_ids[direction])} validated "
              f"case(s)")
    print(f"[bank] {'layers':<10} {list(HISTORICAL_LAYERS)} -> states "
          f"{list(BANK_STATES)}")
    print(f"[bank] {'cosine':<10} repair/safety per layer "
          f"{[round(v, 6) for v in cosine]}")
    print(f"[bank] {'parity':<10} {COMPARATOR_VARIANT} cosine "
          f"{[round(v, 9) for v in parity['per_layer_cosine'].values()]}")
    print(f"[bank] {'artifact':<10} {manifest['outputs']['artifact']['sha256']}")
    print("[bank] no orthogonalisation applied; no strength applied")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True,
                        help=f"the migrated capture directory holding "
                             f"{CAPTURE_FILE} and {CAPTURE_META_FILE}")
    parser.add_argument("--inclusion-manifest", type=Path, required=True,
                        help="manifests/rq1_triplet_inclusion.json, the "
                             "authoritative eligibility record")
    parser.add_argument("--source-case-manifest", type=Path, required=True,
                        help="the RQ2 freeze's source_case_manifest.csv, for "
                             "the third-way prompt-digest cross-check")
    parser.add_argument("--actuator-dir", type=Path, required=True,
                        help="the migrated RQ2 actuator freeze, for the "
                             "historical d1 parity check")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        build(args.capture_dir, args.inclusion_manifest,
              args.source_case_manifest, args.actuator_dir, args.output_dir,
              write=not args.dry_run)
    except ValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
