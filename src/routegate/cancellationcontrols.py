"""Frozen, CPU-only core of the cancellation-control experiment.

The bank is decomposed in float64 without changing its stored repair or H-E
vectors. Derived intervention vectors are float32. The matching claims concern
intended local deltas: neither bfloat16 addition nor downstream model states
are asserted to match exactly. This module never imports Torch, loads a model,
reads historical judgement outputs, writes a file, or normalises a composite.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROTOCOL = "cancellation_controls_v1"
HISTORICAL_LAYERS = (31, 32, 33)
STATES = (32, 33, 34)
BUDGET_REFERENCE_STATES = (31, 32, 33)
ALPHA_R = 0.15
ALPHA_S = 0.10
RANDOM_SEED = 20260910
RANDOM_DRAWS = 3
ARM_IDS = (
    "baseline", "repair_only", "original_combination", "attenuated_repair",
    "orthogonal_full_repair", "orthogonal_only", "random_orthogonal_1",
    "random_orthogonal_2", "random_orthogonal_3",
)
EXPECTED_ROWS = 18
EXPECTED_GENERATIONS = EXPECTED_ROWS * len(ARM_IDS)
BANK_SHA256 = "d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb"
BANK_MANIFEST_SHA256 = "4a0d390ff7c8e621eeefbb83e80ca38de668ca48283fb9000ac1a502356fe538"
DATASET_SHA256 = "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08"
EXTRACTION_MANIFEST_SHA256 = "a3b7444b1a2782b5bc04142f3545ac5d71ad57a5f546edc315d7d51d14947abd"
SELECTION_MANIFEST_SHA256 = "e2a1f5f20c81672f54c8082acd7739c4844d84bdb2e9b8349ce92a7c840ab098"
SELECTED_ROWS_SHA256 = "a54483a4266fb6ea6cae8a0cee3eab1e797cef8c6592f3bf65914c5123949ff1"
SELECTION_RELATIVE_PATH = Path("manifests/focused_direction_validation_v1_phase2_selection.json")
DATASET_RELATIVE_PATH = Path("frozen/corpus/rq3_dataset_final.csv")
BANK_RELATIVE_PATH = Path("frozen/validated_direction_bank")
SELECTED_ROW_IDS = (
    "harmbench_text_test:aa_meeting_drink_encouragement",
    "harmbench_text_test:biden_complicity_hunter_activities_reply",
    "harmbench_text_test:crack_cocaine_synthesis_verification",
    "harmbench_text_test:disability_negative_impact_post",
    "harmbench_text_test:election_fraud_2020_flip_result_reply_1",
    "harmbench_text_test:http_parameter_pollution_bypass_code",
    "harmbench_text_test:lockdown_systems_reply_payment",
    "harmbench_text_test:ransomware_encryption_script",
    "harmbench_text_test:senator_cocaine_usage_claim",
    "harmbench_text_test:zero_day_vulnerabilities_purchase_guide",
    "orbench_toxic:602d88fbc64d", "orbench_toxic:7a88f5d40c8c",
    "strongreject:1619a7a3b628", "strongreject:5f3903556cef",
    "strongreject:669857cdbf49", "strongreject:6afd4b66e096",
    "strongreject:c9871fd41d31", "strongreject:f8143d545807",
)
# Unit-space dot/norm checks after one float32 cast. This is not a model's
# post-bfloat16 tolerance and must never be reused as one.
ANALYTIC_ATOL = 16.0 * float(np.finfo(np.float32).eps)


class CancellationControlsError(ValueError):
    """A frozen input or intended-edit identity did not verify."""


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_bytes(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise CancellationControlsError(
            f"SHA256 mismatch for {path.name}: {actual}; expected {expected}")
    return content


def selected_rows_digest(row_ids: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(sorted(row_ids), ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_original_bank(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = json.loads(_verified_bytes(
        directory / "direction_bank_manifest.json", BANK_MANIFEST_SHA256))
    artifact = directory / "validated_direction_bank.npz"
    _verified_bytes(artifact, BANK_SHA256)
    if manifest["outputs"]["artifact"]["sha256"] != BANK_SHA256:
        raise CancellationControlsError("Original manifest names a different bank")
    if manifest["geometry"]["historical_layers"] != list(HISTORICAL_LAYERS):
        raise CancellationControlsError("Original bank has different intervention layers")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != {"repair_direction", "safety_direction"}:
            raise CancellationControlsError("Unexpected original bank members")
        repair = archive["repair_direction"].copy()
        safety = archive["safety_direction"].copy()
    for name, array in (("repair", repair), ("safety", safety)):
        if array.shape != (3, 2048) or array.dtype != np.dtype("float32"):
            raise CancellationControlsError(f"Wrong shape or dtype for {name}")
        if not np.isfinite(array).all():
            raise CancellationControlsError(f"Non-finite {name} vector")
        if not np.allclose(np.linalg.norm(array.astype(np.float64), axis=1),
                           1.0, rtol=0.0, atol=ANALYTIC_ATOL):
            raise CancellationControlsError(f"Non-unit original {name} vector")
    return repair, safety, manifest


def build_control_bank(bank_directory: Path | str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the pinned bank, returning read-only arrays and JSON-safe metadata.

    Arrays: repair/safety/safety_perp/zero [layer,hidden], q [layer],
    random_orthogonal [draw,layer,hidden]. Each random vector has the length of
    s_perp, rather than unit length, so all second components use alpha_S=.10.
    A parent runner may serialise these arrays; this function writes nothing.
    """
    directory = Path(bank_directory)
    repair, safety, _ = _read_original_bank(directory)
    r64, s64 = repair.astype(np.float64), safety.astype(np.float64)
    rr = np.einsum("ij,ij->i", r64, r64)
    q = np.einsum("ij,ij->i", s64, r64) / rr
    perp64 = s64 - q[:, None] * r64
    perp = perp64.astype(np.float32)
    random_vectors = np.empty((RANDOM_DRAWS, 3, 2048), dtype=np.float32)
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    for index in range(3):
        p = perp64[index]
        pp = float(p @ p)
        if pp <= 1e-12 or ALPHA_R + ALPHA_S * q[index] <= 0.0:
            raise CancellationControlsError("Degenerate safety residual or negative attenuation")
        for draw in range(RANDOM_DRAWS):
            candidate = rng.normal(size=2048)
            # Two Gram-Schmidt passes suppress roundoff before the final cast.
            for _ in range(2):
                candidate -= float(candidate @ r64[index]) / rr[index] * r64[index]
                candidate -= float(candidate @ p) / pp * p
            norm = float(np.linalg.norm(candidate))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise CancellationControlsError("Degenerate fixed random draw")
            random_vectors[draw, index] = (candidate * np.sqrt(pp) / norm).astype(np.float32)
    arrays = {"repair": repair, "safety": safety, "q": q,
              "safety_perp": perp, "random_orthogonal": random_vectors,
              "zero": np.zeros_like(repair)}
    checks = validate_control_bank(arrays)
    for array in arrays.values():
        array.setflags(write=False)
    per_layer = []
    for index, layer in enumerate(HISTORICAL_LAYERS):
        per_layer.append({
            "historical_layer": layer, "output_state": STATES[index],
            "budget_reference_state": BUDGET_REFERENCE_STATES[index],
            "q": float(q[index]),
            "cosine": float(s64[index] @ r64[index]) / float(
                np.linalg.norm(s64[index]) * np.linalg.norm(r64[index])),
            "attenuated_alpha_R": float(ALPHA_R + ALPHA_S * q[index]),
            "safety_perp_norm": float(np.linalg.norm(perp[index].astype(np.float64))),
        })
    metadata = {
        "protocol": PROTOCOL, "arm_ids": list(ARM_IDS),
        "alpha_R": ALPHA_R, "alpha_S": ALPHA_S,
        "historical_layers": list(HISTORICAL_LAYERS), "states": list(STATES),
        "budget_reference_states": list(BUDGET_REFERENCE_STATES),
        "token_position": "final prompt token", "prefill_only": True,
        "combined_delta_renormalised": False,
        "expected_rows": EXPECTED_ROWS, "expected_fresh_generations": EXPECTED_GENERATIONS,
        "all_reference_arms_generated_fresh": True,
        "original_bank_sha256": BANK_SHA256,
        "original_bank_manifest_sha256": BANK_MANIFEST_SHA256,
        "random": {"seed": RANDOM_SEED, "bit_generator": "PCG64", "draws": RANDOM_DRAWS,
                   "order": "layer ascending, then draw ascending", "gram_schmidt_passes": 2,
                   "orthogonal_to": ["repair", "safety_perp"],
                   "norm": "norm of the float64 safety residual before float32 casting",
                   "numpy_version": np.__version__},
        "per_layer": per_layer, "checks": checks,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": _array_hash(value)} for key, value in arrays.items()},
        "scope": "conditional mechanism test on 18 previously selected H regressions; not a population estimate",
        "numerical_limit": "Intended local float32 matching only. Cast deltas and live additions to bfloat16 can change realised projections and norms; earlier edits change later live states.",
        "model_loaded": False, "reads_historical_judgement_outputs": False,
    }
    return arrays, metadata


def arm_components(arrays: Mapping[str, np.ndarray], arm_id: str, state: int
                   ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (repair vector, second vector, alpha_r, alpha_s) for one site.

    Intended delta is the unsteered input norm times the returned weighted sum.
    The returned vectors are not unit-normalised here. Baseline is explicitly
    zero and should cause no intervention hook to apply an edit.
    """
    if arm_id not in ARM_IDS:
        raise CancellationControlsError(f"Unknown arm {arm_id!r}")
    if isinstance(state, bool) or not isinstance(state, (int, np.integer)) or state not in STATES:
        raise CancellationControlsError(f"Unknown output state {state!r}")
    index = STATES.index(state)
    r, zero = arrays["repair"][index], arrays["zero"][index]
    reduced = float(ALPHA_R + ALPHA_S * arrays["q"][index])
    if arm_id == "baseline":
        return zero, zero, 0.0, 0.0
    if arm_id == "repair_only":
        return r, zero, ALPHA_R, 0.0
    if arm_id == "original_combination":
        return r, arrays["safety"][index], ALPHA_R, ALPHA_S
    if arm_id == "attenuated_repair":
        return r, zero, reduced, 0.0
    if arm_id == "orthogonal_full_repair":
        return r, arrays["safety_perp"][index], ALPHA_R, ALPHA_S
    if arm_id == "orthogonal_only":
        return zero, arrays["safety_perp"][index], 0.0, ALPHA_S
    draw = int(arm_id.rsplit("_", 1)[1]) - 1
    return r, arrays["random_orthogonal"][draw, index], reduced, ALPHA_S


def validate_control_bank(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Verify scientific matching identities at float32 realistic tolerances."""
    shapes = {"repair": (3, 2048), "safety": (3, 2048), "q": (3,),
              "safety_perp": (3, 2048), "random_orthogonal": (3, 3, 2048),
              "zero": (3, 2048)}
    if set(arrays) != set(shapes):
        raise CancellationControlsError("Control bank members differ from the fixed schema")
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        dtype = np.dtype("float64" if name == "q" else "float32")
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise CancellationControlsError(f"Invalid array {name}")
    if np.any(arrays["zero"]):
        raise CancellationControlsError("Baseline zero vector is nonzero")
    errors = {name: 0.0 for name in (
        "q", "orthogonal_dot", "decomposition_l2", "float32_decomposition_l2",
        "attenuation_projection", "random_repair_dot", "random_safety_dot",
        "random_norm", "random_composite_norm", "random_composite_projection")}
    for index, state in enumerate(STATES):
        r, s = arrays["repair"][index].astype(np.float64), arrays["safety"][index].astype(np.float64)
        p = arrays["safety_perp"][index].astype(np.float64)
        q = float(arrays["q"][index])
        if not 0.0 < ALPHA_R + ALPHA_S*q < ALPHA_R:
            raise CancellationControlsError("This frozen experiment requires attenuation")
        rr = float(r @ r)
        original = ALPHA_R*r + ALPHA_S*s
        attenuation = (ALPHA_R + ALPHA_S*q)*r
        errors["q"] = max(errors["q"], abs(q-float(s @ r)/rr))
        errors["orthogonal_dot"] = max(errors["orthogonal_dot"], abs(float(p @ r)))
        errors["decomposition_l2"] = max(errors["decomposition_l2"], float(np.linalg.norm(original-attenuation-ALPHA_S*p)))
        original32 = np.float32(ALPHA_R)*arrays["repair"][index] + np.float32(ALPHA_S)*arrays["safety"][index]
        reconstruction32 = np.float32(ALPHA_R+ALPHA_S*q)*arrays["repair"][index] + np.float32(ALPHA_S)*arrays["safety_perp"][index]
        errors["float32_decomposition_l2"] = max(errors["float32_decomposition_l2"], float(np.linalg.norm(original32.astype(np.float64)-reconstruction32)))
        errors["attenuation_projection"] = max(errors["attenuation_projection"], abs(float((original-attenuation) @ r)/rr))
        for vector in arrays["random_orthogonal"][:, index]:
            v = vector.astype(np.float64)
            control = attenuation + ALPHA_S*v
            errors["random_repair_dot"] = max(errors["random_repair_dot"], abs(float(v @ r)))
            errors["random_safety_dot"] = max(errors["random_safety_dot"], abs(float(v @ p)))
            errors["random_norm"] = max(errors["random_norm"], abs(float(np.linalg.norm(v)-np.linalg.norm(p))))
            errors["random_composite_norm"] = max(errors["random_composite_norm"], abs(float(np.linalg.norm(control)-np.linalg.norm(original))))
            errors["random_composite_projection"] = max(errors["random_composite_projection"], abs(float((control-original) @ r)/rr))
    if any(error > ANALYTIC_ATOL for error in errors.values()):
        raise CancellationControlsError(f"Intended-edit matching failed: {errors}")
    return {"all_passed": True, "absolute_tolerance": ANALYTIC_ATOL,
            "arithmetic": "float64 checks of stored float32 vectors plus float32 combination check",
            "maximum_absolute_errors": errors}


def load_selected_rows(repo_root: Path | str, data_root: Path | str) -> list[dict[str, str]]:
    """Return the frozen 18 prompts after source, bank and selection checks.

    Reads the original selection manifest and development CSV, not the
    historical responses or adjudication outputs. Only prompt and provenance
    fields leave this function, despite other columns present in the corpus.
    """
    repo, data = Path(repo_root), Path(data_root)
    selection = json.loads(_verified_bytes(repo / SELECTION_RELATIVE_PATH,
                                           SELECTION_MANIFEST_SHA256))
    row_ids = selection.get("row_ids")
    if row_ids != list(SELECTED_ROW_IDS) or len(set(row_ids)) != EXPECTED_ROWS:
        raise CancellationControlsError("Selection differs from the frozen 18 H rows")
    if (selected_rows_digest(row_ids) != SELECTED_ROWS_SHA256 or
            selection.get("selected_rows_sha256") != SELECTED_ROWS_SHA256):
        raise CancellationControlsError("Selected-row digest failed")
    treatment = selection["treatment_that_produced_them"]
    if (treatment["direction_bank_sha256"] != BANK_SHA256 or
            treatment["historical_layers"] != list(HISTORICAL_LAYERS)):
        raise CancellationControlsError("Selection treatment differs from the bank or sites")
    _read_original_bank(data / BANK_RELATIVE_PATH)
    dataset = data / DATASET_RELATIVE_PATH
    _verified_bytes(dataset, DATASET_SHA256)
    declared = json.loads((repo / "manifests/development_inputs.json").read_text())
    if declared["dataset"]["sha256"] != DATASET_SHA256:
        raise CancellationControlsError("Development manifest names a different corpus")
    extraction_path = data / "frozen/residual_extraction/manifest.csv"
    _verified_bytes(extraction_path, EXTRACTION_MANIFEST_SHA256)
    with extraction_path.open(newline="", encoding="utf-8") as handle:
        extraction = {row["row_id"]: row for row in csv.DictReader(handle)}
    selected: dict[str, dict[str, str]] = {}
    with dataset.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_id = row["row_id"]
            if row_id not in SELECTED_ROW_IDS:
                continue
            if row_id in selected or row["final_ebh"] != "H":
                raise CancellationControlsError("Selected row duplicated or not H")
            prompt = row["prompt_text"]
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            witness = extraction.get(row_id, {})
            if (not prompt or witness.get("prompt_text") != prompt or
                    witness.get("prompt_sha256") != digest or
                    witness.get("source_family") != row["source_family"] or
                    witness.get("final_ebh") != "H"):
                raise CancellationControlsError(f"Prompt provenance failed for {row_id}")
            selected[row_id] = {"row_id": row_id, "prompt_text": prompt,
                                "prompt_sha256": digest, "row_class": "H",
                                "source_family": row["source_family"]}
    if set(selected) != set(SELECTED_ROW_IDS):
        raise CancellationControlsError("Corpus is missing selected prompts")
    return [selected[row_id] for row_id in SELECTED_ROW_IDS]
