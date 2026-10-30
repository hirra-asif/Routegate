"""Input gates: hashes, class counts, shape and dtype, stale and held-out refusal.

Covers requirements 1, 2, 3, 5, 6 and 7 of the test plan.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    MANIFEST_PATH,
    REPO_ROOT,
    needs_real_inputs,
    real_inputs_available,
)
from routegate.data import (
    DEFAULT_MANIFEST,
    load_development_inputs,
    load_input_manifest,
    load_local_paths,
)
from routegate.validation import (
    SEALED_PATH_TOKENS,
    STALE_DIRECTORY_TOKENS,
    STALE_FILENAMES,
    SealedSetError,
    StaleInputError,
    ValidationError,
    assert_config_is_sealed_free,
    assert_hash,
    assert_not_sealed,
    assert_not_stale,
    looks_sealed,
    safe_argmax,
    sha256_file,
)


# ---------------------------------------------------------------- 1. dataset
@needs_real_inputs
def test_dataset_sha256_is_exactly_the_accepted_value(input_manifest, local_paths):
    expected = input_manifest["dataset"]["sha256"]
    assert expected == (
        "0943449697327aa47912ef80337462414ab98255c3ad99b717baa716353e1d08")
    assert sha256_file(local_paths["dataset"]) == expected


# ------------------------------------------------------------- 2. activation
@needs_real_inputs
def test_activation_sha256_is_exactly_the_accepted_value(input_manifest, local_paths):
    expected = input_manifest["activation"]["sha256"]
    assert expected == (
        "af89872d5decd4dc432ec579a941e6247bc7de63ff5690432e9f5812a994e2d8")
    path = Path(local_paths["extraction_dir"]) / input_manifest["activation"]["filename"]
    assert sha256_file(path) == expected


@needs_real_inputs
def test_extraction_manifest_sha256_matches(input_manifest, local_paths):
    spec = input_manifest["extraction_manifest"]
    path = Path(local_paths["extraction_dir"]) / spec["filename"]
    assert sha256_file(path) == spec["sha256"]


def test_a_hash_mismatch_is_refused(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"not the accepted artefact")
    with pytest.raises(ValidationError, match="hash mismatch"):
        assert_hash(target, "0" * 64, "test artefact")


# ----------------------------------------------------------- 3. class counts
def test_manifest_declares_the_expected_counts(input_manifest):
    counts = input_manifest["dataset"]["counts"]
    assert counts["E"] == 89
    assert counts["B"] == 178
    assert counts["H"] == 213
    assert counts["EXCLUDED_PARTIAL_OR_UNCLEAR"] == 29
    assert counts["total"] == 509
    assert counts["analysis_eligible"] == 480
    assert counts["E"] + counts["B"] + counts["H"] == counts["analysis_eligible"]
    assert counts["analysis_eligible"] + counts["EXCLUDED_PARTIAL_OR_UNCLEAR"] == counts["total"]


@needs_real_inputs
def test_real_class_counts_are_exactly_as_declared(inputs):
    assert inputs.n == 267
    assert inputs.n_positive == 89
    assert inputs.n_negative == 178
    assert len(inputs.manifest_rows) == 480


@needs_real_inputs
def test_the_withheld_class_is_present_in_the_corpus_but_not_analysed(inputs):
    labels = {row["final_ebh"] for row in inputs.manifest_rows}
    assert labels == {"E", "B", "H"}
    analysed = {inputs.manifest_rows[i]["final_ebh"] for i in inputs.eb_positions}
    assert analysed == {"E", "B"}
    assert "H" not in analysed


@needs_real_inputs
def test_no_excluded_row_reaches_the_loader(inputs):
    excluded = "EXCLUDED_PARTIAL_OR_UNCLEAR"
    assert all(row["final_ebh"] != excluded for row in inputs.manifest_rows)


# ------------------------------------------------------ 5. shape and dtype
@needs_real_inputs
def test_activation_shape_and_dtype(inputs, input_manifest):
    assert tuple(inputs.activations.shape) == tuple(input_manifest["activation"]["shape"])
    assert tuple(inputs.activations.shape) == (480, 49, 2048)
    assert inputs.activations.dtype.name == "float32"
    assert inputs.n_states == 49
    assert inputs.hidden_size == 2048


@needs_real_inputs
def test_analysed_activations_are_finite(inputs):
    for state in (0, 1, 24, 45, 48):
        assert np.isfinite(inputs.state_matrix(state)).all()


# ------------------------------------------------- 6. stale input rejection
@pytest.mark.parametrize("filename", sorted(STALE_FILENAMES))
def test_stale_filenames_are_refused(filename, tmp_path):
    with pytest.raises(StaleInputError):
        assert_not_stale(tmp_path / filename)


def test_the_two_named_stale_dataset_revisions_are_refused(tmp_path):
    for filename in ("rq3_dataset_final_before_e_axis2_reaudit.csv",
                     "rq3_ebh_final_analysis_dataset.csv"):
        with pytest.raises(StaleInputError):
            assert_not_stale(tmp_path / filename)


def test_the_superseded_preparation_package_is_refused(tmp_path):
    for token in STALE_DIRECTORY_TOKENS:
        path = tmp_path / token / "dataset_full.csv"
        with pytest.raises(StaleInputError):
            assert_not_stale(path)


def test_the_accepted_dataset_name_is_not_refused(tmp_path):
    assert assert_not_stale(tmp_path / "rq3_dataset_final.csv").name == "rq3_dataset_final.csv"


def test_the_manifest_records_which_inputs_are_refused(input_manifest):
    refused = input_manifest["refused_inputs"]
    assert "rq3_dataset_final_before_e_axis2_reaudit.csv" in refused["by_filename"]
    assert "rq3_ebh_final_analysis_dataset.csv" in refused["by_filename"]
    assert "rq3_geometry_prepare_20260803_215218" in refused["by_directory_token"]


@needs_real_inputs
def test_the_loader_refuses_a_stale_dataset_path(local_paths, input_manifest):
    stale = (Path(local_paths["dataset"]).parent
             / "rq3_ebh_final_analysis_dataset.csv")
    paths = dict(local_paths, dataset=str(stale))
    with pytest.raises(StaleInputError):
        load_development_inputs(manifest=input_manifest, paths=paths)


@needs_real_inputs
def test_the_loader_reads_only_the_accepted_dataset(inputs):
    assert inputs.dataset_path.name == "rq3_dataset_final.csv"


# ------------------------------------------------ 7. held-out set rejection
@pytest.mark.parametrize("token", SEALED_PATH_TOKENS)
def test_held_out_tokens_are_detected(token):
    assert looks_sealed(f"/some/dir/{token}_manifest.csv") is not None


@pytest.mark.parametrize("token", SEALED_PATH_TOKENS)
def test_held_out_paths_are_refused(token):
    with pytest.raises(SealedSetError):
        assert_not_sealed(f"/data/{token}/rows.csv", "test path")


def test_a_benign_path_is_not_refused():
    assert_not_sealed("/data/development/rows.csv", "test path")


def test_a_nested_config_with_a_held_out_value_is_refused():
    config = {"inputs": {"a": "/data/ok.csv", "b": ["/data/sealed_rows.csv"]}}
    with pytest.raises(SealedSetError):
        assert_config_is_sealed_free(config, "test config")


def test_the_local_path_file_names_no_held_out_material(local_paths):
    assert_config_is_sealed_free(local_paths, "paths.local.json")
    assert "sealed" not in json.dumps(local_paths).lower()


def test_the_commitment_file_stores_counts_only():
    commitment = json.loads(
        (REPO_ROOT / "manifests" / "sealed_set_commitment.json").read_text())
    assert commitment["expected_row_count"] == 380
    assert commitment["expected_safe_source_count"] == 180
    assert commitment["expected_harmful_source_count"] == 200
    assert (commitment["expected_safe_source_count"]
            + commitment["expected_harmful_source_count"]
            == commitment["expected_row_count"])
    assert commitment["availability"][
        "sealed_prompt_text_available_to_development_code"] is False

    # The commitment records counts and a statement of unavailability. It may
    # name this repository's own files, since that is where the enforcement
    # lives, but it must carry nothing that could locate the held-out material:
    # no absolute path, no reference to the historical workspace, and no
    # prompt or prompt hash.
    blob = json.dumps(commitment)
    assert '"/' not in blob and ": /" not in blob, "no absolute path"
    for needle in ("SteerMoE", "RouteGate-MSc", "results/", "rq3_",
                   "prompt_splits", ".csv", ".jsonl"):
        assert needle not in blob, (
            f"the commitment file must not carry '{needle}'")

    # No stored digest: a per-prompt SHA256 would let a held-out prompt be
    # confirmed by guessing it. The file may *say* it stores no hash -- it may
    # not contain one.
    assert not re.search(r"\b[0-9a-f]{64}\b", blob), (
        "the commitment file must not store a hash of held-out material")


def test_no_development_module_encodes_a_held_out_path():
    """Development modules exclude final inputs; finaljudging has a released adapter."""
    package = REPO_ROOT / "src" / "routegate"
    forbidden = ("rq3_sealed", "sealed_final_manifest", "rq3_prompt_splits")
    for module in sorted(package.glob("*.py")):
        # This exact adapter verifies the released v5 inputs before use.
        # Its boundary is covered by tests/test_final_judging.py.
        if module.name == "finaljudging.py":
            continue
        text = module.read_text(encoding="utf-8").lower()
        for needle in forbidden:
            assert needle not in text, f"{module.name} encodes '{needle}'"


def test_the_development_configs_name_no_held_out_material():
    for name in ("residual_geometry_pilot.json", "residual_geometry.json",
                 "paths.local.example.json"):
        config = json.loads((REPO_ROOT / "configs" / name).read_text())
        assert_config_is_sealed_free(config, name)


def test_development_entry_points_name_no_held_out_material():
    forbidden = ("rq3_sealed", "sealed_final_manifest", "sealed_final",
                 "rq3_prompt_splits")
    for script in sorted((REPO_ROOT / "experiments").rglob("*.py")):
        # The completed final report reads the released adapter. Other scripts,
        # including new files in final_evaluation/, remain covered by this scan.
        if script.relative_to(REPO_ROOT).as_posix() == (
                "experiments/final_evaluation/build_final_thesis_results.py"):
            continue
        text = script.read_text(encoding="utf-8").lower()
        for needle in forbidden:
            assert needle not in text, f"{script.name} references '{needle}'"


@pytest.mark.parametrize("relative", [
    "src/routegate/unexpected_reader.py",
    "src/routegate/finaljudging_helper.py",
    "experiments/final_evaluation/unexpected_reader.py",
    "experiments/outcome_selector_v1/unexpected_reader.py",
])
def test_new_readers_do_not_inherit_the_final_adapter_exception(tmp_path, monkeypatch, relative):
    """A new reader is rejected even beside the authorised final adapter."""
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('INPUT = "rq3_sealed_final_manifest.csv"\n')
    monkeypatch.setitem(globals(), "REPO_ROOT", tmp_path)
    guard = (test_no_development_module_encodes_a_held_out_path
             if relative.startswith("src/")
             else test_development_entry_points_name_no_held_out_material)
    with pytest.raises(AssertionError, match="rq3_sealed"):
        guard()


# --------------------------------------------------------- loader behaviour
def test_a_missing_local_path_file_is_a_clear_refusal(tmp_path):
    with pytest.raises(ValidationError, match="paths.local.example.json"):
        load_local_paths(tmp_path / "absent.json")


def test_the_manifest_loads_and_is_the_default(input_manifest):
    assert DEFAULT_MANIFEST == MANIFEST_PATH
    assert load_input_manifest()["dataset"]["filename"] == "rq3_dataset_final.csv"


def test_the_manifest_records_that_no_activation_is_copied(input_manifest):
    assert input_manifest["activation"]["copied_into_this_repository"] is False


def test_no_activation_array_was_copied_into_the_repository():
    """No tensor blob anywhere in the repository proper.

    ``.venv`` is excluded: it is a gitignored environment, and installed
    packages ship their own ``.npy`` test fixtures. The claim under test is
    about this repository's tracked content, not about scikit-learn's.
    """
    def tracked(pattern):
        return [p for p in REPO_ROOT.rglob(pattern)
                if ".venv" not in p.parts and "outputs" not in p.parts]

    assert not tracked("*.npy"), "activation data must not be copied here"
    assert not tracked("*.pt")
    assert not tracked("*.bin")
    assert not tracked("*.safetensors")


# ------------------------------------------------- non-finite summary guard
def test_a_non_finite_value_cannot_win_a_maximum():
    with pytest.raises(ValidationError, match="non-finite"):
        safe_argmax([0.1, float("nan"), 0.2], "test values")
    assert safe_argmax([0.1, 0.9, 0.2]) == 1
