"""The frozen-input inventory says what is true, and agrees with the gates.

``manifests/frozen_inputs.json`` is a reference document, not a loader: nothing
reads it to decide whether an input is acceptable. That makes it exactly the
kind of file that can drift out of agreement with reality without anything
failing, so these tests pin it against the manifests that *do* gate the inputs
and against the two artifacts committed to this repository.

Everything here runs offline. No test in this module reads
``configs/paths.local.json``, resolves the machine-local data root, or touches
any external file -- the only files opened are three manifests and the two
committed pilot CSVs, all of which live in the repository.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "manifests" / "frozen_inputs.json"
DEVELOPMENT_INPUTS_PATH = REPO_ROOT / "manifests" / "development_inputs.json"
PILOT_REFERENCE_PATH = REPO_ROOT / "manifests" / "pilot_reference.json"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "pilot_reference"

SCHEMA = "routegate/frozen_inputs/1"
EXPECTED_INPUT_COUNT = 6
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

# The static-steering master has no producing script and no enforcing manifest
# of its own yet; it is pinned inside run_baseline_v1.py. Written out literally
# here so that a change to it fails this test rather than passing silently.
STATIC_MASTER_SHA256 = (
    "525e62e0a6fdac3200111deae146d699ea2db9b3459dab2e57b09c705b11b9e4")

# Of the six, exactly these three carry prompt or response text.
PROMPT_BEARING = {
    "development_corpus",
    "residual_extraction_manifest",
    "static_steering_ebh_master",
}

PILOT_COLUMNS = {
    "layerwise_eb_probe_metrics.csv": [
        "layer", "n", "n_E", "n_B", "roc_auc", "pr_auc", "balanced_accuracy",
        "accuracy", "fold_roc_auc_mean", "fold_roc_auc_sd", "fold_roc_auc_min",
        "fold_roc_auc_max", "mean_n_iter", "max_grad_norm", "all_folds_converged",
    ],
    "best_layer_oof_scores.csv": [
        "array_row_index", "row_id", "source_family", "final_ebh", "y_true",
        "fold", "oof_decision_function", "oof_score_p_E", "predicted_label",
        "correct", "prompt_chars",
    ],
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inventory():
    return _load(INVENTORY_PATH)


@pytest.fixture(scope="module")
def by_name(inventory):
    return {entry["logical_name"]: entry for entry in inventory["inputs"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ shape
def test_the_schema_is_the_expected_one(inventory):
    assert inventory["schema"] == SCHEMA


def test_there_are_exactly_six_unique_logical_inputs(inventory):
    names = [entry["logical_name"] for entry in inventory["inputs"]]
    assert len(names) == EXPECTED_INPUT_COUNT
    assert len(set(names)) == EXPECTED_INPUT_COUNT, f"duplicate logical name in {names}"


def test_the_inventory_declares_itself_a_reference_not_a_loader(inventory):
    """It must not be mistaken for a new gate; the real gates are named."""
    assert "role" in inventory
    assert "is_not" in inventory["role"]
    for entry in inventory["inputs"]:
        enforced = entry["hash_enforced_by"]
        assert isinstance(enforced, list) and enforced, entry["logical_name"]


def test_every_sha256_is_64_lowercase_hex(inventory):
    for entry in inventory["inputs"]:
        value = entry["sha256"]
        assert SHA256_RE.match(value), f"{entry['logical_name']}: {value!r}"


def test_every_recorded_size_is_a_positive_integer(inventory):
    for entry in inventory["inputs"]:
        assert isinstance(entry["bytes"], int) and entry["bytes"] > 0


# ------------------------------------------------------------------ paths
def test_all_origin_and_destination_paths_are_relative(inventory):
    for entry in inventory["inputs"]:
        for key in ("origin_relative_path", "local_destination"):
            value = entry[key]
            assert not value.startswith("/"), f"{entry['logical_name']}.{key} is absolute"
            assert not value.startswith("~"), f"{entry['logical_name']}.{key} is home-relative"
            assert ".." not in Path(value).parts, f"{entry['logical_name']}.{key} escapes its root"


def test_no_absolute_hpc_path_occurs_anywhere_in_the_manifest():
    """Not in a path, not in a comment, not in a note."""
    blob = INVENTORY_PATH.read_text(encoding="utf-8")
    for needle in ("/home/", "/myriadfs", "ucabh21", "/scratch/", "C:\\"):
        assert needle not in blob, f"the inventory leaks {needle!r}"


# ------------------------------------- agreement with the enforcing manifests
def test_the_first_three_hashes_match_development_inputs(by_name):
    """The corpus, extraction manifest and residual array are gated there."""
    gate = _load(DEVELOPMENT_INPUTS_PATH)
    pairs = [
        ("development_corpus", gate["dataset"]),
        ("residual_extraction_manifest", gate["extraction_manifest"]),
        ("residual_activations", gate["activation"]),
    ]
    for name, gated in pairs:
        assert by_name[name]["sha256"] == gated["sha256"], name
        assert by_name[name]["origin_relative_path"] == gated["canonical_source_path"], name


def test_the_two_pilot_hashes_match_the_pilot_reference_manifest(by_name):
    gate = _load(PILOT_REFERENCE_PATH)["reference_files"]
    assert by_name["pilot_layerwise_metrics"]["sha256"] == gate["layerwise_metrics"]["sha256"]
    assert (by_name["pilot_oof_scores"]["sha256"]
            == gate["fold_assignment_and_out_of_fold_scores"]["sha256"])


def test_the_static_master_hash_is_exactly_the_pinned_value(by_name):
    assert by_name["static_steering_ebh_master"]["sha256"] == STATIC_MASTER_SHA256


def test_the_static_master_records_where_its_hash_is_enforced(by_name):
    """It has no manifest gate yet; the inventory must say so, not imply one."""
    enforced = " ".join(by_name["static_steering_ebh_master"]["hash_enforced_by"])
    assert "run_baseline_v1.py" in enforced
    assert "step 6" in enforced


# --------------------------------------------- the committed pilot artifacts
@pytest.mark.parametrize("filename", sorted(PILOT_COLUMNS))
def test_the_committed_pilot_file_exists_and_matches_its_recorded_hash(filename, by_name):
    path = ARTIFACTS_DIR / filename
    assert path.is_file(), f"{path} is a committed artifact and must be present"
    logical = ("pilot_layerwise_metrics"
               if filename.startswith("layerwise") else "pilot_oof_scores")
    assert sha256_file(path) == by_name[logical]["sha256"], filename
    assert path.stat().st_size == by_name[logical]["bytes"], filename


@pytest.mark.parametrize("filename", sorted(PILOT_COLUMNS))
def test_the_committed_pilot_file_has_its_exact_expected_columns(filename):
    with open(ARTIFACTS_DIR / filename, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == PILOT_COLUMNS[filename]


@pytest.mark.parametrize("filename", sorted(PILOT_COLUMNS))
def test_the_committed_pilot_file_carries_no_prompt_or_response_column(filename):
    """These two are committed precisely because they carry no study material."""
    with open(ARTIFACTS_DIR / filename, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for column in header:
        assert column not in ("prompt_text", "response_text"), f"{filename}: {column}"
        assert not column.endswith("_response"), f"{filename}: {column}"


# ------------------------------------------------------- text classification
def test_exactly_three_of_the_six_are_marked_prompt_or_response_bearing(inventory):
    marked = {entry["logical_name"] for entry in inventory["inputs"]
              if entry["contains_prompt_or_response_text"]}
    assert marked == PROMPT_BEARING


def test_the_pilot_files_are_not_marked_prompt_bearing(by_name):
    """They are committed to git, so this flag is load-bearing."""
    for name in ("pilot_layerwise_metrics", "pilot_oof_scores"):
        assert by_name[name]["contains_prompt_or_response_text"] is False
        assert by_name[name]["committable_to_git"] is True


def test_no_prompt_bearing_input_is_marked_committable(by_name):
    for name in PROMPT_BEARING:
        assert by_name[name]["committable_to_git"] is False, name


# ------------------------------------------------------------- the exclusion
def test_donor_pairs_with_text_is_recorded_as_explicitly_not_an_input(inventory):
    excluded = inventory["explicitly_not_an_input"]
    paths = [entry["path"] for entry in excluded]
    assert "outputs/causal_repair/donor_pairs_with_text.csv" in paths
    entry = next(e for e in excluded
                 if e["path"] == "outputs/causal_repair/donor_pairs_with_text.csv")
    assert "not one of the six" in entry["reason"]


def test_donor_pairs_is_not_among_the_six_inputs(inventory):
    for entry in inventory["inputs"]:
        assert "donor_pairs" not in entry["origin_relative_path"]
        assert "donor_pairs" not in entry["local_destination"]


# --------------------------------------------------------------- no leakage
def test_the_inventory_names_no_held_out_material():
    blob = INVENTORY_PATH.read_text(encoding="utf-8").lower()
    for needle in ("rq3_sealed", "sealed_final_manifest", "rq3_prompt_splits"):
        assert needle not in blob
