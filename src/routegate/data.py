"""Loading the accepted development inputs, or refusing to load anything.

State numbering, used consistently throughout this package:

    state 0       the embedding output at the final prompt token
    states 1-48   the outputs of transformer blocks 1 to 48

"state 0" is never called "layer 0": it is not a transformer block, and on this
corpus it is constant across every prompt (the rendered chat prompt ends with
the same token for all of them), which makes it a built-in degenerate control
rather than a result.

The activation array is never copied into this repository. It is read in place,
memory-mapped, from the path recorded in the local path file, and only after its
SHA256 matches the value committed in ``manifests/development_inputs.json``.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .validation import (
    Checklist,
    ValidationError,
    assert_config_is_sealed_free,
    assert_hash,
    assert_not_sealed,
    assert_not_stale,
    sha256_file,
    sha256_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "development_inputs.json"
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.json"

# The pilot reference is the one frozen input that lives *inside* the
# repository. Both files are small, carry no prompt text, and are hash-verified
# before use, so they are committed as artifacts rather than resolved through
# the machine-local path file. Any copy still sitting in a private data root is
# a backup, not a runtime input.
PILOT_REFERENCE_DIR = REPO_ROOT / "artifacts" / "pilot_reference"

POSITIVE_LABEL = "E"      # over-refusal; encoded as y = 1
NEGATIVE_LABEL = "B"      # benign compliance; encoded as y = 0
WITHHELD_LABEL = "H"      # present in the corpus, not analysed at this stage

STATE_ZERO_MEANING = "embedding output at the final prompt token"
STATE_BLOCK_MEANING = "transformer-block output"

# Categorical covariates for the control analyses. These are the column names
# that genuinely exist in the accepted dataset, verified by a gate below. No
# label is inferred from prompt text, and no column is renamed on read.
SOURCE_COLUMN = "source_family"
TOPIC_COLUMN = "topic_standardized"
TASK_FRAME_COLUMN = "task_frame"
COVARIATE_COLUMNS: Dict[str, str] = {"source": SOURCE_COLUMN,
                                     "topic": TOPIC_COLUMN,
                                     "task_frame": TASK_FRAME_COLUMN}

# Column names that would carry a saved prompt token count. The extraction
# manifest records row_index, row_id, final_ebh, source_family, prompt_text and
# prompt_sha256 -- and no token count. Nothing here retokenises a prompt, so a
# prompt-length control can only be built from surface text statistics, and it
# is reported as such rather than as a token count.
TOKEN_COUNT_COLUMN_CANDIDATES = ("n_tokens", "num_tokens", "token_count",
                                 "prompt_tokens", "prompt_n_tokens",
                                 "prompt_token_count", "input_tokens",
                                 "n_prompt_tokens", "sequence_length",
                                 "prompt_length_tokens")


def token_count_columns(columns: Sequence[str]) -> List[str]:
    """Any genuinely saved token-count column among ``columns``.

    Used to decide whether a token-length control is available at all. An empty
    result means one is not, and the caller must not manufacture one.
    """
    lowered = {c.lower(): c for c in columns}
    return [lowered[c] for c in TOKEN_COUNT_COLUMN_CANDIDATES if c in lowered]


def describe_state(state: int) -> str:
    """Human-readable meaning of a state index."""
    if state == 0:
        return f"state 0 ({STATE_ZERO_MEANING})"
    return f"state {state} ({STATE_BLOCK_MEANING} {state})"


# ------------------------------------------------------------------- manifest
def load_input_manifest(path: Path | str = DEFAULT_MANIFEST) -> Dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_config_is_sealed_free(manifest, "development input manifest")
    return manifest


def pilot_reference_dir(directory: Path | str = PILOT_REFERENCE_DIR) -> Path:
    """The committed, repo-local pilot reference directory.

    Resolved from the repository, not from ``configs/paths.local.json``: these
    two files are committed artifacts, so the reproduction does not depend on a
    machine-local path being set correctly. Their hashes are still checked by
    the caller against ``manifests/pilot_reference.json`` -- being committed
    makes a file present, not trustworthy.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValidationError(
            f"{directory} does not exist. The pilot reference files are "
            f"committed artifacts; see artifacts/README.md.")
    return directory


def load_local_paths(path: Path | str = DEFAULT_PATHS_FILE) -> Dict[str, str]:
    """Resolve machine-local absolute input paths.

    This file is gitignored. It is the only place an absolute path to the
    historical workspace appears, and it is validated to contain no reference
    to the held-out split.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(
            f"{path} does not exist. Copy configs/paths.local.example.json to "
            f"configs/paths.local.json and fill in the absolute input paths.")
    paths = json.loads(path.read_text(encoding="utf-8"))
    assert_config_is_sealed_free(paths, "local path file")
    for key, value in paths.items():
        if key.startswith("_"):
            continue
        assert_not_stale(value)
    return paths


# --------------------------------------------------------------- the payload
@dataclass
class DevelopmentInputs:
    """The validated development corpus, restricted to the E/B contrast."""

    dataset_path: Path
    activation_path: Path
    extraction_manifest_path: Path

    dataset_sha256: str
    activation_sha256: str
    extraction_manifest_sha256: str

    manifest_rows: List[Dict[str, str]]          # all 480 eligible rows
    activations: np.ndarray                      # memory-mapped [480, 49, 2048]

    eb_positions: np.ndarray                     # indices into the 480 rows
    y: np.ndarray                                # 1 for E, 0 for B
    row_ids: List[str]
    source_family: List[str]
    prompt_text: List[str]
    topic: List[str]                             # dataset topic_standardized
    task_frame: List[str]                        # dataset task_frame

    n_states: int
    hidden_size: int
    dataset_columns: List[str] = field(default_factory=list)
    extraction_manifest_columns: List[str] = field(default_factory=list)
    checklist: Checklist = field(default_factory=lambda: Checklist("inputs"))

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_positive(self) -> int:
        return int((self.y == 1).sum())

    @property
    def n_negative(self) -> int:
        return int((self.y == 0).sum())

    @property
    def states(self) -> List[int]:
        return list(range(self.n_states))

    def state_matrix(self, state: int) -> np.ndarray:
        """The [n, hidden_size] design matrix for one state, E/B rows only.

        Returned contiguous and float64: the memory-mapped array is strided on
        the state axis, and a strided operand takes BLAS off its fast path.
        """
        if not 0 <= state < self.n_states:
            raise ValidationError(
                f"state {state} is outside 0..{self.n_states - 1}")
        block = self.activations[self.eb_positions, state, :]
        return np.ascontiguousarray(np.asarray(block, dtype=np.float64))

    def covariate(self, name: str) -> List[str]:
        """The analysed rows' values for one genuine categorical column.

        ``name`` is a role -- "source", "topic", "task_frame" -- and maps to the
        column of that role in the accepted dataset. Nothing is inferred from
        prompt text.
        """
        if name not in COVARIATE_COLUMNS:
            raise ValidationError(
                f"unknown covariate {name!r}; the roles carried by genuine "
                f"dataset columns are {sorted(COVARIATE_COLUMNS)}")
        return {"source": self.source_family, "topic": self.topic,
                "task_frame": self.task_frame}[name]

    @property
    def saved_token_count_columns(self) -> List[str]:
        """Saved token-count columns in the inputs. Empty on this extraction."""
        return (token_count_columns(self.extraction_manifest_columns)
                + token_count_columns(self.dataset_columns))

    def degenerate_states(self) -> List[int]:
        """States whose design matrix is constant across every analysed row."""
        found = []
        for state in self.states:
            if float(self.state_matrix(state).std(axis=0).max()) == 0.0:
                found.append(state)
        return found


# ----------------------------------------------------------------- the loader
def load_development_inputs(
    manifest: Optional[Dict[str, Any]] = None,
    paths: Optional[Dict[str, str]] = None,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    paths_path: Path | str = DEFAULT_PATHS_FILE,
    verify_hashes: bool = True,
) -> DevelopmentInputs:
    """Load the accepted inputs, or raise.

    Every gate below is mandatory. There is no permissive mode: ``verify_hashes``
    exists only so unit tests can exercise the structural gates against small
    synthetic fixtures, and it never relaxes a structural check.
    """
    manifest = manifest if manifest is not None else load_input_manifest(manifest_path)
    paths = paths if paths is not None else load_local_paths(paths_path)

    checks = Checklist("development_inputs")

    dataset_path = assert_not_stale(Path(paths["dataset"]))
    extraction_dir = Path(paths["extraction_dir"])
    activation_path = assert_not_stale(extraction_dir / manifest["activation"]["filename"])
    extraction_manifest_path = assert_not_stale(
        extraction_dir / manifest["extraction_manifest"]["filename"])

    for candidate in (dataset_path, activation_path, extraction_manifest_path):
        assert_not_sealed(candidate, "development input path")

    expected_dataset = manifest["dataset"]
    expected_activation = manifest["activation"]
    expected_manifest = manifest["extraction_manifest"]

    # --- gate 1: the dataset is the accepted file, by name and by content ----
    checks.require(
        "dataset_filename_is_the_accepted_one",
        dataset_path.name == expected_dataset["filename"],
        f"{dataset_path.name} (expected {expected_dataset['filename']})")
    for path in (dataset_path, activation_path, extraction_manifest_path):
        checks.require(f"{path.name}_exists", path.exists(), str(path))

    dataset_sha = (assert_hash(dataset_path, expected_dataset["sha256"], "dataset")
                   if verify_hashes else sha256_file(dataset_path))
    checks.record("dataset_sha256", not verify_hashes or True, dataset_sha)

    manifest_sha = (assert_hash(extraction_manifest_path,
                                expected_manifest["sha256"], "extraction manifest")
                    if verify_hashes else sha256_file(extraction_manifest_path))
    checks.record("extraction_manifest_sha256", True, manifest_sha)

    activation_sha = (assert_hash(activation_path, expected_activation["sha256"],
                                  "activation array")
                      if verify_hashes else sha256_file(activation_path))
    checks.record("activation_sha256", True, activation_sha)

    # --- gate 2: dataset class counts ---------------------------------------
    dataset_rows = _read_csv(dataset_path)
    counts = Counter(row["final_ebh"] for row in dataset_rows)
    expected_counts = expected_dataset["counts"]
    checks.require(
        "dataset_total_rows",
        len(dataset_rows) == expected_counts["total"],
        f"{len(dataset_rows)} (expected {expected_counts['total']})")
    for label in (POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL,
                  expected_dataset["excluded_label"]):
        checks.require(
            f"dataset_count_{label}",
            counts.get(label, 0) == expected_counts[label],
            f"{counts.get(label, 0)} (expected {expected_counts[label]})")
    eligible = [r for r in dataset_rows
                if r["final_ebh"] in (POSITIVE_LABEL, NEGATIVE_LABEL, WITHHELD_LABEL)]
    checks.require(
        "dataset_analysis_eligible_rows",
        len(eligible) == expected_counts["analysis_eligible"],
        f"{len(eligible)} (expected {expected_counts['analysis_eligible']})")

    # --- gate 3: the extraction manifest is exactly the eligible rows -------
    manifest_rows = _read_csv(extraction_manifest_path)
    checks.require(
        "extraction_manifest_row_count",
        len(manifest_rows) == expected_counts["analysis_eligible"],
        f"{len(manifest_rows)} (expected {expected_counts['analysis_eligible']})")
    manifest_ids = [r["row_id"] for r in manifest_rows]
    checks.require("extraction_manifest_row_ids_unique",
                   len(set(manifest_ids)) == len(manifest_ids),
                   f"{len(set(manifest_ids))} distinct row_id values")
    checks.require("extraction_manifest_sorted_by_row_id",
                   manifest_ids == sorted(manifest_ids),
                   "manifest is in ascending row_id order")
    checks.require(
        "extraction_manifest_is_exactly_the_eligible_set",
        sorted(manifest_ids) == sorted(r["row_id"] for r in eligible),
        f"{len(manifest_ids)} row_id values, matched against the "
        f"{len(eligible)} analysis-eligible dataset rows")
    checks.require(
        "row_index_is_the_array_position",
        [int(r["row_index"]) for r in manifest_rows] == list(range(len(manifest_rows))),
        "row_index runs 0..n-1 in file order, so manifest line i is array row i")

    excluded_label = expected_dataset["excluded_label"]
    checks.require(
        "no_excluded_row_in_the_extraction_manifest",
        all(r["final_ebh"] != excluded_label for r in manifest_rows),
        f"no row carries {excluded_label}")

    # --- gate 4: per-row alignment against the dataset -----------------------
    by_row_id = {r["row_id"]: r for r in dataset_rows}
    label_bad = [r["row_id"] for r in manifest_rows
                 if by_row_id[r["row_id"]]["final_ebh"] != r["final_ebh"]]
    text_bad = [r["row_id"] for r in manifest_rows
                if by_row_id[r["row_id"]]["prompt_text"] != r["prompt_text"]]
    source_bad = [r["row_id"] for r in manifest_rows
                  if by_row_id[r["row_id"]]["source_family"] != r["source_family"]]
    hash_bad = [r["row_id"] for r in manifest_rows
                if sha256_text(r["prompt_text"]) != r["prompt_sha256"]]
    checks.require("labels_match_the_dataset", not label_bad,
                   f"{len(manifest_rows)} rows agree on final_ebh")
    checks.require("prompt_text_matches_the_dataset", not text_bad,
                   f"{len(manifest_rows)} rows agree on prompt_text")
    checks.require("source_family_matches_the_dataset", not source_bad,
                   f"{len(manifest_rows)} rows agree on source_family")
    checks.require("prompt_hashes_recompute", not hash_bad,
                   "every prompt_sha256 recomputes from its own prompt_text")

    # --- gate 4b: the covariate columns exist, and are populated -------------
    dataset_columns = list(dataset_rows[0])
    manifest_columns = list(manifest_rows[0])
    for role, column in COVARIATE_COLUMNS.items():
        checks.require(
            f"dataset_has_the_{role}_column",
            column in dataset_columns,
            f"{column!r} is present in the accepted dataset")
    analysed_labels = (POSITIVE_LABEL, NEGATIVE_LABEL)
    for role, column in COVARIATE_COLUMNS.items():
        blank = [r["row_id"] for r in dataset_rows
                 if r["final_ebh"] in analysed_labels and not r[column].strip()]
        distinct = sorted({r[column] for r in dataset_rows
                           if r["final_ebh"] in analysed_labels})
        checks.require(
            f"{role}_column_is_populated_for_every_analysed_row",
            not blank,
            f"{column}: {len(distinct)} distinct value(s), no blank")

    # Requirement, recorded rather than assumed: a token-length control may use
    # a token count only if one was genuinely saved. Nothing here retokenises.
    saved_token_columns = (token_count_columns(manifest_columns)
                           + token_count_columns(dataset_columns))
    checks.record(
        "saved_prompt_token_count_columns", True,
        f"{saved_token_columns or 'none'} — extraction manifest columns are "
        f"{manifest_columns}; with no saved token count, a prompt-length "
        f"control may use surface text statistics only and must be reported "
        f"as such")

    # --- gate 5: the array itself -------------------------------------------
    activations = np.load(activation_path, mmap_mode="r")
    expected_shape = tuple(expected_activation["shape"])
    checks.require("activation_shape", tuple(activations.shape) == expected_shape,
                   f"{tuple(activations.shape)} (expected {expected_shape})")
    checks.require("activation_dtype",
                   activations.dtype.name == expected_activation["dtype"],
                   f"{activations.dtype.name} (expected {expected_activation['dtype']})")
    checks.require("activation_rows_match_the_manifest",
                   activations.shape[0] == len(manifest_rows),
                   f"{activations.shape[0]} array rows, {len(manifest_rows)} manifest rows")

    # --- gate 6: the analysed contrast --------------------------------------
    eb_positions = np.array(
        [i for i, r in enumerate(manifest_rows)
         if r["final_ebh"] in (POSITIVE_LABEL, NEGATIVE_LABEL)], dtype=int)
    y = np.array([1 if manifest_rows[i]["final_ebh"] == POSITIVE_LABEL else 0
                  for i in eb_positions], dtype=int)
    checks.require(
        "eb_class_counts",
        int((y == 1).sum()) == expected_counts[POSITIVE_LABEL]
        and int((y == 0).sum()) == expected_counts[NEGATIVE_LABEL],
        f"E={int((y == 1).sum())} B={int((y == 0).sum())}")
    checks.require(
        "withheld_class_is_not_analysed",
        all(manifest_rows[i]["final_ebh"] != WITHHELD_LABEL for i in eb_positions),
        f"{expected_counts[WITHHELD_LABEL]} {WITHHELD_LABEL} rows are present in "
        f"the corpus and excluded from this contrast")

    inputs = DevelopmentInputs(
        dataset_path=dataset_path,
        activation_path=activation_path,
        extraction_manifest_path=extraction_manifest_path,
        dataset_sha256=dataset_sha,
        activation_sha256=activation_sha,
        extraction_manifest_sha256=manifest_sha,
        manifest_rows=manifest_rows,
        activations=activations,
        eb_positions=eb_positions,
        y=y,
        row_ids=[manifest_rows[i]["row_id"] for i in eb_positions],
        source_family=[manifest_rows[i]["source_family"] for i in eb_positions],
        prompt_text=[manifest_rows[i]["prompt_text"] for i in eb_positions],
        topic=[by_row_id[manifest_rows[i]["row_id"]][TOPIC_COLUMN]
               for i in eb_positions],
        task_frame=[by_row_id[manifest_rows[i]["row_id"]][TASK_FRAME_COLUMN]
                    for i in eb_positions],
        n_states=int(activations.shape[1]),
        hidden_size=int(activations.shape[2]),
        dataset_columns=dataset_columns,
        extraction_manifest_columns=manifest_columns,
        checklist=checks,
    )

    # --- gate 7: finiteness over the analysed block --------------------------
    block = np.asarray(activations[eb_positions], dtype=np.float64)
    checks.require("analysed_activations_are_finite",
                   bool(np.isfinite(block).all()),
                   f"{block.shape} block, all finite")
    del block

    return inputs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
