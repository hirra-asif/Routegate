"""Row alignment between the corpus, the extraction manifest and the array.

Covers requirement 4 of the test plan: manifest line *i* is array row *i*, the
manifest names exactly the analysis-eligible rows, and every joined field
agrees with the authoritative corpus.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_real_inputs
from routegate.data import describe_state
from routegate.validation import sha256_text

pytestmark = needs_real_inputs


def _dataset_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_row_index_is_the_array_position(inputs):
    indices = [int(row["row_index"]) for row in inputs.manifest_rows]
    assert indices == list(range(len(inputs.manifest_rows)))


def test_row_ids_are_unique_and_sorted(inputs):
    ids = [row["row_id"] for row in inputs.manifest_rows]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_the_manifest_is_exactly_the_analysis_eligible_rows(inputs):
    dataset = _dataset_rows(inputs.dataset_path)
    eligible = sorted(row["row_id"] for row in dataset
                      if row["final_ebh"] in ("E", "B", "H"))
    assert sorted(row["row_id"] for row in inputs.manifest_rows) == eligible
    assert len(eligible) == 480


def test_every_joined_field_agrees_with_the_corpus(inputs):
    dataset = {row["row_id"]: row for row in _dataset_rows(inputs.dataset_path)}
    for row in inputs.manifest_rows:
        authoritative = dataset[row["row_id"]]
        assert authoritative["final_ebh"] == row["final_ebh"]
        assert authoritative["prompt_text"] == row["prompt_text"]
        assert authoritative["source_family"] == row["source_family"]


def test_prompt_hashes_recompute_from_their_own_text(inputs):
    for row in inputs.manifest_rows:
        assert sha256_text(row["prompt_text"]) == row["prompt_sha256"]
        assert (hashlib.sha256(row["prompt_text"].encode("utf-8")).hexdigest()
                == row["prompt_sha256"])


def test_the_analysed_selection_keeps_the_manifest_order(inputs):
    assert np.all(np.diff(inputs.eb_positions) > 0)
    for position, array_row in enumerate(inputs.eb_positions):
        row = inputs.manifest_rows[array_row]
        assert row["row_id"] == inputs.row_ids[position]
        assert row["source_family"] == inputs.source_family[position]
        assert row["prompt_text"] == inputs.prompt_text[position]


def test_label_encoding_is_e_one_and_b_zero(inputs):
    for position, array_row in enumerate(inputs.eb_positions):
        label = inputs.manifest_rows[array_row]["final_ebh"]
        assert int(inputs.y[position]) == (1 if label == "E" else 0)
    assert int((inputs.y == 1).sum()) == 89
    assert int((inputs.y == 0).sum()) == 178


def test_the_state_matrix_pulls_the_rows_it_claims_to(inputs):
    state = 45
    matrix = inputs.state_matrix(state)
    assert matrix.shape == (267, 2048)
    raw = np.asarray(inputs.activations)
    for position in (0, 1, 133, 266):
        expected = np.asarray(raw[inputs.eb_positions[position], state, :],
                              dtype=np.float64)
        np.testing.assert_array_equal(matrix[position], expected)


def test_the_state_matrix_is_contiguous_float64(inputs):
    matrix = inputs.state_matrix(24)
    assert matrix.dtype == np.float64
    assert matrix.flags["C_CONTIGUOUS"]


def test_every_input_gate_passed(inputs):
    assert inputs.checklist.all_passed
    assert inputs.checklist.failures == []


def test_state_numbering_is_explicit():
    assert describe_state(0) == "state 0 (embedding output at the final prompt token)"
    assert describe_state(1) == "state 1 (transformer-block output 1)"
    assert describe_state(48) == "state 48 (transformer-block output 48)"
    assert "layer 0" not in describe_state(0)


def test_an_out_of_range_state_is_refused(inputs):
    from routegate.validation import ValidationError
    with pytest.raises(ValidationError):
        inputs.state_matrix(49)
    with pytest.raises(ValidationError):
        inputs.state_matrix(-1)
