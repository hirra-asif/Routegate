"""The cross-fitted, H-orthogonalised direction construction, on numbers by hand.

Everything here runs on synthetic arrays a few units wide. No GPU, no model, no
download, and -- apart from two tests that read a CSV already committed to this
repository -- no real data at all: the arithmetic under test is small enough to
check against values worked out on paper, which is the only way to be sure the
projection is a projection and the sign is the sign.

What is being pinned:

* ``false_refusal`` is E − B and ``true_refusal`` is H − B, exactly;
* the projection removes the component along ``true_refusal`` and nothing else;
* the repair direction points *away* from what is left, and is unit length;
* a fold's direction is a function of that fold's training rows only, so
  perturbing a held-out row cannot move it;
* the inherited E/B folds survive the addition of H unchanged;
* the H assignment is a pure function of the row identities;
* a degenerate or non-finite site fails loudly rather than quietly returning
  something that would be applied as though it were a direction.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest

from routegate.ebh import (
    EBHError,
    EBHFrame,
    assign_withheld_folds,
    build_ebh_fold_assignment,
    load_frozen_eb_folds,
)
from routegate.steering import (
    ARRAY_NAMES,
    BANK_ARTIFACTS,
    CAUSAL_SCAN_STATES,
    FROZEN_SCAN_STRENGTH,
    MANIFEST_FILENAME,
    N_STORED_STATES,
    STATUS_OK,
    STATUS_ZERO_ORTHOGONAL,
    STATUS_ZERO_TRUE_REFUSAL,
    SteeringError,
    build_direction_bank,
    class_means,
    direction_sha256,
    load_direction_bank,
    population_vectors,
    write_direction_bank,
)
from routegate.validation import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_EB_FOLDS = (REPO_ROOT / "outputs" / "residual_geometry" / "corrected"
                 / "cv_assignments.csv")


# --------------------------------------------------------------- 1. the maths
def test_false_refusal_is_exactly_e_minus_b():
    vectors = population_vectors(e_mean=[3.0, 5.0, 9.0],
                                 b_mean=[1.0, 2.0, 4.0],
                                 h_mean=[1.0, 2.0, 6.0])
    assert np.allclose(vectors.false_refusal, [2.0, 3.0, 5.0])


def test_true_refusal_is_exactly_h_minus_b():
    vectors = population_vectors(e_mean=[3.0, 5.0, 9.0],
                                 b_mean=[1.0, 2.0, 4.0],
                                 h_mean=[1.0, 2.0, 6.0])
    assert np.allclose(vectors.true_refusal, [0.0, 0.0, 2.0])


def test_the_projection_removes_the_component_along_true_refusal():
    # false_refusal = (2, 3, 5); true_refusal = (0, 0, 2).
    # coefficient = (2*0 + 3*0 + 5*2) / 4 = 2.5, projection = (0, 0, 5),
    # orthogonal = (2, 3, 0).
    vectors = population_vectors(e_mean=[3.0, 5.0, 9.0],
                                 b_mean=[1.0, 2.0, 4.0],
                                 h_mean=[1.0, 2.0, 6.0])
    assert vectors.projection_coefficient == pytest.approx(2.5)
    assert np.allclose(vectors.orthogonal_false_refusal, [2.0, 3.0, 0.0])


def test_the_orthogonalised_vector_is_perpendicular_to_the_true_refusal_vector():
    rng = np.random.default_rng(20260807)
    for _ in range(20):
        b = rng.normal(size=32)
        vectors = population_vectors(e_mean=b + rng.normal(size=32),
                                     b_mean=b,
                                     h_mean=b + rng.normal(size=32))
        assert vectors.status == STATUS_OK
        assert abs(vectors.orthogonality_residual_cosine) < 1e-12
        assert float(np.dot(vectors.orthogonal_false_refusal,
                            vectors.true_refusal)) == pytest.approx(0.0, abs=1e-9)


def test_the_repair_direction_is_unit_length_and_points_away_from_the_e_signature():
    vectors = population_vectors(e_mean=[3.0, 5.0, 9.0],
                                 b_mean=[1.0, 2.0, 4.0],
                                 h_mean=[1.0, 2.0, 6.0])
    # orthogonal is (2, 3, 0), norm sqrt(13); repair is its negative unit vector.
    assert vectors.norm_repair_direction == pytest.approx(1.0)
    assert np.allclose(vectors.repair_direction,
                       -np.array([2.0, 3.0, 0.0]) / np.sqrt(13.0))
    # The sign, stated as an identity rather than as a direction of inequality:
    # moving along repair moves against the whole orthogonalised E signature.
    assert float(np.dot(vectors.repair_direction, vectors.false_refusal)) == \
        pytest.approx(-vectors.norm_orthogonal_false_refusal)
    assert float(np.dot(vectors.repair_direction,
                        vectors.orthogonal_false_refusal)) < 0.0


def test_the_repair_direction_is_orthogonal_to_the_true_refusal_direction():
    vectors = population_vectors(e_mean=[3.0, 5.0, 9.0],
                                 b_mean=[1.0, 2.0, 4.0],
                                 h_mean=[1.0, 2.0, 6.0])
    assert float(np.dot(vectors.repair_direction,
                        vectors.true_refusal)) == pytest.approx(0.0, abs=1e-12)


def test_the_cosine_between_the_two_population_vectors_is_recorded():
    vectors = population_vectors(e_mean=[3.0, 5.0, 9.0],
                                 b_mean=[1.0, 2.0, 4.0],
                                 h_mean=[1.0, 2.0, 6.0])
    expected = 10.0 / (np.sqrt(38.0) * 2.0)
    assert vectors.cosine_false_true_refusal == pytest.approx(expected)


# ----------------------------------------------------- 2. loud failure modes
def test_a_zero_true_refusal_vector_is_reported_not_divided_by():
    vectors = population_vectors(e_mean=[2.0, 0.0], b_mean=[1.0, 0.0],
                                 h_mean=[1.0, 0.0])
    assert vectors.status == STATUS_ZERO_TRUE_REFUSAL
    assert np.allclose(vectors.repair_direction, 0.0)


def test_a_false_refusal_vector_lying_entirely_along_h_leaves_nothing_to_steer():
    vectors = population_vectors(e_mean=[3.0, 0.0], b_mean=[1.0, 0.0],
                                 h_mean=[2.0, 0.0])
    assert vectors.status == STATUS_ZERO_ORTHOGONAL
    assert np.allclose(vectors.repair_direction, 0.0)


def test_identical_class_means_are_degenerate_in_both_ways():
    vectors = population_vectors(e_mean=[1.0, 1.0], b_mean=[1.0, 1.0],
                                 h_mean=[1.0, 1.0])
    assert vectors.status == STATUS_ZERO_TRUE_REFUSAL


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_class_mean_is_refused(bad):
    with pytest.raises(SteeringError, match="non-finite"):
        population_vectors(e_mean=[1.0, bad], b_mean=[0.0, 0.0],
                           h_mean=[0.0, 1.0])


def test_class_means_disagreeing_on_shape_are_refused():
    with pytest.raises(SteeringError, match="disagree on shape"):
        population_vectors(e_mean=[1.0, 2.0], b_mean=[0.0], h_mean=[0.0, 1.0])


def test_a_class_mean_over_zero_rows_is_refused():
    activations = np.zeros((4, 3, 2), dtype=np.float32)
    with pytest.raises(SteeringError, match="empty"):
        class_means(activations, [])


def test_a_non_finite_activation_is_refused_by_the_mean():
    activations = np.zeros((4, 3, 2), dtype=np.float32)
    activations[1, 0, 0] = np.nan
    with pytest.raises(SteeringError, match="non-finite"):
        class_means(activations, [0, 1])


def test_class_means_do_not_depend_on_the_order_the_rows_were_named_in():
    rng = np.random.default_rng(7)
    activations = rng.normal(size=(9, 4, 5)).astype(np.float32)
    first = class_means(activations, [5, 1, 8, 3])
    second = class_means(activations, [8, 3, 1, 5])
    assert np.array_equal(first, second)


# --------------------------------------------------------- 3. direction hashes
def test_a_direction_hash_is_deterministic_and_distinguishing():
    vector = np.array([0.5, -0.25, 0.125])
    assert direction_sha256(vector) == direction_sha256(vector.copy())
    assert direction_sha256(vector) == direction_sha256(list(vector))
    assert direction_sha256(vector) != direction_sha256(-vector)
    assert len(direction_sha256(vector)) == 64


# ------------------------------------------------------------ 4. the fold rule
def test_the_withheld_assignment_does_not_depend_on_the_input_order():
    rows = [(f"h:{i:03d}", "family_a" if i % 3 else "family_b") for i in range(37)]
    forward = assign_withheld_folds(rows, n_splits=5)
    backward = assign_withheld_folds(list(reversed(rows)), n_splits=5)
    shuffled = list(rows)
    random.Random(20260807).shuffle(shuffled)
    assert forward == backward == assign_withheld_folds(shuffled, n_splits=5)


def test_every_withheld_row_is_assigned_exactly_once_and_evenly():
    rows = [(f"h:{i:03d}", f"family_{i % 3}") for i in range(213)]
    assignment = assign_withheld_folds(rows, n_splits=5)
    assert len(assignment) == 213
    sizes = [sum(1 for fold in assignment.values() if fold == k) for k in range(5)]
    assert sum(sizes) == 213
    assert max(sizes) - min(sizes) <= 1


def test_each_withheld_source_family_is_spread_across_the_folds():
    rows = ([(f"a:{i:03d}", "family_a") for i in range(32)]
            + [(f"b:{i:03d}", "family_b") for i in range(62)]
            + [(f"c:{i:03d}", "family_c") for i in range(119)])
    assignment = assign_withheld_folds(rows, n_splits=5)
    for family in ("a", "b", "c"):
        sizes = [sum(1 for rid, fold in assignment.items()
                     if rid.startswith(family) and fold == k) for k in range(5)]
        assert max(sizes) - min(sizes) <= 1


def test_a_repeated_withheld_row_is_refused():
    with pytest.raises(EBHError, match="twice"):
        assign_withheld_folds([("h:1", "f"), ("h:1", "f")], n_splits=5)


# ------------------------------------------------- 5. the frame and its folds
def synthetic_frame(n_e=10, n_b=20, n_h=25, n_states=4, hidden=6, seed=20260807):
    """A small three-class frame with a planted, H-entangled E signature."""
    rng = np.random.default_rng(seed)
    labels = ["E"] * n_e + ["B"] * n_b + ["H"] * n_h
    row_ids = ([f"e_src:{i:03d}" for i in range(n_e)]
               + [f"b_src:{i:03d}" for i in range(n_b)]
               + [f"h_src:{i:03d}" for i in range(n_h)])
    families = (["fam_e"] * n_e + ["fam_b"] * n_b
                + [f"fam_h{i % 3}" for i in range(n_h)])
    n = n_e + n_b + n_h

    activations = rng.normal(size=(n, n_states, hidden)).astype(np.float32)
    shared = rng.normal(size=(n_states, hidden)).astype(np.float32)
    only_e = rng.normal(size=(n_states, hidden)).astype(np.float32)
    activations[:n_e] += 2.0 * shared + 3.0 * only_e          # E
    activations[n_e + n_b:] += 2.0 * shared                   # H

    return EBHFrame(
        row_ids=row_ids, labels=labels, source_family=families,
        topic=["TOPIC"] * n, task_frame=["FRAME"] * n,
        prompt_text=[f"prompt {i}" for i in range(n)],
        original_response=[f"response {i}" for i in range(n)],
        activations=activations, n_states=n_states, hidden_size=hidden,
        dataset_path=Path("synthetic.csv"), activation_path=Path("synthetic.npy"),
        extraction_manifest_path=Path("synthetic_manifest.csv"),
        dataset_sha256="dataset-sha", activation_sha256="activation-sha",
        extraction_manifest_sha256="manifest-sha")


def write_frozen_eb_folds(path: Path, frame: EBHFrame, n_splits=5) -> Path:
    """A stand-in for the residual-geometry run's cv_assignments.csv."""
    columns = ["repeat", "seed", "position", "row_id", "y_true", "final_ebh",
               "fold", "split", "array_row_index", "source_family"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for repeat in range(2):
            position = 0
            for index, (rid, label) in enumerate(zip(frame.row_ids, frame.labels)):
                if label not in ("E", "B"):
                    continue
                writer.writerow({
                    "repeat": repeat, "seed": 20260805 + repeat,
                    "position": position, "row_id": rid,
                    "y_true": 1 if label == "E" else 0, "final_ebh": label,
                    # repeat 1 deliberately differs, so a test that reads the
                    # wrong repeat cannot accidentally pass.
                    "fold": (index + repeat) % n_splits,
                    "split": "held_out", "array_row_index": index,
                    "source_family": frame.source_family[index]})
                position += 1
    return path


@pytest.fixture
def frame():
    return synthetic_frame()


@pytest.fixture
def assignment(frame, tmp_path):
    folds_path = write_frozen_eb_folds(tmp_path / "cv_assignments.csv", frame)
    return build_ebh_fold_assignment(frame, folds_path, repeat=0, n_splits=5)


def test_the_inherited_eb_folds_are_carried_through_unchanged(frame, assignment):
    built, frozen_eb = assignment
    assert frozen_eb
    for row_id, fold in frozen_eb.items():
        assert built.fold_of(row_id) == fold
    built.audit(frame, frozen_eb=frozen_eb)          # raises on any violation


def test_the_named_repeat_is_the_one_that_is_inherited(frame, tmp_path):
    folds_path = write_frozen_eb_folds(tmp_path / "cv_assignments.csv", frame)
    first, _ = build_ebh_fold_assignment(frame, folds_path, repeat=0)
    second, _ = build_ebh_fold_assignment(frame, folds_path, repeat=1)
    assert first.fold_of_row != second.fold_of_row


def test_a_repeat_the_file_does_not_carry_is_refused(frame, tmp_path):
    folds_path = write_frozen_eb_folds(tmp_path / "cv_assignments.csv", frame)
    with pytest.raises(EBHError, match="no repeat 9"):
        build_ebh_fold_assignment(frame, folds_path, repeat=9)


def test_every_row_of_every_class_is_held_out_exactly_once(frame, assignment):
    built, _ = assignment
    seen = {}
    for fold in range(built.n_splits):
        for row_id in built.held_out_row_ids(fold):
            seen[row_id] = seen.get(row_id, 0) + 1
    assert set(seen) == set(frame.row_ids)
    assert set(seen.values()) == {1}


def test_training_and_held_out_rows_never_overlap(frame, assignment):
    built, _ = assignment
    for fold in range(built.n_splits):
        assert not (set(built.training_row_ids(fold))
                    & set(built.held_out_row_ids(fold)))


def test_the_assignment_is_audited_against_the_frame(frame, assignment):
    built, frozen_eb = assignment
    checks = built.audit(frame, frozen_eb=frozen_eb)
    assert checks.all_passed
    assert any(entry["check"] == "inherited_E_and_B_folds_are_unchanged"
               for entry in checks.entries)


def test_an_audit_fails_when_an_inherited_fold_was_moved(frame, assignment):
    built, frozen_eb = assignment
    moved = dict(frozen_eb)
    victim = sorted(moved)[0]
    moved[victim] = (moved[victim] + 1) % built.n_splits
    with pytest.raises(ValidationError,
                       match="inherited_E_and_B_folds_are_unchanged"):
        built.audit(frame, frozen_eb=moved)


def test_a_frozen_file_carrying_a_withheld_row_is_refused(frame, tmp_path):
    path = tmp_path / "cv_assignments.csv"
    write_frozen_eb_folds(path, frame)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        handle.write("0,20260805,999,h_src:000,0,H,0,held_out,999,fam_h0\n")
    with pytest.raises(EBHError, match="covers"):
        load_frozen_eb_folds(path, frame.row_ids_of("E") + frame.row_ids_of("B"))


# ------------------------------------------------------- 6. cross-fitted bank
def training_positions_from(frame, built):
    return [built.training_positions(frame, fold)
            for fold in range(built.n_splits)]


def test_a_direction_is_built_from_training_rows_only(frame, assignment):
    """Perturbing a held-out row must not move that fold's direction at all."""
    built, _ = assignment
    causal = tuple(range(1, frame.n_states))
    first = build_direction_bank(frame.activations,
                                 training_positions_from(frame, built),
                                 n_states=frame.n_states, causal_states=causal)

    held_out = built.held_out_row_ids(0)
    position = frame.position_of[held_out[0]]
    perturbed = np.array(frame.activations, copy=True)
    perturbed[position] += 25.0
    second = build_direction_bank(perturbed, training_positions_from(frame, built),
                                  n_states=frame.n_states, causal_states=causal)

    assert np.array_equal(first.arrays["repair_direction"][0],
                          second.arrays["repair_direction"][0])
    # ... and the folds that *were* allowed to see that row did move.
    assert not np.array_equal(first.arrays["repair_direction"][1],
                              second.arrays["repair_direction"][1])


def test_the_bank_covers_every_fold_and_every_stored_state(frame, assignment):
    built, _ = assignment
    causal = tuple(range(1, frame.n_states))
    bank = build_direction_bank(frame.activations,
                                training_positions_from(frame, built),
                                n_states=frame.n_states, causal_states=causal)
    assert bank.n_folds == built.n_splits
    for name in ARRAY_NAMES:
        assert bank.arrays[name].shape == (built.n_splits, frame.n_states,
                                           frame.hidden_size)
    sites = {(row["fold"], row["state"]) for row in bank.rows}
    assert sites == {(f, s) for f in range(built.n_splits)
                     for s in range(frame.n_states)}


def test_the_state_zero_row_is_named_as_an_embedding_not_as_a_block(frame,
                                                                    assignment):
    built, _ = assignment
    bank = build_direction_bank(frame.activations,
                                training_positions_from(frame, built),
                                n_states=frame.n_states,
                                causal_states=tuple(range(1, frame.n_states)))
    embedding = bank.row(0, 0)
    assert embedding["site"] == "embedding_output"
    assert embedding["transformer_block_index"] == ""
    assert embedding["in_causal_scan"] is False
    for state in range(1, frame.n_states):
        row = bank.row(0, state)
        assert row["site"] == "block_output"
        assert row["transformer_block_index"] == state - 1
        assert row["in_causal_scan"] is True


def test_a_degenerate_transformer_block_state_stops_the_build(frame, assignment):
    built, _ = assignment
    flat = np.array(frame.activations, copy=True)
    flat[:, 2, :] = 1.0                       # state 2 identical for every row
    with pytest.raises(SteeringError, match="degenerate"):
        build_direction_bank(flat, training_positions_from(frame, built),
                             n_states=frame.n_states,
                             causal_states=tuple(range(1, frame.n_states)))


def test_a_degenerate_state_zero_is_recorded_rather_than_fatal(frame, assignment):
    built, _ = assignment
    flat = np.array(frame.activations, copy=True)
    flat[:, 0, :] = 1.0                       # the real corpus behaves this way
    bank = build_direction_bank(flat, training_positions_from(frame, built),
                                n_states=frame.n_states,
                                causal_states=tuple(range(1, frame.n_states)))
    assert bank.row(0, 0)["status"] == STATUS_ZERO_TRUE_REFUSAL
    assert np.allclose(bank.repair_direction(0, 0), 0.0)
    for state in range(1, frame.n_states):
        assert bank.row(0, state)["status"] == STATUS_OK


def test_state_zero_may_not_be_declared_part_of_the_causal_scan(frame, assignment):
    built, _ = assignment
    with pytest.raises(SteeringError, match="state 0"):
        build_direction_bank(frame.activations,
                             training_positions_from(frame, built),
                             n_states=frame.n_states,
                             causal_states=(0, 1))


def test_an_empty_training_class_is_refused(frame, assignment):
    built, _ = assignment
    positions = training_positions_from(frame, built)
    positions[0]["H"] = np.array([], dtype=int)
    with pytest.raises(SteeringError, match="empty class"):
        build_direction_bank(frame.activations, positions,
                             n_states=frame.n_states,
                             causal_states=tuple(range(1, frame.n_states)))


def test_a_missing_training_class_is_refused(frame, assignment):
    built, _ = assignment
    positions = training_positions_from(frame, built)
    del positions[1]["B"]
    with pytest.raises(SteeringError, match="no training rows declared"):
        build_direction_bank(frame.activations, positions,
                             n_states=frame.n_states,
                             causal_states=tuple(range(1, frame.n_states)))


def test_one_fold_is_not_cross_fitting(frame, assignment):
    built, _ = assignment
    with pytest.raises(SteeringError, match="at least two folds"):
        build_direction_bank(frame.activations,
                             training_positions_from(frame, built)[:1],
                             n_states=frame.n_states,
                             causal_states=tuple(range(1, frame.n_states)))


def test_the_same_inputs_produce_the_same_bank_bit_for_bit(frame, assignment):
    built, _ = assignment
    causal = tuple(range(1, frame.n_states))
    first = build_direction_bank(frame.activations,
                                 training_positions_from(frame, built),
                                 n_states=frame.n_states, causal_states=causal)
    second = build_direction_bank(frame.activations,
                                  training_positions_from(frame, built),
                                  n_states=frame.n_states, causal_states=causal)
    for name in ARRAY_NAMES:
        assert np.array_equal(first.arrays[name], second.arrays[name])
    assert [r["direction_sha256"] for r in first.rows] == \
        [r["direction_sha256"] for r in second.rows]


# ------------------------------------------------- 7. writing and reading back
@pytest.fixture
def written_bank(frame, assignment, tmp_path):
    built, _ = assignment
    causal = tuple(range(1, frame.n_states))
    arrays = build_direction_bank(frame.activations,
                                  training_positions_from(frame, built),
                                  n_states=frame.n_states, causal_states=causal)
    directory = tmp_path / "direction_bank"
    manifest = write_direction_bank(
        directory, arrays,
        fold_assignment_rows=built.as_rows(frame),
        training_row_ids={fold: {label: built.training_row_ids(fold, label)
                                 for label in ("E", "B", "H")}
                          for fold in range(built.n_splits)},
        held_out_row_ids={fold: {label: built.held_out_row_ids(fold, label)
                                 for label in ("E", "B", "H")}
                          for fold in range(built.n_splits)},
        counts={"rows": frame.n},
        input_hashes={"dataset": {"sha256": frame.dataset_sha256}},
        config={"strength": FROZEN_SCAN_STRENGTH,
                "causal_scan_states": list(causal)},
        created_utc="2026-08-07T00:00:00Z")
    return directory, manifest, arrays, built


def test_every_declared_artefact_is_written_and_hashed(written_bank):
    directory, manifest, _arrays, _built = written_bank
    for name in BANK_ARTIFACTS:
        assert (directory / name).exists()
        assert name in manifest["artifacts"]
    assert (directory / MANIFEST_FILENAME).exists()


def test_the_bank_reads_back_with_its_directions_intact(written_bank):
    directory, manifest, arrays, _built = written_bank
    bank = load_direction_bank(directory, manifest["manifest_sha256"])
    assert bank.manifest_sha256 == manifest["manifest_sha256"]
    for fold in range(bank.n_folds):
        for state in range(1, arrays.n_states):
            assert np.array_equal(bank.repair_direction(fold, state),
                                  arrays.arrays["repair_direction"][fold, state])
            assert bank.direction_sha256(fold, state) == \
                arrays.row(fold, state)["direction_sha256"]


def test_reading_a_bank_never_unpickles_anything(written_bank):
    directory, _manifest, _arrays, _built = written_bank
    with np.load(directory / "direction_bank.npz", allow_pickle=False) as handle:
        assert set(handle.files) == set(ARRAY_NAMES)


def test_a_bank_whose_hash_is_not_the_expected_one_is_refused(written_bank):
    directory, _manifest, _arrays, _built = written_bank
    with pytest.raises(SteeringError, match="not the bank this run"):
        load_direction_bank(directory, "0" * 64)


def test_a_bank_edited_after_freezing_is_refused(written_bank):
    directory, manifest, _arrays, _built = written_bank
    payload = json.loads((directory / MANIFEST_FILENAME).read_text())
    payload["counts"]["rows"] = 999
    (directory / MANIFEST_FILENAME).write_text(json.dumps(payload))
    with pytest.raises(SteeringError, match="modified since it was frozen"):
        load_direction_bank(directory)


def test_an_artefact_edited_after_freezing_is_refused(written_bank):
    directory, manifest, _arrays, _built = written_bank
    (directory / "counts.json").write_text('{"rows": 1}\n')
    with pytest.raises(SteeringError, match="hashes to"):
        load_direction_bank(directory, manifest["manifest_sha256"])


def test_state_zero_has_no_direction_to_hand_out(written_bank):
    directory, manifest, _arrays, _built = written_bank
    bank = load_direction_bank(directory, manifest["manifest_sha256"])
    with pytest.raises(SteeringError, match="embedding output"):
        bank.repair_direction(0, 0)


def test_a_held_out_row_may_use_its_own_folds_direction(written_bank):
    directory, manifest, _arrays, built = written_bank
    bank = load_direction_bank(directory, manifest["manifest_sha256"])
    for fold in range(bank.n_folds):
        for row_id in built.held_out_row_ids(fold):
            bank.assert_held_out(row_id, fold)      # raises if it would leak


def test_a_training_row_may_not_be_steered_with_its_own_folds_direction(
        written_bank):
    directory, manifest, _arrays, built = written_bank
    bank = load_direction_bank(directory, manifest["manifest_sha256"])
    training = built.training_row_ids(0)
    with pytest.raises(SteeringError, match="training row"):
        bank.assert_held_out(training[0], 0)


def test_a_row_that_is_not_in_the_assignment_is_refused(written_bank):
    directory, manifest, _arrays, _built = written_bank
    bank = load_direction_bank(directory, manifest["manifest_sha256"])
    with pytest.raises(SteeringError, match="not in the frozen fold assignment"):
        bank.fold_of("nobody:000")


# ------------------------------- 8. the real inherited assignment, as committed
@pytest.mark.skipif(not REAL_EB_FOLDS.exists(),
                    reason="the corrected residual-geometry run is not present")
def test_the_committed_eb_assignment_is_five_folds_over_the_two_analysed_classes():
    with open(REAL_EB_FOLDS, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["repeat"] == "0"]
    row_ids = [row["row_id"] for row in rows]
    folds, n_splits = load_frozen_eb_folds(REAL_EB_FOLDS, row_ids, repeat=0)
    assert n_splits == 5
    assert len(folds) == len(row_ids) == len(set(row_ids))
    assert {row["final_ebh"] for row in rows} == {"E", "B"}
    sizes = [sum(1 for fold in folds.values() if fold == k) for k in range(5)]
    assert sum(sizes) == len(row_ids)


@pytest.mark.skipif(not REAL_EB_FOLDS.exists(),
                    reason="the corrected residual-geometry run is not present")
def test_adding_the_withheld_class_would_not_disturb_the_committed_folds():
    """The composition step, exercised against the real E/B file.

    The withheld rows here are stand-ins -- the real ones need the corpus, which
    this test does not load -- but the property under test is about composition,
    not about which prompts they are: adding a disjoint set of rows must leave
    every inherited assignment exactly where it was.
    """
    with open(REAL_EB_FOLDS, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["repeat"] == "0"]
    inherited, _ = load_frozen_eb_folds(REAL_EB_FOLDS,
                                        [row["row_id"] for row in rows], repeat=0)
    withheld = assign_withheld_folds(
        [(f"withheld:{i:03d}", f"family_{i % 3}") for i in range(213)], 5)
    assert not set(inherited) & set(withheld)
    combined = {**inherited, **withheld}
    assert all(combined[row_id] == fold for row_id, fold in inherited.items())
    assert len(combined) == len(inherited) + len(withheld)


# ------------------------------------------------- 9. the declared scan extent
def test_the_causal_scan_is_the_forty_eight_transformer_blocks():
    assert CAUSAL_SCAN_STATES == tuple(range(1, 49))
    assert len(CAUSAL_SCAN_STATES) == 48
    assert 0 not in CAUSAL_SCAN_STATES
    assert N_STORED_STATES == 49


def test_the_scan_strength_is_the_frozen_pilot_value():
    assert FROZEN_SCAN_STRENGTH == 0.2
