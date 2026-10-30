"""The deployable safety-first gate: weights, the live path, and its limits.

The completed sensor run kept out-of-fold probabilities and no weights, so the
gate could only be applied to a development row by looking that row up. These
tests cover the artifact that closes that gap and the guards that keep it from
becoming something else: a checkpoint that carries a prompt, a gate whose
constants drifted, a refit quoted as an evaluation, or a path that reaches
held-out material.

Every fit here is on synthetic data built in the test. No frozen artifact is
refitted, nothing is downloaded and no sealed material is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from routegate.gatefreeze import (
    ACTUATOR,
    CAPABILITY,
    DEVELOPMENT_ESTIMATES,
    DIGEST_FIELD,
    NAME,
    OUTPUT_DIR,
    PREDECLARATION,
    SCHEMA,
    SEALED,
    SOURCES,
    build_predeclaration,
    record_sha256,
    verify_predeclaration,
)
from routegate.gatesensors import (
    ACTIVATIONS_SHA256,
    CHECKPOINT_FIELDS,
    CHECKPOINT_FILES,
    CHECKPOINT_SCHEMA,
    C_GRID,
    CLASS_WEIGHT,
    EXPECTED_ROWS,
    HIDDEN_SIZE,
    KIND_FOLD,
    KIND_REFIT,
    MAX_ITER,
    OOF_PREDICTIONS_SHA256,
    PENALTY,
    READ_STATE,
    SEED,
    SENSOR_E,
    SENSOR_H,
    SENSOR_NEGATIVE,
    SENSOR_POSITIVE,
    SOLVER,
    THRESHOLDS_SHA256,
    GateBundle,
    GateSensorCheckpoint,
    GateSensorError,
    assert_checkpoint_is_metadata_only,
    assert_not_a_development_reading,
    fit_sensor,
    load_bundle,
    load_thresholds,
    select_C,
    sensor_design,
)
from routegate.validation import Checklist, ValidationError, looks_sealed
from conftest import needs_gate_thresholds

REPO = Path(__file__).resolve().parents[1]
WIDTH = 24


def _design(n=120, width=WIDTH, seed=0):
    """A separable synthetic problem with all three classes present."""
    g = np.random.default_rng(seed)
    labels = np.array(["E"] * (n // 3) + ["B"] * (n // 3)
                      + ["H"] * (n - 2 * (n // 3)))
    X = g.normal(size=(n, width))
    X[labels == "E"] += 1.5
    X[labels == "H"] -= 1.5
    return X, labels


def _fit(sensor=SENSOR_E, kind=KIND_REFIT, seed=0):
    X, labels = _design(seed=seed)
    Xf, y, _mask = sensor_design(sensor, X, labels)
    return fit_sensor(sensor, Xf, y, 1.0, kind), X, labels


# ------------------------------------------- deterministic reload / identity
def test_a_checkpoint_round_trips_through_json_unchanged():
    ckpt, X, _ = _fit()
    blob = json.loads(json.dumps(ckpt.as_dict()))
    back = GateSensorCheckpoint.from_dict(blob)
    assert back.weights_sha256() == ckpt.weights_sha256()
    for row in X[:20]:
        assert back.probability(row) == ckpt.probability(row)


def test_refitting_the_same_design_twice_is_bit_identical():
    a, _, _ = _fit(seed=3)
    b, _, _ = _fit(seed=3)
    assert a.weights_sha256() == b.weights_sha256()
    assert np.array_equal(a.coef, b.coef)
    assert a.intercept == b.intercept


def test_a_checkpoint_that_misdescribes_its_own_weights_is_refused():
    ckpt, _, _ = _fit()
    blob = ckpt.as_dict()
    blob["coef"] = [x + 1.0 for x in blob["coef"]]
    with pytest.raises(GateSensorError, match="does not describe its own"):
        GateSensorCheckpoint.from_dict(blob)


@needs_gate_thresholds
def test_a_written_bundle_reloads_to_the_same_decisions(tmp_path):
    e, X, _ = _fit(SENSOR_E, seed=5)
    h, _, _ = _fit(SENSOR_H, seed=5)
    thresholds = load_thresholds()
    for sensor, ckpt in ((SENSOR_E, e), (SENSOR_H, h)):
        (tmp_path / CHECKPOINT_FILES[sensor]).write_text(
            json.dumps(ckpt.as_dict(), indent=2, sort_keys=True) + "\n")
    reloaded = load_bundle(tmp_path, thresholds)
    original = GateBundle(e_sensor=e, h_veto=h, thresholds=thresholds)
    for row in X[:30]:
        assert reloaded.decide(row).as_dict() == original.decide(row).as_dict()


# ------------------------------------------- live inference without row ids
@needs_gate_thresholds
def test_the_gate_decides_from_an_activation_alone():
    e, X, _ = _fit(SENSOR_E, seed=7)
    h, _, _ = _fit(SENSOR_H, seed=7)
    bundle = GateBundle(e_sensor=e, h_veto=h, thresholds=load_thresholds())
    decision = bundle.decide(X[0])
    assert isinstance(decision.fires, bool)
    assert 0.0 <= decision.p_E <= 1.0 and 0.0 <= decision.p_H <= 1.0


def test_decide_accepts_no_row_identifier():
    import inspect
    params = list(inspect.signature(GateBundle.decide).parameters)
    assert params == ["self", "activation"]
    assert not any("row" in p or "id" in p for p in params[1:])


def test_an_unseen_activation_is_scored_without_any_table():
    e, X, _ = _fit(SENSOR_E, seed=11)
    unseen = np.random.default_rng(99).normal(size=WIDTH)
    p = e.probability(unseen)
    assert 0.0 <= p <= 1.0


def test_no_checkpoint_field_carries_a_row_id_or_text():
    ckpt, _, _ = _fit()
    blob = ckpt.as_dict()
    assert_checkpoint_is_metadata_only(blob)
    assert blob["contains_row_ids"] is False
    assert blob["contains_prompt_or_response_text"] is False
    assert blob["contains_per_row_probabilities"] is False
    assert set(blob) <= set(CHECKPOINT_FIELDS)


@pytest.mark.parametrize("field", [
    "prompt_text", "response_text", "row_id", "p_E", "screen_row_ids"])
def test_a_checkpoint_carrying_forbidden_material_is_refused(field):
    blob = _fit()[0].as_dict()
    blob[field] = "anything"
    with pytest.raises(GateSensorError):
        assert_checkpoint_is_metadata_only(blob)


@pytest.mark.parametrize("field", [
    "contains_prompt_or_response_text", "contains_per_row_probabilities",
    "contains_row_ids"])
def test_a_checkpoint_that_declares_it_carries_them_is_refused(field):
    """The negative declarations are checked, not merely present."""
    blob = _fit()[0].as_dict()
    blob[field] = True
    with pytest.raises(GateSensorError, match="declares each of these False"):
        assert_checkpoint_is_metadata_only(blob)


def test_the_refit_may_never_be_quoted_as_a_development_reading():
    with pytest.raises(GateSensorError, match="training-set prediction"):
        assert_not_a_development_reading(KIND_REFIT, "scoring a dev row")
    assert_not_a_development_reading(KIND_FOLD, "scoring a held-out row")


@needs_gate_thresholds
def test_a_bundle_may_not_mix_a_fold_sensor_with_a_refit_sensor():
    e, _, _ = _fit(SENSOR_E, kind=KIND_REFIT)
    h, _, _ = _fit(SENSOR_H, kind=KIND_FOLD)
    with pytest.raises(GateSensorError, match="mixes a"):
        GateBundle(e_sensor=e, h_veto=h, thresholds=load_thresholds())


def test_a_width_mismatch_is_refused_rather_than_broadcast():
    ckpt, _, _ = _fit()
    with pytest.raises(GateSensorError, match="different state"):
        ckpt.probability(np.zeros(WIDTH + 1))


# ------------------------------------------------ unchanged gate constants
@needs_gate_thresholds
def test_the_thresholds_and_rule_are_the_frozen_ones():
    t = load_thresholds()
    assert t.threshold_E == 0.29185573339357973
    assert t.threshold_H == 0.06975384354575279
    assert t.gate_rule == "gate_fire = (p_E >= threshold_E) AND (p_H < threshold_H)"
    assert t.source_sha256 == THRESHOLDS_SHA256


@needs_gate_thresholds
def test_the_gate_rule_is_evaluated_not_restated():
    t = load_thresholds()
    assert t.fires(t.threshold_E, 0.0) is True             # >= is inclusive
    assert t.fires(t.threshold_E - 1e-12, 0.0) is False
    assert t.fires(1.0, t.threshold_H) is False            # < is exclusive
    assert t.fires(1.0, t.threshold_H - 1e-12) is True


@needs_gate_thresholds
def test_an_edited_threshold_file_is_refused(tmp_path):
    blob = json.loads((REPO / "outputs/routegate_residual_v1"
                       / "thresholds.json").read_text())
    blob["threshold_H"] = 0.5
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps(blob))
    with pytest.raises(GateSensorError, match="different gate"):
        load_thresholds(p)


def test_the_frozen_protocol_constants_have_not_drifted():
    assert READ_STATE == 45
    assert HIDDEN_SIZE == 2048 and EXPECTED_ROWS == 480
    assert (PENALTY, SOLVER, CLASS_WEIGHT, MAX_ITER) == (
        "l2", "lbfgs", "balanced", 5000)
    assert C_GRID == (0.01, 0.1, 1.0, 10.0, 100.0)
    assert SEED == 20260811
    assert SENSOR_POSITIVE == {SENSOR_E: "E", SENSOR_H: "H"}
    assert SENSOR_NEGATIVE == {SENSOR_E: ("B",), SENSOR_H: ("E", "B")}


def test_the_actuator_is_the_one_that_produced_the_outcome_table():
    assert ACTUATOR["direction_variant"] == "consensus23_global_frozen"
    assert ACTUATOR["donors"] == ["B_d2", "B_d3"]
    assert ACTUATOR["alpha_R"] == 0.15 and ACTUATOR["alpha_S"] == 0.0
    assert ACTUATOR["historical_layers"] == [31, 32, 33]
    assert ACTUATOR["prefill_only"] is True
    assert "historical_d1_global_frozen" in ACTUATOR["is_not"]


def test_C_selection_is_deterministic_and_prefers_the_smallest_on_a_tie():
    X, labels = _design(seed=13)
    Xf, y, _ = sensor_design(SENSOR_E, X, labels)
    first, scores = select_C(Xf, y)
    second, _ = select_C(Xf, y)
    assert first == second
    assert first in C_GRID
    assert set(scores) == {str(c) for c in C_GRID}


def test_each_head_is_fitted_on_its_declared_rows():
    X, labels = _design(seed=17)
    Xe, ye, mask_e = sensor_design(SENSOR_E, X, labels)
    assert set(labels[mask_e]) == {"E", "B"}
    assert ye.sum() == int((labels == "E").sum())
    Xh, yh, mask_h = sensor_design(SENSOR_H, X, labels)
    assert mask_h.all()
    assert yh.sum() == int((labels == "H").sum())


# ------------------------------------------------- prohibited sealed access
@needs_gate_thresholds
def test_the_predeclaration_reads_no_sealed_material():
    d = build_predeclaration()
    assert d["reads_the_sealed_set"] is False
    assert d["sealed_evaluation"]["opened_by_this_stage"] is False
    assert d["sealed_evaluation"]["prompt_content_read"] is False
    for flag in ("generates_responses", "calls_a_judge", "downloads_anything",
                 "modifies_sigr", "contains_row_ids",
                 "contains_prompt_or_response_text"):
        assert d[flag] is False


def test_no_declared_source_path_reads_like_held_out_material():
    for logical, spec in SOURCES.items():
        assert not looks_sealed(spec["path"]), logical
        assert not Path(spec["path"]).is_absolute(), logical


def test_the_sealed_set_is_the_existing_frozen_380():
    """The composition is pinned by the final protocol; this record defers."""
    assert SEALED["expected_row_count"] == 380
    assert SEALED["protocol"] == "routegate_final_evaluation_v1"
    from routegate.finaleval import DATASET_SPEC, DS_SEALED
    spec = DATASET_SPEC[DS_SEALED]
    assert spec["expected_safe_source_count"] == 180
    assert spec["expected_harmful_source_count"] == 200
    commitment = json.loads(
        (REPO / "manifests" / "sealed_set_commitment.json").read_text())
    assert commitment["expected_row_count"] == SEALED["expected_row_count"]
    assert commitment["availability"][
        "sealed_prompt_text_available_to_development_code"] is False


# ------------------------------------------------------- exact provenance
@needs_gate_thresholds
def test_the_predeclaration_pins_every_source_by_digest():
    d = build_predeclaration()
    assert set(d["sources"]) == set(SOURCES)
    for logical, entry in d["sources"].items():
        assert len(entry["sha256"]) == 64, logical
    assert d["sources"]["residual_activations"]["sha256"] == ACTIVATIONS_SHA256
    assert (d["sources"]["sensor_oof_predictions"]["sha256"]
            == OOF_PREDICTIONS_SHA256)
    assert d["sources"]["sensor_thresholds"]["sha256"] == THRESHOLDS_SHA256


def test_the_pinned_digests_match_the_frozen_input_registry():
    registry = {e["logical_name"]: e["sha256"] for e in json.loads(
        (REPO / "manifests" / "frozen_inputs.json").read_text())["inputs"]}
    assert registry["residual_activations"] == ACTIVATIONS_SHA256
    assert (registry["development_corpus"]
            == SOURCES["development_corpus"]["sha256"])
    assert (registry["static_steering_ebh_master"]
            == SOURCES["static_steering_master"]["sha256"])


@needs_gate_thresholds
def test_the_development_estimates_are_the_out_of_fold_ones():
    d = build_predeclaration()["development_estimates"]
    assert d["E_substantive_repair"] == [37, 89]
    assert d["B_material_degradation"] == [0, 178]
    assert d["H_actionable_regression"] == [0, 213]
    assert d["steered_rows"] == 59
    assert d["standing"].startswith("out-of-fold")
    assert build_predeclaration()["does_not_claim"][
        "the_refit_is_an_evaluation"] is False


def test_the_capability_scope_defers_to_the_final_protocol():
    assert CAPABILITY["protocol"] == "routegate_final_evaluation_v1"
    assert CAPABILITY["protocol_manifest"] == (
        "manifests/routegate_final_evaluation_v1.json")


@needs_gate_thresholds
def test_the_output_directory_is_declared_and_edits_no_existing_run():
    d = build_predeclaration()["output"]
    assert d["directory"] == OUTPUT_DIR == "outputs/routegate_gate_deployment_v1"
    assert d["modifies_any_existing_run"] is False
    assert d["kind"] == KIND_REFIT
    assert d["checkpoint_schema"] == CHECKPOINT_SCHEMA


@needs_gate_thresholds
def test_the_predeclaration_is_deterministic_and_self_describing():
    a, b = build_predeclaration(), build_predeclaration()
    assert a == b
    assert a[DIGEST_FIELD] == record_sha256(a)
    assert a["schema"] == SCHEMA and a["name"] == NAME


@needs_gate_thresholds
@pytest.mark.skipif(not PREDECLARATION.is_file(),
                    reason="the predeclaration has not been frozen yet")
def test_the_committed_predeclaration_verifies():
    checks = Checklist("test")
    verify_predeclaration(PREDECLARATION, checks)
    assert checks.all_passed
