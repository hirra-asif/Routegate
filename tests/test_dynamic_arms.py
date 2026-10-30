"""The one-direction dynamic arm: real out-of-fold generation, and its gates.

Every fixture is synthetic and lives in ``tmp_path``, except the handful marked
``needs_real_inputs``, which read the frozen corpus and the committed fold
assignment. Nothing here loads a language model, generates a response, calls a
judge, reads the sealed set or touches completed judging evidence.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import (                                            # noqa: E402
    needs_dynamic_arm_inputs,
    needs_real_inputs,
)

import routegate.dynamicarms as D                                # noqa: E402
import routegate.gatecontroller as G                             # noqa: E402
import routegate.livecontroller as L                             # noqa: E402
from routegate.dualsteer import make_record                      # noqa: E402
from routegate.validation import Checklist                       # noqa: E402

_RUNNER = (Path(__file__).resolve().parents[1] / "experiments"
           / "single_direction_controller" / "run_dynamic_arms.py")


def _runner():
    spec = importlib.util.spec_from_file_location("_dyn_arms_v2", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                              # type: ignore
    return module


R = _runner()


# ------------------------------------------------------------------ fixtures
def _checkpoint(kind=L.KIND_FOLD, fold=0, readout=None):
    """A fold controller with its own readout, so folds score differently.

    ``build_controller`` zeroes the readout, so without this every fold returns
    the same gate on every row and a test that binds a coefficient to a
    controller would bind nothing.
    """
    pytest.importorskip("torch")
    net = G.build_controller(G.seed_for(fold if fold is not None else 0))
    state = {name: value.detach().numpy().copy()
             for name, value in net.state_dict().items()}
    if readout is not None:
        state["3.weight"] = readout
    return L.build_checkpoint(
        kind, fold, state, mu=np.zeros(G.HIDDEN_SIZE),
        sd=np.ones(G.HIDDEN_SIZE), tau_hi=1.5, selected_step=42,
        trained_on_rows=384)


def _bundle_dir(tmp_path, with_refit=True):
    rng = np.random.default_rng(4242)
    for k in range(G.N_SPLITS):
        payload = _checkpoint(L.KIND_FOLD, k,
                              readout=rng.normal(size=(1, G.CONTROLLER_HIDDEN)))
        (tmp_path / f"controller_fold_{k}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    if with_refit:
        (tmp_path / "controller_all_development_refit.json").write_text(
            json.dumps(_checkpoint(L.KIND_REFIT, None,
                                   readout=rng.normal(
                                       size=(1, G.CONTROLLER_HIDDEN)))),
            encoding="utf-8")
    return tmp_path


CLAUSES = ("row_set_matches", "coefficients_agree_within_the_float32_budget",
           D.CHECKPOINT_CLAUSE)


def _write_reproduction(directory, bundle, per_fold=None, *, passed=True,
                        clauses=CLAUSES, tolerance=None, rows=480,
                        checkpoints=None, agreement=None):
    recorded = checkpoints if checkpoints is not None else {
        f"controller_fold_{k}.json": c["weights_sha256"]
        for k, c in bundle.folds.items()}
    if checkpoints is None and bundle.refit is not None:
        recorded["controller_all_development_refit.json"] = \
            bundle.refit["weights_sha256"]
    summary = {
        "schema": D.REPRODUCTION_SCHEMA,
        "reproduces": "gate_oof_v1",
        "rows_compared": rows,
        "contains_per_row_coefficients": False,
        "checkpoints": recorded,
        "per_fold": per_fold or [{"fold": k, "held_out_mean_g": 0.1,
                                  "n_held_out": 96}
                                 for k in range(G.N_SPLITS)],
        "equivalence": {
            "passed": passed, "rule_version": 1,
            "tolerance": {"gate_abs_tolerance":
                          D.GATE_TOLERANCE if tolerance is None else tolerance},
            "clauses": [{"clause": name, "status": "PASS"}
                        for name in clauses],
        },
    }
    diagnostic = {
        "schema": D.EQUIVALENCE_SCHEMA,
        "checkpoint_agreement": agreement if agreement is not None else {
            "scored_out_of_fold_only": True, "rows_scored": 480,
            "max_abs_error_against_saved": 2.384185791015625e-07},
    }
    (Path(directory) / D.SUMMARY_FILENAME).write_text(
        json.dumps(summary), encoding="utf-8")
    (Path(directory) / D.DIAGNOSTIC_FILENAME).write_text(
        json.dumps(diagnostic), encoding="utf-8")
    return summary, diagnostic


def _prepared(tmp_path, n=20):
    """A bundle, activations, folds, and the reference coefficients."""
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    rng = np.random.default_rng(7)
    rows = [f"row{i:03d}" for i in range(n)]
    activations = {rid: rng.normal(size=G.HIDDEN_SIZE) for rid in rows}
    fold_of = {rid: i % G.N_SPLITS for i, rid in enumerate(rows)}
    reference = D.reference_gates(activations, fold_of, bundle)
    per_fold = []
    for k in range(G.N_SPLITS):
        held = [rid for rid in rows if fold_of[rid] == k]
        per_fold.append({"fold": k, "n_held_out": len(held),
                         "held_out_mean_g": float(np.mean(
                             [reference[rid] for rid in held]))})
    _write_reproduction(tmp_path, bundle, per_fold)
    return bundle, activations, fold_of, reference, rows


# ================================================================ the bundle
def test_a_bundle_is_bound_to_the_weight_digests_the_reproduction_recorded(
        tmp_path):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    _write_reproduction(tmp_path, bundle)
    checks = Checklist("t")
    provenance = D.verify_bundle(tmp_path, bundle, checks)
    assert checks.all_passed
    assert provenance["reproduces"] == "gate_oof_v1"
    assert provenance["equivalence_passed"] is True
    assert provenance["gate_abs_tolerance"] == D.GATE_TOLERANCE
    # the summary records weights, and the file digest is carried beside it
    for name, digests in provenance["checkpoint_sha256"].items():
        assert len(digests["weights_sha256"]) == 64
        assert len(digests["file_sha256"]) == 64
        assert digests["weights_sha256"] != digests["file_sha256"]


def test_a_checkpoint_whose_weights_are_not_the_recorded_ones_is_refused(
        tmp_path):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    recorded = {f"controller_fold_{k}.json": c["weights_sha256"]
                for k, c in bundle.folds.items()}
    recorded["controller_all_development_refit.json"] = \
        bundle.refit["weights_sha256"]
    recorded["controller_fold_2.json"] = "0" * 64
    _write_reproduction(tmp_path, bundle, checkpoints=recorded)
    with pytest.raises(Exception, match="controller_fold_2.json"):
        D.verify_bundle(tmp_path, bundle, Checklist("t"))


def test_a_bundle_with_no_reproduction_record_is_refused(tmp_path):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    with pytest.raises(D.DynamicArmError, match="is absent"):
        D.verify_bundle(tmp_path, bundle, Checklist("t"))


def test_a_checkpoint_the_reproduction_never_recorded_is_refused(tmp_path):
    """A sixth controller nobody reproduced is not quietly deployed."""
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    recorded = {f"controller_fold_{k}.json": c["weights_sha256"]
                for k, c in bundle.folds.items()}
    _write_reproduction(tmp_path, bundle, checkpoints=recorded)
    with pytest.raises(Exception, match="did not record"):
        D.verify_bundle(tmp_path, bundle, Checklist("t"))


def test_a_reproduction_that_did_not_pass_is_refused(tmp_path):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    _write_reproduction(tmp_path, bundle, passed=False)
    with pytest.raises(Exception, match="reproduction_passed"):
        D.verify_bundle(tmp_path, bundle, Checklist("t"))


def test_a_bundle_that_never_proved_its_checkpoints_score_the_corpus_is_refused(
        tmp_path):
    """The clause that makes the weights evidence rather than just weights."""
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    _write_reproduction(tmp_path, bundle,
                        clauses=tuple(c for c in CLAUSES
                                      if c != D.CHECKPOINT_CLAUSE))
    with pytest.raises(Exception,
                       match="checkpoints_score_the_corpus_the_same_way"):
        D.verify_bundle(tmp_path, bundle, Checklist("t"))


def test_a_reproduction_gated_at_another_tolerance_is_refused(tmp_path):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    _write_reproduction(tmp_path, bundle, tolerance=1e-9)
    with pytest.raises(Exception, match="tolerance"):
        D.verify_bundle(tmp_path, bundle, Checklist("t"))


def test_an_in_fold_score_is_refused_by_the_bundle_itself(tmp_path):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    _write_reproduction(tmp_path, bundle, agreement={
        "scored_out_of_fold_only": False, "rows_scored": 480,
        "max_abs_error_against_saved": 0.0})
    with pytest.raises(Exception, match="out_of_fold"):
        D.verify_bundle(tmp_path, bundle, Checklist("t"))


# ================================================== the reference coefficients
def test_the_reference_is_scored_by_the_controller_that_never_saw_the_row(
        tmp_path):
    """The row's own fold reproduces it; no other fold comes near it.

    ``reference_gates`` scores a whole fold in one batched float32 pass and the
    live path scores one row at a time, so the two reduction orders differ by a
    few float32 ULPs. That is the disagreement ``GATE_TOLERANCE`` is a budget
    for, and it is three orders of magnitude smaller than the separation
    between two different controllers scoring the same row.
    """
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path)
    for rid in rows:
        k = fold_of[rid]
        one_at_a_time = {
            j: float(L.gate_from_activation(
                activations[rid].reshape(1, -1), bundle.folds[j])[0])
            for j in range(G.N_SPLITS)}
        assert abs(reference[rid] - one_at_a_time[k]) <= D.GATE_TOLERANCE
        assert all(abs(reference[rid] - one_at_a_time[j]) > D.GATE_TOLERANCE
                   for j in range(G.N_SPLITS) if j != k)


def test_the_deployment_refit_may_never_score_a_development_row(tmp_path):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    rng = np.random.default_rng(3)
    activations = {"row000": rng.normal(size=G.HIDDEN_SIZE)}
    with pytest.raises(L.LiveControllerError):
        D.reference_gates(activations, {"row000": G.N_SPLITS}, bundle)


def test_the_five_held_out_means_are_re_derived_and_must_agree(tmp_path):
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path)
    summary, _ = D.load_reproduction(tmp_path)
    checks = Checklist("t")
    out = D.assert_reference_reproduces_the_run(reference, fold_of, summary,
                                                checks)
    assert checks.all_passed
    assert out["max_abs_difference"] <= D.MEAN_GATE_TOLERANCE
    assert out["scored_out_of_fold_only"] is True
    assert [entry["fold"] for entry in out["per_fold"]] == list(
        range(G.N_SPLITS))


def test_a_held_out_mean_that_does_not_reproduce_stops_the_run(tmp_path):
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path)
    summary, _ = D.load_reproduction(tmp_path)
    summary["per_fold"][2]["held_out_mean_g"] += 1e-4
    with pytest.raises(Exception, match="fold_2_held_out_mean_coefficient"):
        D.assert_reference_reproduces_the_run(reference, fold_of, summary,
                                              Checklist("t"))


def test_a_permuted_fold_assignment_moves_a_held_out_mean(tmp_path):
    """The check is not vacuous: rotate the folds and it fails."""
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path)
    summary, _ = D.load_reproduction(tmp_path)
    moved = {rid: (f + 1) % G.N_SPLITS for rid, f in fold_of.items()}
    with pytest.raises(Exception, match="held_out_mean_coefficient"):
        D.assert_reference_reproduces_the_run(
            D.reference_gates(activations, moved, bundle), moved, summary,
            Checklist("t"))


# ---------------------------------------------- the live coefficient's gate
def test_a_live_coefficient_inside_the_budget_is_accepted():
    delta = D.assert_gate_is_faithful("row000", 0.5, 0.5 + D.GATE_TOLERANCE / 2)
    assert delta == pytest.approx(D.GATE_TOLERANCE / 2)


def test_a_live_coefficient_outside_the_budget_stops_the_run():
    with pytest.raises(D.DynamicArmError, match="faithfully"):
        D.assert_gate_is_faithful("row000", 0.5, 0.5 + 1e-3)


def test_the_budget_is_the_reproduction_s_own_and_is_not_a_new_number():
    from routegate.reproequivalence import (GATE_ABS_TOLERANCE,
                                            GATE_EPSILON_BUDGET)
    assert D.GATE_TOLERANCE == GATE_ABS_TOLERANCE
    assert D.GATE_TOLERANCE_EXPRESSION == f"{GATE_EPSILON_BUDGET} * float32_eps"


# ================================================================ the direction
def _bank(repair="B_d1 - E", orthogonalised=False):
    return SimpleNamespace(
        artifact_sha256="a" * 64, manifest_sha256="b" * 64,
        manifest={"construction": {"subtraction_order":
                                   {"repair": repair, "safety": "H - E"}},
                  "orthogonalisation": {"applied": orthogonalised}})


def test_the_repair_direction_must_be_b_d1_minus_e():
    checks = Checklist("t")
    out = D.assert_repair_direction_is_b_d1_minus_e(_bank(), checks)
    assert out["subtraction_order"] == "B_d1 - E" == D.REPAIR_SUBTRACTION_ORDER
    assert checks.all_passed


def test_a_reversed_repair_subtraction_is_refused():
    with pytest.raises(D.DynamicArmError, match="subtraction order"):
        D.assert_repair_direction_is_b_d1_minus_e(_bank("E - B_d1"))


def test_the_orthogonalised_bank_is_not_substituted_for_the_validated_one():
    with pytest.raises(D.DynamicArmError, match="orthogonalisation"):
        D.assert_repair_direction_is_b_d1_minus_e(_bank(orthogonalised=True))


# =================================================================== records
def _locked(arm=D.ARM_RAW):
    return D.locked_fields(
        direction_bank_sha256="a" * 64, dataset_sha256="d" * 64,
        generation_config_sha256="g" * 64, model="Qwen/Qwen3-30B-A3B",
        revision="r" * 40, tokenizer_config_sha256="t" * 64,
        chat_template_sha256="c" * 64,
        intervention_type="dynamic_two_site_repair",
        budget_reference="block_input", arm_id=arm)


def _corpus(rows):
    return {rid: {"row_id": rid, "final_ebh": "B",
                  "prompt_sha256": f"{i:064d}", "prompt_text": "p"}
            for i, rid in enumerate(rows)}


def _make(rid, g_r, fold, locked, corpus, checkpoint):
    block = D.controller_block(fold, checkpoint, g_r, g_r, True)
    per_state = {str(state): {"state": state, "status": "ok",
                              "repair_direction_sha256": f"{state:064d}",
                              "controller": dict(block)}
                 for state in D.CANDIDATE_STATES}
    return make_record(
        row_id=rid, row_class=corpus[rid]["final_ebh"],
        prompt_sha256=corpus[rid]["prompt_sha256"],
        alpha_R=float(G.ALPHA_R * g_r), status="ok", error_message="",
        per_state=per_state, hooks_expected=2, hooks_fired=2,
        applied_by_state={"32": 1.0, "33": 1.0}, created_utc="now",
        response_text="answer", **locked)


def _one(tmp_path):
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path, n=5)
    locked, corpus = _locked(), _corpus(rows)
    rid = rows[0]
    record = _make(rid, reference[rid], fold_of[rid], locked, corpus,
                   bundle.folds[fold_of[rid]])
    return record, locked, corpus, fold_of, reference, rid, bundle


def test_a_well_formed_record_validates(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_the_applied_coefficient_is_alpha_r_times_the_gate(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    assert record["alpha_R"] == G.ALPHA_R * reference[rid]
    record["alpha_R"] = 0.15
    with pytest.raises(D.DynamicArmError, match="alpha_R"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_alpha_r_is_not_a_run_locked_field(tmp_path):
    """Two rows at different coefficients are one arm, not two."""
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path, n=5)
    locked, corpus = _locked(), _corpus(rows)
    assert "alpha_R" not in D.RUN_LOCKED_FIELDS
    for rid in rows[:2]:
        D.validate_record(
            _make(rid, reference[rid], fold_of[rid], locked, corpus,
                  bundle.folds[fold_of[rid]]),
            locked, corpus, fold_of, reference, rid)
    assert len({reference[rid] for rid in rows[:2]}) == 2


def test_two_coefficients_at_the_two_sites_is_a_different_intervention(
        tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    record["per_state"]["33"]["controller"]["g_R"] += 1e-9
    with pytest.raises(D.DynamicArmError, match="different controller blocks"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_a_row_scored_by_another_fold_s_controller_is_refused(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    for state in ("32", "33"):
        record["per_state"][state]["controller"]["fold"] = \
            (fold_of[rid] + 1) % G.N_SPLITS
    with pytest.raises(D.DynamicArmError, match="never saw it"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_a_coefficient_that_does_not_reproduce_is_refused(tmp_path):
    record, locked, corpus, fold_of, reference, rid, bundle = _one(tmp_path)
    moved = dict(reference)
    moved[rid] = reference[rid] + 1e-3
    with pytest.raises(D.DynamicArmError, match="does not reproduce"):
        D.validate_record(record, locked, corpus, fold_of, moved, "r")


def test_a_record_written_at_other_sites_is_refused(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    record["historical_layers"] = [31, 32, 33]
    with pytest.raises(D.DynamicArmError, match="historical_layers"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_block_33_is_refused_even_when_the_run_itself_declares_it(tmp_path):
    """The site that failed the one-pass budget test stays out."""
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    widened = {**locked, "historical_layers": [31, 32, 33],
               "states": [32, 33, 34]}
    record["historical_layers"] = [31, 32, 33]
    record["states"] = [32, 33, 34]
    with pytest.raises(D.DynamicArmError, match="states"):
        D.validate_record(record, widened, corpus, fold_of, reference, "r")
    # and with the states left alone it is the block-33 clause that fires
    record["states"] = list(D.CANDIDATE_STATES)
    with pytest.raises(D.DynamicArmError, match="block 33"):
        D.validate_record(
            record, {**widened, "states": list(D.CANDIDATE_STATES)},
            corpus, fold_of, reference, "r")


def test_a_record_with_no_controller_block_cannot_be_traced(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    del record["per_state"]["32"]["controller"]
    with pytest.raises(D.DynamicArmError, match="no controller block"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_a_record_whose_edit_never_fired_is_not_a_steered_record(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    record["hooks_fired"] = 1
    with pytest.raises(D.DynamicArmError, match="hook"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_a_record_from_a_different_run_is_refused(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    record["direction_bank_sha256"] = "z" * 64
    with pytest.raises(D.DynamicArmError, match="direction_bank_sha256"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_a_record_that_applied_the_safety_direction_is_refused(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    record["alpha_S"] = 0.1
    with pytest.raises(D.DynamicArmError, match="alpha_S"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


def test_a_record_for_another_prompt_is_refused(tmp_path):
    record, locked, corpus, fold_of, reference, rid, _ = _one(tmp_path)
    record["prompt_sha256"] = "9" * 64
    with pytest.raises(D.DynamicArmError, match="prompt digest"):
        D.validate_record(record, locked, corpus, fold_of, reference, "r")


# ================================================================= the shard
def _write(path, records):
    from routegate.dualsteer import append_record
    for record in records:
        append_record(path, record)


def test_existing_records_are_validated_before_they_are_skipped(tmp_path):
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path, n=5)
    locked, corpus = _locked(), _corpus(rows)
    path = tmp_path / D.shard_records_name(1, 1)
    _write(path, [_make(rid, reference[rid], fold_of[rid], locked, corpus,
                        bundle.folds[fold_of[rid]]) for rid in rows[:3]])
    checks = Checklist("t")
    done = D.validate_existing_records(path, rows, locked, corpus, fold_of,
                                       reference, checks)
    assert sorted(done) == sorted(rows[:3])
    assert checks.all_passed


def test_a_duplicate_row_in_a_shard_file_is_refused(tmp_path):
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path, n=5)
    locked, corpus = _locked(), _corpus(rows)
    path = tmp_path / D.shard_records_name(1, 1)
    record = _make(rows[0], reference[rows[0]], fold_of[rows[0]], locked,
                   corpus, bundle.folds[fold_of[rows[0]]])
    _write(path, [record, record])
    with pytest.raises(D.DynamicArmError, match="more than once"):
        D.validate_existing_records(path, rows, locked, corpus, fold_of,
                                    reference)


def test_a_row_from_another_shard_means_two_processes_shared_a_file(tmp_path):
    bundle, activations, fold_of, reference, rows = _prepared(tmp_path, n=5)
    locked, corpus = _locked(), _corpus(rows)
    path = tmp_path / D.shard_records_name(1, 2)
    _write(path, [_make(rows[4], reference[rows[4]], fold_of[rows[4]], locked,
                        corpus, bundle.folds[fold_of[rows[4]]])])
    with pytest.raises(D.DynamicArmError, match="not one of this shard"):
        D.validate_existing_records(path, rows[:2], locked, corpus, fold_of,
                                    reference)


def test_a_finished_shard_is_not_re_entered():
    with pytest.raises(D.DynamicArmError, match="already complete"):
        D.assert_shard_is_not_complete({"a": {}, "b": {}}, ["a", "b"])
    D.assert_shard_is_not_complete({"a": {}}, ["a", "b"])


def test_resume_claims_only_a_directory_this_shard_wrote(tmp_path):
    allowed = D.shard_filenames(1, 1)
    out = tmp_path / "arm"
    out.mkdir()
    (out / allowed[0]).write_text("", encoding="utf-8")
    assert D.prepare_output_dir(out, allowed, True) == out
    with pytest.raises(D.DynamicArmError, match="--resume was not given"):
        D.prepare_output_dir(out, allowed, False)
    (out / "someone_elses.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(D.DynamicArmError, match="did not write"):
        D.prepare_output_dir(out, allowed, True)


# ============================================================== completion
def _complete(tmp_path, n=D.EXPECTED_ROWS):
    bundle = L.load_bundle(_bundle_dir(tmp_path))
    rng = np.random.default_rng(19)
    rows = [f"row{i:04d}" for i in range(n)]
    activations = {rid: rng.normal(size=G.HIDDEN_SIZE) for rid in rows}
    fold_of = {rid: i % G.N_SPLITS for i, rid in enumerate(rows)}
    reference = D.reference_gates(activations, fold_of, bundle)
    locked, corpus = _locked(), _corpus(rows)
    records = {rid: _make(rid, reference[rid], fold_of[rid], locked, corpus,
                          bundle.folds[fold_of[rid]]) for rid in rows}
    return records, rows, locked, corpus, fold_of, reference


def test_a_finished_arm_is_exactly_the_480_rows_once_each(tmp_path):
    records, rows, *_ = _complete(tmp_path)
    checks = Checklist("t")
    reading = D.assert_complete(records, rows, checks)
    assert checks.all_passed
    assert reading["records"] == reading["rows"] == D.EXPECTED_ROWS
    assert 0.0 < reading["min_applied_alpha_R"]
    assert reading["max_applied_alpha_R"] <= G.ALPHA_R
    assert reading["distinct_applied_coefficients"] > 1


def test_479_records_is_not_a_finished_arm(tmp_path):
    records, rows, *_ = _complete(tmp_path)
    records.pop(rows[17])
    with pytest.raises(Exception, match="completed_records"):
        D.assert_complete(records, rows, Checklist("t"))


def test_a_constant_coefficient_would_mean_the_gate_was_never_read(tmp_path):
    records, rows, locked, corpus, fold_of, reference = _complete(tmp_path)
    for record in records.values():
        record["alpha_R"] = 0.15
    with pytest.raises(Exception, match="prompt_specific"):
        D.assert_complete(records, rows, Checklist("t"))


def test_shards_merge_into_one_result_and_may_not_overlap(tmp_path):
    records, rows, locked, corpus, fold_of, reference = _complete(tmp_path,
                                                                  n=10)
    out = tmp_path / "arm"
    out.mkdir()
    _write(out / D.shard_records_name(1, 2),
           [records[rid] for rid in rows[:5]])
    _write(out / D.shard_records_name(2, 2),
           [records[rid] for rid in rows[5:]])
    merged, files = D.collect_records(out, locked, corpus, fold_of, reference,
                                      Checklist("t"))
    assert sorted(merged) == sorted(rows)
    assert files["records"] == 10
    _write(out / D.shard_records_name(2, 2), [records[rows[0]]])
    with pytest.raises(D.DynamicArmError, match="did not\\s+partition"):
        D.collect_records(out, locked, corpus, fold_of, reference)


def test_the_coefficient_table_is_derived_from_the_records(tmp_path):
    records, rows, locked, corpus, fold_of, reference = _complete(tmp_path,
                                                                  n=6)
    table = D.coefficient_rows(records, rows)
    assert [r["row_id"] for r in table] == rows
    assert set(table[0]) == set(D.COEFFICIENT_COLUMNS)
    for row in table:
        assert row["fold"] == fold_of[row["row_id"]]
        assert row["applied_alpha_R"] == pytest.approx(
            G.ALPHA_R * row["g_R"], rel=0, abs=0)
        assert row["apply_at_historical_blocks"] == "31 32"


# ================================================================ the runner
def test_the_veto_arm_refuses_to_generate(tmp_path):
    args = R.build_parser().parse_args([
        "--direction-bank", str(tmp_path), "--controllers", str(tmp_path),
        "--output-dir", str(tmp_path / "out"), "--arm", D.ARM_VETO])
    with pytest.raises(D.DynamicArmError, match="deterministic recombination"):
        R.run(args)


def test_dry_run_and_verify_are_not_the_same_question(tmp_path):
    args = R.build_parser().parse_args([
        "--direction-bank", str(tmp_path), "--controllers", str(tmp_path),
        "--output-dir", str(tmp_path / "out"), "--dry-run", "--verify"])
    with pytest.raises(D.DynamicArmError, match="not the same question"):
        R.run(args)


def test_the_runner_reads_the_gate_from_the_live_capture_not_a_lookup():
    """The mechanism, pinned in the source that implements it."""
    source = _RUNNER.read_text(encoding="utf-8")
    # the coefficient comes from the activation this pass captured
    assert "capture[READ_STATE]" in source
    assert "plan_live_edit(activation, fold, bundle)" in source
    # read before write: inside the generation loop the capture happens before
    # any plan is built. Measured over the loop, not the import block.
    body = source[source.index("def generate("):]
    assert body.index("capture_prompt_states(") < body.index(
        "build_two_site_plans(")
    assert body.index("plan_live_edit(") < body.index("build_two_site_plans(")
    # and the frozen coefficient column is never opened
    assert "oof_coefficients" not in source


def test_the_runner_orders_and_shards_on_the_corpus_records():
    source = _RUNNER.read_text(encoding="utf-8")
    assert "deterministic_row_order(corpus_rows)" in source
    assert "deterministic_row_order(list(frame.row_ids))" not in source
    assert "deterministic_row_order(inputs.row_ids)" not in source


def test_the_runner_applies_one_coefficient_at_the_two_passing_sites():
    from routegate.twosite import CANDIDATE_BLOCKS, CANDIDATE_STATES
    assert tuple(CANDIDATE_BLOCKS) == (31, 32)
    assert tuple(CANDIDATE_STATES) == (32, 33)
    assert tuple(L.APPLY_BLOCKS) == (31, 32)
    assert G.ALPHA_R == 0.15 and G.ALPHA_S == 0.0


# ---------------------------------------------------------- the real inputs
@needs_real_inputs
def test_the_frozen_folds_cover_the_corpus_exactly_once():
    from routegate.data import load_development_inputs
    from routegate.twositecontinuation import deterministic_row_order

    inputs = load_development_inputs()
    order = deterministic_row_order(list(inputs.manifest_rows))
    checks = Checklist("t")
    fold_of, source = D.fold_map(order, checks=checks)
    assert checks.all_passed
    assert len(fold_of) == D.EXPECTED_ROWS == 480
    assert sorted(set(fold_of.values())) == list(range(G.N_SPLITS))
    assert sum(source["held_out_rows_per_fold"].values()) == 480


# ====================================================== the generation loop
#
# The loop that actually runs on the GPU job, exercised without one. The model,
# the hooks and the steering plans are stubs; everything the runner itself is
# responsible for -- the read-before-write order, the coefficient, the
# faithfulness gate, the record, the resume, the table and the manifest -- is
# the real code.

class _FakePlan:
    def __init__(self, state: int, strength: float) -> None:
        self.state, self.strength, self.status = state, strength, "ok"

    def diagnostics(self):
        return {"state": self.state, "status": self.status,
                "strength": self.strength}


class _FakeController:
    def __init__(self, model, states, budget_reference="block_input") -> None:
        self.states = list(states)
        self.budget_reference = budget_reference
        self.captures = {}
        self.closed = False

    def capture_for(self, name):
        return self.captures[name]

    def close(self):
        self.closed = True


def _install_fakes(monkeypatch, activations, *, drift=0.0):
    import routegate.hooks as H
    import routegate.runtime as RT

    def capture(model, tokenizer, prompt, controller, capture_name,
                example_id, config=None):
        vector = np.asarray(activations[example_id], dtype=np.float32)
        if drift:
            vector = (vector + np.float32(drift)).astype(np.float32)
        controller.captures[capture_name] = {31: vector, 32: vector,
                                             33: vector}
        return 11

    def generate_response(model, tokenizer, prompt, config=None,
                          controller=None, mode="off", plans=None,
                          example_id=None):
        return {"response_text": f"answer for {example_id}",
                "prompt_token_count": 11, "completion_token_count": 3,
                "eos_reached": True, "hit_max_new_tokens": False,
                "elapsed_seconds": 0.5,
                "applied_by_state": {str(s): 1.0 for s in (plans or {})}}

    monkeypatch.setattr(H, "PatchController", _FakeController)
    monkeypatch.setattr(RT, "capture_prompt_states", capture)
    monkeypatch.setattr(RT, "generate_response", generate_response)
    monkeypatch.setattr(RT, "load_model_and_tokenizer",
                        lambda config=None: (SimpleNamespace(
                            eval=lambda: None), object()))
    monkeypatch.setattr(R, "build_two_site_plans",
                        lambda bank, capture, alpha, g_r, blocks: {
                            int(state): _FakePlan(int(state),
                                                  float(alpha) * float(g_r))
                            for state in D.CANDIDATE_STATES})


def _generation_inputs(tmp_path, n=6):
    from routegate.runtime import GenerationConfig

    bundle = L.load_bundle(_bundle_dir(tmp_path))
    _write_reproduction(tmp_path, bundle)
    rng = np.random.default_rng(31)
    rows = [f"row{i:03d}" for i in range(n)]
    frozen = {rid: rng.normal(size=G.HIDDEN_SIZE).astype(np.float32)
              for rid in rows}
    fold_of = {rid: i % G.N_SPLITS for i, rid in enumerate(rows)}
    reference = D.reference_gates(frozen, fold_of, bundle)
    locked, corpus = _locked(), _corpus(rows)
    bank = SimpleNamespace(
        artifact_sha256="a" * 64,
        direction_sha256=lambda name, state: f"{state:064d}")
    prepared = {
        "bundle": bundle, "bank": bank, "corpus": corpus, "order": rows,
        "fold_of": fold_of, "fold_source": {"n_splits": G.N_SPLITS},
        "reference": reference, "frozen": frozen, "shard": rows,
        "generation": GenerationConfig(), "locked": locked,
        "provenance": {"reproduces": "gate_oof_v1"},
        "direction": {"subtraction_order": "B_d1 - E"},
        "reproduction": {"max_abs_difference": 0.0},
        "dataset_sha256": "d" * 64,
    }
    return prepared, rows, locked, corpus, fold_of, reference


def _args(tmp_path, out, **overrides):
    argv = ["--direction-bank", str(tmp_path), "--controllers", str(tmp_path),
            "--output-dir", str(out)]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}"] + (
            [] if value is True else [str(value)])
    return R.build_parser().parse_args(argv)


def test_the_generation_loop_writes_one_reproducing_record_per_row(
        tmp_path, monkeypatch):
    prepared, rows, locked, corpus, fold_of, reference = _generation_inputs(
        tmp_path)
    _install_fakes(monkeypatch, prepared["frozen"])
    out = tmp_path / "arm"
    checks = Checklist("t")
    assert R.generate(_args(tmp_path, out), prepared, checks) == 0

    records = D.collect_records(out, locked, corpus, fold_of, reference,
                                Checklist("t"))[0]
    assert sorted(records) == sorted(rows)
    for rid, record in records.items():
        block = record["per_state"]["32"]["controller"]
        assert block["fold"] == fold_of[rid]
        assert block["is_out_of_fold"] is True
        assert block["read_state"] == G.READ_STATE == 32
        assert block["activation_bit_identical_to_the_frozen_capture"] is True
        assert block["abs_gate_difference"] <= D.GATE_TOLERANCE
        # one coefficient, written at both sites
        assert record["per_state"]["33"]["controller"] == block
        assert record["alpha_R"] == pytest.approx(
            G.ALPHA_R * block["g_R"], rel=0, abs=0)
        assert record["response_text"] == f"answer for {rid}"
    # the coefficient is the prompt's, not a constant
    assert len({r["alpha_R"] for r in records.values()}) == len(rows)


def test_the_run_records_its_coefficients_and_its_mechanism(tmp_path,
                                                            monkeypatch):
    prepared, rows, locked, corpus, fold_of, reference = _generation_inputs(
        tmp_path)
    _install_fakes(monkeypatch, prepared["frozen"])
    out = tmp_path / "arm"
    R.generate(_args(tmp_path, out), prepared, Checklist("t"))

    import csv
    with (out / D.COEFFICIENTS_FILENAME).open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert [r["row_id"] for r in table] == rows
    assert list(table[0]) == list(D.COEFFICIENT_COLUMNS)
    assert all(r["apply_at_historical_blocks"] == "31 32" for r in table)

    manifest = json.loads((out / D.OUTPUT_MANIFEST).read_text(
        encoding="utf-8"))
    assert manifest["arm"] == D.ARM_RAW
    assert manifest["generation_forward_passes_per_prompt"] == 1
    assert manifest["coefficient"]["alpha_R"] == 0.15
    assert manifest["coefficient"]["applied_at_historical_blocks"] == [31, 32]
    assert manifest["coefficient"]["same_coefficient_at_every_site"] is True
    assert manifest["coefficient"]["gate_tolerance"] == D.GATE_TOLERANCE
    assert manifest["coefficient"][
        "live_captures_bit_identical_to_the_frozen_extraction"] == len(rows)
    assert "no row-id coefficient lookup" in manifest["coefficient"][
        "mechanism"]
    assert manifest["trains_anything"] is False
    assert manifest["judges_responses"] is False
    assert manifest["reads_the_sealed_set"] is False
    assert manifest["errors"] == []


def test_resume_regenerates_nothing_and_refuses_a_finished_shard(
        tmp_path, monkeypatch):
    prepared, rows, locked, corpus, fold_of, reference = _generation_inputs(
        tmp_path)
    _install_fakes(monkeypatch, prepared["frozen"])
    out = tmp_path / "arm"

    partial = dict(prepared, shard=rows[:3])
    assert R.generate(_args(tmp_path, out), partial, Checklist("t")) == 0
    first = (out / D.shard_records_name(1, 1)).read_text(encoding="utf-8")

    assert R.generate(_args(tmp_path, out, resume=True), prepared,
                      Checklist("t")) == 0
    text = (out / D.shard_records_name(1, 1)).read_text(encoding="utf-8")
    assert text.startswith(first)                 # nothing was regenerated
    assert len(text.strip().splitlines()) == len(rows)

    with pytest.raises(D.DynamicArmError, match="already complete"):
        R.generate(_args(tmp_path, out, resume=True), prepared, Checklist("t"))


def test_a_run_that_did_not_ask_to_resume_will_not_touch_a_used_directory(
        tmp_path, monkeypatch):
    prepared, rows, locked, corpus, fold_of, reference = _generation_inputs(
        tmp_path)
    _install_fakes(monkeypatch, prepared["frozen"])
    out = tmp_path / "arm"
    R.generate(_args(tmp_path, out), dict(prepared, shard=rows[:2]),
               Checklist("t"))
    with pytest.raises(D.DynamicArmError, match="--resume was not given"):
        R.generate(_args(tmp_path, out), prepared, Checklist("t"))


def test_a_controller_that_cannot_score_the_activation_stops_the_run(
        tmp_path, monkeypatch):
    """Not an error record, not a warning: the run stops with nothing extra."""
    prepared, rows, locked, corpus, fold_of, reference = _generation_inputs(
        tmp_path)
    _install_fakes(monkeypatch, prepared["frozen"], drift=0.5)
    out = tmp_path / "arm"
    with pytest.raises(D.DynamicArmError, match="faithfully"):
        R.generate(_args(tmp_path, out), prepared, Checklist("t"))
    written = (out / D.shard_records_name(1, 1))
    assert not written.exists() or not written.read_text(
        encoding="utf-8").strip()


def test_a_finished_directory_verifies_and_writes_nothing(tmp_path,
                                                          monkeypatch):
    prepared, rows, locked, corpus, fold_of, reference = _generation_inputs(
        tmp_path)
    _install_fakes(monkeypatch, prepared["frozen"])
    out = tmp_path / "arm"
    R.generate(_args(tmp_path, out), prepared, Checklist("t"))
    before = sorted(p.name for p in out.iterdir())

    monkeypatch.setattr(D, "EXPECTED_ROWS", len(rows))
    monkeypatch.setattr(R, "EXPECTED_ROWS", len(rows))
    checks = Checklist("t")
    assert R.verify(_args(tmp_path, out, verify=True), prepared, checks) == 0
    assert checks.all_passed
    assert sorted(p.name for p in out.iterdir()) == before


def test_verification_refuses_a_directory_that_is_one_row_short(tmp_path,
                                                                monkeypatch):
    prepared, rows, locked, corpus, fold_of, reference = _generation_inputs(
        tmp_path)
    _install_fakes(monkeypatch, prepared["frozen"])
    out = tmp_path / "arm"
    R.generate(_args(tmp_path, out), dict(prepared, shard=rows[:-1]),
               Checklist("t"))
    monkeypatch.setattr(D, "EXPECTED_ROWS", len(rows))
    with pytest.raises(Exception, match="completed_records"):
        R.verify(_args(tmp_path, out, verify=True), prepared, Checklist("t"))


@needs_dynamic_arm_inputs
def test_the_real_bundle_reproduces_gate_oof_v1_on_the_frozen_activations(
        direction_bank_dir, controller_bundle_dir):
    """The evidence the whole arm rests on, asserted rather than described."""
    from routegate.data import load_development_inputs
    from routegate.ebh import frame_from_inputs
    from routegate.probeheads import state_matrix
    from routegate.twositecontinuation import deterministic_row_order

    bundle = L.load_bundle(controller_bundle_dir)
    checks = Checklist("t")
    provenance = D.verify_bundle(controller_bundle_dir, bundle, checks)
    assert checks.all_passed
    assert provenance["reproduces"] == "gate_oof_v1"
    assert provenance["rows_compared"] == D.EXPECTED_ROWS == 480
    assert provenance["max_abs_error_against_saved"] <= D.GATE_TOLERANCE
    assert len(provenance["checkpoint_sha256"]) == G.N_SPLITS + 1

    inputs = load_development_inputs()
    frame = frame_from_inputs(inputs)
    order = deterministic_row_order(list(inputs.manifest_rows))
    fold_of, _ = D.fold_map(order, checks=checks)
    design = state_matrix(frame, G.READ_STATE)
    position = frame.position_of
    reference = D.reference_gates(
        {rid: design[position[rid]] for rid in order}, fold_of, bundle)

    summary, _ = D.load_reproduction(controller_bundle_dir)
    out = D.assert_reference_reproduces_the_run(reference, fold_of, summary,
                                                checks)
    assert checks.all_passed
    assert out["max_abs_difference"] <= D.MEAN_GATE_TOLERANCE

    # Every row's applied strength must be strictly positive: the two-site
    # actuator refuses a strength outside (0, alpha_R], so a coefficient that
    # underflowed to exactly zero would stop the run mid-corpus.
    applied = np.array([G.ALPHA_R * reference[rid] for rid in order])
    assert len(applied) == 480
    assert applied.min() > 0.0
    assert applied.max() <= G.ALPHA_R

    # The coefficient is prompt-specific, but it is not injective: a handful of
    # rows land on the same float32 gate, most of them near the initialisation
    # value. That is the controller's resolution, not a duplicated row -- every
    # prompt in the corpus is distinct, and so is every prompt inside each
    # colliding group.
    import collections
    prompt_of = {r["row_id"]: r["prompt_sha256"] for r in inputs.manifest_rows}
    assert len(set(prompt_of.values())) == 480
    by_gate = collections.defaultdict(list)
    for rid in order:
        by_gate[reference[rid]].append(rid)
    assert len(by_gate) >= 450
    for rows in by_gate.values():
        assert len({prompt_of[rid] for rid in rows}) == len(rows)
