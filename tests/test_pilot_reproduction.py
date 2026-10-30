"""Reproduction of the frozen pilot.

Covers requirements 14 and 15 of the test plan: the legacy values reproduce with
scikit-learn under the pilot's own folds, and a run saves one prediction per row
per state.

The gate runs on state 45 alone so the suite stays fast; the full 49-state sweep
is the job of the experiment entry point, which the suite gates.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT, needs_pilot_reference

sys.path.insert(0, str(REPO_ROOT / "experiments" / "residual_geometry"))

import reproduce_pilot as R                                        # noqa: E402
from routegate.data import pilot_reference_dir
from routegate.metrics import evaluate_out_of_fold                 # noqa: E402
from routegate.probes import cross_val_out_of_fold, make_probe     # noqa: E402

pytestmark = needs_pilot_reference

GATE_STATE = 45
TOLERANCE = 1e-6


# ------------------------------------------------ 14. legacy value recovery
def test_the_reference_declares_the_expected_state_45_values(pilot_manifest):
    expected = pilot_manifest["expected_values"]["state_45"]
    assert expected["roc_auc"] == pytest.approx(0.952594, abs=5e-7)
    assert expected["pr_auc"] == pytest.approx(0.894166, abs=5e-7)
    assert expected["balanced_accuracy"] == pytest.approx(0.865169, abs=5e-7)


def test_state_45_reproduces_the_pilot_within_tolerance(inputs, frozen_folds,
                                                        local_paths, pilot_manifest):
    reference_file = (pilot_reference_dir()
                      / pilot_manifest["reference_files"]["layerwise_metrics"]["filename"])
    reference = R.read_reference_metrics(
        reference_file,
        pilot_manifest["reference_files"]["layerwise_metrics"]["state_column"])

    result = cross_val_out_of_fold(inputs.state_matrix(GATE_STATE), inputs.y,
                                   frozen_folds, probe_factory=make_probe,
                                   keep_fold_detail=False)
    metrics = evaluate_out_of_fold(inputs.y, result.scores, result.predictions,
                                   state=GATE_STATE, degenerate=result.degenerate)

    for name in ("roc_auc", "pr_auc", "balanced_accuracy"):
        difference = abs(getattr(metrics, name) - reference[GATE_STATE][name])
        assert difference <= TOLERANCE, (
            f"state {GATE_STATE} {name}: reproduced "
            f"{getattr(metrics, name):.9f} vs pilot "
            f"{reference[GATE_STATE][name]:.9f}, difference {difference:.3e}")


def test_state_45_out_of_fold_scores_match_the_pilot(inputs, frozen_folds,
                                                     local_paths, pilot_manifest):
    spec = pilot_manifest["reference_files"]["fold_assignment_and_out_of_fold_scores"]
    path = pilot_reference_dir() / spec["filename"]
    reference = R.read_reference_scores(path, spec["score_column"])
    aligned = np.array([reference[rid] for rid in inputs.row_ids])

    result = cross_val_out_of_fold(inputs.state_matrix(GATE_STATE), inputs.y,
                                   frozen_folds, keep_fold_detail=False)
    assert np.max(np.abs(result.scores - aligned)) < 1e-3


def test_the_analysed_corpus_matches_the_pilot(inputs, frozen_folds):
    assert inputs.n == 267
    assert inputs.n_positive == 89
    assert inputs.n_negative == 178
    assert frozen_folds.n == 267
    assert frozen_folds.n_splits == 5


def test_the_row_order_is_the_pilots(inputs, local_paths, pilot_manifest):
    spec = pilot_manifest["reference_files"]["fold_assignment_and_out_of_fold_scores"]
    path = pilot_reference_dir() / spec["filename"]
    with open(path, newline="", encoding="utf-8") as handle:
        reference_order = [row["row_id"] for row in csv.DictReader(handle)]
    assert reference_order == inputs.row_ids


def test_state_zero_reproduces_as_a_degenerate_control(inputs, frozen_folds,
                                                       local_paths, pilot_manifest):
    reference_file = (pilot_reference_dir()
                      / pilot_manifest["reference_files"]["layerwise_metrics"]["filename"])
    reference = R.read_reference_metrics(reference_file)

    result = cross_val_out_of_fold(inputs.state_matrix(0), inputs.y, frozen_folds,
                                   keep_fold_detail=False)
    assert result.degenerate is True
    metrics = evaluate_out_of_fold(inputs.y, result.scores, result.predictions,
                                   state=0, degenerate=True)
    assert metrics.roc_auc == pytest.approx(reference[0]["roc_auc"], abs=TOLERANCE)
    assert metrics.pr_auc == pytest.approx(reference[0]["pr_auc"], abs=1e-6)
    assert metrics.balanced_accuracy == pytest.approx(
        reference[0]["balanced_accuracy"], abs=TOLERANCE)
    assert np.isfinite(result.scores).all()


# --------------------------------------- 15. predictions for every row/state
def test_a_run_saves_one_prediction_per_row_per_state(tmp_path):
    states = [0, 45]
    outcome = R.reproduce(output_dir=tmp_path / "run", states=states)
    output = tmp_path / "run"

    for name in ("config.json", "environment.json", "input_hashes.json",
                 "fold_assignments.csv", "oof_predictions_all_states.csv",
                 "layerwise_metrics.csv", "comparison_with_legacy.csv",
                 "validation.json", "VALIDATION.md", "output_hashes.json"):
        assert (output / name).exists(), f"missing {name}"

    with open(output / "oof_predictions_all_states.csv", newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(states) * 267
    for state in states:
        at_state = [r for r in rows if int(r["state"]) == state]
        assert len(at_state) == 267
        assert len({r["row_id"] for r in at_state}) == 267
        assert sum(1 for r in at_state if r["final_ebh"] == "E") == 89
        assert sum(1 for r in at_state if r["final_ebh"] == "B") == 178
        assert all(np.isfinite(float(r["oof_score_p_E"])) for r in at_state)
        assert all(0.0 <= float(r["oof_score_p_E"]) <= 1.0 for r in at_state)
        assert {int(r["fold"]) for r in at_state} == set(range(5))

    with open(output / "fold_assignments.csv", newline="", encoding="utf-8") as handle:
        folds = list(csv.DictReader(handle))
    assert len(folds) == 267
    assert [r["row_id"] for r in folds] == [r["row_id"] for r in rows[:267]]

    validation = json.loads((output / "validation.json").read_text())
    assert validation["all_passed"] is True
    assert validation["summary"]["gate_passed"] is True
    assert validation["summary"]["n"] == 267
    assert validation["summary"]["degenerate_states"] == [0]

    hashes = json.loads((output / "output_hashes.json").read_text())
    assert "layerwise_metrics.csv" in hashes["files"]
    assert outcome["summary"]["gate_passed"] is True


def test_the_run_records_the_verified_input_hashes(tmp_path):
    R.reproduce(output_dir=tmp_path / "run", states=[45])
    hashes = json.loads((tmp_path / "run" / "input_hashes.json").read_text())
    assert hashes["dataset"]["matches"] is True
    assert hashes["activation"]["matches"] is True
    assert hashes["extraction_manifest"]["matches"] is True
    assert hashes["activation"]["copied"] is False
    assert hashes["inputs_modified"] is False
    assert hashes["activation"]["shape"] == [480, 49, 2048]
    assert hashes["activation"]["dtype"] == "float32"


def test_the_run_records_that_scikit_learn_was_used(tmp_path):
    R.reproduce(output_dir=tmp_path / "run", states=[45])
    environment = json.loads((tmp_path / "run" / "environment.json").read_text())
    assert environment["scikit_learn"]
    assert environment["model_loaded"] is False
    assert environment["network_calls"] == 0
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["probe"]["pipeline"] == [
        "sklearn.preprocessing.StandardScaler",
        "sklearn.linear_model.LogisticRegression"]
    assert config["fold_source"].startswith("frozen:")


def test_the_historical_sources_are_untouched_by_a_run(tmp_path, local_paths,
                                                       input_manifest):
    from routegate.validation import sha256_file
    watched = [Path(local_paths["dataset"]),
               Path(local_paths["extraction_dir"]) / input_manifest["activation"]["filename"],
               Path(local_paths["extraction_dir"]) / input_manifest["extraction_manifest"]["filename"]]
    before = {str(p): sha256_file(p) for p in watched}
    R.reproduce(output_dir=tmp_path / "run", states=[45])
    after = {str(p): sha256_file(p) for p in watched}
    assert before == after


def test_the_corrected_analysis_refuses_to_run_without_approval(capsys):
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "residual_geometry"))
    import run_analysis

    assert run_analysis.main([]) == 0
    captured = capsys.readouterr().out
    assert "has NOT been run" in captured
