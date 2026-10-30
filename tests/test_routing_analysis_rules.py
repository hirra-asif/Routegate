"""The routing-analysis stage must reuse routegate_residual_v1's frozen rules.

The strongest available offline check: feed residual_v1's OWN frozen out-of-fold
predictions into this stage's reimplementation of its threshold, gate and metric
rules, and require the frozen answers back. No model is fitted and no feature
matrix is built, so this runs in milliseconds and gates the expensive job.

Skipped where the frozen baseline is not reachable.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "experiments/routegate_routing/analyze_routing_vs_residual_v1.py")


def _module():
    spec = importlib.util.spec_from_file_location("analyze_routing_vs_residual_v1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # safe: no heavy imports at module scope
    return mod


# The script resolves its inputs from configs/paths.local.json when imported.
if not (Path(__file__).resolve().parents[1] / "configs" / "paths.local.json").is_file():
    pytest.skip("configs/paths.local.json is absent", allow_module_level=True)

A = _module()

needs_baseline = pytest.mark.skipif(
    not (A.RESID_V1 / "oof_predictions.csv").is_file(),
    reason="the frozen routegate_residual_v1 baseline is not reachable on this machine")


@pytest.fixture(scope="module")
def frozen():
    rows = list(csv.DictReader(open(A.RESID_V1 / "oof_predictions.csv", newline="",
                                    encoding="utf-8")))
    lab = np.array([r["final_ebh"] for r in rows])
    return dict(
        lab=lab,
        p_E=np.array([float(r["p_E"]) for r in rows]),
        p_H=np.array([float(r["p_H"]) for r in rows]),
        is_E=lab == "E", is_B=lab == "B", is_H=lab == "H",
        thresholds=json.load(open(A.RESID_V1 / "thresholds.json")),
        gate=json.load(open(A.RESID_V1 / "gate_summary.json")),
        metrics=json.load(open(A.RESID_V1 / "metrics.json")),
    )


# residual_v1 stored p at 10 decimal places, so a threshold re-derived from that
# CSV differs from its thresholds.json in the 11th. The achieved criteria and
# every gate count are unaffected; those are asserted exactly below.
CSV_ROUNDING = 1e-9


@needs_baseline
def test_threshold_E_rule_matches_frozen(frozen):
    eb = frozen["is_E"] | frozen["is_B"]
    got = A.threshold_E(frozen["is_E"][eb].astype(int), frozen["p_E"][eb])
    assert got["threshold"] == pytest.approx(frozen["thresholds"]["threshold_E"], abs=CSV_ROUNDING)
    assert got["balanced_accuracy"] == pytest.approx(
        frozen["thresholds"]["threshold_E_balanced_accuracy"], abs=1e-12)


@needs_baseline
def test_threshold_H_rule_matches_frozen(frozen):
    got = A.threshold_H(frozen["is_H"].astype(int), frozen["p_H"])
    assert got["threshold"] == pytest.approx(frozen["thresholds"]["threshold_H"], abs=CSV_ROUNDING)
    assert got["recall"] == pytest.approx(frozen["thresholds"]["threshold_H_recall"], abs=1e-12)
    assert got["safe_specificity"] == pytest.approx(
        frozen["thresholds"]["threshold_H_safe_specificity"], abs=1e-12)
    assert got["recall"] >= A.H_RECALL_FLOOR


@needs_baseline
def test_gate_counts_match_frozen(frozen):
    eb = frozen["is_E"] | frozen["is_B"]
    tE = A.threshold_E(frozen["is_E"][eb].astype(int), frozen["p_E"][eb])["threshold"]
    tH = A.threshold_H(frozen["is_H"].astype(int), frozen["p_H"])["threshold"]

    alone = A.gate_counts(frozen["lab"], frozen["p_E"] >= tE)
    full = A.gate_counts(frozen["lab"], (frozen["p_E"] >= tE) & (frozen["p_H"] < tH))
    f1 = frozen["gate"]["ablation_1_E_sensor_alone"]
    f2 = frozen["gate"]["ablation_2_full_residual_routegate"]
    for c in ("E", "B", "H"):
        assert alone[f"fired_{c}"] == f1[f"fired_{c}"], c
        assert full[f"fired_{c}"] == f2[f"fired_{c}"], c
    assert full["precision"] == pytest.approx(f2["precision"], abs=1e-12)
    assert alone["precision"] == pytest.approx(f1["precision"], abs=1e-12)


@needs_baseline
def test_metrics_match_frozen(frozen):
    eb = frozen["is_E"] | frozen["is_B"]
    e = A.metrics_block(frozen["is_E"][eb].astype(int), frozen["p_E"][eb])
    fe = frozen["metrics"]["E_sensor_E_vs_B"]
    assert e["roc_auc"] == pytest.approx(fe["roc_auc"], abs=1e-12)
    assert e["pr_auc"] == pytest.approx(fe["pr_auc"], abs=1e-12)
    assert e["balanced_accuracy_at_0_5"] == pytest.approx(fe["balanced_accuracy_at_0.5"], abs=1e-12)
    assert e["best_balanced_accuracy"] == pytest.approx(
        fe["balanced_accuracy_at_threshold_E"], abs=1e-12)
    assert (e["n"], e["n_positive"], e["n_negative"]) == (267, 89, 178)

    h = A.metrics_block(frozen["is_H"].astype(int), frozen["p_H"])
    fh = frozen["metrics"]["H_veto_H_vs_EB"]
    assert h["roc_auc"] == pytest.approx(fh["roc_auc"], abs=1e-12)
    assert h["pr_auc"] == pytest.approx(fh["pr_auc"], abs=1e-12)
    assert h["balanced_accuracy_at_0_5"] == pytest.approx(fh["balanced_accuracy_at_0.5"], abs=1e-12)
    assert h["best_balanced_accuracy"] == pytest.approx(
        fh["balanced_accuracy_at_best_threshold"], abs=1e-12)
    assert (h["n"], h["n_positive"], h["n_negative"]) == (480, 213, 267)


# ------------------------------------------------------- rules, on toy inputs
def test_threshold_E_breaks_ties_towards_the_smallest_threshold():
    """The frozen rule is 'first argmax over ascending candidates'."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    got = A.threshold_E(y, p)
    # every t in (0.2, 0.8] separates perfectly; the candidates are the p values,
    # so the smallest maximising candidate is 0.8, not 0.9
    assert got["balanced_accuracy"] == 1.0
    assert got["threshold"] == pytest.approx(0.8)


def test_threshold_H_takes_the_largest_threshold_meeting_the_recall_floor():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])
    got = A.threshold_H(y, p)
    assert got["recall"] == 1.0                     # 0.6 keeps every positive
    assert got["threshold"] == pytest.approx(0.6)
    assert got["safe_specificity"] == 1.0


def test_threshold_H_fails_loudly_when_the_recall_floor_is_unreachable():
    y = np.array([1] * 200 + [0] * 50)
    p = np.concatenate([np.linspace(0.0, 1.0, 200), np.full(50, 1.0)])
    # every candidate above the smallest drops recall below 0.99, and the
    # smallest candidate is itself only reachable at t = 0.0; construct a case
    # with no qualifying threshold by making positives span below the negatives
    y2 = np.array([1, 1, 0, 0])
    p2 = np.array([0.0, 0.1, 0.9, 1.0])
    got = A.threshold_H(y2, p2)                     # t = 0.0 qualifies here
    assert got["recall"] == 1.0
    with pytest.raises(A.AnalysisError, match="99% H-recall floor"):
        A.threshold_H(np.array([1, 0]), np.array([np.nan, 0.5]))


def test_gate_counts_precision_is_E_over_total_fired():
    lab = np.array(["E", "E", "B", "H"])
    fired = np.array([1, 0, 1, 1])
    g = A.gate_counts(lab, fired)
    assert (g["fired_E"], g["fired_B"], g["fired_H"], g["fired_total"]) == (1, 1, 1, 3)
    assert g["precision"] == pytest.approx(1 / 3)
    assert g["rate_E"] == pytest.approx(0.5)


def test_gate_counts_precision_is_zero_when_nothing_fires():
    g = A.gate_counts(np.array(["E", "B", "H"]), np.array([0, 0, 0]))
    assert g["fired_total"] == 0 and g["precision"] == 0.0
    assert all(g[f"rate_{c}"] == 0.0 for c in "EBH")


def test_gate_counts_handles_an_absent_class_without_a_nan():
    """Degenerate input only: the frozen 480 always populates E, B and H."""
    g = A.gate_counts(np.array(["E", "B"]), np.array([1, 0]))
    assert g["fired_H"] == 0 and g["rate_H"] == 0.0
    assert not np.isnan(g["rate_H"])


def test_paired_bootstrap_is_deterministic_and_skips_single_class_replicates():
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 30)
    p_a = np.clip(y * 0.6 + rng.normal(0, 0.2, 60), 0, 1)
    p_b = np.clip(y * 0.3 + rng.normal(0, 0.2, 60), 0, 1)
    r1 = A.paired_bootstrap(y, p_a, p_b, n_rep=200, seed=7)
    r2 = A.paired_bootstrap(y, p_a, p_b, n_rep=200, seed=7)
    assert r1 == r2, "same seed must give the identical interval"
    assert r1["n_used"] + r1["n_skipped_single_class"] == 200
    assert r1["ci95_low"] <= r1["delta"] <= r1["ci95_high"] or r1["n_used"] < 200
