"""When a re-run of the coefficient controller counts as the same computation.

The rule under test replaced ``max |dg| <= 1e-9``, which refused a
reproduction that had matched every fold, every early-stopping step and every
held-out mean, and missed by two float32 epsilons. These tests hold the
replacement to the only standard that makes the correction legitimate: it must
still refuse everything the old gate would have refused *for a real reason*,
and it must refuse things the old gate never looked at.

Organised by the failure each group prevents:

    1. the constant   the budget is read off the observation, or off nothing
    2. drift          real movement in the parameters or the coefficients passes
    3. decisions      a row changes side of a boundary inside the budget
    4. structure      folds, stopping steps, thresholds or probes move
    5. actuator       the applied edit moves by something the model could see
    6. evidence       a refused run leaves nothing to inspect
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import routegate.gatecontroller as G                               # noqa: E402
import routegate.reproequivalence as E                             # noqa: E402
from routegate.validation import Checklist, ValidationError        # noqa: E402

#: What the real reproduction actually landed at: 2**-22, two float32
#: epsilons. Used to characterise the regime, never to set a tolerance.
OBSERVED_MAX = 2.384185791015625e-07

#: Gate values that sit clear of every reporting cutoff, so a test that means
#: to exercise one clause does not trip another by accident.
SAFE_BANDS = ((0.015, 0.040), (0.150, 0.220), (0.300, 0.450),
              (0.560, 0.700), (0.790, 0.870), (0.920, 0.990))
TAU = 1.5
BLOCKS = (31, 32)


def _safe_gates(n: int, seed: int = 20260828) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        low, high = SAFE_BANDS[i % len(SAFE_BANDS)]
        out.append(float(rng.uniform(low, high)))
    return np.array(out, dtype=np.float64)


def _observation(n: int = 60, seed: int = 20260828):
    """A synthetic run that reproduces itself exactly, and its own record."""
    gates = _safe_gates(n, seed)
    row_ids = [f"row:{i:04d}" for i in range(n)]
    folds = {rid: i % G.N_SPLITS for i, rid in enumerate(row_ids)}
    labels = {rid: ("E", "B", "H")[i % 3] for i, rid in enumerate(row_ids)}
    gate = {rid: float(gates[i]) for i, rid in enumerate(row_ids)}
    # every third row is vetoed, and the clamp binds on every fourth; both
    # decision surfaces sit far from where any coefficient lands.
    vetoed = {rid: (i % 3 == 0) for i, rid in enumerate(row_ids)}
    ceiling = {rid: (gate[rid] - 0.05 if i % 4 == 0 else gate[rid] + 0.05)
               for i, rid in enumerate(row_ids)}
    z_safety = {rid: (TAU + 1.0 if vetoed[rid] else TAU - 1.0)
                for rid in row_ids}
    saved = {rid: {"fold": folds[rid], "g_R": gate[rid],
                   "g_R_after_veto": 0.0 if vetoed[rid] else gate[rid],
                   "g_R_after_clamp": min(gate[rid], ceiling[rid]),
                   "vetoed": vetoed[rid], "final_ebh": labels[rid],
                   "z_safety_before": z_safety[rid]}
             for rid in row_ids}
    steps = {k: 300 + 7 * k for k in range(G.N_SPLITS)}
    tau = {k: TAU for k in range(G.N_SPLITS)}
    digests = {"oof_predictions.csv": "a" * 64, "summary.json": "b" * 64}
    budgets = {31: np.full(n, 25.0), 32: np.full(n, 26.0)}
    written = {31: np.full(n, 26.0), 32: np.full(n, 27.0)}
    return dict(row_ids=row_ids, saved=saved, reproduced_gate=dict(gate),
                reproduced_fold=dict(folds), reproduced_label=dict(labels),
                reproduced_vetoed=dict(vetoed),
                reproduced_clamp_ceiling=dict(ceiling),
                reproduced_safety_logit=dict(z_safety),
                saved_steps=dict(steps), reproduced_steps=dict(steps),
                saved_tau=dict(tau), reproduced_tau=dict(tau),
                pinned_probe_digests=dict(digests),
                actual_probe_digests=dict(digests),
                budgets=budgets, written_state_norms=written,
                alpha=G.ALPHA_R)


def _failures(report):
    return {c.name for c in report.failures}


def _nudge(observation, delta):
    """Move every reproduced coefficient by ``delta``, alternating sign."""
    out = dict(observation)
    gate = dict(observation["reproduced_gate"])
    for i, rid in enumerate(observation["row_ids"]):
        gate[rid] = gate[rid] + (delta if i % 2 == 0 else -delta)
    out["reproduced_gate"] = gate
    return out


# =============================================== 1. where the constant comes from
def test_the_budget_is_stated_in_float32_epsilons_not_in_decimal():
    assert E.GATE_ABS_TOLERANCE == E.GATE_EPSILON_BUDGET * E.FLOAT32_EPS
    assert E.FLOAT32_EPS == 2.0 ** -23
    # a whole number of epsilons, so the constant means the same thing on any
    # machine that has float32 -- which a decimal constant does not
    assert float(E.GATE_ABS_TOLERANCE / E.FLOAT32_EPS).is_integer()


def test_the_superseded_gate_asked_for_more_than_float32_can_represent():
    """1e-9 is about a hundredth of one float32 ULP at the top of the range."""
    ulp_at_one = E.float32_ulp(1.0)
    assert 1e-9 < ulp_at_one / 100.0
    # so no two distinct float32 gates can ever be that close without being
    # the same number: the old gate could only pass a bit-identical re-run
    assert E.float32_ulp(0.9) > 1e-9 * 50


def test_the_budget_is_not_the_observed_maximum():
    assert E.GATE_ABS_TOLERANCE != OBSERVED_MAX
    assert E.GATE_ABS_TOLERANCE > OBSERVED_MAX
    # comfortably above the regime that was observed, so the rule is not a
    # fingerprint of the node the reproduction first ran on
    assert E.GATE_ABS_TOLERANCE / OBSERVED_MAX >= 8.0
    assert E.describe_tolerance()["derived_from_the_observed_maximum"] is False


def test_the_rule_refuses_itself_on_a_corpus_that_decides_too_finely():
    tight = {"gate_space_margin": E.GATE_ABS_TOLERANCE * 0.5,
             "gate_space_binding_boundary": "counterfactual_clamp",
             "safety_veto_logit_margin": 1.0}
    with pytest.raises(ValidationError) as exc:
        E.assert_rule_is_admissible(tight, Checklist("t"))
    assert "admissible" in str(exc.value)

    roomy = {"gate_space_margin": 1e-3,
             "gate_space_binding_boundary": "reporting_cutoff",
             "safety_veto_logit_margin": 1.0}
    assert E.assert_rule_is_admissible(roomy, Checklist("t"))["admissible"]


def test_the_budget_may_be_tightened_and_never_loosened():
    assert E.assert_tolerance_is_not_loosened(E.GATE_ABS_TOLERANCE / 4)
    with pytest.raises(ValidationError) as exc:
        E.assert_tolerance_is_not_loosened(E.GATE_ABS_TOLERANCE * 2)
    assert "looser than the declared rule" in str(exc.value)


def test_the_superseded_rule_and_its_result_are_recorded():
    assert E.SUPERSEDED_RULES
    record = E.SUPERSEDED_RULES[0]
    assert "1e-9" in record["rule"]
    assert record["inherited_from_a_frozen_protocol"] is False
    assert record["outcome"].startswith("REFUSED")
    assert record["observed_max_abs_coefficient_error"] == OBSERVED_MAX
    assert record["observed_in_float32_epsilons"] == 2.0
    assert "1571" in record["run"] and "280" in record["run"]


def test_the_cutoff_grid_is_the_trainer_s_own():
    import importlib.util
    path = (REPO_ROOT / "experiments" / "single_direction_controller"
            / "train_gate_controller.py")
    spec = importlib.util.spec_from_file_location("_trainer_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(module.CUTOFFS) == E.REPORTING_CUTOFFS


# ============================================================= 2. genuine drift
def test_a_float32_equivalent_replay_passes():
    """The regime the real reproduction landed in, and one at the budget."""
    report = E.evaluate(**_nudge(_observation(), OBSERVED_MAX / 2))
    assert report.passed, _failures(report)
    assert report.coefficient_error["max"] <= E.GATE_ABS_TOLERANCE
    assert report.coefficient_error["max_in_float32_epsilons"] == pytest.approx(1.0)

    at_budget = E.evaluate(**_nudge(_observation(), E.GATE_ABS_TOLERANCE))
    assert at_budget.passed, _failures(at_budget)


def test_a_bit_identical_replay_passes_and_says_so():
    report = E.evaluate(**_observation())
    assert report.passed, _failures(report)
    assert report.coefficient_error["max"] == 0.0
    assert report.coefficient_error["rows_with_any_difference"] == 0
    assert report.ordering["exact_global_order_preserved"] is True


def test_a_coefficient_one_epsilon_past_the_budget_fails():
    over = _nudge(_observation(), E.GATE_ABS_TOLERANCE + E.FLOAT32_EPS)
    report = E.evaluate(**over)
    assert not report.passed
    assert "coefficients_agree_within_the_float32_budget" in _failures(report)


def test_genuine_coefficient_drift_fails():
    report = E.evaluate(**_nudge(_observation(), 1e-4))
    assert not report.passed
    assert "coefficients_agree_within_the_float32_budget" in _failures(report)
    assert report.coefficient_error["rows_above_the_tolerance"] == 60


def test_genuine_parameter_drift_fails():
    """A real controller, a real weight perturbation, the real forward pass."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20260828)
    n = 40
    x = rng.normal(size=(n, G.HIDDEN_SIZE))
    net = G.build_controller(G.seed_for(0))
    with torch.no_grad():
        # a trained controller has a non-zero second layer; the initialiser
        # zeroes it, which would make every gate identical
        net[3].weight.copy_(torch.as_tensor(
            rng.normal(scale=0.5, size=tuple(net[3].weight.shape)),
            dtype=torch.float32))
    net.eval()
    inputs = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        before = G.gate(net, inputs).numpy().astype(np.float64)
        # one weight moves by 1e-4 -- far below anything a training run would
        # call a difference, and far above anything float32 rounding produces
        net[3].weight[0, 0] += 1e-4
        after = G.gate(net, inputs).numpy().astype(np.float64)

    drift = float(np.abs(after - before).max())
    assert drift > E.GATE_ABS_TOLERANCE, "the perturbation must actually move g"

    base = _observation(n=n)
    ids = base["row_ids"]
    base["saved"] = {rid: dict(base["saved"][rid], g_R=float(before[i]),
                               g_R_after_clamp=float(before[i]))
                     for i, rid in enumerate(ids)}
    base["reproduced_gate"] = {rid: float(after[i]) for i, rid in enumerate(ids)}
    base["reproduced_clamp_ceiling"] = {rid: 1.0 for rid in ids}
    report = E.evaluate(**base)
    assert not report.passed
    assert "coefficients_agree_within_the_float32_budget" in _failures(report)


# ================================================== 3. decisions inside the budget
def test_a_reporting_bucket_change_fails_even_inside_the_numeric_budget():
    base = _observation()
    rid = base["row_ids"][0]
    step = E.GATE_ABS_TOLERANCE / 4.0
    base["saved"][rid] = dict(base["saved"][rid], g_R=0.50 + step,
                              g_R_after_clamp=0.50 + step)
    base["reproduced_gate"][rid] = 0.50 - step
    base["reproduced_clamp_ceiling"][rid] = 1.0
    report = E.evaluate(**base)
    # the numeric clause is satisfied and the run still fails
    assert report.coefficient_error["max"] <= E.GATE_ABS_TOLERANCE
    assert not report.passed
    assert "no_row_changes_reporting_bucket" in _failures(report)
    assert "no_row_changes_intervention_decision" in _failures(report)


def test_a_veto_decision_change_fails():
    base = _observation()
    rid = base["row_ids"][0]
    base["reproduced_vetoed"][rid] = not base["saved"][rid]["vetoed"]
    report = E.evaluate(**base)
    assert not report.passed
    assert "no_row_changes_veto_decision" in _failures(report)


def test_a_clamp_decision_change_fails():
    base = _observation()
    rid = base["row_ids"][0]          # a row the clamp binds on in the record
    assert base["saved"][rid]["g_R_after_clamp"] < base["saved"][rid]["g_R"]
    base["reproduced_clamp_ceiling"][rid] = 1.0     # it no longer binds
    report = E.evaluate(**base)
    assert not report.passed
    assert "no_row_changes_clamp_decision" in _failures(report)


def test_an_inversion_the_tolerance_can_separate_fails():
    """Two rows swap order while every row stays inside the numeric budget."""
    base = _observation()
    a, b = base["row_ids"][1], base["row_ids"][7]
    tol = E.GATE_ABS_TOLERANCE
    lo, hi = 0.30, 0.30 + 1.5 * tol
    for rid, saved_value, repro_value in ((a, lo, lo + tol),
                                          (b, hi, hi - tol)):
        base["saved"][rid] = dict(base["saved"][rid], g_R=saved_value,
                                  g_R_after_clamp=saved_value)
        base["reproduced_gate"][rid] = repro_value
        base["reproduced_clamp_ceiling"][rid] = 1.0
    report = E.evaluate(**base)
    assert report.coefficient_error["max"] <= tol
    assert not report.passed
    assert "no_inversion_the_tolerance_can_see" in _failures(report)


def test_inversions_below_the_tolerance_inside_one_bucket_are_not_a_failure():
    """480 real coefficients include pairs 1e-16 apart; exact order is not claimed."""
    base = _observation()
    a, b = base["row_ids"][1], base["row_ids"][7]
    tiny = E.GATE_ABS_TOLERANCE / 1000.0
    base["saved"][a] = dict(base["saved"][a], g_R=0.30, g_R_after_clamp=0.30)
    base["saved"][b] = dict(base["saved"][b], g_R=0.30 + tiny,
                            g_R_after_clamp=0.30 + tiny)
    base["reproduced_gate"][a] = 0.30 + tiny
    base["reproduced_gate"][b] = 0.30
    base["reproduced_clamp_ceiling"][a] = 1.0
    base["reproduced_clamp_ceiling"][b] = 1.0
    report = E.evaluate(**base)
    assert report.ordering["inversions"] > 0
    assert report.ordering["unsafe_inversions"] == 0
    assert report.passed, _failures(report)


# ================================================================= 4. structure
def test_a_changed_fold_assignment_fails():
    base = _observation()
    rid = base["row_ids"][0]
    base["reproduced_fold"][rid] = (base["saved"][rid]["fold"] + 1) % G.N_SPLITS
    report = E.evaluate(**base)
    assert not report.passed
    assert "fold_assignment_is_identical" in _failures(report)


def test_a_changed_stopping_step_fails():
    base = _observation()
    base["reproduced_steps"][2] = base["saved_steps"][2] + 1
    report = E.evaluate(**base)
    assert not report.passed
    assert "early_stopping_steps_are_identical" in _failures(report)
    assert 2 in report.structure["selected_step_disagreements"]


def test_a_changed_label_fails():
    base = _observation()
    rid = base["row_ids"][0]
    base["reproduced_label"][rid] = "H" if base["saved"][rid]["final_ebh"] != "H" else "E"
    report = E.evaluate(**base)
    assert not report.passed
    assert "labels_are_identical" in _failures(report)


def test_a_changed_veto_threshold_fails_and_a_one_ulp_wobble_does_not():
    base = _observation()
    base["reproduced_tau"][1] = TAU + 1e-3
    report = E.evaluate(**base)
    assert not report.passed
    assert "veto_thresholds_agree_within_the_ulp_budget" in _failures(report)

    wobble = _observation()
    wobble["reproduced_tau"][1] = TAU + E.float32_ulp(TAU)
    assert E.evaluate(**wobble).passed


def test_a_changed_probe_digest_fails():
    base = _observation()
    base["actual_probe_digests"]["oof_predictions.csv"] = "c" * 64
    report = E.evaluate(**base)
    assert not report.passed
    assert "probe_digests_are_identical" in _failures(report)


def test_a_missing_row_fails_rather_than_being_compared_around():
    base = _observation()
    base["saved"].pop(base["row_ids"][0])
    report = E.evaluate(**base)
    assert not report.passed
    assert "row_set_matches" in _failures(report)


def test_checkpoints_that_score_differently_fail():
    base = _observation()
    base["checkpoint_gate"] = {rid: base["reproduced_gate"][rid]
                               for rid in base["row_ids"]}
    assert E.evaluate(**base).passed
    base["checkpoint_gate"][base["row_ids"][0]] += 1e-3
    report = E.evaluate(**base)
    assert not report.passed
    assert "the_written_checkpoints_score_the_corpus_the_same_way" in _failures(report)


# ================================================================== 5. actuator
def test_the_edit_clause_binds_when_the_budget_is_large_enough_to_matter():
    base = _nudge(_observation(), E.GATE_ABS_TOLERANCE)
    n = len(base["row_ids"])
    base["budgets"] = {31: np.full(n, 1.0e5), 32: np.full(n, 1.0e5)}
    base["written_state_norms"] = {31: np.full(n, 1.0), 32: np.full(n, 1.0)}
    report = E.evaluate(**base)
    assert not report.passed
    assert "the_applied_edit_is_below_the_actuator_resolution" in _failures(report)


def test_the_edit_difference_is_reported_in_the_actuator_s_own_dtype():
    n = 8
    edit = E.edit_difference(OBSERVED_MAX, G.ALPHA_R,
                             {31: np.full(n, 25.5), 32: np.full(n, 26.4)},
                             {31: np.full(n, 26.4), 32: np.full(n, 27.4)})
    assert edit["max_edit_norm_difference"] == pytest.approx(
        G.ALPHA_R * OBSERVED_MAX * 26.4)
    assert edit["max_edit_difference_in_bf16_ulps"] < E.EDIT_BF16_ULP_TOLERANCE
    # the gate would have to move ~2.6e-2 to shift the written state by one
    # bfloat16 step -- five orders of magnitude above anything at issue here
    assert edit["by_site"]["31"]["gate_difference_for_one_bf16_ulp"] > 1e-2


def test_the_veto_does_not_read_the_gate():
    margins = E.decision_margins([0.4, 0.6], [1.0, 1.0], [0.5, 2.5],
                                 [1.5, 1.5])
    assert margins["safety_veto_reads_the_gate"] is False
    assert margins["safety_veto_logit_margin"] == pytest.approx(1.0)


# ================================================================== 6. evidence
def test_a_refused_comparison_is_written_before_anything_returns(tmp_path):
    report = E.evaluate(**_nudge(_observation(), 1e-4))
    assert not report.passed
    path = E.write_diagnostic(tmp_path, report, {"reproduces": "gate_oof_v1"})
    assert path.name == E.DIAGNOSTIC_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "coefficients_agree_within_the_float32_budget" in payload["failing_clauses"]
    # the numbers a future reader needs to diagnose it, not just the verdict
    error = payload["coefficient_error"]
    assert error["rows_compared"] == 60
    assert set(error["quantiles"]) >= {"p50", "p95", "p99"}
    assert payload["per_fold"] and len(payload["per_fold"]) == G.N_SPLITS
    assert payload["decision_margins"]["gate_space_margin"] > 0
    assert payload["edit_difference"]["by_site"]
    assert payload["superseded_rules"][0]["outcome"].startswith("REFUSED")
    assert payload["context"]["reproduces"] == "gate_oof_v1"


def test_the_report_carries_every_number_the_audit_asked_for():
    report = E.evaluate(**_nudge(_observation(), OBSERVED_MAX / 2))
    error = report.coefficient_error
    for key in ("max", "mean", "median", "quantiles", "rows_with_any_difference",
                "rows_above_one_float32_eps", "max_in_float32_epsilons"):
        assert key in error
    assert len(report.per_fold) == G.N_SPLITS
    for entry in report.per_fold:
        assert "max_abs_error" in entry and "saved_selected_step" in entry
    assert report.edit_difference["max_applied_alpha_difference"] == pytest.approx(
        G.ALPHA_R * error["max"])


def test_the_runner_holds_no_tolerance_constant_of_its_own():
    """The 1e-9 lived in the runner. Nothing numeric may live there again."""
    source = (REPO_ROOT / "experiments" / "single_direction_controller"
              / "reproduce_gate_controller.py").read_text(encoding="utf-8")
    import re
    # the module-level binding, not the docstring that explains the history
    assert re.search(r"^COEFFICIENT_TOLERANCE = GATE_ABS_TOLERANCE$",
                     source, re.M)
    assert re.search(r"^COEFFICIENT_TOLERANCE\s*=\s*1e-0*9", source, re.M) is None
    # the supersession stays documented where the runner is read
    assert "COEFFICIENT_TOLERANCE = 1e-9" in source
    # the runner must reach the rule, not restate it
    assert "assert_tolerance_is_not_loosened" in source
    assert "write_diagnostic" in source


def test_the_correction_is_in_the_decision_log():
    log = (REPO_ROOT / "docs" / "decision_log.md").read_text(encoding="utf-8")
    entry = log[log.index("## 2026-08-28 — The controller reproduction"):]
    assert "1e-9" in entry
    assert "2.384186e-07" in entry
    assert "1571" in entry and "280" in entry
    assert "2.368883e-05" in entry          # the margin the budget is held to
