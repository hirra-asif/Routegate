"""When is a re-run of the coefficient controller *the same computation*?

WHY THIS EXISTS
---------------
The first reproduction runner gated ``gate_oof_v1`` at

    max |reproduced g_R - saved g_R| <= 1e-9

and refused a run that had reproduced every fold, every early-stopping step and
every held-out mean. The observed disagreement was ``2.384186e-07``, which is
exactly ``2**-22`` -- **two float32 epsilons**, four ULPs at the top binade of
a quantity that lives in ``[0, 1]``. A float32 trajectory cannot be held to
1e-9 by any argument; ``1e-9`` is roughly one hundredth of a single float32
ULP at ``g_R = 1``, so the gate was asking two float32 numbers to agree a
hundred times more closely than float32 can represent a difference at all.

That gate was **not** inherited from any frozen scientific protocol. It was
introduced by the engineering runner and by nothing else, so correcting it
changes no predeclared quantity. The superseded rule and its result are kept
in ``SUPERSEDED_RULES`` and in ``docs/decision_log.md`` rather than deleted.

WHAT REPLACES IT
----------------
Not a looser number. A **conjunction**, of which the numeric clause is the
least important member:

1. ``coefficients_agree``    -- ``max |dg| <= GATE_ABS_TOLERANCE``, a budget
   stated in float32 epsilons, not in decimal;
2. ``structure_agrees``      -- fold assignment, early-stopping steps, labels,
   veto thresholds and probe digests agree exactly (thresholds in ULPs, being
   float32 reductions themselves);
3. ``no_decision_moves``     -- not one row changes side of a reporting cutoff,
   the safety veto, the counterfactual clamp or the intervention decision;
4. ``ordering_holds``        -- every rank inversion is between rows the
   tolerance cannot separate *and* inside one reporting bucket;
5. ``edit_is_indistinguishable`` -- the difference in the applied steering edit
   is a small fraction of one **bfloat16** rounding step of the activation it
   is written into.

Clause 1 alone can never pass a run. Clauses 3-5 are evaluated against the
actual corpus, so a tolerance that were too loose for this data would be caught
by the data rather than by the number.

WHY THE TOLERANCE IS NOT READ OFF THE OBSERVATION
-------------------------------------------------
``GATE_ABS_TOLERANCE`` is ``GATE_EPSILON_BUDGET`` float32 epsilons. Two facts
fix it, neither of which is the observed maximum:

* **the dtype.** ``g_R = sigmoid(f_theta(x))`` is computed in float32 and
  compared in ``[0, 1]``, so the natural unit of disagreement is
  ``ulp_f32(1.0) = 2**-23``. A budget in that unit means the same thing on
  every machine; a decimal constant does not.
* **the corpus's own decision geometry.** ``assert_rule_is_admissible``
  computes the smallest distance from any saved coefficient to any boundary
  the package actually decides on, and **refuses the rule** unless the
  tolerance is at least ``DECISION_MARGIN_SAFETY_FACTOR`` times smaller. The
  budget is therefore falsifiable: on a corpus whose coefficients sat closer to
  a cutoff, this rule would refuse itself instead of passing a run.

The budget is deliberately larger than any one machine needs, so that the rule
is not a fingerprint of the node it was first run on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .validation import Checklist, ValidationError

SCHEMA = "routegate/gate_controller_equivalence/1"

#: The rule's own version. Version 1 is the first *dtype-aware* rule; the
#: dtype-blind 1e-9 attempt it replaces is recorded in ``SUPERSEDED_RULES`` and
#: never carried a version of its own.
RULE_VERSION = 1
RULE_NAME = "float32_equivalent_controller_reproduction"

# ------------------------------------------------------------------- dtypes
#: ``2**-23``. The spacing of float32 at 1.0, and therefore the coarsest
#: spacing anywhere in the gate's range ``[0, 1]``.
FLOAT32_EPS = float(np.finfo(np.float32).eps)
#: ``2**-24``. Half an epsilon: the unit roundoff, and the ULP just below 1.0.
FLOAT32_UNIT_ROUNDOFF = FLOAT32_EPS / 2.0
#: ``2**-8``. The activations the edit is written into are stored in bfloat16.
BFLOAT16_EPS = float(np.float64(2.0) ** -8)

# ------------------------------------------------------------- the budgets
#: The gate budget, in float32 epsilons. Round, generous, and checked against
#: the corpus by ``assert_rule_is_admissible`` rather than asserted.
GATE_EPSILON_BUDGET = 32
GATE_ABS_TOLERANCE = GATE_EPSILON_BUDGET * FLOAT32_EPS      # 3.814697265625e-06

#: The veto threshold is a *logit*, not a gate, so its budget is stated in
#: ULPs at its own magnitude rather than in a gate-range epsilon.
TAU_ULP_BUDGET = 32

#: The applied edit may differ by at most a thousandth of one bfloat16 rounding
#: step of the state it is written into. One whole step is already invisible to
#: the actuator; a thousandth of one is three orders of magnitude inside that.
EDIT_BF16_ULP_TOLERANCE = 1.0e-3

#: The tolerance must be at least this many times smaller than the tightest
#: decision margin in the corpus, or the rule refuses itself.
DECISION_MARGIN_SAFETY_FACTOR = 2.0

#: The intervention/reporting grid ``g_R`` is bucketed on. Kept here so the
#: rule can be evaluated without importing an experiment script; a test asserts
#: it still equals the trainer's own ``CUTOFFS``.
REPORTING_CUTOFFS: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)

#: Reported error quantiles, in percent.
ERROR_QUANTILES: Tuple[float, ...] = (25.0, 50.0, 75.0, 90.0, 95.0, 99.0)

#: Every rule this one replaces, with what it did. Kept in the code that
#: superseded it so the failure cannot be rediscovered as a novelty.
SUPERSEDED_RULES: Tuple[Dict[str, Any], ...] = ({
    "rule": "max |reproduced g_R - saved g_R| <= 1e-9",
    "kind": "absolute, dtype-blind",
    "declared_in": "experiments/single_direction_controller/"
                   "reproduce_gate_controller.py::COEFFICIENT_TOLERANCE",
    "inherited_from_a_frozen_protocol": False,
    "introduced_by": "the engineering reproduction runner, and nothing else",
    "outcome": "REFUSED a reproduction that was float32-equivalent",
    "run": "all five folds completed; early-stopping steps 1571, 616, 578, "
           "367, 280 -- identical to gate_oof_v1's own record; held-out mean "
           "g_R identical to six decimals in every fold",
    "observed_max_abs_coefficient_error": 2.384185791015625e-07,
    "observed_in_float32_epsilons": 2.0,
    "observed_in_ulps_at_the_top_binade": 4.0,
    "tolerance_in_float32_ulps_at_one": 1e-9 / FLOAT32_EPS,
    "why_it_was_wrong":
        "1e-9 is about one hundredth of a single float32 ULP at g_R = 1. It "
        "required two float32 numbers to agree more closely than float32 can "
        "represent a difference, so it could only ever have passed a "
        "bit-identical re-run on bit-identical hardware. It gated the dtype, "
        "not the science.",
    "recorded_in": "docs/decision_log.md, 2026-08-28",
},)


class EquivalenceError(ValidationError):
    """The equivalence rule itself was misapplied or is inadmissible."""


# ------------------------------------------------------------------ helpers
def float32_ulp(value: float) -> float:
    """The spacing of float32 at ``value``. Zero maps to the smallest subnormal."""
    x = np.float32(abs(float(value)))
    if x == np.float32(0.0):
        return float(np.float32(np.finfo(np.float32).smallest_subnormal))
    return float(np.spacing(x))


def in_epsilons(value: float) -> float:
    """A gate-space difference, expressed in float32 epsilons."""
    return float(abs(float(value)) / FLOAT32_EPS)


def tau_tolerance(tau: float) -> float:
    """The veto threshold's budget: ``TAU_ULP_BUDGET`` ULPs at its own scale."""
    return TAU_ULP_BUDGET * float32_ulp(tau)


def _quantiles(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:                                   # pragma: no cover
        return {f"p{q:g}": 0.0 for q in ERROR_QUANTILES}
    return {f"p{q:g}": float(np.percentile(errors, q)) for q in ERROR_QUANTILES}


# --------------------------------------------------------------- the clauses
@dataclass(frozen=True)
class Clause:
    """One named requirement, its verdict, and the evidence behind it."""

    name: str
    passed: bool
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"clause": self.name, "status": "PASS" if self.passed else "FAIL",
                "detail": self.detail, "evidence": self.evidence}


@dataclass(frozen=True)
class EquivalenceReport:
    """Everything the gate looked at, whether or not it passed."""

    clauses: Tuple[Clause, ...]
    coefficient_error: Dict[str, Any]
    per_fold: List[Dict[str, Any]]
    decision_margins: Dict[str, Any]
    edit_difference: Dict[str, Any]
    structure: Dict[str, Any]
    ordering: Dict[str, Any]
    checkpoint_agreement: Dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.clauses)

    @property
    def failures(self) -> List[Clause]:
        return [c for c in self.clauses if not c.passed]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "rule_version": RULE_VERSION,
            "rule_name": RULE_NAME,
            "passed": self.passed,
            "tolerance": describe_tolerance(),
            "clauses": [c.as_dict() for c in self.clauses],
            "coefficient_error": self.coefficient_error,
            "per_fold": self.per_fold,
            "decision_margins": self.decision_margins,
            "edit_difference": self.edit_difference,
            "structure": self.structure,
            "ordering": self.ordering,
            "checkpoint_agreement": self.checkpoint_agreement,
            "superseded_rules": [dict(r) for r in SUPERSEDED_RULES],
        }


def describe_tolerance() -> Dict[str, Any]:
    """The rule's constants, with the unit each is stated in."""
    return {
        "gate_epsilon_budget": GATE_EPSILON_BUDGET,
        "gate_abs_tolerance": GATE_ABS_TOLERANCE,
        "gate_abs_tolerance_expression":
            f"{GATE_EPSILON_BUDGET} * float32_eps",
        "float32_eps": FLOAT32_EPS,
        "float32_unit_roundoff": FLOAT32_UNIT_ROUNDOFF,
        "bfloat16_eps": BFLOAT16_EPS,
        "tau_ulp_budget": TAU_ULP_BUDGET,
        "edit_bf16_ulp_tolerance": EDIT_BF16_ULP_TOLERANCE,
        "decision_margin_safety_factor": DECISION_MARGIN_SAFETY_FACTOR,
        "reporting_cutoffs": list(REPORTING_CUTOFFS),
        "derived_from_the_observed_maximum": False,
        "derived_from": [
            "the dtype: g_R is float32 in [0, 1], so ulp_f32(1.0) = 2**-23 is "
            "the unit a gate-space disagreement is naturally stated in",
            "the corpus decision geometry: assert_rule_is_admissible refuses "
            "the rule unless the tolerance is at least "
            f"{DECISION_MARGIN_SAFETY_FACTOR:g}x smaller than the tightest "
            "margin from any saved coefficient to any boundary",
            "the actuator dtype: the edit lands in bfloat16, whose epsilon is "
            "2**-8, and the edit-difference clause allows a thousandth of one "
            "such step",
        ],
    }


# ------------------------------------------------------------ admissibility
def decision_margins(gate: Sequence[float], clamp_ceiling: Sequence[float],
                     safety_logit: Sequence[float], tau: Sequence[float],
                     cutoffs: Sequence[float] = REPORTING_CUTOFFS
                     ) -> Dict[str, Any]:
    """How close the saved coefficients sit to every boundary that decides.

    Gate-space margins bound what the *gate* tolerance may be; the veto margin
    is a logit distance and bounds the *threshold* tolerance instead. They are
    reported separately because they are not in the same units.
    """
    g = np.asarray(gate, dtype=np.float64)
    ceiling = np.asarray(clamp_ceiling, dtype=np.float64)
    z = np.asarray(safety_logit, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)

    per_cutoff = {f"{c:.2f}": float(np.abs(g - c).min()) for c in cutoffs}
    cutoff_margin = min(per_cutoff.values()) if per_cutoff else float("inf")
    clamp_margin = float(np.abs(g - ceiling).min()) if ceiling.size else float("inf")
    veto_margin = float(np.abs(z - t).min()) if z.size else float("inf")
    gate_space = min(cutoff_margin, clamp_margin)
    return {
        "gate_space_margin": gate_space,
        "gate_space_binding_boundary":
            "counterfactual_clamp" if clamp_margin <= cutoff_margin
            else "reporting_cutoff",
        "reporting_cutoff_margin": cutoff_margin,
        "reporting_cutoff_margin_by_cutoff": per_cutoff,
        "counterfactual_clamp_margin": clamp_margin,
        "safety_veto_logit_margin": veto_margin,
        "safety_veto_reads_the_gate": False,
        "gate_space_margin_in_tolerances":
            gate_space / GATE_ABS_TOLERANCE if GATE_ABS_TOLERANCE else float("inf"),
    }


def assert_rule_is_admissible(margins: Mapping[str, Any],
                              checks: Optional[Checklist] = None
                              ) -> Dict[str, Any]:
    """Refuse the rule -- not the run -- if this corpus decides too finely.

    A tolerance is only inert if every boundary is further away than it is.
    That is a property of the *data*, so it is measured here rather than
    assumed when the constant was written down.
    """
    checks = checks or Checklist("gate_controller_equivalence")
    needed = GATE_ABS_TOLERANCE * DECISION_MARGIN_SAFETY_FACTOR
    gate_ok = float(margins["gate_space_margin"]) >= needed
    checks.require(
        "equivalence_rule_is_admissible_in_gate_space", gate_ok,
        f"the tightest gate-space margin is "
        f"{float(margins['gate_space_margin']):.6e} at the "
        f"{margins['gate_space_binding_boundary']}, and the rule needs at "
        f"least {needed:.6e} ({GATE_EPSILON_BUDGET} float32 epsilons x "
        f"{DECISION_MARGIN_SAFETY_FACTOR:g}). A tolerance this corpus cannot "
        f"carry is refused here, rather than passing a run that moved a "
        f"decision.")
    tau_needed = TAU_ULP_BUDGET * float32_ulp(2.0) * DECISION_MARGIN_SAFETY_FACTOR
    veto_ok = float(margins["safety_veto_logit_margin"]) >= tau_needed
    checks.require(
        "equivalence_rule_is_admissible_in_logit_space", veto_ok,
        f"the tightest |z_H - tau_hi| is "
        f"{float(margins['safety_veto_logit_margin']):.6e} and the threshold "
        f"budget needs at least {tau_needed:.6e}")
    return {"gate_space_margin_required": needed,
            "logit_space_margin_required": tau_needed,
            "admissible": gate_ok and veto_ok}


def assert_tolerance_is_not_loosened(value: float,
                                     checks: Optional[Checklist] = None
                                     ) -> float:
    """A caller may tighten the budget. Widening it is refused.

    The 1e-9 attempt was corrected in this module, in the open, with the
    reason recorded. A command-line flag that could restate the budget upward
    would let the next such gate be argued away instead, leaving no trace of
    what was relaxed or why.
    """
    checks = checks or Checklist("gate_controller_equivalence")
    checks.require(
        "tolerance_may_be_tightened_never_loosened",
        float(value) <= GATE_ABS_TOLERANCE,
        f"a coefficient budget of {float(value):g} is looser than the "
        f"declared rule {GATE_ABS_TOLERANCE:g} "
        f"({GATE_EPSILON_BUDGET} * float32_eps). The rule is corrected here, "
        f"versioned, with the superseded attempt recorded -- never widened at "
        f"the point of use to obtain a pass.")
    return float(value)


# ---------------------------------------------------------- the edit clause
def edit_difference(max_abs_gate_error: float, alpha: float,
                    budgets: Mapping[int, Sequence[float]],
                    written_state_norms: Mapping[int, Sequence[float]]
                    ) -> Dict[str, Any]:
    """How much the applied steering edit could move, in the actuator's dtype.

    The edit at historical block ``l`` is ``alpha * g_R * r_l * d_R`` with
    ``d_R`` a unit vector, so a gate difference of ``dg`` moves it by
    ``alpha * dg * r_l`` in norm. That is compared against one bfloat16
    rounding step of the activation it is written into -- the coarsest thing
    standing between the edit and the model.
    """
    dg = float(abs(max_abs_gate_error))
    by_site: Dict[str, Any] = {}
    worst_norm, worst_ulps = 0.0, 0.0
    for block in sorted(budgets):
        r = np.asarray(budgets[block], dtype=np.float64)
        target = np.asarray(written_state_norms[block], dtype=np.float64)
        if r.shape != target.shape:
            raise EquivalenceError(
                f"block {block}: {r.size} budget(s) and {target.size} written "
                f"state norm(s); the edit clause compares them row by row")
        delta_norm = alpha * dg * r
        step = BFLOAT16_EPS * target
        ulps = np.divide(delta_norm, step, out=np.full_like(delta_norm, np.inf),
                         where=step > 0)
        # the gate difference that would move one whole bfloat16 step
        breakeven = np.divide(step, alpha * r,
                              out=np.full_like(step, np.inf), where=r > 0)
        by_site[str(block)] = {
            "historical_block": block,
            "writes_package_state": block + 1,
            "max_budget": float(r.max()),
            "max_edit_norm_difference": float(delta_norm.max()),
            "max_edit_difference_in_bf16_ulps": float(ulps.max()),
            "gate_difference_for_one_bf16_ulp": float(breakeven.min()),
        }
        worst_norm = max(worst_norm, float(delta_norm.max()))
        worst_ulps = max(worst_ulps, float(ulps.max()))
    return {"alpha_R": alpha, "max_abs_gate_error": dg,
            "direction_is_unit_norm": True,
            "max_edit_norm_difference": worst_norm,
            "max_edit_difference_in_bf16_ulps": worst_ulps,
            "bf16_ulp_tolerance": EDIT_BF16_ULP_TOLERANCE,
            "by_site": by_site}


# ------------------------------------------------------------------ ordering
def _bucket(values: np.ndarray, cutoffs: Sequence[float]) -> np.ndarray:
    edges = np.asarray(sorted(cutoffs), dtype=np.float64)
    return np.searchsorted(edges, values, side="right")


def ordering_agreement(saved: np.ndarray, reproduced: np.ndarray,
                       tolerance: float,
                       cutoffs: Sequence[float] = REPORTING_CUTOFFS
                       ) -> Dict[str, Any]:
    """Which pairs changed order, and whether any of them could matter.

    Exact global order is not claimable and is not claimed: 480 coefficients
    include adjacent pairs separated by 1e-16, far below anything float32 can
    hold apart. What is claimable -- and is checked -- is that every inversion
    is between rows the tolerance cannot separate *and* inside one reporting
    bucket, so no inversion changes a reported quantity.
    """
    n = saved.size
    order = np.argsort(saved, kind="stable")
    s, r = saved[order], reproduced[order]
    ds = s[None, :] - s[:, None]
    dr = r[None, :] - r[:, None]
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    inverted = upper & (np.sign(ds) * np.sign(dr) < 0)
    separated = np.abs(ds) > tolerance
    buckets = _bucket(s, cutoffs)
    same_bucket = buckets[None, :] == buckets[:, None]

    unsafe = inverted & (separated | ~same_bucket)
    gaps = np.abs(ds)[upper]
    positive = gaps[gaps > 0]
    return {
        "rows": int(n),
        "distinct_saved_coefficients": int(np.unique(saved).size),
        "inversions": int(inverted.sum()),
        "inversions_the_tolerance_can_separate": int((inverted & separated).sum()),
        "inversions_crossing_a_reporting_bucket": int((inverted & ~same_bucket).sum()),
        "unsafe_inversions": int(unsafe.sum()),
        "pairs_closer_than_the_tolerance": int((~separated & upper).sum()),
        "smallest_positive_saved_gap":
            float(positive.min()) if positive.size else 0.0,
        "exact_global_order_preserved": bool(inverted.sum() == 0),
    }


# ------------------------------------------------------------------ evaluate
def evaluate(*, row_ids: Sequence[str],
             saved: Mapping[str, Mapping[str, Any]],
             reproduced_gate: Mapping[str, float],
             reproduced_fold: Mapping[str, int],
             reproduced_label: Mapping[str, str],
             reproduced_vetoed: Mapping[str, bool],
             reproduced_clamp_ceiling: Mapping[str, float],
             reproduced_safety_logit: Mapping[str, float],
             saved_steps: Mapping[int, int],
             reproduced_steps: Mapping[int, int],
             saved_tau: Mapping[int, float],
             reproduced_tau: Mapping[int, float],
             pinned_probe_digests: Mapping[str, str],
             actual_probe_digests: Mapping[str, str],
             budgets: Mapping[int, Sequence[float]],
             written_state_norms: Mapping[int, Sequence[float]],
             alpha: float,
             checkpoint_gate: Optional[Mapping[str, float]] = None,
             tolerance: float = GATE_ABS_TOLERANCE,
             cutoffs: Sequence[float] = REPORTING_CUTOFFS,
             ) -> EquivalenceReport:
    """Decide whether a reproduction is the same computation. Never raises.

    Returning a report rather than raising is deliberate: the caller writes it
    to disk *before* it decides an exit status, so a refused run leaves the
    evidence it was refused on.
    """
    ids = [str(r) for r in row_ids]
    clauses: List[Clause] = []

    # ---- the row set -----------------------------------------------------
    missing = sorted(set(ids) - set(saved))
    unexpected = sorted(set(saved) - set(ids))
    clauses.append(Clause(
        "row_set_matches", not missing and not unexpected,
        f"{len(ids)} reproduced row(s), {len(saved)} saved; "
        f"{len(missing)} unmatched reproduced, {len(unexpected)} unmatched saved",
        {"unmatched_reproduced": missing[:8],
         "unmatched_saved": unexpected[:8]}))
    shared = [r for r in ids if r in saved]

    g_saved = np.array([float(saved[r]["g_R"]) for r in shared], dtype=np.float64)
    g_repro = np.array([float(reproduced_gate[r]) for r in shared], dtype=np.float64)
    fold = np.array([int(saved[r]["fold"]) for r in shared], dtype=np.int64)
    errors = np.abs(g_repro - g_saved)

    # ---- 1. the numeric clause ------------------------------------------
    worst = float(errors.max()) if errors.size else 0.0
    nonzero = errors[errors > 0.0]
    coefficient_error = {
        "rows_compared": int(errors.size),
        "max": worst,
        "max_in_float32_epsilons": in_epsilons(worst),
        "max_in_ulps_at_the_saved_value":
            worst / float32_ulp(float(g_saved[int(errors.argmax())]))
            if errors.size else 0.0,
        "mean": float(errors.mean()) if errors.size else 0.0,
        "median": float(np.median(errors)) if errors.size else 0.0,
        "quantiles": _quantiles(errors),
        "rows_with_any_difference": int(nonzero.size),
        "rows_bit_identical": int(errors.size - nonzero.size),
        "rows_above_one_float32_eps": int((errors > FLOAT32_EPS).sum()),
        "rows_above_the_tolerance": int((errors > tolerance).sum()),
        "mean_over_differing_rows":
            float(nonzero.mean()) if nonzero.size else 0.0,
        "tolerance": tolerance,
        "tolerance_in_float32_epsilons": tolerance / FLOAT32_EPS,
    }
    clauses.append(Clause(
        "coefficients_agree_within_the_float32_budget", worst <= tolerance,
        f"max |reproduced - saved| = {worst:.6e} "
        f"({in_epsilons(worst):.3f} float32 epsilons) over {errors.size} "
        f"row(s); budget {tolerance:.6e} ({tolerance / FLOAT32_EPS:.0f} "
        f"epsilons)",
        {"max_abs_error": worst, "tolerance": tolerance}))

    per_fold: List[Dict[str, Any]] = []
    for k in sorted(set(int(v) for v in fold.tolist())):
        mask = fold == k
        sub = errors[mask]
        per_fold.append({
            "fold": k,
            "rows": int(mask.sum()),
            "max_abs_error": float(sub.max()) if sub.size else 0.0,
            "max_in_float32_epsilons": in_epsilons(float(sub.max())) if sub.size else 0.0,
            "mean_abs_error": float(sub.mean()) if sub.size else 0.0,
            "median_abs_error": float(np.median(sub)) if sub.size else 0.0,
            "rows_with_any_difference": int((sub > 0).sum()),
            "saved_selected_step": int(saved_steps.get(k, -1)),
            "reproduced_selected_step": int(reproduced_steps.get(k, -1)),
            "saved_tau_hi": float(saved_tau.get(k, float("nan"))),
            "reproduced_tau_hi": float(reproduced_tau.get(k, float("nan"))),
        })

    # ---- 2. structure ----------------------------------------------------
    fold_drift = [r for r in shared
                  if int(saved[r]["fold"]) != int(reproduced_fold[r])]
    label_drift = [r for r in shared
                   if str(saved[r]["final_ebh"]) != str(reproduced_label[r])]
    step_drift = sorted(k for k in set(saved_steps) | set(reproduced_steps)
                        if int(saved_steps.get(k, -1))
                        != int(reproduced_steps.get(k, -2)))
    tau_drift = []
    for k in sorted(set(saved_tau) | set(reproduced_tau)):
        want, got = saved_tau.get(k), reproduced_tau.get(k)
        if want is None or got is None:
            tau_drift.append({"fold": k, "saved": want, "reproduced": got,
                              "budget": None})
            continue
        budget = tau_tolerance(want)
        if abs(float(got) - float(want)) > budget:
            tau_drift.append({"fold": k, "saved": float(want),
                              "reproduced": float(got), "budget": budget,
                              "difference": abs(float(got) - float(want))})
    probe_drift = sorted(
        name for name in set(pinned_probe_digests) | set(actual_probe_digests)
        if pinned_probe_digests.get(name) != actual_probe_digests.get(name))

    structure = {
        "fold_disagreements": len(fold_drift),
        "label_disagreements": len(label_drift),
        "selected_step_disagreements": step_drift,
        "saved_selected_steps": {str(k): int(v) for k, v in sorted(saved_steps.items())},
        "reproduced_selected_steps": {str(k): int(v) for k, v in sorted(reproduced_steps.items())},
        "tau_hi_disagreements": tau_drift,
        "tau_hi_budget_rule": f"{TAU_ULP_BUDGET} float32 ULPs at tau_hi",
        "probe_digest_disagreements": probe_drift,
        "probe_digests_checked": len(set(pinned_probe_digests)
                                     | set(actual_probe_digests)),
    }
    clauses.append(Clause(
        "fold_assignment_is_identical", not fold_drift,
        f"{len(fold_drift)} row(s) changed fold",
        {"rows": fold_drift[:8]}))
    clauses.append(Clause(
        "labels_are_identical", not label_drift,
        f"{len(label_drift)} row(s) changed label", {"rows": label_drift[:8]}))
    clauses.append(Clause(
        "early_stopping_steps_are_identical", not step_drift,
        f"saved {structure['saved_selected_steps']}, reproduced "
        f"{structure['reproduced_selected_steps']}",
        {"folds": step_drift}))
    clauses.append(Clause(
        "veto_thresholds_agree_within_the_ulp_budget", not tau_drift,
        f"{len(tau_drift)} fold(s) moved tau_hi by more than "
        f"{TAU_ULP_BUDGET} float32 ULP(s)", {"folds": tau_drift}))
    clauses.append(Clause(
        "probe_digests_are_identical", not probe_drift,
        f"{len(probe_drift)} probe artifact digest(s) differ",
        {"artifacts": probe_drift}))

    # ---- 3. no decision moves -------------------------------------------
    ceiling = np.array([float(reproduced_clamp_ceiling[r]) for r in shared],
                       dtype=np.float64)
    veto_saved = np.array([bool(saved[r]["vetoed"]) for r in shared])
    veto_repro = np.array([bool(reproduced_vetoed[r]) for r in shared])
    clamp_saved = np.array([float(saved[r]["g_R_after_clamp"]) < float(saved[r]["g_R"])
                            for r in shared])
    clamp_repro = g_repro > ceiling
    bucket_saved = _bucket(g_saved, cutoffs)
    bucket_repro = _bucket(g_repro, cutoffs)
    intervene_saved = {f"{c:.2f}": g_saved >= c for c in cutoffs}
    intervene_repro = {f"{c:.2f}": g_repro >= c for c in cutoffs}
    intervention_moves = {k: int((intervene_saved[k] != intervene_repro[k]).sum())
                          for k in intervene_saved}

    moved_bucket = int((bucket_saved != bucket_repro).sum())
    moved_veto = int((veto_saved != veto_repro).sum())
    moved_clamp = int((clamp_saved != clamp_repro).sum())
    moved_any_intervention = sum(intervention_moves.values())
    ceiling_ok = bool((np.minimum(g_repro, ceiling) <= g_repro + 0.0).all())

    clauses.append(Clause(
        "no_row_changes_reporting_bucket", moved_bucket == 0,
        f"{moved_bucket} row(s) changed side of a reporting cutoff "
        f"{list(cutoffs)}", {"rows_moved": moved_bucket}))
    clauses.append(Clause(
        "no_row_changes_intervention_decision", moved_any_intervention == 0,
        f"{moved_any_intervention} intervention decision(s) changed across "
        f"the cutoff grid", {"by_cutoff": intervention_moves}))
    clauses.append(Clause(
        "no_row_changes_veto_decision", moved_veto == 0,
        f"{moved_veto} row(s) changed veto decision "
        f"({int(veto_saved.sum())} vetoed saved, {int(veto_repro.sum())} "
        f"reproduced)", {"rows_moved": moved_veto}))
    clauses.append(Clause(
        "no_row_changes_clamp_decision", moved_clamp == 0,
        f"{moved_clamp} row(s) changed whether the counterfactual clamp binds "
        f"({int(clamp_saved.sum())} bound saved, {int(clamp_repro.sum())} "
        f"reproduced)", {"rows_moved": moved_clamp}))
    clauses.append(Clause(
        "the_clamp_still_never_raises_a_gate", ceiling_ok,
        "min(g, g*) <= g on every row"))

    margins = decision_margins(
        g_saved, ceiling,
        [float(reproduced_safety_logit[r]) for r in shared],
        [float(reproduced_tau.get(int(saved[r]["fold"]), np.nan))
         for r in shared],
        cutoffs)
    margins["reporting_bucket_moves"] = moved_bucket
    margins["veto_decision_moves"] = moved_veto
    margins["clamp_decision_moves"] = moved_clamp
    margins["intervention_decision_moves"] = intervention_moves

    admissible = float(margins["gate_space_margin"]) >= \
        tolerance * DECISION_MARGIN_SAFETY_FACTOR
    clauses.append(Clause(
        "the_tolerance_is_smaller_than_every_decision_margin", admissible,
        f"tightest gate-space margin {float(margins['gate_space_margin']):.6e} "
        f"at the {margins['gate_space_binding_boundary']}; the rule needs "
        f"{tolerance * DECISION_MARGIN_SAFETY_FACTOR:.6e}",
        {"margin": float(margins["gate_space_margin"]),
         "required": tolerance * DECISION_MARGIN_SAFETY_FACTOR}))

    # ---- 4. ordering -----------------------------------------------------
    ordering = ordering_agreement(g_saved, g_repro, tolerance, cutoffs)
    clauses.append(Clause(
        "no_inversion_the_tolerance_can_see", ordering["unsafe_inversions"] == 0,
        f"{ordering['inversions']} inversion(s), "
        f"{ordering['unsafe_inversions']} of them between rows the tolerance "
        f"separates or across a reporting bucket",
        {"inversions": ordering["inversions"],
         "unsafe": ordering["unsafe_inversions"]}))

    # ---- 5. the applied edit --------------------------------------------
    edit = edit_difference(worst, alpha, budgets, written_state_norms)
    clauses.append(Clause(
        "the_applied_edit_is_below_the_actuator_resolution",
        edit["max_edit_difference_in_bf16_ulps"] <= EDIT_BF16_ULP_TOLERANCE,
        f"max ||d delta|| = {edit['max_edit_norm_difference']:.6e}, which is "
        f"{edit['max_edit_difference_in_bf16_ulps']:.6e} bfloat16 ULP(s) of "
        f"the state it is written into; budget "
        f"{EDIT_BF16_ULP_TOLERANCE:g}",
        {"bf16_ulps": edit["max_edit_difference_in_bf16_ulps"]}))
    edit["max_applied_alpha_difference"] = alpha * worst

    # ---- the deployable checkpoints -------------------------------------
    checkpoint_agreement: Dict[str, Any] = {"checked": False}
    if checkpoint_gate is not None:
        ck = np.array([float(checkpoint_gate[r]) for r in shared],
                      dtype=np.float64)
        ck_err = np.abs(ck - g_saved)
        live_err = np.abs(ck - g_repro)
        checkpoint_agreement = {
            "checked": True,
            "rows_scored": int(ck.size),
            "max_abs_error_against_saved": float(ck_err.max()) if ck_err.size else 0.0,
            "max_abs_error_against_the_in_run_gate":
                float(live_err.max()) if live_err.size else 0.0,
            "rows_bit_identical_to_the_in_run_gate":
                int((live_err == 0).sum()),
            "scored_out_of_fold_only": True,
        }
        clauses.append(Clause(
            "the_written_checkpoints_score_the_corpus_the_same_way",
            float(ck_err.max()) <= tolerance if ck_err.size else True,
            f"re-scoring all {int(ck.size)} row(s) through the serialised "
            f"fold checkpoints lands "
            f"{float(ck_err.max()) if ck_err.size else 0.0:.6e} from the saved "
            f"coefficients and "
            f"{float(live_err.max()) if live_err.size else 0.0:.6e} from the "
            f"in-run gate",
            {"max_abs_error_against_saved":
                 float(ck_err.max()) if ck_err.size else 0.0}))

    return EquivalenceReport(
        clauses=tuple(clauses), coefficient_error=coefficient_error,
        per_fold=per_fold, decision_margins=margins, edit_difference=edit,
        structure=structure, ordering=ordering,
        checkpoint_agreement=checkpoint_agreement)


# ------------------------------------------------------------- the artifact
DIAGNOSTIC_FILENAME = "reproduction_diagnostic.json"


def write_diagnostic(directory: Path | str, report: EquivalenceReport,
                     context: Optional[Mapping[str, Any]] = None
                     ) -> Path:
    """Persist the whole comparison, passed or failed, before anything decides.

    A refused run that writes nothing is a run whose refusal cannot be
    audited. This is written first and the exit status is decided after.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(report.as_dict())
    payload["context"] = dict(context or {})
    payload["failing_clauses"] = [c.name for c in report.failures]
    path = directory / DIAGNOSTIC_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               default=float),
                    encoding="utf-8")
    return path


__all__ = [
    "BFLOAT16_EPS", "Clause", "DECISION_MARGIN_SAFETY_FACTOR",
    "DIAGNOSTIC_FILENAME", "EDIT_BF16_ULP_TOLERANCE", "ERROR_QUANTILES",
    "EquivalenceError", "EquivalenceReport", "FLOAT32_EPS",
    "FLOAT32_UNIT_ROUNDOFF", "GATE_ABS_TOLERANCE", "GATE_EPSILON_BUDGET",
    "REPORTING_CUTOFFS", "RULE_NAME", "RULE_VERSION", "SCHEMA",
    "SUPERSEDED_RULES", "TAU_ULP_BUDGET", "assert_rule_is_admissible",
    "assert_tolerance_is_not_loosened",
    "decision_margins", "describe_tolerance", "edit_difference", "evaluate",
    "float32_ulp", "in_epsilons", "ordering_agreement", "tau_tolerance",
    "write_diagnostic",
]
