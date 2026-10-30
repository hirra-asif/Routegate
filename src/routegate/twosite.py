"""The one-pass budget result, closed; and the two-site actuator it implies.

The one-pass budget equivalence question is settled. Its answer is **no** for
the validated three-block band, and the reason is structural rather than
numerical:

* blocks 31 and 32 read their budget before anything upstream of it is edited,
  so both conventions compute the same number and the measured disagreement is
  float32 rounding -- 3.5e-07 and 3.2e-07, nothing above the 1e-2 gate;
* block 33 reads its budget from block 32's output, and by then block 32 has
  already consumed an edited state 32. There is no ordering of hooks within one
  pass that recovers the unsteered value, because the unsteered value no longer
  exists anywhere in that pass. Measured: 7.455261e-02 maximum, 382 of 480 rows
  over tolerance.

The alternative convention -- budgeting block 33 against its *actual* runtime
input, which also carries delta_32 -- was measured alongside and is **worse**:
1.093370e-01 maximum, 432 of 480 rows over tolerance. It is recorded as
rejected, not adopted.

No tolerance was changed at any point. The 1e-2 gate is the one predeclared
before the first run, and both v1 and v2 were decided against it.

What follows, and what does not
-------------------------------
What follows is a *candidate*: an actuator that intervenes at historical blocks
31 and 32 only, which are exactly the two sites the equivalence test passed. It
is deliberately not called validated. The three-block band is what phase 1 and
phase 2 measured causally; dropping a site changes the intervention, and an
intervention that has not been run on the corpus has no causal evidence
whatever the arithmetic says about its budget. This module declares the
candidate and binds it to the evidence; a new static causal validation is what
would earn it a claim.

What this module does not do
----------------------------
It generates nothing, judges nothing, trains nothing, loads no model and
selects nothing. It holds two immutable records and the gates that refuse a
tree which has drifted from them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .dualsteer import COMBINED_ALPHA_REPAIR, INTERVENTION_TYPE
from .onepassbudget import (
    ACTUATOR_BLOCKS,
    ALPHA_R,
    TOLERANCE,
    WORST_CASE_G_R,
    block_input_state,
    block_output_state,
)
from .validation import Checklist, ValidationError, sha256_file, sha256_text

REPO_ROOT = Path(__file__).resolve().parents[2]

CLOSURE_SCHEMA = "routegate/one_pass_budget_equivalence_closure/1"
CANDIDATE_SCHEMA = "routegate/two_site_actuator_candidate/1"

CLOSURE_VERSION = 1
CANDIDATE_VERSION = 1

CLOSURE_NAME = "one_pass_budget_equivalence_closure_v1"
CANDIDATE_NAME = "two_site_actuator_candidate_v1"

CLOSURE_MANIFEST = (REPO_ROOT / "manifests"
                    / "one_pass_budget_equivalence_closure_v1.json")
CANDIDATE_MANIFEST = (REPO_ROOT / "manifests"
                      / "two_site_actuator_candidate_v1.json")

#: The field holding each artifact's digest of itself, excluded from it.
DIGEST_FIELD = "record_sha256"


class TwoSiteError(ValidationError):
    """The two-site candidate was asked for something the evidence forbids."""


# ================================================================= the result
#
# The v2 run's five artifacts, by digest. The run itself lives under outputs/
# and is not committed -- these digests are its committed identity, and the
# closure is bound to them.

V2_RUN_DIRECTORY = "outputs/single_direction_controller/budget_equivalence_v2"

V2_ARTIFACTS: Dict[str, str] = {
    "per_row_budgets.csv":
        "400b5ce76e8750c401a7734692be39a256d458babfa534f194ce4c437b46fd6c",
    "input_hashes.json":
        "9d9594bb322416442f0b63f47eb2e0365671565914a5d8adf33ef4b1d0ead69d",
    "run_manifest.json":
        "96723b0d9d473ad5aeab87df438a48bd88c260e21b86469d99a7a441de71c170",
    "summary.json":
        "30caa46961ee254a9b7faa3b10b4820cfe46c00a3015d5df3f33d013c3b13276",
    "output_hashes.json":
        "0c6d032b2c267324dc6fd618a56b623097227ea102738dfc360617942ad886c0",
}

#: The one artifact the candidate is bound to: the summary that carries the
#: per-block decision. A candidate whose sites are not the sites this summary
#: passed is not this candidate.
V2_SUMMARY_SHA256 = V2_ARTIFACTS["summary.json"]

V2_ROWS = 480

#: What v2 measured, per historical block. Restated here so the closure states
#: its own numbers; ``verify_v2_result`` checks every one of them against the
#: artifact rather than trusting this copy.
V2_PER_BLOCK: Dict[int, Dict[str, Any]] = {
    31: {"max": 3.527094567313233e-07,
         "mean": 1.6718764661011732e-07,
         "n_exceeding_tolerance": 0, "within_tolerance": True},
    32: {"max": 3.212717735920706e-07,
         "mean": 1.550493432109504e-07,
         "n_exceeding_tolerance": 0, "within_tolerance": True},
    33: {"max": 0.07455261363665293,
         "mean": 0.026922416048923058,
         "n_exceeding_tolerance": 382, "within_tolerance": False},
}

V2_RUNTIME_INPUT_VARIANT: Dict[str, Any] = {
    "block": 33,
    "max": 0.109336969786945,
    "mean": 0.04418197085681989,
    "n_exceeding_tolerance": 432,
    "within_tolerance": False,
    "status": "REJECTED",
    "why": "budgeting block 33 against its actual runtime input -- which "
           "carries delta_32 as well as the propagated delta_31 -- is strictly "
           "worse than the declared pre-write convention on both counts: "
           "1.093370e-01 against 7.455261e-02 at the maximum, and 432 of 480 "
           "rows over tolerance against 382. It was measured alongside the "
           "gate so the unfavourable convention is on the record, and it is "
           "not adopted.",
}

#: The blocks v2 cleared, and the one it did not. Derived from the result, not
#: chosen: ``verify_v2_result`` recomputes both from the artifact.
PASSING_BLOCKS: Tuple[int, ...] = (31, 32)
FAILING_BLOCKS: Tuple[int, ...] = (33,)


def _canonical(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != DIGEST_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def record_sha256(payload: Mapping[str, Any]) -> str:
    """A record's digest of itself, over its canonical form."""
    return sha256_text(_canonical(payload))


def verify_v2_result(directory: Path | str | None = None,
                     checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The v2 run on disk is the run this closure was taken from.

    Digest first, then content. The run directory is an output and is not
    committed, so its absence is a fact about this machine; its PRESENCE with a
    different digest, or with different numbers, is refused.
    """
    checks = checks or Checklist("two_site_candidate")
    path = Path(directory) if directory is not None else (
        REPO_ROOT / V2_RUN_DIRECTORY)
    if not path.is_dir():
        checks.record(
            "v2_run_reachable", True,
            f"{V2_RUN_DIRECTORY} is not present on this machine; it is an "
            f"output and is not committed. The closure pins its digests.")
        return {"reachable": False, "directory": str(path)}

    for name, expected in sorted(V2_ARTIFACTS.items()):
        artifact = path / name
        checks.require(f"v2_artifact_present_{name}", artifact.is_file(),
                       f"{artifact} is missing from the v2 run")
        actual = sha256_file(artifact)
        checks.require(
            f"v2_artifact_sha256_{name}", actual == expected,
            f"{name} hashes to {actual}, the closure pins {expected}. The run "
            f"this candidate rests on has changed since it was closed; it is "
            f"refused, not re-pinned.")

    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    checks.require(
        "v2_decision_is_blocked",
        summary.get("decision", {}).get("decision") == "BLOCKED",
        f"the v2 summary records decision "
        f"{summary.get('decision', {}).get('decision')!r}; this closure exists "
        f"for BLOCKED and is not edited to fit another outcome")
    checks.require(
        "v2_tolerance_is_the_predeclared_one",
        float(summary["decision"]["tolerance"]) == TOLERANCE,
        f"the v2 decision was taken against tolerance "
        f"{summary['decision']['tolerance']}; the predeclared gate is "
        f"{TOLERANCE} and no run in this line relaxed it")
    checks.require("v2_rows", summary.get("rows") == V2_ROWS,
                   f"{summary.get('rows')} row(s), expected {V2_ROWS}")
    checks.require(
        "v2_generated_no_token", summary.get("generated_tokens") == 0,
        f"{summary.get('generated_tokens')} generated token(s); the budget "
        f"check is prefill only")
    checks.require(
        "v2_read_before_write_held",
        summary.get("same_forward_read_before_write", {}).get("holds") is True,
        "the v2 run's same-forward read-before-write invariant did not hold, "
        "so its budgets are not the ones it says they are")

    for block, expected in sorted(V2_PER_BLOCK.items()):
        observed = summary["per_block"][str(block)]
        for field in ("max", "mean", "n_exceeding_tolerance",
                      "within_tolerance"):
            checks.require(
                f"v2_block_{block}_{field}", observed[field] == expected[field],
                f"block {block} {field}: artifact says {observed[field]!r}, "
                f"this closure records {expected[field]!r}")

    variant = summary["block_33_runtime_input_variant"]
    for field in ("max", "mean", "n_exceeding_tolerance"):
        checks.require(
            f"v2_runtime_input_variant_{field}",
            variant[field] == V2_RUNTIME_INPUT_VARIANT[field],
            f"the runtime-input variant's {field}: artifact says "
            f"{variant[field]!r}, this closure records "
            f"{V2_RUNTIME_INPUT_VARIANT[field]!r}")
    checks.require(
        "v2_runtime_input_variant_is_worse",
        variant["max"] > summary["per_block"]["33"]["max"],
        f"the runtime-input variant ({variant['max']:.6e}) is not worse than "
        f"the declared convention "
        f"({summary['per_block']['33']['max']:.6e}); the closure records it as "
        f"the worse alternative and that claim must hold in the artifact")

    return {"reachable": True, "directory": str(path), "summary": summary,
            "checks": checks}


def passing_blocks_from_summary(summary: Mapping[str, Any]) -> Tuple[int, ...]:
    """The blocks the v2 run actually cleared, read out of the artifact.

    The candidate's sites are not asserted and then justified. They are
    *derived* here, from the run's own per-block verdicts, and
    ``assert_sites_are_the_passing_sites`` refuses any candidate whose sites
    are not exactly what this returns.
    """
    return tuple(sorted(int(block)
                        for block, entry in summary["per_block"].items()
                        if entry.get("within_tolerance") is True))


# =============================================================== the candidate
CANDIDATE_ARM_ID = "repair_015_blocks31_32"

#: The historical decoder blocks the candidate intervenes at, and the package
#: states that follow from them. Never restated as literals downstream: the
#: states come from ``block_output_state`` and the budget references from
#: ``block_input_state``, so the two conventions cannot drift apart.
CANDIDATE_BLOCKS: Tuple[int, ...] = PASSING_BLOCKS
CANDIDATE_STATES: Tuple[int, ...] = tuple(
    block_output_state(b) for b in CANDIDATE_BLOCKS)
CANDIDATE_BUDGET_STATES: Tuple[int, ...] = tuple(
    block_input_state(b) for b in CANDIDATE_BLOCKS)

#: The block the candidate drops, and the states that go with it.
DROPPED_BLOCKS: Tuple[int, ...] = FAILING_BLOCKS
DROPPED_STATES: Tuple[int, ...] = tuple(
    block_output_state(b) for b in DROPPED_BLOCKS)

#: The later preregistered ablation, preserved and NOT run here: historical
#: block 32 alone, which is package state 33. Named so the distinction between
#: the two numbering systems is written down rather than remembered.
ABLATION_BLOCKS: Tuple[int, ...] = (32,)
ABLATION_STATES: Tuple[int, ...] = tuple(
    block_output_state(b) for b in ABLATION_BLOCKS)


def assert_sites_are_the_passing_sites(
        blocks: Sequence[int], summary: Mapping[str, Any],
        checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """The candidate's sites are exactly the sites v2 passed. Mechanically.

    Three separate refusals, because they are three different mistakes:
    a site that v2 never cleared, a site v2 cleared that the candidate drops,
    and a site v2 explicitly failed.
    """
    checks = checks or Checklist("two_site_candidate")
    proposed = tuple(sorted(int(b) for b in blocks))
    passing = passing_blocks_from_summary(summary)
    failing = tuple(sorted(int(block)
                           for block, entry in summary["per_block"].items()
                           if entry.get("within_tolerance") is not True))

    # Ordered from the most specific diagnosis to the least. A site that was
    # measured and FAILED is told so; only a site the run never cleared for
    # some other reason -- never measured at all, say -- falls through to the
    # general message.
    included_failures = [b for b in proposed if b in failing]
    checks.require(
        "no_failing_site_is_included", not included_failures,
        f"block(s) {included_failures} were measured and FAILED the "
        f"equivalence test at tolerance {TOLERANCE}, and are in the candidate. "
        f"A site whose one-pass budget is not the validated budget is not made "
        f"one by being declared.")
    smuggled = [b for b in proposed if b not in passing]
    checks.require(
        "every_candidate_site_passed_the_equivalence_test", not smuggled,
        f"block(s) {smuggled} are in the candidate and are not among the "
        f"blocks v2 cleared {list(passing)}; the run carries no verdict for "
        f"them at all, and an unmeasured site is not a cleared one.")
    dropped_but_passing = [b for b in passing if b not in proposed]
    checks.require(
        "no_passing_site_is_dropped_without_saying_so", not dropped_but_passing,
        f"block(s) {dropped_but_passing} cleared the equivalence test and are "
        f"absent from the candidate; dropping a site that passed is a "
        f"decision to record, not an omission")
    checks.require(
        "the_candidate_is_bound_to_the_v2_summary",
        proposed == passing == CANDIDATE_BLOCKS,
        f"the candidate declares {list(proposed)}, the v2 summary cleared "
        f"{list(passing)}, and the package pins {list(CANDIDATE_BLOCKS)}")
    return {"blocks": list(proposed), "passing": list(passing),
            "failing": list(failing), "checks": checks}


def candidate_intervention() -> Dict[str, Any]:
    """The edit the candidate applies, stated once and derived everywhere."""
    return {
        "arm_id": CANDIDATE_ARM_ID,
        "formula": "delta(i, l) = alpha_R * g_R * r(i, l) * d_R(l)",
        "alpha_R": ALPHA_R,
        "g_R": WORST_CASE_G_R,
        "g_R_note": "fixed at 1 for the causal validation: this is a STATIC "
                    "arm and no controller predicts anything in it. A gate is "
                    "what a later controller would supply.",
        "applied_strength": ALPHA_R * WORST_CASE_G_R,
        "alpha_S": 0.0,
        "alpha_S_note": "0 permanently; d_S = H - E is retired as an actuator "
                        "by the phase-2 closure and no coefficient scales it.",
        "historical_blocks": list(CANDIDATE_BLOCKS),
        "package_states_edited": list(CANDIDATE_STATES),
        "budget_reference_states": list(CANDIDATE_BUDGET_STATES),
        "budget_reference": "block_input",
        "numbering_rule": "historical block l is edited at package state l + 1 "
                          "and budgeted against package state l. The two "
                          "numbering systems are never used interchangeably.",
        "site": "residual_stream",
        "site_kind": "block_output",
        "token_position": "final prompt token",
        "prefill_only": True,
        "generated_token_positions_are_never_edited": True,
        "operation": "additive",
        "forward_passes": 1,
        "forward_passes_note": "one prefill forward pass. Each block's budget "
                               "is snapshotted before that block's own edit is "
                               "written, which the v2 run proved holds on all "
                               "480 rows.",
        "orthogonalisation_applied": False,
        "intervention_type": INTERVENTION_TYPE,
        "dropped_blocks": list(DROPPED_BLOCKS),
        "dropped_states": list(DROPPED_STATES),
        "why_dropped": "block 33's one-pass budget cannot reproduce the "
                       "validated two-pass budget: it is read from block 32's "
                       "output, and by then block 32 has consumed an edited "
                       "state 32. v2 measured 7.455261e-02 maximum, 382 of "
                       "480 rows over the 1e-2 gate.",
    }


def build_two_site_plans(bank: Any, capture: Mapping[int, Any],
                         alpha: float = ALPHA_R,
                         g_r: float = WORST_CASE_G_R,
                         blocks: Sequence[int] = CANDIDATE_BLOCKS
                         ) -> Dict[int, Any]:
    """One plan per candidate site, through the validated actuator's own code.

    Deliberately not ``dualsteer.build_plans``: that iterates ``bank.states``
    and so always plans all three sites, which is the intervention this result
    ruled out. The arithmetic is unchanged --
    ``steering.build_steering_plan`` is the validated actuator itself -- and
    only the set of sites differs.
    """
    from .steering import build_steering_plan

    strength = float(alpha) * float(g_r)
    if not 0.0 < strength <= ALPHA_R:
        raise TwoSiteError(
            f"applied strength {strength} is outside (0, {ALPHA_R}]; the "
            f"candidate inherits the frozen validated repair strength and does "
            f"not exceed it")
    plans: Dict[int, Any] = {}
    for block in blocks:
        state = block_output_state(int(block))
        reference_state = block_input_state(int(block))
        if reference_state not in capture:
            raise TwoSiteError(
                f"historical block {block} is edited at state {state} and "
                f"budgeted against state {reference_state}, which the capture "
                f"does not hold")
        repair, _safety = bank.directions_for_state(state)
        plans[state] = build_steering_plan(
            state=state, direction=repair, strength=strength,
            reference_vector=capture[reference_state],
            budget_reference="block_input")
    return plans


# ============================================================ the two records
def build_closure_record() -> Dict[str, Any]:
    """The immutable closure of the one-pass budget equivalence question."""
    payload: Dict[str, Any] = {
        "schema": CLOSURE_SCHEMA,
        "name": CLOSURE_NAME,
        "version": CLOSURE_VERSION,
        "status": "CLOSED",
        "immutable": True,
        "written_after_the_run": True,
        "question": "does the one-pass actuator budget reproduce the validated "
                    "two-pass budget at historical blocks 31, 32 and 33?",
        "answer": "no, at block 33; yes, at blocks 31 and 32",
        "decision": "BLOCKED",
        "tolerance": TOLERANCE,
        "tolerance_note": "the 1e-2 maximum relative difference was "
                          "predeclared before the first run and was not "
                          "changed, relaxed or re-derived at any point in this "
                          "line. Both v1 and v2 were decided against it.",
        "rows": V2_ROWS,
        "g_R": WORST_CASE_G_R,
        "g_R_note": "the worst case: the edit, and so the divergence it "
                    "causes, is largest at a fully open gate.",
        "alpha_R": ALPHA_R,
        "alpha_S": 0.0,
        "failed_specification": {
            "historical_blocks": list(ACTUATOR_BLOCKS),
            "package_states_edited": [block_output_state(b)
                                      for b in ACTUATOR_BLOCKS],
            "budget_reference_states": [block_input_state(b)
                                        for b in ACTUATOR_BLOCKS],
            "what_failed": "the three-block band cannot be driven from one "
                           "prefill forward pass under the validated "
                           "block-input budget convention.",
            "the_band_itself_is_not_invalidated": "phases 1 and 2 measured the "
                                                  "31-33 band causally, in TWO "
                                                  "passes, and those results "
                                                  "stand. What failed is the "
                                                  "one-pass driving of it.",
        },
        "per_block": {str(b): dict(V2_PER_BLOCK[b]) for b in sorted(V2_PER_BLOCK)},
        "block_33_is_infeasible": {
            "block": 33,
            "state_edited": block_output_state(33),
            "budget_reference_state": block_input_state(33),
            "max_relative_difference": V2_PER_BLOCK[33]["max"],
            "rows_over_tolerance": V2_PER_BLOCK[33]["n_exceeding_tolerance"],
            "structural_reason": "block 33's budget is the norm of block 32's "
                                 "output. In one pass that tensor has already "
                                 "been computed from an edited state 32, so "
                                 "the unsteered value the validated convention "
                                 "requires does not exist anywhere in the "
                                 "pass. No ordering of hooks recovers it.",
            "is_not_a_numerical_artifact": "the disagreement is 7.5e-02, five "
                                           "orders of magnitude above the "
                                           "float32 rounding that explains "
                                           "blocks 31 and 32.",
        },
        "rejected_alternative": dict(V2_RUNTIME_INPUT_VARIANT),
        "same_forward_read_before_write": {
            "holds": True,
            "blocks": list(PASSING_BLOCKS),
            "rows": V2_ROWS,
            "statement": "the budget is read before the activation is edited, "
                         "proved within one forward pass on every row. The "
                         "block-33 failure is not a hook-ordering fault.",
        },
        "v2_run": {
            "directory": V2_RUN_DIRECTORY,
            "artifacts": dict(V2_ARTIFACTS),
            "summary_sha256": V2_SUMMARY_SHA256,
            "not_committed": "the run is an output. The digests above are its "
                             "committed identity.",
            "per_row_read_before_write_column": False,
            "per_row_read_before_write_note":
                "the invariant was verified per row at run time on all 480 "
                "rows and its per-block aggregate is in summary.json; the "
                "per-row separations are not a column in per_row_budgets.csv.",
        },
        "v1_run": {
            "directory": "outputs/single_direction_controller/"
                         "budget_equivalence_v1",
            "outcome": "BLOCKED (exit 2)",
            "why": "v1 required blocks 31 and 32 to be bit-identical across "
                   "two separate forward passes whose norms are reduced on "
                   "different devices. Its 3.527095e-07 was float32 rounding, "
                   "not a fault in the actuator.",
            "wrote_no_artifacts": True,
            "preserved_not_overwritten": True,
            "recorded_in": "docs/decision_log.md, 2026-08-26",
        },
        "consequence": {
            "candidate": CANDIDATE_NAME,
            "remove_block": 33,
            "keep_blocks": list(CANDIDATE_BLOCKS),
            "status": "CANDIDATE, NOT VALIDATED",
            "subject_to": "a new static causal validation over the "
                          "development corpus. The 31-33 band is what phases 1 "
                          "and 2 measured; a two-site actuator is a different "
                          "intervention and carries none of that evidence "
                          "forward. Budget arithmetic is not a causal result.",
            "nothing_is_claimed_yet": True,
        },
        "contains_prompt_or_response_text": False,
    }
    payload[DIGEST_FIELD] = record_sha256(payload)
    return payload


def build_candidate_record() -> Dict[str, Any]:
    """The versioned two-site actuator candidate, bound to the v2 evidence."""
    payload: Dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "name": CANDIDATE_NAME,
        "version": CANDIDATE_VERSION,
        "status": "DECLARED_NOT_RUN",
        "status_meaning": "the arm is specified and no generation exists for "
                          "it. It is a candidate: the sites are the ones the "
                          "budget test cleared, and nothing about its causal "
                          "effect has been measured.",
        "supersedes": {
            "specification": "historical blocks 31, 32 and 33 driven from one "
                             "forward pass",
            "why": "block 33 failed the one-pass budget equivalence test at "
                   "the predeclared 1e-2 tolerance",
            "closure": CLOSURE_NAME,
            "the_previous_specification_is_not_rewritten": True,
        },
        "bound_to": {
            "run": V2_RUN_DIRECTORY,
            "summary_sha256": V2_SUMMARY_SHA256,
            "artifacts": dict(V2_ARTIFACTS),
            "rule": "the candidate's sites are derived from this summary's "
                    "per-block verdicts, not asserted beside it. A summary "
                    "that clears a different set of blocks is a different "
                    "candidate.",
        },
        "intervention": candidate_intervention(),
        "direction": {
            "repair": "d_R = B_d1 - E",
            "source": "validated_direction_bank.npz, array repair_direction",
            "artifact_sha256":
                "d759574b5be630ea5170fb440f8663d8c4bfaa32b479ce5cf641b6dbd73516bb",
            "unchanged": True,
            "safety_direction_is_not_used": True,
            "safety_direction_note": "d_S = H - E is retired as an actuator. "
                                     "The bank still carries it; nothing in "
                                     "this candidate reads it.",
        },
        "controller": {
            "used_in_this_validation": False,
            "statement": "no controller is trained, loaded or consulted. This "
                         "is a STATIC arm at a fixed strength, which is what "
                         "makes it comparable to the phase-1 repair_015 arm.",
            "g_R_is_fixed_at": WORST_CASE_G_R,
        },
        "inherited_frozen": {
            "corpus": "the 480-row E/B/H development frame",
            "folds": "the frozen five-fold division; no fold is used by a "
                     "static arm and none is re-derived",
            "model": "Qwen/Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "generation": "routegate.runtime.GenerationConfig, reused verbatim",
            "tokenizer_and_chat_template": "the frozen ones, by digest",
            "note": "every value is inherited. Nothing here is re-chosen, and "
                    "a run whose records disagree with any of them is refused "
                    "rather than recorded.",
        },
        "sealed_set": {
            "accessible": False,
            "statement": "prohibited. This validation runs on development "
                         "prompts only and no option it exposes can reach "
                         "held-out material.",
        },
        "preregistered_ablation": {
            "name": "historical block 32 only",
            "historical_blocks": list(ABLATION_BLOCKS),
            "package_states_edited": list(ABLATION_STATES),
            "budget_reference_states": [block_input_state(b)
                                        for b in ABLATION_BLOCKS],
            "status": "PRESERVED_NOT_RUN",
            "run_when": "after the two-site arm has a causal result, as a "
                        "preregistered ablation beside it",
            "numbering_warning": "historical block 32 is package state 33. It "
                                 "is NOT package state 32, which is the output "
                                 "of historical block 31. The two numbering "
                                 "systems differ by one and are never used "
                                 "interchangeably.",
        },
        "what_this_does_not_claim": {
            "a_causal_result": False,
            "that_the_31_33_band_is_invalid": False,
            "that_dropping_block_33_is_free": False,
            "statement": "the budget test says the two-site actuator can be "
                         "driven from one pass. It says nothing about whether "
                         "it repairs anything, and the phase-1 evidence for "
                         "the three-block band does not transfer to it.",
        },
        "contains_prompt_or_response_text": False,
    }
    payload[DIGEST_FIELD] = record_sha256(payload)
    return payload


COMPARED_FIELDS: Tuple[str, ...] = (
    "schema", "name", "version", "status", "intervention", "direction",
    "controller", "inherited_frozen", "sealed_set", "preregistered_ablation",
    "bound_to", "supersedes", "what_this_does_not_claim",
    "failed_specification", "per_block", "block_33_is_infeasible",
    "rejected_alternative", "consequence", "v2_run", "v1_run", "tolerance",
    "same_forward_read_before_write", "decision", "question", "answer",
)


def _verify_record(path: Path, derived: Mapping[str, Any], label: str,
                   checks: Checklist) -> Dict[str, Any]:
    if not path.is_file():
        raise TwoSiteError(
            f"{path} does not exist; it is the committed {label}. Write it "
            f"with experiments/single_direction_controller/"
            f"freeze_two_site_candidate.py")
    committed = json.loads(path.read_text(encoding="utf-8"))
    recorded = committed.get(DIGEST_FIELD)
    recomputed = record_sha256(committed)
    checks.require(
        f"{label}_digest_recomputes", recorded == recomputed,
        f"{path.name} records {DIGEST_FIELD} {recorded}, its own contents hash "
        f"to {recomputed}. It has been edited since it was written; an "
        f"immutable record is not repaired in place.")
    for field in COMPARED_FIELDS:
        if field not in derived:
            continue
        checks.require(
            f"{label}_{field}", committed.get(field) == derived[field],
            f"{path.name}: '{field}' differs from routegate.twosite. A record "
            f"that has moved since it was written is refused; neither copy is "
            f"preferred over the other.")
    checks.require(
        f"{label}_digest_matches_the_package",
        recorded == derived[DIGEST_FIELD],
        f"{path.name} hashes to {recorded}, the package derives "
        f"{derived[DIGEST_FIELD]}")
    return committed


def verify_records(closure_path: Path | str = CLOSURE_MANIFEST,
                   candidate_path: Path | str = CANDIDATE_MANIFEST,
                   checks: Optional[Checklist] = None) -> Dict[str, Any]:
    """Both committed records are what this package declares, and untampered."""
    checks = checks or Checklist("two_site_candidate")
    closure = _verify_record(Path(closure_path), build_closure_record(),
                             "closure", checks)
    candidate = _verify_record(Path(candidate_path), build_candidate_record(),
                               "candidate", checks)
    checks.require(
        "the_candidate_names_the_closure_it_follows_from",
        candidate["supersedes"]["closure"] == closure["name"],
        "the candidate does not name the closure it follows from")
    checks.require(
        "both_records_agree_on_the_bound_summary",
        candidate["bound_to"]["summary_sha256"]
        == closure["v2_run"]["summary_sha256"],
        "the candidate and the closure name different v2 summaries")
    return {"closure": closure, "candidate": candidate, "checks": checks}
