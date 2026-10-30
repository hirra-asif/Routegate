"""The intervention arithmetic, on synthetic tensors.

Everything here is checked against the recovered historical definition:

    applied = unit(donor - target) * strength * ||reference||

added at one position, never substituted, with the donor and the target taken at
their own final prompt tokens so their prompts may be different lengths.

torch is required to build the tensors, but no model, no weights and no GPU are.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from routegate.intervention import (
    EPS,
    STATE_EMBEDDING,
    InterventionError,
    apply_delta_at_position,
    build_plan,
    final_prompt_token_index,
    replace_at_position,
    unit_scaled,
    vector_norm,
)


def vec(*values):
    return torch.tensor(values, dtype=torch.float32)


def activation(sequence_length, hidden=4, fill=0.0):
    return torch.full((1, sequence_length, hidden), float(fill),
                      dtype=torch.float32)


# ------------------------------------------------------- 1. the maths itself
def test_the_applied_delta_is_the_unit_donor_direction_at_the_target_budget():
    target = vec(1.0, 0.0, 0.0, 0.0)
    donor = vec(1.0, 3.0, 0.0, 0.0)          # raw delta is (0, 3, 0, 0)
    reference = vec(0.0, 0.0, 4.0, 0.0)      # norm 4
    plan = build_plan(state=5, intervention_type="donor_direction",
                      strength=0.25, target_state_vector=target,
                      donor_state_vector=donor, reference_vector=reference)

    assert plan.status == "ok"
    assert plan.reference_norm == pytest.approx(4.0)
    assert plan.budget == pytest.approx(1.0)          # 0.25 * 4
    assert plan.raw_delta_norm == pytest.approx(3.0)
    assert plan.applied_delta_norm == pytest.approx(1.0)
    assert torch.allclose(plan.delta, vec(0.0, 1.0, 0.0, 0.0))


def test_the_direction_is_normalised_so_the_raw_delta_size_does_not_matter():
    target = vec(0.0, 0.0)
    reference = vec(3.0, 4.0)                # norm 5
    small = build_plan(state=1, intervention_type="donor_direction",
                       strength=0.2, target_state_vector=target,
                       donor_state_vector=vec(0.001, 0.0),
                       reference_vector=reference)
    large = build_plan(state=1, intervention_type="donor_direction",
                       strength=0.2, target_state_vector=target,
                       donor_state_vector=vec(1000.0, 0.0),
                       reference_vector=reference)
    assert torch.allclose(small.delta, large.delta)
    assert small.applied_delta_norm == pytest.approx(1.0)     # 0.2 * 5


def test_the_budget_scales_linearly_with_strength():
    target, donor = vec(0.0, 0.0), vec(1.0, 1.0)
    reference = vec(10.0, 0.0)
    norms = []
    for strength in (0.1, 0.2, 0.4):
        plan = build_plan(state=1, intervention_type="donor_direction",
                          strength=strength, target_state_vector=target,
                          donor_state_vector=donor, reference_vector=reference)
        norms.append(plan.applied_delta_norm)
    assert norms == pytest.approx([1.0, 2.0, 4.0])


def test_the_reference_defaults_to_the_target_when_none_is_given():
    target = vec(3.0, 4.0)                   # norm 5
    plan = build_plan(state=1, intervention_type="donor_direction",
                      strength=0.2, target_state_vector=target,
                      donor_state_vector=vec(3.0, 5.0))
    assert plan.reference_norm == pytest.approx(5.0)
    assert plan.applied_delta_norm == pytest.approx(1.0)


def test_the_donor_never_sets_the_budget():
    """A huge donor must not buy a huge edit: the budget is target-relative."""
    target = vec(1.0, 0.0)
    reference = vec(2.0, 0.0)
    modest = build_plan(state=1, intervention_type="donor_direction",
                        strength=0.5, target_state_vector=target,
                        donor_state_vector=vec(1.0, 1.0),
                        reference_vector=reference)
    enormous = build_plan(state=1, intervention_type="donor_direction",
                          strength=0.5, target_state_vector=target,
                          donor_state_vector=vec(1.0, 10_000.0),
                          reference_vector=reference)
    assert modest.applied_delta_norm == pytest.approx(1.0)
    assert enormous.applied_delta_norm == pytest.approx(1.0)


def test_unit_scaled_refuses_a_direction_that_does_not_exist():
    assert unit_scaled(vec(0.0, 0.0), 1.0) is None
    assert unit_scaled(vec(EPS / 10, 0.0), 1.0) is None
    assert unit_scaled(vec(float("nan"), 0.0), 1.0) is None
    assert unit_scaled(vec(3.0, 4.0), 10.0) is not None


def test_vector_norm_is_computed_in_float32():
    assert vector_norm(vec(3.0, 4.0)) == pytest.approx(5.0)
    assert vector_norm(torch.zeros(8)) == pytest.approx(0.0)


# --------------------------------------------- 2. self-patch is exactly zero
def test_a_self_patch_produces_no_intervention():
    target = vec(1.5, -2.0, 0.25)
    plan = build_plan(state=17, intervention_type="self_patch", strength=0.2,
                      target_state_vector=target,
                      donor_state_vector=None,
                      reference_vector=vec(1.0, 0.0, 0.0))
    assert plan.status == "zero_delta"
    assert plan.delta is None
    assert plan.replacement is None
    assert plan.applies_anything is False
    assert plan.raw_delta_norm == pytest.approx(0.0)
    assert plan.applied_delta_norm == pytest.approx(0.0)


def test_a_self_patch_ignores_any_donor_it_is_handed():
    target = vec(2.0, 2.0)
    with_donor = build_plan(state=3, intervention_type="self_patch",
                            strength=0.3, target_state_vector=target,
                            donor_state_vector=vec(99.0, -99.0))
    assert with_donor.status == "zero_delta"
    assert with_donor.applies_anything is False


def test_a_zero_delta_donor_is_recorded_not_forced():
    """Two identical states -- state 0 on this corpus -- apply nothing."""
    identical = vec(0.7, 0.7, 0.7)
    plan = build_plan(state=STATE_EMBEDDING,
                      intervention_type="donor_direction", strength=0.2,
                      target_state_vector=identical,
                      donor_state_vector=identical.clone())
    assert plan.status == "zero_delta"
    assert plan.applies_anything is False


# ---------------------------------------------- 3. the other two conditions
def test_the_no_intervention_condition_plans_nothing():
    plan = build_plan(state=12, intervention_type="none", strength=0.0)
    assert plan.status == "no_intervention"
    assert plan.applies_anything is False


def test_donor_replace_substitutes_the_whole_state():
    target, donor = vec(1.0, 1.0), vec(5.0, -5.0)
    plan = build_plan(state=8, intervention_type="donor_replace", strength=0.2,
                      target_state_vector=target, donor_state_vector=donor)
    assert plan.status == "ok"
    assert plan.delta is None
    assert torch.allclose(plan.replacement, donor)


def test_an_unknown_intervention_type_is_refused():
    with pytest.raises(InterventionError, match="unknown intervention"):
        build_plan(state=1, intervention_type="scale_it_up", strength=0.2,
                   target_state_vector=vec(1.0))


def test_a_non_positive_strength_is_refused_for_a_real_intervention():
    for strength in (0.0, -0.1):
        with pytest.raises(InterventionError, match="strength"):
            build_plan(state=1, intervention_type="donor_direction",
                       strength=strength, target_state_vector=vec(1.0),
                       donor_state_vector=vec(2.0))


def test_a_missing_donor_is_refused():
    with pytest.raises(InterventionError, match="donor"):
        build_plan(state=1, intervention_type="donor_direction", strength=0.2,
                   target_state_vector=vec(1.0))


def test_a_missing_target_is_refused():
    with pytest.raises(InterventionError, match="target"):
        build_plan(state=1, intervention_type="donor_direction", strength=0.2,
                   donor_state_vector=vec(1.0))


# ------------------------------- 4. donor and target of different lengths
def test_donor_and_target_prompts_may_be_any_lengths():
    """Each side is read at its own final prompt token, so lengths are free."""
    hidden = 4
    target_activations = torch.randn(1, 9, hidden)     # a 9-token prompt
    donor_activations = torch.randn(1, 41, hidden)     # a 41-token prompt

    target_vector = target_activations[0, final_prompt_token_index(9)]
    donor_vector = donor_activations[0, final_prompt_token_index(41)]
    assert target_vector.shape == donor_vector.shape == (hidden,)

    plan = build_plan(state=20, intervention_type="donor_direction",
                      strength=0.2, target_state_vector=target_vector,
                      donor_state_vector=donor_vector)
    assert plan.status in {"ok", "zero_delta"}
    if plan.status == "ok":
        assert plan.delta.shape == (hidden,)


def test_the_captured_vectors_are_the_last_row_of_each_prompt():
    target_activations = torch.arange(6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    donor_activations = torch.arange(2 * 3, dtype=torch.float32).reshape(1, 2, 3)
    assert torch.allclose(target_activations[0, final_prompt_token_index(6)],
                          torch.tensor([15.0, 16.0, 17.0]))
    assert torch.allclose(donor_activations[0, final_prompt_token_index(2)],
                          torch.tensor([3.0, 4.0, 5.0]))


def test_state_vectors_of_different_widths_are_refused():
    with pytest.raises(InterventionError, match="shape"):
        build_plan(state=1, intervention_type="donor_direction", strength=0.2,
                   target_state_vector=torch.zeros(4),
                   donor_state_vector=torch.zeros(8))


# ------------------------------------------- 5. applying it to an activation
def test_the_delta_lands_on_exactly_one_position():
    tensor = activation(5, hidden=3, fill=1.0)
    delta = vec(10.0, 0.0, -10.0)
    changed = apply_delta_at_position(tensor, final_prompt_token_index(5), delta)

    assert torch.allclose(changed[0, 4], torch.tensor([11.0, 1.0, -9.0]))
    assert torch.allclose(changed[0, :4], tensor[0, :4])
    assert torch.allclose(tensor, activation(5, hidden=3, fill=1.0))  # untouched


def test_replacement_overwrites_exactly_one_position():
    tensor = activation(4, hidden=2, fill=2.0)
    changed = replace_at_position(tensor, final_prompt_token_index(4),
                                  vec(-1.0, -1.0))
    assert torch.allclose(changed[0, 3], vec(-1.0, -1.0))
    assert torch.allclose(changed[0, :3], tensor[0, :3])


def test_applying_to_a_batch_is_refused():
    tensor = torch.zeros(2, 5, 3)
    with pytest.raises(InterventionError, match="batch size must be 1"):
        apply_delta_at_position(tensor, 4, vec(1.0, 1.0, 1.0))


def test_applying_to_a_non_activation_is_refused():
    with pytest.raises(InterventionError, match="batch, sequence, hidden"):
        apply_delta_at_position(torch.zeros(5, 3), 4, vec(1.0, 1.0, 1.0))


def test_the_delta_is_cast_to_the_activation_dtype():
    tensor = torch.zeros(1, 3, 2, dtype=torch.bfloat16)
    changed = apply_delta_at_position(tensor, 2, vec(1.0, 2.0))
    assert changed.dtype == torch.bfloat16
    assert changed[0, 2].float().tolist() == [1.0, 2.0]
