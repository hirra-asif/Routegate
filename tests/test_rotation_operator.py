"""The rotation operator: its identities, its precision, and what it refuses.

Every claim ``routegate.rotation`` makes is checked here on synthetic tensors.
No model, no bank, no corpus, no GPU: the operator is arithmetic and its
properties do not depend on where the activation came from.

The numbered comments correspond to the mathematical tests the screen's
specification requires. They are grouped by claim rather than by number so a
reader can follow the argument.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from routegate.rotation import (
    BETA_MAX,
    EDGE_ALREADY_ALIGNED,
    EDGE_BETA_ZERO,
    EDGE_NONE,
    ROTATION_TOLERANCES,
    TOLERANCES_VERSION,
    RotationError,
    RotationTolerances,
    assert_rotation_invariants,
    rotate_toward,
)

HIDDEN = 2048
BETA_GRID = (0.05, 0.10, 0.15, 0.20)


def vectors(seed: int = 20260830, scale: float = 7.5, dim: int = HIDDEN):
    """One activation and one unit direction, reproducibly."""
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(dim, generator=generator, dtype=torch.float32) * scale
    direction = torch.randn(dim, generator=generator, dtype=torch.float32)
    return h, direction / torch.linalg.vector_norm(direction)


def angle_between(a, b) -> float:
    cosine = float(torch.dot(a.float() / torch.linalg.vector_norm(a.float()),
                             b.float() / torch.linalg.vector_norm(b.float())))
    return math.acos(max(-1.0, min(1.0, cosine)))


# ============================================================ 1. beta = 0
def test_beta_zero_returns_the_original_tensor_exactly():
    h, direction = vectors()
    outcome = rotate_toward(h, direction, 0.0)
    assert outcome.tensor is h                       # not merely equal
    assert bool(torch.equal(outcome.tensor, h))
    assert outcome.applied is False
    assert outcome.edge_case == EDGE_BETA_ZERO


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16,
                                   torch.float16])
def test_beta_zero_is_bitwise_identity_in_every_dtype(dtype):
    h, direction = vectors()
    cast = h.to(dtype)
    outcome = rotate_toward(cast, direction, 0.0)
    assert bool(torch.equal(outcome.tensor, cast))
    assert outcome.tensor.dtype == dtype


def test_beta_zero_computes_nothing_and_says_so():
    """The identity return precedes the arithmetic, so no angle is recorded."""
    h, direction = vectors()
    diagnostics = rotate_toward(h, direction, 0.0).diagnostics
    assert diagnostics["theta"] is None
    assert diagnostics["cosine_before"] is None
    assert diagnostics["beta"] == 0.0
    assert diagnostics["tolerances_version"] == TOLERANCES_VERSION


# ================================================= 2. beta = 1 lands on y
def test_beta_one_points_toward_the_direction_and_preserves_the_norm():
    h, direction = vectors()
    before = float(torch.linalg.vector_norm(h))
    outcome = rotate_toward(h, direction, BETA_MAX)
    after = float(torch.linalg.vector_norm(outcome.tensor.float()))

    assert outcome.applied is True
    assert outcome.diagnostics["cosine_after_float32"] == pytest.approx(
        1.0, abs=1e-5)
    assert angle_between(outcome.tensor, direction) == pytest.approx(
        0.0, abs=1e-3)
    assert after == pytest.approx(
        before, rel=ROTATION_TOLERANCES.norm_preservation_rtol_float32)


def test_beta_above_one_is_not_a_rotation_toward_the_direction():
    h, direction = vectors()
    with pytest.raises(RotationError, match="FRACTION"):
        rotate_toward(h, direction, 1.5)


def test_a_negative_beta_is_refused():
    h, direction = vectors()
    with pytest.raises(RotationError, match="FRACTION"):
        rotate_toward(h, direction, -0.05)


# ========================== 3. positive beta never decreases the cosine
@pytest.mark.parametrize("beta", BETA_GRID)
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_positive_beta_never_decreases_the_cosine_with_the_direction(beta,
                                                                     seed):
    h, direction = vectors(seed=seed)
    outcome = rotate_toward(h, direction, beta)
    before = float(outcome.diagnostics["cosine_before"])
    after = float(outcome.diagnostics["cosine_after_float32"])
    assert after >= before - ROTATION_TOLERANCES.cosine_non_decrease_atol
    # And strictly increases when there is any angle at all to traverse.
    assert after > before


def test_a_cosine_that_fell_is_refused_by_the_invariants():
    """The runtime gate, not the operator: a flipped sign must not record."""
    h, direction = vectors()
    diagnostics = dict(rotate_toward(h, direction, 0.15).diagnostics)
    diagnostics["cosine_after_float32"] = (
        float(diagnostics["cosine_before"]) - 0.01)
    with pytest.raises(RotationError, match="sign convention"):
        assert_rotation_invariants(diagnostics)


# ================================ 4. the remaining angle is (1-beta)*theta
@pytest.mark.parametrize("beta", BETA_GRID + (0.5, 1.0))
def test_the_remaining_angular_distance_is_one_minus_beta_times_theta(beta):
    h, direction = vectors()
    outcome = rotate_toward(h, direction, beta)
    theta = float(outcome.diagnostics["theta"])
    remaining = angle_between(outcome.tensor, direction)
    assert remaining == pytest.approx(
        (1.0 - beta) * theta,
        rel=ROTATION_TOLERANCES.remaining_angle_rtol,
        abs=ROTATION_TOLERANCES.remaining_angle_atol)
    assert float(outcome.diagnostics["remaining_angle"]) == pytest.approx(
        (1.0 - beta) * theta,
        rel=ROTATION_TOLERANCES.remaining_angle_rtol,
        abs=ROTATION_TOLERANCES.remaining_angle_atol)


# ============================== 5. the relative edit norm is the chord
@pytest.mark.parametrize("beta", BETA_GRID)
@pytest.mark.parametrize("scale", [1e-2, 1e0, 1e2])
def test_the_relative_edit_norm_is_two_sin_half_beta_theta(beta, scale):
    h, direction = vectors(scale=scale)
    outcome = rotate_toward(h, direction, beta)
    theta = float(outcome.diagnostics["theta"])
    measured = float(torch.linalg.vector_norm(outcome.tensor.float() - h)
                     / torch.linalg.vector_norm(h))
    chord = 2.0 * math.sin(beta * theta / 2.0)
    assert measured == pytest.approx(chord,
                                     rel=ROTATION_TOLERANCES.chord_rtol,
                                     abs=ROTATION_TOLERANCES.chord_atol)
    assert float(outcome.diagnostics["theoretical_chord"]) == pytest.approx(
        chord, rel=1e-12)
    assert float(outcome.diagnostics["relative_edit_norm_float32"]) == \
        pytest.approx(chord, rel=ROTATION_TOLERANCES.chord_rtol,
                      abs=ROTATION_TOLERANCES.chord_atol)


def test_an_edit_that_is_not_the_chord_is_refused_by_the_invariants():
    h, direction = vectors()
    diagnostics = dict(rotate_toward(h, direction, 0.15).diagnostics)
    diagnostics["relative_edit_norm_float32"] = (
        float(diagnostics["theoretical_chord"]) * 1.5)
    with pytest.raises(RotationError, match="relative edit norm"):
        assert_rotation_invariants(diagnostics)


# ============ 6. norm preservation, before and after the cast, separately
@pytest.mark.parametrize("dtype,label", [(torch.float32, "torch.float32"),
                                         (torch.bfloat16, "torch.bfloat16"),
                                         (torch.float16, "torch.float16")])
@pytest.mark.parametrize("beta", BETA_GRID)
def test_norm_preservation_is_checked_before_and_after_the_cast(dtype, label,
                                                                beta):
    h, direction = vectors()
    cast = h.to(dtype)
    outcome = rotate_toward(cast, direction, beta)
    diagnostics = outcome.diagnostics

    assert diagnostics["dtype_out"] == label
    before = float(diagnostics["norm_before_float32"])

    # Before the cast: the operator's own float32 arithmetic.
    pre = float(diagnostics["norm_after_float32"])
    assert pre == pytest.approx(
        before, rel=ROTATION_TOLERANCES.norm_preservation_rtol_float32)
    assert float(diagnostics["relative_norm_error_float32"]) <= \
        ROTATION_TOLERANCES.norm_preservation_rtol_float32

    # After the cast: the dtype's own unit roundoff, a different tolerance.
    post = float(diagnostics["norm_after_cast_float32"])
    rtol = ROTATION_TOLERANCES.rtol_after_cast(label)
    assert post == pytest.approx(before, rel=rtol)
    assert float(diagnostics["relative_norm_error_after_cast"]) <= rtol
    assert bool(torch.equal(
        outcome.tensor,
        outcome.tensor.to(torch.float32).to(dtype)))     # already in dtype
    assert_rotation_invariants(diagnostics)


def test_the_two_norm_tolerances_are_different_numbers():
    """A single tolerance would let the cast's error hide the operator's."""
    assert (ROTATION_TOLERANCES.rtol_after_cast("torch.bfloat16")
            > ROTATION_TOLERANCES.norm_preservation_rtol_float32)


def test_an_undeclared_dtype_has_no_tolerance_and_is_refused():
    with pytest.raises(RotationError, match="predeclared"):
        ROTATION_TOLERANCES.rtol_after_cast("torch.float8_e4m3fn")


def test_a_post_cast_norm_beyond_its_dtype_tolerance_is_refused():
    h, direction = vectors()
    diagnostics = dict(
        rotate_toward(h.to(torch.bfloat16), direction, 0.15).diagnostics)
    diagnostics["relative_norm_error_after_cast"] = 0.5
    with pytest.raises(RotationError, match="unit roundoff"):
        assert_rotation_invariants(diagnostics)


def test_a_float32_norm_beyond_tolerance_is_refused():
    h, direction = vectors()
    diagnostics = dict(rotate_toward(h, direction, 0.15).diagnostics)
    diagnostics["relative_norm_error_float32"] = 1e-3
    with pytest.raises(RotationError, match="does not preserve the norm"):
        assert_rotation_invariants(diagnostics)


# ============================ 7. an already-aligned vector is unchanged
def test_an_aligned_activation_is_returned_unchanged():
    _, direction = vectors()
    h = direction * 4.0                              # exactly parallel
    outcome = rotate_toward(h, direction, 0.15)
    assert outcome.tensor is h
    assert bool(torch.equal(outcome.tensor, h))
    assert outcome.applied is False
    assert outcome.edge_case == EDGE_ALREADY_ALIGNED
    assert float(outcome.diagnostics["theta"]) <= \
        ROTATION_TOLERANCES.angle_eps


def test_a_near_parallel_activation_fails_closed_rather_than_guessing():
    """arccos is ill-conditioned at c = 1, and the operator does not pretend.

    A vector a hair off the direction has a true angle far below what float32
    can read back out of its cosine: ``acos`` of ``1 - 1.19e-7`` is about
    ``4.9e-4``, which is three orders larger than the angle really is. The
    angle gate therefore does not catch this case -- but the in-plane
    component does, because ``||y - c x||`` IS the sine and is measured
    directly rather than reconstructed. The rotation plane genuinely is not
    determined at that separation, so the operator refuses instead of
    rotating through an angle it cannot trust.
    """
    _, direction = vectors()
    perturbation = torch.zeros_like(direction)
    perturbation[0] = 1e-9
    h = (direction + perturbation) * 3.0
    with pytest.raises(RotationError, match="not unique"):
        rotate_toward(h, direction, 0.20)


def test_an_edge_case_return_has_nothing_for_the_invariants_to_check():
    _, direction = vectors()
    outcome = rotate_toward(direction * 2.0, direction, 0.15)
    assert_rotation_invariants(outcome.diagnostics)      # does not raise


# ================================================ 8. the fail-closed cases
def test_an_antiparallel_activation_fails_closed():
    _, direction = vectors()
    with pytest.raises(RotationError, match="not unique"):
        rotate_toward(direction * -3.0, direction, 0.15)


def test_a_zero_norm_activation_fails_closed():
    _, direction = vectors()
    with pytest.raises(RotationError, match="zero-length activation"):
        rotate_toward(torch.zeros(HIDDEN), direction, 0.15)


def test_a_zero_norm_direction_fails_closed():
    h, _ = vectors()
    with pytest.raises(RotationError, match="nothing to rotate toward"):
        rotate_toward(h, torch.zeros(HIDDEN), 0.15)


def test_a_nan_activation_fails_closed():
    h, direction = vectors()
    h = h.clone()
    h[17] = float("nan")
    with pytest.raises(RotationError, match="non-finite"):
        rotate_toward(h, direction, 0.15)


def test_an_inf_activation_fails_closed():
    h, direction = vectors()
    h = h.clone()
    h[3] = float("inf")
    with pytest.raises(RotationError, match="non-finite"):
        rotate_toward(h, direction, 0.15)


def test_a_nan_direction_fails_closed():
    h, direction = vectors()
    direction = direction.clone()
    direction[5] = float("nan")
    with pytest.raises(RotationError, match="non-finite"):
        rotate_toward(h, direction, 0.15)


def test_an_inf_direction_fails_closed():
    h, direction = vectors()
    direction = direction.clone()
    direction[9] = float("inf")
    with pytest.raises(RotationError, match="non-finite"):
        rotate_toward(h, direction, 0.15)


def test_a_non_finite_beta_fails_closed():
    h, direction = vectors()
    with pytest.raises(RotationError, match="finite"):
        rotate_toward(h, direction, float("nan"))


def test_a_shape_mismatch_fails_closed():
    h, _ = vectors()
    with pytest.raises(RotationError, match="mismatch"):
        rotate_toward(h, torch.ones(HIDDEN // 2), 0.15)


def test_a_two_dimensional_direction_fails_closed():
    h, _ = vectors()
    with pytest.raises(RotationError, match="1-D"):
        rotate_toward(h, torch.ones(2, HIDDEN), 0.15)


def test_a_non_tensor_activation_fails_closed():
    _, direction = vectors()
    with pytest.raises(RotationError, match="live activation tensor"):
        rotate_toward([1.0] * HIDDEN, direction, 0.15)


def test_a_boolean_beta_fails_closed():
    h, direction = vectors()
    with pytest.raises(RotationError, match="beta must be a number"):
        rotate_toward(h, direction, True)


# ============================================ the direction is not rotated
def test_the_direction_is_never_modified():
    h, direction = vectors()
    original = direction.clone()
    rotate_toward(h, direction, 0.20)
    assert bool(torch.equal(direction, original))


def test_the_activation_is_never_modified_in_place():
    h, direction = vectors()
    original = h.clone()
    outcome = rotate_toward(h, direction, 0.20)
    assert bool(torch.equal(h, original))
    assert outcome.tensor is not h


# ================================================= the operator is a rotation
@pytest.mark.parametrize("beta", BETA_GRID)
def test_the_rotation_stays_in_the_plane_the_two_vectors_span(beta):
    """The edit is a combination of x and y and of nothing else."""
    h, direction = vectors(dim=64)
    outcome = rotate_toward(h, direction, beta)
    x = h / torch.linalg.vector_norm(h)
    y = direction / torch.linalg.vector_norm(direction)
    basis = torch.stack([x, y])
    # Project the result onto the span and check nothing is left over.
    gram = basis @ basis.T
    coefficients = torch.linalg.solve(gram, basis @ outcome.tensor.float())
    residual = outcome.tensor.float() - coefficients @ basis
    assert float(torch.linalg.vector_norm(residual)) < 1e-4 * float(
        torch.linalg.vector_norm(h))


def test_the_rotation_is_monotone_in_beta():
    """A larger beta lands closer to the direction, at every step of the grid."""
    h, direction = vectors()
    start = float(rotate_toward(h, direction, BETA_GRID[0])
                  .diagnostics["cosine_before"])
    cosines = [start] + [
        float(rotate_toward(h, direction, beta)
              .diagnostics["cosine_after_float32"])
        for beta in BETA_GRID]
    assert cosines == sorted(cosines)
    assert cosines[-1] > cosines[0]


# ============================================================ the tolerances
def test_the_tolerances_are_versioned_and_recordable():
    recorded = ROTATION_TOLERANCES.as_dict()
    assert recorded["version"] == TOLERANCES_VERSION
    for key in ("norm_eps", "angle_eps", "orthogonal_eps",
                "norm_preservation_rtol_float32",
                "norm_preservation_rtol_after_cast", "chord_rtol",
                "chord_atol", "remaining_angle_rtol", "remaining_angle_atol",
                "cosine_non_decrease_atol"):
        assert key in recorded
    assert "torch.bfloat16" in recorded["norm_preservation_rtol_after_cast"]


def test_a_caller_may_supply_its_own_tolerances():
    """So a tightened gate is a decision at a call site, not a global edit."""
    h, direction = vectors()
    strict = RotationTolerances(norm_preservation_rtol_float32=1e-12)
    diagnostics = rotate_toward(h, direction, 0.15,
                                tolerances=strict).diagnostics
    assert diagnostics["tolerances_version"] == strict.version
    with pytest.raises(RotationError, match="does not preserve the norm"):
        assert_rotation_invariants(diagnostics, strict)


def test_an_unknown_edge_case_label_is_refused():
    h, direction = vectors()
    diagnostics = dict(rotate_toward(h, direction, 0.15).diagnostics)
    diagnostics["edge_case"] = "made_up"
    with pytest.raises(RotationError, match="unknown edge case"):
        assert_rotation_invariants(diagnostics)


def test_a_performed_rotation_carries_no_edge_case():
    h, direction = vectors()
    assert rotate_toward(h, direction, 0.15).edge_case == EDGE_NONE
