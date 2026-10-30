"""Norm-preserving rotation of a residual activation toward a fixed direction.

This module is arithmetic. It holds no model, opens no file, registers no hook
and reads no manifest, so every claim it makes is checkable on synthetic
tensors.

What the operator does
----------------------
Given an activation ``h`` and a fixed unit-ish direction ``s``, it rotates
``h`` **toward** ``s`` through a fraction ``beta`` of the angle that separates
them, inside the plane the two vectors span::

    r     = ||h||                      the norm that is preserved
    x     = h / r                      the activation's own unit vector
    y     = s / ||s||                  the target direction's unit vector
    c     = clamp(<x, y>, -1, 1)
    theta = arccos(c)                  the whole angle, per prompt and per layer
    v_hat = unit(y - c x)              the in-plane direction orthogonal to x
    phi   = beta * theta               the fraction actually traversed
    h_rot = r * (cos(phi) x + sin(phi) v_hat)

Three things follow from that form and are what the tests check:

* ``||h_rot|| == ||h||`` -- the edit moves the activation on the sphere and
  never changes how long it is. This is the whole difference from the additive
  screen, where ``h + alpha * ||block_input|| * d`` changes both;
* the remaining angle to ``y`` is ``(1 - beta) * theta``, so ``beta`` is a
  *fraction of a prompt-specific angle* and not a coefficient in the additive
  sense. ``beta = 0.15`` and ``alpha_S = 0.15`` are not the same amount of
  anything and are never compared as though they were;
* the size of the edit is the chord ``||h_rot - h|| = 2 r sin(beta theta / 2)``,
  which is a *derived* quantity here rather than a budget that was chosen.

**The direction is not rotated.** ``s`` is the frozen layer-specific
``d_S,l = H - E`` vector read from the validated bank. Only the activation
moves.

Precision
---------
All normalisation and trigonometry happen in float32, and the result is cast
back to the activation's own dtype at the end -- once, at the boundary. The
model runs in bfloat16, whose unit roundoff is about ``2 ** -8``; doing the
arccos there would put the angle's error above the smallest ``beta`` on the
grid. The cast is measured rather than assumed: norm preservation is recorded
before and after it, against two different predeclared tolerances, because
they are two different claims.

Fail-closed cases
-----------------
Four inputs have no defined rotation and are refused rather than approximated:
a non-finite activation or direction, a zero-length activation, a zero-length
direction, and an antiparallel pair -- where ``y - c x`` vanishes, the plane is
not unique, and every rotation "toward" ``y`` is as good as every other.

Two inputs are no-ops and return the original tensor **unchanged, bitwise**:
``beta == 0`` and an activation already aligned with the direction. Both are
recorded as the edge cases they are; neither is a silent pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .validation import ValidationError


class RotationError(ValidationError):
    """The rotation operator was asked for something it must not produce."""


# ------------------------------------------------------------- the tolerances
#
# Versioned and predeclared. They are written into the frozen manifest before
# any generation exists, so a later run cannot loosen one to make a record
# pass. Every value is chosen for float32 over 2048 hidden units; the
# post-cast entry is chosen for the dtype the model actually runs in.

#: Bumped whenever any number below changes. A run records the version it ran
#: under, so two runs at different tolerances are visibly different runs.
TOLERANCES_VERSION = 1


@dataclass(frozen=True)
class RotationTolerances:
    """Every numerical threshold the operator and its invariants use."""

    version: int = TOLERANCES_VERSION

    #: A vector shorter than this carries no direction. Dividing by its norm
    #: would turn rounding noise into a unit vector that looks exactly like a
    #: real one. Matches ``directionbank.MIN_DELTA_NORM``.
    norm_eps: float = 1e-6

    #: An angle at or below this is treated as already aligned. arccos is
    #: ill-conditioned near c = 1: at theta = 1e-6 the float32 cosine is
    #: 1 - 5e-13, far below float32 resolution, so the angle is not
    #: recoverable and the rotation is not attempted.
    angle_eps: float = 1e-6

    #: ``||y - c x||`` at or below this means the plane is not unique. It is
    #: sin(theta) analytically, so this is the antiparallel and the aligned
    #: case both; aligned is caught earlier by ``angle_eps`` and antiparallel
    #: is refused here.
    orthogonal_eps: float = 1e-6

    #: Norm preservation BEFORE the cast: the operator's own float32
    #: arithmetic. A 2048-term float32 reduction plus a sin/cos pair lands
    #: two orders inside this.
    norm_preservation_rtol_float32: float = 1e-5

    #: Norm preservation AFTER the cast, per dtype. This is the dtype's own
    #: unit roundoff and not the operator's error, which is why it is a
    #: separate number and a separate check: bfloat16 has 8 significand bits,
    #: so 2 * 2 ** -8 bounds it with room to spare.
    norm_preservation_rtol_after_cast: Mapping[str, float] = field(
        default_factory=lambda: {
            "torch.float64": 1e-9,
            "torch.float32": 1e-5,
            "torch.float16": 2e-3,
            "torch.bfloat16": 8e-3,
        })

    #: ``||h_rot - h|| / ||h||`` against ``2 sin(beta theta / 2)``.
    chord_rtol: float = 2e-5
    chord_atol: float = 1e-7

    #: The remaining angle against ``(1 - beta) theta``. Looser than the chord
    #: check because it passes through a second arccos.
    remaining_angle_rtol: float = 1e-3
    remaining_angle_atol: float = 1e-5

    #: How far the cosine with the target direction may DECREASE before a
    #: positive-beta rotation is refused. Zero is the claim; this is the
    #: float32 slack around it.
    cosine_non_decrease_atol: float = 1e-6

    def rtol_after_cast(self, dtype_name: str) -> float:
        """The post-cast norm tolerance for one dtype, or a refusal.

        An unlisted dtype is refused rather than given the float32 number: a
        dtype nobody predeclared a tolerance for is a dtype this operator has
        not been characterised on.
        """
        if dtype_name not in self.norm_preservation_rtol_after_cast:
            raise RotationError(
                f"no post-cast norm tolerance is predeclared for dtype "
                f"{dtype_name!r}; this operator is characterised on "
                f"{sorted(self.norm_preservation_rtol_after_cast)}. A dtype "
                f"without a declared tolerance is not given another dtype's.")
        return float(self.norm_preservation_rtol_after_cast[dtype_name])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "norm_eps": self.norm_eps,
            "angle_eps": self.angle_eps,
            "orthogonal_eps": self.orthogonal_eps,
            "norm_preservation_rtol_float32":
                self.norm_preservation_rtol_float32,
            "norm_preservation_rtol_after_cast":
                dict(sorted(self.norm_preservation_rtol_after_cast.items())),
            "chord_rtol": self.chord_rtol,
            "chord_atol": self.chord_atol,
            "remaining_angle_rtol": self.remaining_angle_rtol,
            "remaining_angle_atol": self.remaining_angle_atol,
            "cosine_non_decrease_atol": self.cosine_non_decrease_atol,
        }


#: The one instance every caller uses unless a test passes its own.
ROTATION_TOLERANCES = RotationTolerances()

# ------------------------------------------------------------- the edge cases
#: ``beta`` was exactly zero: the original tensor is returned, bitwise.
EDGE_BETA_ZERO = "beta_zero"
#: The activation already points along the direction: nothing to traverse.
EDGE_ALREADY_ALIGNED = "already_aligned"
#: No edge case; the rotation was performed.
EDGE_NONE = ""

EDGE_CASES = (EDGE_NONE, EDGE_BETA_ZERO, EDGE_ALREADY_ALIGNED)

#: ``beta`` is a fraction of an angle, so it lives in [0, 1]. 1 lands exactly
#: on the direction; anything above it would overshoot and is not "toward".
BETA_MIN = 0.0
BETA_MAX = 1.0


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is present in-container
        raise RotationError(
            f"the rotation operator needs torch: {exc}") from exc
    return torch


@dataclass
class RotationOutcome:
    """What one rotation produced: the tensor, and everything about it."""

    tensor: Any
    applied: bool
    edge_case: str
    diagnostics: Dict[str, Any]

    @property
    def is_identity(self) -> bool:
        return not self.applied


def _as_vector(value: Any, label: str, like: Any = None) -> Any:
    """One 1-D float32 tensor, on ``like``'s device when one is given."""
    torch = _require_torch()
    import numpy as np

    if not torch.is_tensor(value):
        value = torch.as_tensor(
            np.ascontiguousarray(np.asarray(value, dtype=np.float32)))
    if value.dim() != 1:
        raise RotationError(
            f"{label} has shape {tuple(value.shape)}; the rotation operates on "
            f"one hidden-state vector at one token position, which is 1-D")
    tensor = value.float()
    if like is not None:
        tensor = tensor.to(device=like.device)
    return tensor


def rotate_toward(h: Any, direction: Any, beta: float, *,
                  tolerances: RotationTolerances = ROTATION_TOLERANCES,
                  label: str = "activation") -> RotationOutcome:
    """Rotate ``h`` toward ``direction`` through ``beta`` of the angle.

    ``h`` is returned in its own dtype and on its own device. The direction is
    never modified, never renormalised in place, and never rotated: it is the
    fixed target ``d_S,l`` this screen was built around.
    """
    torch = _require_torch()

    if not torch.is_tensor(h):
        raise RotationError(
            f"{label}: the rotation is applied to a live activation tensor, "
            f"got {type(h).__name__}")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise RotationError(f"beta must be a number, got {beta!r}")
    beta = float(beta)
    if not math.isfinite(beta):
        raise RotationError(f"beta is {beta}; a fraction of an angle is finite")
    if not BETA_MIN <= beta <= BETA_MAX:
        raise RotationError(
            f"beta is {beta}; it is the FRACTION of the prompt-specific angle "
            f"that is traversed and lives in [{BETA_MIN}, {BETA_MAX}]. A value "
            f"outside that is not a rotation toward the direction.")

    h_original = h
    dtype_name = str(h_original.dtype)

    base: Dict[str, Any] = {
        "beta": beta,
        "tolerances_version": tolerances.version,
        "dtype_in": dtype_name,
        "dtype_out": dtype_name,
        "theta": None,
        "phi": None,
        "cosine_before": None,
        "cosine_after_float32": None,
        "cosine_after_cast": None,
        "norm_before_float32": None,
        "norm_after_float32": None,
        "norm_after_cast_float32": None,
        "relative_norm_error_float32": None,
        "relative_norm_error_after_cast": None,
        "relative_edit_norm_float32": None,
        "relative_edit_norm_after_cast": None,
        "theoretical_chord": None,
        "chord_absolute_error": None,
        "remaining_angle": None,
        "expected_remaining_angle": None,
        "edge_case": EDGE_NONE,
    }

    # ---- beta == 0 is the exact identity, before anything is computed ------
    #
    # Transcribed from the specification rather than reasoned to: returning
    # ``h`` itself is the only way "bitwise identity" is guaranteed regardless
    # of dtype, and computing an angle first would make that depend on the
    # arithmetic not perturbing anything.
    if beta == 0.0:
        return RotationOutcome(tensor=h_original, applied=False,
                               edge_case=EDGE_BETA_ZERO,
                               diagnostics={**base,
                                            "edge_case": EDGE_BETA_ZERO})

    h32 = h_original.float()
    s32 = _as_vector(direction, "the H - E direction", like=h32)
    if tuple(h32.shape) != tuple(s32.shape):
        raise RotationError(
            f"{label}: the activation has shape {tuple(h32.shape)} and the "
            f"direction {tuple(s32.shape)}; a mismatch means two different "
            f"states or two different models")

    if not bool(torch.isfinite(h32).all()):
        bad = int((~torch.isfinite(h32)).sum())
        raise RotationError(
            f"{label}: {bad} non-finite value(s) in the activation. A NaN or "
            f"an infinity is refused, never zeroed, dropped or rotated.")
    if not bool(torch.isfinite(s32).all()):
        bad = int((~torch.isfinite(s32)).sum())
        raise RotationError(
            f"{label}: {bad} non-finite value(s) in the H - E direction; the "
            f"frozen bank is finite and a direction that is not is not it")

    r = float(torch.linalg.vector_norm(h32).item())
    s_norm = float(torch.linalg.vector_norm(s32).item())
    if not math.isfinite(r) or not math.isfinite(s_norm):
        raise RotationError(
            f"{label}: a norm overflowed float32 (activation {r}, direction "
            f"{s_norm}); refusing rather than rotating an infinity")
    if not r > tolerances.norm_eps:
        raise RotationError(
            f"{label}: the activation's norm is {r:.6e}, at or below "
            f"{tolerances.norm_eps}. A zero-length activation has no direction "
            f"to rotate and no norm to preserve.")
    if not s_norm > tolerances.norm_eps:
        raise RotationError(
            f"{label}: the H - E direction's norm is {s_norm:.6e}, at or below "
            f"{tolerances.norm_eps}. There is nothing to rotate toward.")

    x = h32 / r
    y = s32 / s_norm

    c = float(torch.dot(x, y).item())
    c = max(-1.0, min(1.0, c))
    theta = math.acos(c)

    base.update({
        "theta": theta,
        "cosine_before": c,
        "norm_before_float32": r,
    })

    # ---- already aligned: unchanged, and said so -------------------------
    if theta <= tolerances.angle_eps:
        return RotationOutcome(
            tensor=h_original, applied=False, edge_case=EDGE_ALREADY_ALIGNED,
            diagnostics={**base, "edge_case": EDGE_ALREADY_ALIGNED,
                         "phi": beta * theta,
                         "cosine_after_float32": c,
                         "cosine_after_cast": c,
                         "norm_after_float32": r,
                         "norm_after_cast_float32": r,
                         "relative_norm_error_float32": 0.0,
                         "relative_norm_error_after_cast": 0.0,
                         "relative_edit_norm_float32": 0.0,
                         "relative_edit_norm_after_cast": 0.0,
                         "theoretical_chord": 2.0 * math.sin(beta * theta / 2),
                         "chord_absolute_error": abs(
                             2.0 * math.sin(beta * theta / 2)),
                         "remaining_angle": theta,
                         "expected_remaining_angle": (1.0 - beta) * theta})

    v = y - c * x
    v_norm = float(torch.linalg.vector_norm(v).item())
    if not v_norm > tolerances.orthogonal_eps:
        raise RotationError(
            f"{label}: the in-plane component ||y - c x|| is {v_norm:.6e}, at "
            f"or below {tolerances.orthogonal_eps}, with cosine {c:+.6f}. The "
            f"activation and the direction are (anti)parallel, so the plane "
            f"they span is not unique and every rotation 'toward' the "
            f"direction is as good as every other. Refused rather than "
            f"picking one.")

    v_hat = v / v_norm
    phi = beta * theta
    z = math.cos(phi) * x + math.sin(phi) * v_hat
    h_rot32 = r * z
    h_rot = h_rot32.to(h_original.dtype)

    # ---- what was actually produced, measured and not assumed ------------
    h_cast32 = h_rot.float()
    norm_after = float(torch.linalg.vector_norm(h_rot32).item())
    norm_after_cast = float(torch.linalg.vector_norm(h_cast32).item())
    cos_after = float(torch.dot(h_rot32 / max(norm_after, tolerances.norm_eps),
                                y).item())
    cos_after = max(-1.0, min(1.0, cos_after))
    cos_after_cast = float(
        torch.dot(h_cast32 / max(norm_after_cast, tolerances.norm_eps),
                  y).item())
    cos_after_cast = max(-1.0, min(1.0, cos_after_cast))
    edit_norm = float(torch.linalg.vector_norm(h_rot32 - h32).item())
    edit_norm_cast = float(torch.linalg.vector_norm(h_cast32 - h32).item())
    chord = 2.0 * math.sin(phi / 2.0)

    diagnostics = {
        **base,
        "phi": phi,
        "cosine_after_float32": cos_after,
        "cosine_after_cast": cos_after_cast,
        "norm_after_float32": norm_after,
        "norm_after_cast_float32": norm_after_cast,
        "relative_norm_error_float32": abs(norm_after - r) / r,
        "relative_norm_error_after_cast": abs(norm_after_cast - r) / r,
        "relative_edit_norm_float32": edit_norm / r,
        "relative_edit_norm_after_cast": edit_norm_cast / r,
        "theoretical_chord": chord,
        "chord_absolute_error": abs(edit_norm / r - chord),
        "remaining_angle": math.acos(cos_after),
        "expected_remaining_angle": (1.0 - beta) * theta,
        "edge_case": EDGE_NONE,
    }
    return RotationOutcome(tensor=h_rot, applied=True, edge_case=EDGE_NONE,
                           diagnostics=diagnostics)


def assert_rotation_invariants(
        diagnostics: Mapping[str, Any],
        tolerances: RotationTolerances = ROTATION_TOLERANCES,
        label: str = "rotation") -> None:
    """Every property the operator claims, checked on what it produced.

    Called at run time as well as in tests. A rotation that preserved the norm
    in a unit test and did not preserve it on a real activation is the failure
    this exists to make impossible to record, so the check travels with the
    edit rather than standing beside it.

    The float32 quantities carry the operator's claims. The post-cast norm is
    checked against its own dtype-specific tolerance, because the cast's error
    is the dtype's and not the operator's.
    """
    edge = diagnostics.get("edge_case", EDGE_NONE)
    if edge not in EDGE_CASES:
        raise RotationError(f"{label}: unknown edge case {edge!r}")
    if edge != EDGE_NONE:
        return                      # an identity return has nothing to verify

    beta = float(diagnostics["beta"])
    theta = float(diagnostics["theta"])
    r = float(diagnostics["norm_before_float32"])

    norm_error = float(diagnostics["relative_norm_error_float32"])
    if not norm_error <= tolerances.norm_preservation_rtol_float32:
        raise RotationError(
            f"{label}: the rotation changed the float32 norm by "
            f"{norm_error:.3e} relative, more than the predeclared "
            f"{tolerances.norm_preservation_rtol_float32:.3e}. A rotation that "
            f"does not preserve the norm is not the operator this screen "
            f"declared.")

    cast_error = float(diagnostics["relative_norm_error_after_cast"])
    cast_rtol = tolerances.rtol_after_cast(str(diagnostics["dtype_out"]))
    if not cast_error <= cast_rtol:
        raise RotationError(
            f"{label}: after casting to {diagnostics['dtype_out']} the norm "
            f"moved by {cast_error:.3e} relative, more than the "
            f"{cast_rtol:.3e} that dtype's unit roundoff explains")

    if beta > 0.0:
        before = float(diagnostics["cosine_before"])
        after = float(diagnostics["cosine_after_float32"])
        if after < before - tolerances.cosine_non_decrease_atol:
            raise RotationError(
                f"{label}: the cosine with H - E fell from {before:+.9f} to "
                f"{after:+.9f} under a positive beta. Positive strength moves "
                f"TOWARD the direction; a decrease means the sign convention "
                f"has inverted and nothing is generated under it.")

    remaining = float(diagnostics["remaining_angle"])
    expected = float(diagnostics["expected_remaining_angle"])
    if abs(remaining - expected) > (tolerances.remaining_angle_atol
                                    + tolerances.remaining_angle_rtol
                                    * abs(expected)):
        raise RotationError(
            f"{label}: the remaining angle is {remaining:.9f} and "
            f"(1 - beta) * theta is {expected:.9f} (beta {beta}, theta "
            f"{theta:.9f}); the traversal is not the declared fraction")

    edit = float(diagnostics["relative_edit_norm_float32"])
    chord = float(diagnostics["theoretical_chord"])
    if abs(edit - chord) > (tolerances.chord_atol
                            + tolerances.chord_rtol * abs(chord)):
        raise RotationError(
            f"{label}: the relative edit norm is {edit:.9e} and the chord "
            f"2 sin(beta theta / 2) is {chord:.9e}; the edit is not the one "
            f"this rotation implies (norm {r:.6e})")
