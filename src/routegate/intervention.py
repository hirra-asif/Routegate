"""The causal-patching intervention: what is applied, where, and how big it is.

This module is the arithmetic and the bookkeeping. It holds no model, opens no
file and registers no hook, so every claim it makes can be tested on synthetic
tensors.

The operation is recovered from the historical RQ1 implementation
(`pathway_probe.build_site_plan`, site ``residual_stream``) and from the RQ2
frozen actuator, which reuses that same function verbatim. It is **not** a full
replacement and **not** a raw donor-minus-target addition:

    raw_delta = donor_state - target_state          (at the patched state)
    unit      = raw_delta / ||raw_delta||
    budget    = strength * ||reference||            (a target-side norm)
    applied   = unit * budget                       (added, not substituted)

Two details of the historical version matter enough to be named.

**The reference norm is not the tensor being patched.** RQ1 computed the budget
from the decoder block's *input* hidden state -- captured by a forward
pre-hook -- while applying the delta to the block's *output*. That gave one
common residual-space budget across four different intervention sites, so their
edit sizes were comparable. It is preserved here as the default, because
changing it would change every number's meaning, and it is written into the run
record as ``budget_reference`` rather than left implicit.

**State numbering is not the historical layer numbering.** RQ1 spoke of decoder
blocks by zero-based index: "layers 31, 32, 33". This package numbers *states*,
where state 0 is the embedding output and states 1..48 are transformer-block
outputs, so historical block 31 is state 32. The translation is done once, here,
by :func:`state_to_site`, and never in prose.

State 0 is patchable and is expected to do nothing: on this corpus the rendered
chat prompt ends with the same token for every prompt, so the donor and target
embedding outputs are identical, ``raw_delta`` is zero and the plan records
``zero_delta``. That is a built-in control, not a failure.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # torch is needed to apply an intervention, not to reason about one
    import torch

    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from .validation import ValidationError

EPS = 1e-8

# ------------------------------------------------------------- state numbering
STATE_EMBEDDING = 0
EMBEDDING_SITE = "embedding_output"
BLOCK_OUTPUT_SITE = "block_output"

# The four conditions this stage can execute. "none" is the untouched baseline;
# "self_patch" builds the delta from the target against itself, which is zero by
# construction and therefore a control that proves the machinery is inert when
# it should be; "donor_direction" is the historical RQ1/RQ2 operation;
# "donor_replace" is the historical exact-donor-replay positive control, kept
# available because it is the strongest possible intervention and bounds what
# the direction edit could ever achieve.
INTERVENTION_TYPES = ("none", "self_patch", "donor_direction", "donor_replace")

# Where the budget norm is read from. The historical default is the patched
# block's input; the alternative is the tensor actually being modified.
BUDGET_REFERENCES = ("block_input", "block_output")


class InterventionError(ValidationError):
    """The intervention was asked for something it must not do."""


@dataclass(frozen=True)
class HookSite:
    """Which module to hook, and whether its input or its output carries the state."""

    state: int
    kind: str                 # EMBEDDING_SITE or BLOCK_OUTPUT_SITE
    block_index: Optional[int]  # zero-based decoder block, None for the embedding
    tensor: str               # always "output": every state is a module output

    def as_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        if record["block_index"] is None:
            record["block_index"] = ""
        return record


def n_states_for(n_layers: int) -> int:
    """49 for a 48-block model: the embedding output plus one per block."""
    if n_layers < 1:
        raise InterventionError(f"a model needs at least one block, got {n_layers}")
    return n_layers + 1


def state_to_site(state: int, n_layers: int) -> HookSite:
    """The module whose **output** is this state.

    state 0      -> the embedding module's output
    state s >= 1 -> decoder block ``s - 1``'s output

    The historical code indexed decoder blocks directly, so its "layer 31" is
    this package's state 32. Nothing else in this package converts between the
    two numbering schemes.
    """
    if not isinstance(state, int) or isinstance(state, bool):
        raise InterventionError(f"state must be an int, got {state!r}")
    if not 0 <= state < n_states_for(n_layers):
        raise InterventionError(
            f"state {state} is outside 0..{n_states_for(n_layers) - 1} for a "
            f"{n_layers}-block model")
    if state == STATE_EMBEDDING:
        return HookSite(state=state, kind=EMBEDDING_SITE, block_index=None,
                        tensor="output")
    return HookSite(state=state, kind=BLOCK_OUTPUT_SITE, block_index=state - 1,
                    tensor="output")


def historical_layer_to_state(block_index: int) -> int:
    """Translate a historical zero-based decoder-block index into a state.

    RQ1's band ``[31, 32, 33]`` is states ``[32, 33, 34]``.
    """
    if block_index < 0:
        raise InterventionError(f"block index {block_index} is negative")
    return block_index + 1


def budget_reference_state(state: int, reference: str = "block_input") -> Tuple[int, str]:
    """(state to read the budget norm from, a note about what that is).

    For ``block_input`` -- the historical default -- the reference for state
    ``s`` is state ``s - 1``, because the input of decoder block ``s - 1`` *is*
    the previous state. State 0 has no predecessor, so it references itself;
    that is a deviation from the historical rule for a state the historical run
    never touched, and it is returned as an explicit note rather than hidden.
    """
    if reference not in BUDGET_REFERENCES:
        raise InterventionError(
            f"unknown budget reference {reference!r}; valid: {list(BUDGET_REFERENCES)}")
    if reference == "block_output":
        return state, "the patched tensor's own norm"
    if state == STATE_EMBEDDING:
        return STATE_EMBEDDING, ("state 0 has no preceding state, so it "
                                 "references its own norm; this is a documented "
                                 "deviation from the historical block-input rule")
    return state - 1, ("the patched block's input hidden state, which is the "
                       "preceding state; this is the historical RQ1/RQ2 rule")


def capture_states_for(states: Sequence[int], reference: str = "block_input"
                       ) -> List[int]:
    """Every state a run must capture: the patched ones and their references."""
    wanted = set()
    for state in states:
        wanted.add(int(state))
        wanted.add(budget_reference_state(int(state), reference)[0])
    return sorted(wanted)


# --------------------------------------------------------------- the operation
def vector_norm(vector: Any) -> float:
    """L2 norm in float32, as the historical implementation computed it."""
    _require_torch()
    return float(torch.linalg.vector_norm(vector.float()).item())


def unit_scaled(delta: Any, budget: float) -> Optional[Any]:
    """``delta`` rescaled to L2 norm ``budget``, or None if it has no direction.

    A zero (or non-finite) delta has no direction to scale, and inventing one
    would turn a control into an intervention. The caller records ``zero_delta``
    and applies nothing -- which is exactly what should happen at state 0.
    """
    _require_torch()
    norm = vector_norm(delta)
    if not math.isfinite(norm) or norm < EPS:
        return None
    return delta.float() * (budget / norm)


@dataclass
class InterventionPlan:
    """What to do at one state, for one target, in one condition."""

    state: int
    intervention_type: str
    strength: float
    status: str                       # "ok" | "zero_delta" | "no_intervention"
    delta: Optional[Any] = None       # the vector to add, float32, CPU
    replacement: Optional[Any] = None  # the vector to substitute, float32, CPU
    budget: float = 0.0
    budget_reference_state: Optional[int] = None
    budget_reference_note: str = ""
    reference_norm: float = 0.0
    raw_delta_norm: float = 0.0
    applied_delta_norm: float = 0.0

    @property
    def applies_anything(self) -> bool:
        return self.delta is not None or self.replacement is not None

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "intervention_type": self.intervention_type,
            "strength": self.strength,
            "status": self.status,
            "budget": self.budget,
            "budget_reference_state": (self.budget_reference_state
                                       if self.budget_reference_state is not None
                                       else ""),
            "budget_reference_note": self.budget_reference_note,
            "reference_norm": self.reference_norm,
            "raw_delta_norm": self.raw_delta_norm,
            "applied_delta_norm": self.applied_delta_norm,
        }


def build_plan(state: int,
               intervention_type: str,
               strength: float,
               target_state_vector: Optional[Any] = None,
               donor_state_vector: Optional[Any] = None,
               reference_vector: Optional[Any] = None,
               budget_reference: str = "block_input") -> InterventionPlan:
    """The historical operation, for one state.

    ``target_state_vector`` and ``donor_state_vector`` are the captured vectors
    at the patched state, each taken at its **own** final prompt token, so the
    two prompts may be any lengths at all. ``reference_vector`` is the vector
    the budget norm is read from -- the target's, always: the edit is sized
    against the thing being edited, never against the donor.
    """
    _require_torch()
    if intervention_type not in INTERVENTION_TYPES:
        raise InterventionError(
            f"unknown intervention type {intervention_type!r}; valid: "
            f"{list(INTERVENTION_TYPES)}")
    if intervention_type != "none" and strength <= 0.0:
        raise InterventionError(
            f"strength must be positive for {intervention_type!r}, got {strength}")

    reference_state, note = budget_reference_state(state, budget_reference)
    plan = InterventionPlan(state=state, intervention_type=intervention_type,
                            strength=float(strength), status="no_intervention",
                            budget_reference_state=reference_state,
                            budget_reference_note=note)

    if intervention_type == "none":
        return plan

    if target_state_vector is None:
        raise InterventionError(
            f"{intervention_type!r} needs the target's captured state vector")

    if intervention_type == "donor_replace":
        if donor_state_vector is None:
            raise InterventionError("donor_replace needs the donor's state vector")
        _require_same_shape(target_state_vector, donor_state_vector)
        plan.replacement = donor_state_vector.float().clone()
        plan.status = "ok"
        plan.raw_delta_norm = vector_norm(
            donor_state_vector.float() - target_state_vector.float())
        plan.applied_delta_norm = plan.raw_delta_norm
        return plan

    # Both remaining types are the same arithmetic; only the donor differs.
    # self_patch uses the target as its own donor, so raw_delta is exactly zero
    # and nothing is applied -- the control that proves an inert run is inert.
    donor = (target_state_vector if intervention_type == "self_patch"
             else donor_state_vector)
    if donor is None:
        raise InterventionError("donor_direction needs the donor's state vector")
    _require_same_shape(target_state_vector, donor)

    reference = (reference_vector if reference_vector is not None
                 else target_state_vector)
    plan.reference_norm = vector_norm(reference)
    plan.budget = float(strength) * plan.reference_norm

    raw_delta = donor.float() - target_state_vector.float()
    plan.raw_delta_norm = vector_norm(raw_delta)
    scaled = unit_scaled(raw_delta, plan.budget)
    if scaled is None:
        plan.status = "zero_delta"
        return plan

    plan.delta = scaled
    plan.applied_delta_norm = vector_norm(scaled)
    plan.status = "ok"
    return plan


def apply_delta_at_position(tensor: Any, position: int, delta: Any) -> Any:
    """Add ``delta`` at one sequence position of a ``[batch, seq, hidden]`` tensor.

    Returns a modified copy: the original activation is never mutated in place,
    so a hook cannot corrupt a tensor another hook is still reading.
    """
    _require_torch()
    _require_activation_shape(tensor)
    changed = tensor.clone()
    changed[0, position] = changed[0, position] + delta.to(
        device=tensor.device, dtype=tensor.dtype)
    return changed


def replace_at_position(tensor: Any, position: int, replacement: Any) -> Any:
    """Substitute one sequence position of a ``[batch, seq, hidden]`` tensor."""
    _require_torch()
    _require_activation_shape(tensor)
    changed = tensor.clone()
    changed[0, position] = replacement.to(device=tensor.device, dtype=tensor.dtype)
    return changed


def final_prompt_token_index(sequence_length: int) -> int:
    """The row to patch: the last token of the rendered prompt.

    Batch size is one throughout, so there is no padding and the last row is
    unambiguously the final prompt token -- the position immediately before the
    first generated answer token, which is the position whose hidden state
    decides that token. With a batch and left padding it would not be, which is
    why the batch size is pinned rather than merely defaulted.
    """
    if sequence_length < 1:
        raise InterventionError(
            f"a prompt cannot have {sequence_length} tokens")
    return sequence_length - 1


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised only without torch
        raise InterventionError(
            f"this operation needs torch, which failed to import: "
            f"{_TORCH_IMPORT_ERROR}")


def _require_same_shape(first: Any, second: Any) -> None:
    if tuple(first.shape) != tuple(second.shape):
        raise InterventionError(
            f"donor and target state vectors disagree on shape: "
            f"{tuple(first.shape)} vs {tuple(second.shape)}. A state vector is "
            f"one row of hidden units; a mismatch means two different states or "
            f"two different models were compared.")


def _require_activation_shape(tensor: Any) -> None:
    if tensor.dim() != 3:
        raise InterventionError(
            f"expected a [batch, sequence, hidden] activation, got shape "
            f"{tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise InterventionError(
            f"batch size must be 1 so that the final row is the final prompt "
            f"token and never a pad token; got batch {tensor.shape[0]}")
