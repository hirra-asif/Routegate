"""The one-pass actuator budget, measured against the validated two-pass one.

The validated actuator sizes its edit against the target's own residual norm at
the *block input* -- the preceding state -- read from an **unsteered** capture
taken in a separate forward pass::

    delta(i, l) = alpha_R * ||h^(l)|| * d_R(l)          l = 31, 32, 33

A deployable controller wants one forward pass, not two. In one pass the
budgets are read as the prefill goes past, and the question this module answers
is how far that departs from the validated convention.

The declared one-pass convention: **snapshot the norm before writing.**

    block 31's budget  ||block 30's output||   nothing upstream is edited,
                                               so this IS the validated value
    block 32's budget  ||block 31's output||   read at the block-31 hook BEFORE
                                               delta_31 is added, so this is
                                               again the validated value
    block 33's budget  ||block 32's output||   read before delta_32 is added --
                                               but block 32 has already
                                               consumed an edited state 32, so
                                               this one is NOT the validated
                                               value and cannot be made into it
                                               without a second pass

Blocks 31 and 32 are therefore *mathematically* identical under the two
conventions. They are not bit-identical, and this module does not ask them to
be: the two numbers are produced by two separate forward passes, and one norm
is reduced on the accelerator while the other is reduced on the CPU the capture
was moved to. Two float32 reductions of the same 2048 values in different
orders disagree in the last bits. Version 1 of this check demanded bit-identity
across those two passes and refused a complete 480-prompt run whose worst
block-31 disagreement was 3.527095e-07 -- about three float32 epsilons, and
some four orders of magnitude below the bfloat16 resolution the activations are
even stored at. That was a false negative produced by the check, not a fault in
the actuator.

What is exact, and is proved per row inside a single forward pass, is that the
budget is read before the activation is edited: at block ``b``'s hook the
snapshot taken for block ``b + 1`` and the tensor block ``b``'s edit lands on
are the same tensor, so if the read moved after the write their difference
would collapse to zero. It does not, and ``verify_read_before_write`` refuses
the run if it ever does. That invariant lives inside one pass, one device and
one dtype, where no reduction order can reach it.

Every cross-forward comparison -- blocks 31, 32 and 33 alike -- is gated at the
predeclared ``TOLERANCE``. Block 33 is the whole question, and answering it
needs the real frozen model: the deviation there is whatever block 32's
attention and MLP do to a perturbed input, which no closed form supplies.

A stricter variant is measured alongside it at block 33 -- the norm of block
33's *actual* runtime input, which carries delta_32 as well -- so the gate is
reported against the favourable convention and the unfavourable one is on the
record beside it.

What this module does not do
----------------------------
It trains nothing, generates no token, judges nothing and selects nothing. It
reads budgets and subtracts them. The edit arithmetic is
``steering.build_steering_plan`` -- the validated actuator itself, not a
restatement of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hooks import (
    PREFILL_FORWARD_INDEX,
    first_tensor,
    replace_first_tensor,
    resolve_modules,
)
from .intervention import (
    apply_delta_at_position,
    final_prompt_token_index,
    vector_norm,
)
from .validation import Checklist, ValidationError

SCHEMA = "routegate/one_pass_budget_equivalence/1"

#: The validated actuator's sites, as historical decoder blocks.
ACTUATOR_BLOCKS: Tuple[int, ...] = (31, 32, 33)

#: The frozen validated repair strength. g_R = 1 is the worst case for this
#: check: the edit, and therefore the divergence it causes, is largest there.
ALPHA_R = 0.15
WORST_CASE_G_R = 1.0

#: The frozen tolerance. Not a knob: a larger relative deviation than this
#: means the one-pass budget is not the validated budget, and the actuator is
#: not silently redefined to make it pass.
TOLERANCE = 1e-2

#: Blocks whose budget is *mathematically* identical under both conventions,
#: because nothing upstream of their reference has been edited when it is read.
#:
#: Their measured difference is NOT expected to be bit-identical. The two
#: numbers come from two separate forward passes, and one norm is reduced on
#: the accelerator while the other is reduced on the CPU after the capture has
#: been moved there. Two float32 reductions over 2048 elements in different
#: orders differ in the last bits, which is a property of floating-point
#: addition and not of this pipeline. Demanding bit-identity across two
#: forwards asserts something the hardware never promised; v1 of this check did
#: exactly that and refused a run whose worst disagreement was 3.527095e-07,
#: about three float32 epsilons.
#:
#: What IS exact, and is proved per row inside a single forward pass, is that
#: the budget is read before the activation is edited. See
#: ``verify_read_before_write``.
MATHEMATICALLY_IDENTICAL_BLOCKS: Tuple[int, ...] = (31, 32)

#: Blocks at which the snapshot and the edit touch the same tensor, so
#: read-before-write can be proved within one forward pass.
SAME_FORWARD_INVARIANT_BLOCKS: Tuple[int, ...] = (31, 32)

#: The block the check exists for: the only one where the two conventions
#: differ by construction rather than by rounding.
MEASURED_BLOCK = 33

#: The float32 scale at which two differently ordered reductions of the same
#: vector may disagree. Recorded so a reader can tell "rounding" from "a
#: different activation" at a glance. It is NOT a gate and nothing is compared
#: against it: every cross-forward comparison is gated at ``TOLERANCE``.
FLOAT32_REDUCTION_SCALE = 1e-5


class BudgetEquivalenceError(ValidationError):
    """The budget comparison was asked for something it must not do."""


# ------------------------------------------------------------ layer numbering
def block_input_state(block: int) -> int:
    """The package state carrying historical block ``block``'s input.

    Block ``l``'s input is the output of block ``l - 1``, which this package
    stores as state ``l``. The budget reference for an edit at block ``l`` is
    therefore state ``l`` -- one below the state the edit lands on.
    """
    if block < 1:
        raise BudgetEquivalenceError(
            f"historical block {block} has no preceding decoder block, so it "
            f"has no block-input budget reference in this convention")
    return int(block)


def block_output_state(block: int) -> int:
    """The package state carrying historical block ``block``'s output."""
    if block < 0:
        raise BudgetEquivalenceError(f"{block} is not a decoder block")
    return int(block) + 1


def budget_reference_states(blocks: Sequence[int] = ACTUATOR_BLOCKS
                            ) -> Tuple[int, ...]:
    """Every state a two-pass reference capture must hold."""
    return tuple(block_input_state(b) for b in blocks)


def budget_norm(vector: Any) -> float:
    """The one way a budget norm is computed, on either side of the comparison.

    Both conventions are measured through this function and through no other,
    so a difference between them is a difference in the activation and never a
    difference in how its norm was taken. ``intervention.vector_norm``
    upcasts to float32 exactly as the historical implementation did.
    """
    return vector_norm(vector)


# ------------------------------------------------------------ the measurement
@dataclass
class RowBudgets:
    """One prompt's budgets under both conventions."""

    row_id: str
    two_pass: Dict[int, float] = field(default_factory=dict)
    one_pass: Dict[int, float] = field(default_factory=dict)
    #: Block 33 only: the norm of block 33's ACTUAL runtime input, which also
    #: carries delta_32. Strictly worse than the declared convention.
    one_pass_runtime_input: Dict[int, float] = field(default_factory=dict)
    #: Same-forward proof, keyed by the state whose budget was snapshotted:
    #: ``|| post-write activation - snapshot ||`` at that position. Zero would
    #: mean the snapshot was taken after the edit. See
    #: ``verify_read_before_write``.
    read_before_write: Dict[int, float] = field(default_factory=dict)
    prefill_tokens: int = 0
    generated_tokens: int = 0

    def relative(self, block: int) -> float:
        ref = self.two_pass[block]
        if ref <= 0.0:
            raise BudgetEquivalenceError(
                f"{self.row_id}: the two-pass budget at block {block} is "
                f"{ref}; a relative difference against it is undefined")
        return abs(self.one_pass[block] - ref) / ref

    def relative_runtime(self, block: int) -> Optional[float]:
        if block not in self.one_pass_runtime_input:
            return None
        ref = self.two_pass[block]
        return abs(self.one_pass_runtime_input[block] - ref) / ref

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"row_id": self.row_id,
                               "prefill_tokens": self.prefill_tokens,
                               "generated_tokens": self.generated_tokens}
        for block in sorted(self.two_pass):
            out[f"two_pass_r{block}"] = self.two_pass[block]
            out[f"one_pass_r{block}"] = self.one_pass[block]
            out[f"rel_diff_block_{block}"] = self.relative(block)
            runtime = self.relative_runtime(block)
            if runtime is not None:
                out[f"one_pass_runtime_input_r{block}"] = \
                    self.one_pass_runtime_input[block]
                out[f"rel_diff_runtime_input_block_{block}"] = runtime
        return out


def summarise(rows: Sequence[RowBudgets], block: int,
              tolerance: float = TOLERANCE,
              runtime_input: bool = False) -> Dict[str, Any]:
    """The per-block distribution of relative differences, and who exceeded."""
    if not rows:
        raise BudgetEquivalenceError(
            "no rows to summarise; a comparison over nothing is not a result")
    if runtime_input:
        pairs = [(r.row_id, r.relative_runtime(block)) for r in rows]
        pairs = [(rid, v) for rid, v in pairs if v is not None]
        if not pairs:
            return {"block": block, "variant": "runtime_input",
                    "measured": False}
    else:
        pairs = [(r.row_id, r.relative(block)) for r in rows]
    values = np.array([v for _, v in pairs], dtype=np.float64)
    exceeding = sorted(rid for rid, v in pairs if v > tolerance)
    return {
        "block": block,
        "state_edited": block_output_state(block),
        "budget_reference_state": block_input_state(block),
        "variant": "runtime_input" if runtime_input else "declared_one_pass",
        "measured": True,
        "rows": int(values.size),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "tolerance": float(tolerance),
        "n_exceeding_tolerance": len(exceeding),
        "row_ids_exceeding_tolerance": exceeding,
        "within_tolerance": bool(values.max() <= tolerance),
    }


def verify_read_before_write(rows: Sequence[RowBudgets],
                             blocks: Sequence[int]
                             = SAME_FORWARD_INVARIANT_BLOCKS,
                             checks: Optional[Checklist] = None
                             ) -> Dict[int, Dict[str, Any]]:
    """The budget was read before the activation was edited. Proved, not assumed.

    This is the invariant v1 was reaching for and expressed in the wrong
    currency. It is checked **within one forward pass**, on one device, in one
    dtype, so no floating-point reduction order can move it:

    at block ``b``'s hook the snapshot for block ``b + 1``'s budget is taken
    from the very tensor block ``b``'s own edit then lands on. If the snapshot
    were taken after the write, the snapshot and the post-write activation
    would be the same vector and their difference would be exactly zero. It is
    instead the applied delta, whose norm the actuator already reports.

    So the invariant is: for every row, at every block where both happen,
    ``|| post_write - snapshot || == || applied delta ||`` and is nonzero. A
    zero means the read moved after the write, which is a real bug and is
    refused here.
    """
    checks = checks or Checklist("one_pass_budget")
    out: Dict[int, Dict[str, Any]] = {}
    for block in blocks:
        successor = block + 1
        separations = [row.read_before_write.get(successor) for row in rows]
        present = [v for v in separations if v is not None]
        checks.require(
            f"block_{block}_read_before_write_evidence_exists",
            len(present) == len(rows),
            f"block {block}: {len(rows) - len(present)} row(s) carry no "
            f"read-before-write evidence; the invariant cannot be waived by "
            f"failing to record it")
        collapsed = [i for i, v in enumerate(present) if v <= 0.0]
        checks.require(
            f"block_{block}_budget_is_read_before_the_activation_is_edited",
            not collapsed,
            f"block {block}: on {len(collapsed)} row(s) the snapshot taken for "
            f"block {successor}'s budget is identical to the activation AFTER "
            f"block {block}'s edit. The read moved after the write, so the "
            f"budget being reported is the edited one and every number in this "
            f"run is measuring the wrong thing.")
        out[block] = {
            "successor": successor,
            "rows": len(present),
            "min_separation": float(min(present)) if present else 0.0,
            "max_separation": float(max(present)) if present else 0.0,
            "statement": f"the snapshot for block {successor}'s budget differs "
                         f"from the post-edit activation at that position by "
                         f"the applied delta, on every row, within one forward "
                         f"pass",
        }
    return out


def decide(summaries: Mapping[int, Mapping[str, Any]],
           tolerance: float = TOLERANCE) -> Dict[str, Any]:
    """PASS only if every measured block is inside the frozen tolerance."""
    failing = sorted(int(b) for b, s in summaries.items()
                     if s.get("measured") and not s.get("within_tolerance"))
    passed = not failing
    return {
        "tolerance": float(tolerance),
        "blocks_measured": sorted(int(b) for b in summaries),
        "blocks_exceeding_tolerance": failing,
        "decision": "PASS" if passed else "BLOCKED",
        "statement": (
            f"every measured block is within {tolerance:g} relative; the "
            f"one-pass budget reproduces the validated two-pass budget"
            if passed else
            f"block(s) {failing} exceed {tolerance:g} relative. The one-pass "
            f"budget is not the validated budget at those sites. The actuator "
            f"is not redefined to make this pass: either the run adopts a "
            f"convention that reproduces it exactly, as a new versioned "
            f"specification, or the deviation is accepted deliberately and on "
            f"the record."),
    }


# ------------------------------------------------------------------ the hooks
class OnePassBudgetRecorder:
    """Steered prefill hooks that record what each budget would have been.

    Installed on historical blocks ``l - 1`` for every actuator block ``l``, so
    each block's budget is the norm of the tensor produced immediately before
    it -- snapshotted, and only then written to.

    Deliberately separate from ``hooks.PatchController``: that controller
    applies plans built from an earlier capture, which is the two-pass
    convention. This one builds its plan from what the running pass has in
    hand, which is the question.
    """

    def __init__(self, model: Any, directions: Mapping[int, Any],
                 blocks: Sequence[int] = ACTUATOR_BLOCKS,
                 alpha: float = ALPHA_R, g_r: float = WORST_CASE_G_R) -> None:
        if not 0.0 < g_r <= 1.0:
            raise BudgetEquivalenceError(
                f"g_R is {g_r}; this check runs the worst case g_R = 1 and a "
                f"gate is in (0, 1]")
        missing = [b for b in blocks if b not in directions]
        if missing:
            raise BudgetEquivalenceError(
                f"no direction for historical block(s) {missing}")
        self.blocks = tuple(int(b) for b in blocks)
        self.alpha = float(alpha)
        self.g_r = float(g_r)
        self.strength = self.alpha * self.g_r
        self.directions = {int(b): directions[b] for b in self.blocks}
        modules = resolve_modules(model)
        self.model = model
        self.layers = modules["layers"]
        self.handles: List[Any] = []
        self.armed = False
        self.forward_index = -1
        self.reference: Dict[int, Any] = {}
        self.one_pass: Dict[int, float] = {}
        self.runtime_input: Dict[int, float] = {}
        self.applied: Dict[int, float] = {}
        self.read_before_write: Dict[int, float] = {}
        self.prefill_tokens = 0
        #: Forward passes seen beyond the prefill while armed. A decode step
        #: would be one. This stage generates nothing, so it must stay 0, and
        #: it is counted rather than assumed.
        self.decode_forwards = 0
        self._install()

    # -- lifecycle ---------------------------------------------------------
    def _install(self) -> None:
        def count_forward(module: Any, args: Any) -> None:
            self.forward_index += 1

        self.handles.append(self.model.register_forward_pre_hook(count_forward))

        # A hook on block b sees block b's output = state b + 1, which is the
        # budget reference for an edit at block b + 1.
        hooked = sorted({b - 1 for b in self.blocks} | set(self.blocks))
        for block in hooked:
            if not 0 <= block < len(self.layers):
                raise BudgetEquivalenceError(
                    f"historical block {block} is outside the model's "
                    f"{len(self.layers)} decoder blocks")

            def hook(module: Any, args: Any, output: Any,
                     block: int = block) -> Any:
                return self._on_block_output(block, output)

            self.handles.append(
                self.layers[block].register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "OnePassBudgetRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def reset(self) -> None:
        self.forward_index = -1
        self.reference.clear()
        self.one_pass.clear()
        self.runtime_input.clear()
        self.applied.clear()
        self.read_before_write.clear()
        self.prefill_tokens = 0

    def arm(self) -> None:
        """Begin one steered prefill. Everything from the previous one is dropped.

        Arming rather than re-installing: the hooks are registered once, in the
        constructor, so the *order* in which they fire is fixed for the whole
        run. Tearing them down and rebuilding them per row would put that
        ordering -- which is the entire subject of this check -- at the mercy of
        registration order on every prompt.
        """
        self.reset()
        self.armed = True

    def disarm(self) -> None:
        """Stand down: the hooks stay installed and do nothing.

        Used for the unsteered reference pass, which must see the model exactly
        as the validated two-pass convention saw it.
        """
        self.armed = False

    # -- the pass ----------------------------------------------------------
    def _on_block_output(self, block: int, output: Any) -> Any:
        if not self.armed:
            return output
        # Decode steps are never touched, and this stage generates no token at
        # all -- so a forward beyond the prefill is not merely skipped, it is
        # counted and reported, because its existence would mean this run
        # decoded something it says it did not.
        if self.forward_index != PREFILL_FORWARD_INDEX:
            self.decode_forwards += 1
            return output
        tensor = first_tensor(output)
        if tensor is None or tensor.dim() != 3:
            return output
        if tensor.shape[0] != 1:
            raise BudgetEquivalenceError(
                f"block {block}: batch size {tensor.shape[0]}; this check runs "
                f"one prompt at a time so the final row is the final prompt "
                f"token and never a pad token")
        position = final_prompt_token_index(int(tensor.shape[1]))
        self.prefill_tokens = int(tensor.shape[1])

        # This block's output is state block+1, which is the budget reference
        # for an edit at block+1. Snapshot BEFORE any write to it.
        successor = block + 1
        snapshot = None
        if successor in self.blocks:
            snapshot = tensor[0, position].detach().clone()
            self.reference[successor] = snapshot
            self.one_pass[successor] = vector_norm(snapshot)

        if block not in self.blocks:
            return output

        reference = self.reference.get(block)
        if reference is None:
            raise BudgetEquivalenceError(
                f"block {block} was reached without its budget reference; the "
                f"hook on block {block - 1} did not fire first, so the hook "
                f"ordering this check depends on does not hold")

        # The validated actuator's own arithmetic, not a restatement of it.
        from .steering import build_steering_plan
        plan = build_steering_plan(
            state=block_output_state(block),
            direction=self.directions[block],
            strength=self.strength,
            reference_vector=reference,
            budget_reference="block_input")
        if plan.delta is None:
            raise BudgetEquivalenceError(
                f"block {block}: the validated actuator produced no delta at "
                f"strength {self.strength}")
        changed = apply_delta_at_position(tensor, position, plan.delta)
        self.applied[block] = float(plan.applied_delta_norm)

        # Block block+1's ACTUAL runtime input, which carries this edit too.
        # Recorded as the stricter variant; the declared convention is the
        # pre-write snapshot above.
        if successor in self.blocks and snapshot is not None:
            self.runtime_input[successor] = vector_norm(changed[0, position])
            # The same-forward proof that the snapshot above was taken before
            # this write: the two vectors differ by exactly the applied delta.
            # Had the snapshot moved after the write they would be one vector
            # and this separation would be zero. Same pass, same device, same
            # dtype -- no reduction order can move it.
            separation = vector_norm(changed[0, position] - snapshot)
            self.read_before_write[successor] = separation
            if separation <= 0.0:
                raise BudgetEquivalenceError(
                    f"block {block}: the snapshot taken for block {successor}'s "
                    f"budget is identical to the activation after this block's "
                    f"edit. The budget is being read after the write, so it is "
                    f"the edited norm and not the validated one.")
        return replace_first_tensor(output, changed)
