"""Installing, using and removing the activation hooks.

One controller owns every hook a run installs. It has three modes:

    off       the model runs untouched -- the baseline condition
    capture   record one vector per requested state, at the final prompt token
    patch     add or substitute at the final prompt token, then get out of the way

Three properties this file exists to guarantee.

**Only the prefill forward is touched.** ``generate`` calls the model once for
the prompt and once per generated token. The hidden state that decides the first
answer token is the last row of the *prefill* forward, so a counter on the
top-level model marks forward 0 and every hook ignores the rest. Without that,
a hook keyed on "the last row" would edit every generated token in turn, which
is a different experiment.

**Nothing carries between examples.** Captures, plans, counters and hit records
are cleared by :meth:`PatchController.reset_run`, and a run must name the
example it is running; a plan built for one target cannot be applied while
another is in flight. The hooks themselves are stateless closures over the
controller.

**Hooks come off.** Every handle is kept and :meth:`close` removes all of them,
so a controller cannot leave a model quietly modified for whatever runs next.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch

    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from .intervention import (
    BLOCK_OUTPUT_SITE,
    EMBEDDING_SITE,
    InterventionError,
    InterventionPlan,
    apply_delta_at_position,
    capture_states_for,
    final_prompt_token_index,
    n_states_for,
    replace_at_position,
    state_to_site,
)

PREFILL_FORWARD_INDEX = 0
MODES = ("off", "capture", "patch")


def first_tensor(output: Any) -> Optional[Any]:
    """The hidden-state tensor in a module output that may be a tuple."""
    if torch is not None and torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch is not None \
            and torch.is_tensor(output[0]):
        return output[0]
    return None


def replace_first_tensor(output: Any, tensor: Any) -> Any:
    """Put a modified hidden state back into whatever container it arrived in."""
    if torch is not None and torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor,) + tuple(output[1:])
    if isinstance(output, list):
        return [tensor] + list(output[1:])
    raise InterventionError(
        f"cannot replace the hidden state in a module output of type "
        f"{type(output)}")


def clone_cpu(tensor: Any) -> Any:
    """A detached float32 CPU copy, as the historical capture stored them."""
    return tensor.detach().float().cpu().clone()


def resolve_modules(model: Any) -> Dict[str, Any]:
    """The embedding module and the decoder block list of a HF causal LM."""
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "layers"):
        raise InterventionError(
            "expected a Hugging Face causal LM exposing model.layers")
    embed = getattr(base, "embed_tokens", None)
    if embed is None:
        raise InterventionError(
            "expected the model to expose model.embed_tokens, whose output is "
            "state 0")
    return {"base": base, "embed": embed, "layers": base.layers}


@dataclass
class RunState:
    """Everything one example's forward pass is allowed to know."""

    mode: str = "off"
    example_id: Optional[str] = None
    capture_name: Optional[str] = None
    plans: Dict[int, InterventionPlan] = field(default_factory=dict)
    forward_index: int = -1
    prefill_sequence_length: Optional[int] = None
    hits: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    applied: Dict[int, float] = field(default_factory=dict)


class PatchController:
    """Hooks for a set of states, on one model, for one example at a time."""

    def __init__(self, model: Any, states: Sequence[int],
                 n_layers: Optional[int] = None,
                 budget_reference: str = "block_input") -> None:
        if torch is None:  # pragma: no cover - exercised only without torch
            raise InterventionError(
                f"PatchController needs torch, which failed to import: "
                f"{_TORCH_IMPORT_ERROR}")
        modules = resolve_modules(model)
        self.model = model
        self.layers = modules["layers"]
        self.embed = modules["embed"]
        self.n_layers = int(n_layers if n_layers is not None else len(self.layers))
        if self.n_layers != len(self.layers):
            raise InterventionError(
                f"the model has {len(self.layers)} decoder blocks but "
                f"{self.n_layers} were declared; refusing to run against a "
                f"model that is not the pinned one")
        self.states = [int(s) for s in states]
        for state in self.states:
            state_to_site(state, self.n_layers)          # validates the range
        self.budget_reference = budget_reference
        self.capture_states = capture_states_for(self.states, budget_reference)
        self.handles: List[Any] = []
        self.captures: Dict[str, Dict[int, Any]] = {}
        self.run = RunState()
        self._install()

    # ------------------------------------------------------------- lifecycle
    def _install(self) -> None:
        # One counter for the whole model: a pre-hook on the top-level module
        # fires once per forward, before any submodule, so every state hook in
        # that forward -- including the embedding -- reads the same index.
        def count_forward(module: Any, args: Any) -> None:
            self.run.forward_index += 1

        self.handles.append(self.model.register_forward_pre_hook(count_forward))

        hooked = sorted(set(self.states) | set(self.capture_states))
        for state in hooked:
            site = state_to_site(state, self.n_layers)
            module = (self.embed if site.kind == EMBEDDING_SITE
                      else self.layers[site.block_index])

            def state_hook(module: Any, args: Any, output: Any,
                           state: int = state) -> Any:
                return self._on_state(state, output)

            self.handles.append(module.register_forward_hook(state_hook))

    def close(self) -> None:
        """Remove every hook. Safe to call more than once."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "PatchController":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def installed(self) -> bool:
        return bool(self.handles)

    # ----------------------------------------------------------------- a run
    def reset_run(self, mode: str = "off", example_id: Optional[str] = None,
                  capture_name: Optional[str] = None,
                  plans: Optional[Dict[int, InterventionPlan]] = None) -> None:
        """Begin one generation. Everything from the previous one is dropped."""
        if mode not in MODES:
            raise InterventionError(f"unknown mode {mode!r}; valid: {list(MODES)}")
        if mode == "capture" and not capture_name:
            raise InterventionError("capture mode needs a capture name")
        if mode == "patch" and not plans:
            raise InterventionError("patch mode needs at least one plan")
        if mode != "off" and not example_id:
            raise InterventionError(
                "every capture or patch must name the example it belongs to, so "
                "a plan built for one target cannot be applied to another")
        for state in (plans or {}):
            if state not in self.states:
                raise InterventionError(
                    f"a plan was given for state {state}, which this controller "
                    f"did not install a hook for (installed: {self.states})")
        self.run = RunState(mode=mode, example_id=example_id,
                            capture_name=capture_name, plans=dict(plans or {}))
        if capture_name is not None:
            self.captures[capture_name] = {}

    def capture_for(self, capture_name: str) -> Dict[int, Any]:
        if capture_name not in self.captures:
            raise InterventionError(
                f"no capture named {capture_name!r}; captured: "
                f"{sorted(self.captures)}")
        return self.captures[capture_name]

    def drop_capture(self, capture_name: str) -> None:
        self.captures.pop(capture_name, None)

    def clear_captures(self) -> None:
        self.captures.clear()

    def verify_applied(self) -> None:
        """Confirm every plan that should have fired did, exactly once.

        A plan that silently never applied would be recorded as an intervention
        and be identical to the baseline, which is the most misleading failure
        this pipeline could have.
        """
        for state, plan in self.run.plans.items():
            expected = 1 if plan.applies_anything else 0
            actual = int(self.run.hits.get(state, 0))
            if actual != expected:
                raise InterventionError(
                    f"state {state}: the plan applied {actual} time(s), expected "
                    f"{expected} (status {plan.status!r}). A patched run that "
                    f"never patched must not be recorded as patched.")

    # ------------------------------------------------------------- the hooks
    def _on_state(self, state: int, output: Any) -> Any:
        if self.run.mode == "off":
            return output
        if self.run.forward_index != PREFILL_FORWARD_INDEX:
            return output          # decode steps are never touched
        tensor = first_tensor(output)
        if tensor is None or tensor.dim() != 3:
            return output
        if tensor.shape[0] != 1:
            raise InterventionError(
                f"state {state}: batch size {tensor.shape[0]}, but this stage "
                f"runs one prompt at a time so the final row is the final "
                f"prompt token and never a pad token")

        position = final_prompt_token_index(int(tensor.shape[1]))
        if self.run.prefill_sequence_length is None:
            self.run.prefill_sequence_length = int(tensor.shape[1])

        if self.run.mode == "capture":
            if state in self.capture_states:
                self.captures[self.run.capture_name][state] = clone_cpu(
                    tensor[0, position])
            return output

        plan = self.run.plans.get(state)
        if plan is None or not plan.applies_anything:
            return output
        if plan.replacement is not None:
            changed = replace_at_position(tensor, position, plan.replacement)
            self.run.applied[state] = float(plan.applied_delta_norm)
        else:
            changed = apply_delta_at_position(tensor, position, plan.delta)
            self.run.applied[state] = float(plan.applied_delta_norm)
        self.run.hits[state] += 1
        return replace_first_tensor(output, changed)
