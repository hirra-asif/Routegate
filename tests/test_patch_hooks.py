"""Hooks: registration, removal, position, prefill-only, and contamination.

The model here is a toy stack of three "decoder blocks" over a small embedding.
It has the two attributes the real hook installer needs -- ``model.layers`` and
``model.embed_tokens`` -- and its blocks return a one-tuple exactly as a Hugging
Face decoder layer does, so the tuple-unwrapping path is exercised rather than
assumed.

No Qwen, no weights, no GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from routegate.hooks import (
    PatchController,
    clone_cpu,
    first_tensor,
    replace_first_tensor,
    resolve_modules,
)
from routegate.intervention import InterventionError, build_plan

HIDDEN = 4
N_BLOCKS = 3
VOCAB = 32


class ToyBlock(nn.Module):
    """A decoder block: adds its own constant and returns a one-tuple."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.register_buffer("bias", torch.full((HIDDEN,), float(index + 1)))

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + self.bias,)


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([ToyBlock(i) for i in range(N_BLOCKS)])

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class ToyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBase()

    def forward(self, input_ids):
        return self.model(input_ids)


@pytest.fixture
def toy():
    torch.manual_seed(20260806)
    model = ToyCausalLM()
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
            / 100.0)
    return model


def ids(*tokens):
    return torch.tensor([list(tokens)], dtype=torch.long)


# ------------------------------------------------------- module resolution
def test_the_installer_finds_the_embedding_and_the_blocks(toy):
    modules = resolve_modules(toy)
    assert modules["embed"] is toy.model.embed_tokens
    assert len(modules["layers"]) == N_BLOCKS


def test_a_model_without_layers_is_refused():
    with pytest.raises(InterventionError, match="model.layers"):
        resolve_modules(nn.Linear(2, 2))


def test_a_declared_block_count_that_disagrees_is_refused(toy):
    with pytest.raises(InterventionError, match="decoder blocks"):
        PatchController(toy, states=[1], n_layers=48)


# ------------------------------------------- 1. registration and removal
def test_hooks_are_registered_and_removed(toy):
    controller = PatchController(toy, states=[2], n_layers=N_BLOCKS)
    # one forward counter, plus one hook per hooked state (2 and its reference 1)
    assert len(controller.handles) == 3
    assert controller.installed

    controller.close()
    assert not controller.installed
    assert controller.handles == []


def test_close_is_idempotent_and_leaves_the_model_untouched(toy):
    before = toy(ids(1, 2, 3)).clone()
    controller = PatchController(toy, states=[1], n_layers=N_BLOCKS)
    controller.close()
    controller.close()
    assert torch.allclose(toy(ids(1, 2, 3)), before)


def test_the_controller_is_a_context_manager(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        assert controller.installed
    assert not controller.installed


def test_an_uninstalled_state_cannot_be_planned_for(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        plan = build_plan(state=3, intervention_type="donor_direction",
                          strength=0.2, target_state_vector=torch.ones(HIDDEN),
                          donor_state_vector=torch.zeros(HIDDEN))
        with pytest.raises(InterventionError, match="did not install"):
            controller.reset_run(mode="patch", example_id="t1", plans={3: plan})


# ---------------------------------------------------------- 2. capturing
def test_capture_records_the_final_prompt_token_of_every_needed_state(toy):
    with PatchController(toy, states=[2], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="capture", example_id="t1",
                             capture_name="target")
        toy(ids(5, 6, 7, 8))
        capture = controller.capture_for("target")

    assert sorted(capture) == [1, 2]              # state 2 and its reference
    for state, vector in capture.items():
        assert vector.shape == (HIDDEN,)
        assert vector.dtype == torch.float32


def test_the_captured_vector_is_the_last_row_not_the_first(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="capture", example_id="t1",
                             capture_name="target")
        token_ids = ids(3, 9)
        toy(token_ids)
        capture = controller.capture_for("target")

    embedded = toy.model.embed_tokens(token_ids)
    expected_state_0 = embedded[0, -1]
    expected_state_1 = (embedded + toy.model.layers[0].bias)[0, -1]
    assert torch.allclose(capture[0], expected_state_0)
    assert torch.allclose(capture[1], expected_state_1)
    assert not torch.allclose(capture[0], embedded[0, 0])


def test_state_zero_is_the_embedding_output(toy):
    with PatchController(toy, states=[0], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="capture", example_id="t1",
                             capture_name="target")
        token_ids = ids(2, 4, 6)
        toy(token_ids)
        capture = controller.capture_for("target")

    assert sorted(capture) == [0]
    assert torch.allclose(capture[0], toy.model.embed_tokens(token_ids)[0, -1])


def test_capture_requires_a_name_and_an_example(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        with pytest.raises(InterventionError, match="capture name"):
            controller.reset_run(mode="capture", example_id="t1")
        with pytest.raises(InterventionError, match="name the example"):
            controller.reset_run(mode="capture", capture_name="target")


# --------------------------------------- 3. donor/target alignment by state
def test_donor_and_target_captures_align_by_state_not_by_length(toy):
    with PatchController(toy, states=[2], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="capture", example_id="target_row",
                             capture_name="target")
        toy(ids(1, 2, 3, 4, 5, 6))                       # a six-token prompt
        target = dict(controller.capture_for("target"))

        controller.reset_run(mode="capture", example_id="donor_row",
                             capture_name="donor")
        donor_ids = ids(7, 8)                            # a two-token prompt
        toy(donor_ids)
        donor = dict(controller.capture_for("donor"))

    assert sorted(target) == sorted(donor) == [1, 2]
    for state in (1, 2):
        assert target[state].shape == donor[state].shape == (HIDDEN,)
    # the donor's vector is its own last row, not the target's position
    embedded = toy.model.embed_tokens(donor_ids)
    expected = (embedded + toy.model.layers[0].bias
                + toy.model.layers[1].bias)[0, -1]
    assert torch.allclose(donor[2], expected)


# ------------------------------------------------------------ 4. patching
def test_a_patch_changes_exactly_one_state_at_exactly_one_position(toy):
    token_ids = ids(1, 2, 3, 4)
    baseline = toy(token_ids).clone()

    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="capture", example_id="t1",
                             capture_name="target")
        toy(token_ids)
        capture = dict(controller.capture_for("target"))

        plan = build_plan(state=1, intervention_type="donor_direction",
                          strength=0.5,
                          target_state_vector=capture[1],
                          donor_state_vector=capture[1] + torch.tensor(
                              [1.0, 0.0, 0.0, 0.0]),
                          reference_vector=capture[0])
        assert plan.status == "ok"

        controller.reset_run(mode="patch", example_id="t1", plans={1: plan})
        patched = toy(token_ids)
        controller.verify_applied()

    difference = (patched - baseline)[0]
    assert torch.allclose(difference[:3], torch.zeros(3, HIDDEN), atol=1e-6)
    assert torch.allclose(difference[3], plan.delta, atol=1e-6)


def test_a_zero_delta_plan_applies_nothing(toy):
    token_ids = ids(4, 5, 6)
    baseline = toy(token_ids).clone()

    with PatchController(toy, states=[2], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="capture", example_id="t1",
                             capture_name="target")
        toy(token_ids)
        capture = dict(controller.capture_for("target"))

        plan = build_plan(state=2, intervention_type="self_patch", strength=0.2,
                          target_state_vector=capture[2],
                          reference_vector=capture[1])
        assert plan.status == "zero_delta"

        controller.reset_run(mode="patch", example_id="t1", plans={2: plan})
        patched = toy(token_ids)
        controller.verify_applied()          # zero hits expected, and allowed

    assert torch.allclose(patched, baseline)
    assert controller.run.hits.get(2, 0) == 0


def test_only_the_prefill_forward_is_patched(toy):
    """Decode steps must run untouched, or the edit lands on every token."""
    token_ids = ids(2, 3, 4)
    baseline = toy(token_ids).clone()

    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        plan = build_plan(state=1, intervention_type="donor_direction",
                          strength=1.0,
                          target_state_vector=torch.zeros(HIDDEN),
                          donor_state_vector=torch.tensor([0.0, 5.0, 0.0, 0.0]),
                          reference_vector=torch.tensor([1.0, 0.0, 0.0, 0.0]))
        controller.reset_run(mode="patch", example_id="t1", plans={1: plan})

        prefill = toy(token_ids)             # forward 0: patched
        decode = toy(ids(9))                 # forward 1: untouched
        decode_again = toy(ids(10))          # forward 2: untouched

        assert controller.run.hits[1] == 1
        assert not torch.allclose(prefill, baseline)
        assert torch.allclose(decode, toy_unhooked_forward(toy, ids(9)))
        assert torch.allclose(decode_again, toy_unhooked_forward(toy, ids(10)))


def toy_unhooked_forward(model, token_ids):
    """The same arithmetic the toy model performs, computed without hooks."""
    hidden = model.model.embed_tokens(token_ids)
    for layer in model.model.layers:
        hidden = hidden + layer.bias
    return hidden


def test_a_patch_that_never_fires_is_a_failure(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        plan = build_plan(state=1, intervention_type="donor_direction",
                          strength=0.5,
                          target_state_vector=torch.zeros(HIDDEN),
                          donor_state_vector=torch.ones(HIDDEN),
                          reference_vector=torch.ones(HIDDEN))
        controller.reset_run(mode="patch", example_id="t1", plans={1: plan})
        # the model is never run, so the plan cannot have applied
        with pytest.raises(InterventionError, match="never patched"):
            controller.verify_applied()


def test_patch_mode_requires_a_plan_and_an_example(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        with pytest.raises(InterventionError, match="at least one plan"):
            controller.reset_run(mode="patch", example_id="t1")


def test_an_unknown_mode_is_refused(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        with pytest.raises(InterventionError, match="unknown mode"):
            controller.reset_run(mode="meddle", example_id="t1")


# ------------------------------------------ 5. no cross-example contamination
def test_a_new_example_inherits_no_plan_from_the_previous_one(toy):
    token_ids = ids(1, 2, 3)
    baseline = toy(token_ids).clone()

    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        plan = build_plan(state=1, intervention_type="donor_direction",
                          strength=1.0,
                          target_state_vector=torch.zeros(HIDDEN),
                          donor_state_vector=torch.tensor([3.0, 0.0, 0.0, 0.0]),
                          reference_vector=torch.ones(HIDDEN))
        controller.reset_run(mode="patch", example_id="first", plans={1: plan})
        first = toy(token_ids)
        assert not torch.allclose(first, baseline)

        controller.reset_run(mode="off", example_id="second")
        second = toy(token_ids)

    assert torch.allclose(second, baseline)
    assert controller.run.plans == {}
    assert dict(controller.run.hits) == {}


def test_two_examples_get_their_own_deltas(toy):
    token_ids = ids(5, 6)
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        outputs = []
        for name, direction in (("a", [4.0, 0.0, 0.0, 0.0]),
                                ("b", [0.0, 4.0, 0.0, 0.0])):
            plan = build_plan(state=1, intervention_type="donor_direction",
                              strength=1.0,
                              target_state_vector=torch.zeros(HIDDEN),
                              donor_state_vector=torch.tensor(direction),
                              reference_vector=torch.ones(HIDDEN))
            controller.reset_run(mode="patch", example_id=name, plans={1: plan})
            outputs.append(toy(token_ids).clone())
            assert controller.run.hits[1] == 1

    assert not torch.allclose(outputs[0], outputs[1])


def test_the_forward_counter_restarts_for_every_example(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="off", example_id="first")
        toy(ids(1, 2))
        toy(ids(3))
        assert controller.run.forward_index == 1

        controller.reset_run(mode="off", example_id="second")
        assert controller.run.forward_index == -1
        toy(ids(4, 5))
        assert controller.run.forward_index == 0


def test_captures_are_kept_per_name_and_can_be_dropped(toy):
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="capture", example_id="a", capture_name="a")
        toy(ids(1, 2))
        controller.reset_run(mode="capture", example_id="b", capture_name="b")
        toy(ids(3, 4))

        assert sorted(controller.captures) == ["a", "b"]
        assert not torch.allclose(controller.capture_for("a")[1],
                                  controller.capture_for("b")[1])

        controller.drop_capture("a")
        assert sorted(controller.captures) == ["b"]
        with pytest.raises(InterventionError, match="no capture named"):
            controller.capture_for("a")

        controller.clear_captures()
        assert controller.captures == {}


def test_off_mode_captures_nothing_and_patches_nothing(toy):
    token_ids = ids(1, 2, 3)
    baseline = toy(token_ids).clone()
    with PatchController(toy, states=[1], n_layers=N_BLOCKS) as controller:
        controller.reset_run(mode="off")
        assert torch.allclose(toy(token_ids), baseline)
        assert controller.captures == {}


# --------------------------------------------------------------- utilities
def test_first_tensor_unwraps_what_a_decoder_layer_returns():
    tensor = torch.zeros(1, 2, 3)
    assert first_tensor(tensor) is tensor
    assert first_tensor((tensor, "cache")) is tensor
    assert first_tensor([tensor, None]) is tensor
    assert first_tensor("not a tensor") is None
    assert first_tensor(()) is None


def test_replace_first_tensor_preserves_the_container():
    tensor, new = torch.zeros(1, 2, 3), torch.ones(1, 2, 3)
    assert replace_first_tensor(tensor, new) is new
    replaced = replace_first_tensor((tensor, "cache"), new)
    assert isinstance(replaced, tuple) and replaced[0] is new and replaced[1] == "cache"
    replaced_list = replace_first_tensor([tensor, "cache"], new)
    assert isinstance(replaced_list, list) and replaced_list[0] is new
    with pytest.raises(InterventionError, match="cannot replace"):
        replace_first_tensor({"hidden": tensor}, new)


def test_clone_cpu_detaches_to_float32():
    source = torch.ones(3, dtype=torch.bfloat16, requires_grad=False)
    copied = clone_cpu(source)
    assert copied.dtype == torch.float32
    assert copied.device.type == "cpu"
    copied[0] = 99.0
    assert source[0].item() == 1.0
